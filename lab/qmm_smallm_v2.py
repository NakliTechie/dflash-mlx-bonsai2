"""v2: threadgroup of 8 SIMD-groups = 8 output rows; K processed in 512-wide chunks; the M x 512 activation
chunk is staged in threadgroup memory once per chunk and shared by the 8 rows; each lane owns exactly one
packed word (16 k) per chunk, dequantizes it once, and dots it against all M rows."""
import sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, 'lab')
from qmm_smallm_kernel import qmm_smallm as qmm_v1

SRC = r"""
    const uint tid   = thread_index_in_threadgroup;      // 0..255
    const uint lane  = thread_index_in_simdgroup;        // 0..31
    const uint sg    = simdgroup_index_in_threadgroup;   // 0..7
    const uint n     = threadgroup_position_in_grid.x * 8 + sg;
    threadgroup half xs[MMAX * 512];
    const uint nchunks = K / 512;
    const uint groups  = K / 128;
    float acc[MMAX];
    for (uint m = 0; m < MMAX; ++m) acc[m] = 0.0f;
    const device uint* wrow = w + (size_t)n * (K / 16);
    const device T* srow = scales + (size_t)n * groups;
    const device T* brow = biases + (size_t)n * groups;
    for (uint c = 0; c < nchunks; ++c) {
        // stage x[:, c*512 : c*512+512] -> xs (M*512 halfs, 256 threads)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < M * 512; idx += 256) {
            const uint m = idx / 512, kk = idx % 512;
            xs[idx] = half(x[(size_t)m * K + c * 512 + kk]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (n < N) {
            const uint word = c * 32 + lane;
            const uint g = word / 8;
            const float s = float(srow[g]);
            const float b = float(brow[g]);
            const uint packed = wrow[word];
            float wv[16];
            #pragma unroll
            for (uint i = 0; i < 16; ++i) wv[i] = float((packed >> (2u * i)) & 3u) * s + b;
            const threadgroup half* xl = xs + lane * 16;
            for (uint m = 0; m < M; ++m) {
                const threadgroup half* xr = xl + m * 512;
                float p = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 16; ++i) p += wv[i] * float(xr[i]);
                acc[m] += p;
            }
        }
    }
    if (n < N) {
        for (uint m = 0; m < M; ++m) {
            const float r = simd_sum(acc[m]);
            if (lane == 0) out[(size_t)m * N + n] = T(r);
        }
    }
"""
_k = mx.fast.metal_kernel(name="qmm_smallm_v2", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC)

def qmm_v2(x2d, w, scales, biases, mmax=16):
    M, K = x2d.shape; N = w.shape[0]
    assert M <= mmax and K % 512 == 0 and N % 8 == 0
    return _k(inputs=[x2d, w, scales, biases], template=[("T", x2d.dtype), ("M", M), ("MMAX", mmax), ("N", N), ("K", K)],
              grid=(N // 8 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[x2d.dtype])[0]

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors')
    mod = lambda p: (W[p + '.weight'], W[p + '.scales'], W[p + '.biases'])
    tests = {'mlp.gate_proj L0 (17408x5120)': mod('language_model.model.layers.0.mlp.gate_proj'),
             'mlp.down_proj L0 (5120x17408)': mod('language_model.model.layers.0.mlp.down_proj'),
             'gdn.in_proj_qkv L0 (10240x5120)': mod('language_model.model.layers.0.linear_attn.in_proj_qkv'),
             'lm_head (248320x5120)': mod('language_model.lm_head')}
    def bench(fn, reps=25):
        for _ in range(3): mx.eval(fn())
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
        return st.median(ts)
    for name, (w, s, b) in tests.items():
        mx.eval(w, s, b); N, K = w.shape[0], w.shape[1] * 16
        print(name)
        for M in (1, 2, 4, 8, 16):
            x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16)
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2)
            got = qmm_v2(x, w, s, b); mx.eval(ref, got)
            err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
            t_ref = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
            t_v1 = bench(lambda: qmm_v1(x, w, s, b)); t_v2 = bench(lambda: qmm_v2(x, w, s, b))
            print(f'   M={M:2d}: stock {t_ref:6.2f}  v1 {t_v1:6.2f}  v2 {t_v2:6.2f} ms  (v2 {t_ref/t_v2:4.2f}x stock)  max|Δ| {err:.4f}')
