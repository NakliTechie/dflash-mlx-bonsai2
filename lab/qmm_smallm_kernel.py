"""v1 small-M batched 2-bit affine qmm Metal kernel (MLX layout: weight uint32 [N, K/16], 16 x 2-bit per word,
scales/biases fp16 [N, K/128]). One SIMD-group (32 lanes) per output row n; lanes stride over K in 128-wide
groups; each lane keeps M fp32 accumulators; simd_sum reduces. Reads each weight exactly once."""
import sys, time, statistics as st
import mlx.core as mx

SRC = r"""
    // grid: (N * 32, 1, 1); threadgroup (32*ROWS_PER_TG, 1, 1)
    const uint lane = thread_index_in_simdgroup;
    const uint n = thread_position_in_grid.x / 32;
    if (n >= N) return;
    const uint groups = K / 128;              // 2-bit: 8 words per 128-group
    float acc[MMAX];
    for (uint m = 0; m < MMAX; ++m) acc[m] = 0.0f;
    const device uint* wrow = w + (size_t)n * (K / 16);
    const device T* srow = scales + (size_t)n * groups;
    const device T* brow = biases + (size_t)n * groups;
    // lane handles words lane, lane+32, ... (each word = 16 consecutive k)
    for (uint word = lane; word < K / 16; word += 32) {
        const uint g = word / 8;              // 128-group index
        const float s = float(srow[g]);
        const float b = float(brow[g]);
        const uint packed = wrow[word];
        const uint k0 = word * 16;
        for (uint m = 0; m < M; ++m) {
            const device T* xrow = x + (size_t)m * K + k0;
            float partial = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                const float q = float((packed >> (2u * i)) & 3u);
                partial += (q * s + b) * float(xrow[i]);
            }
            acc[m] += partial;
        }
    }
    for (uint m = 0; m < M; ++m) {
        const float r = simd_sum(acc[m]);
        if (lane == 0) out[(size_t)m * N + n] = T(r);
    }
"""
_kernel = mx.fast.metal_kernel(name="qmm_smallm_v1", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC)

def qmm_smallm(x2d, w, scales, biases, mmax=16):
    M, K = x2d.shape; N = w.shape[0]
    assert M <= mmax and K % 128 == 0
    return _kernel(inputs=[x2d, w, scales, biases], template=[("T", x2d.dtype), ("M", M), ("MMAX", mmax), ("N", N), ("K", K)],
                   grid=(N * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[x2d.dtype])[0]

if __name__ == "__main__":
    sys.path.insert(0, 'lab')
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    # load only one module's tensors from the safetensors (no full model): gate_proj of layer 0 + lm_head
    W = mx.load(PACK + '/model.safetensors')
    def mod(prefix):
        return W[prefix + '.weight'], W[prefix + '.scales'], W[prefix + '.biases']
    tests = {'mlp.gate_proj L0': mod('language_model.model.layers.0.mlp.gate_proj'),
             'gdn.in_proj_qkv L0': mod('language_model.model.layers.0.linear_attn.in_proj_qkv'),
             'lm_head': mod('language_model.lm_head')}
    def bench(fn, reps=20):
        for _ in range(3): mx.eval(fn())
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
        return st.median(ts)
    for name, (w, s, b) in tests.items():
        mx.eval(w, s, b); N, K = w.shape[0], w.shape[1] * 16
        print(f'{name}: N={N} K={K}')
        for M in (1, 2, 4, 8, 16):
            x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16)
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2)
            got = qmm_smallm(x, w, s, b); mx.eval(ref, got)
            err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max()); scale = float(mx.abs(ref).max())
            t_ref = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
            t_new = bench(lambda: qmm_smallm(x, w, s, b))
            print(f'   M={M:2d}: stock {t_ref:6.2f} ms  v1 {t_new:6.2f} ms  ({t_ref/t_new:4.2f}x)   max|Δ| {err:.4f} of {scale:.1f}')
