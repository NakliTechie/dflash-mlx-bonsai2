"""v3: v1's structure (one SIMD-group streams one output row, lanes stride packed words) with the 16-weight
dequant hoisted out of the M loop into four float4 registers, activations loaded as half4 vectors and dotted
with float4 FMAs, two words in flight per lane for ILP."""
import sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, 'lab')
from qmm_smallm_kernel import qmm_smallm as qmm_v1

SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint n = thread_position_in_grid.x / 32;
    if (n >= N) return;
    const uint words = K / 16;
    float acc[MMAX];
    for (uint m = 0; m < MMAX; ++m) acc[m] = 0.0f;
    const device uint* wrow = w + (size_t)n * words;
    const device T* srow = scales + (size_t)n * (K / 128);
    const device T* brow = biases + (size_t)n * (K / 128);
    for (uint word = lane; word < words; word += 32) {
        const uint g = word >> 3;
        const float s = float(srow[g]);
        const float b = float(brow[g]);
        const uint p = wrow[word];
        const float4 w0 = float4(float((p      ) & 3u), float((p >> 2u) & 3u), float((p >> 4u) & 3u), float((p >> 6u) & 3u)) * s + b;
        const float4 w1 = float4(float((p >> 8u) & 3u), float((p >> 10u) & 3u), float((p >> 12u) & 3u), float((p >> 14u) & 3u)) * s + b;
        const float4 w2 = float4(float((p >> 16u) & 3u), float((p >> 18u) & 3u), float((p >> 20u) & 3u), float((p >> 22u) & 3u)) * s + b;
        const float4 w3 = float4(float((p >> 24u) & 3u), float((p >> 26u) & 3u), float((p >> 28u) & 3u), float((p >> 30u) & 3u)) * s + b;
        const device T* xb = x + (size_t)word * 16;
        for (uint m = 0; m < M; ++m) {
            const device vec<T, 4>* xr = (const device vec<T, 4>*)(xb + (size_t)m * K);
            acc[m] += dot(w0, float4(xr[0])) + dot(w1, float4(xr[1])) + dot(w2, float4(xr[2])) + dot(w3, float4(xr[3]));
        }
    }
    for (uint m = 0; m < M; ++m) {
        const float r = simd_sum(acc[m]);
        if (lane == 0) out[(size_t)m * N + n] = T(r);
    }
"""
_k = mx.fast.metal_kernel(name="qmm_smallm_v3", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC)

def qmm_v3(x2d, w, scales, biases, mmax=16):
    M, K = x2d.shape; N = w.shape[0]
    assert M <= mmax and K % 128 == 0
    return _k(inputs=[x2d, w, scales, biases], template=[("T", x2d.dtype), ("M", M), ("MMAX", mmax), ("N", N), ("K", K)],
              grid=(N * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[x2d.dtype])[0]

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors')
    mod = lambda p: (W[p + '.weight'], W[p + '.scales'], W[p + '.biases'])
    tests = {'gate_proj (17408x5120)': mod('language_model.model.layers.0.mlp.gate_proj'),
             'down_proj (5120x17408)': mod('language_model.model.layers.0.mlp.down_proj'),
             'in_proj_qkv (10240x5120)': mod('language_model.model.layers.0.linear_attn.in_proj_qkv'),
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
            got = qmm_v3(x, w, s, b); mx.eval(ref, got)
            err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
            t_ref = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
            t_v1 = bench(lambda: qmm_v1(x, w, s, b)); t_v3 = bench(lambda: qmm_v3(x, w, s, b))
            print(f'   M={M:2d}: stock {t_ref:6.2f}  v1 {t_v1:6.2f}  v3 {t_v3:6.2f} ms  (v3 {t_ref/t_v3:4.2f}x stock, {t_v1/t_v3:4.2f}x v1)  max|Δ| {err:.4f}')
