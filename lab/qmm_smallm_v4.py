"""v4: M=8 batched 2-bit affine qmm using simdgroup_matrix 8x8 MMA. One SIMD-group owns 8 output rows (n0..n0+7);
per K-step of 16 the 32 lanes dequantize the 8 rows' packed words (4 values per lane) into a threadgroup half tile
[8 n][16 k], then two simdgroup_load(transposed) + two MMAs against the X tiles [8 m][8 k] read straight from device
memory. Accumulators are simdgroup_float8x8; output float32 [8, N]."""
import sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, 'lab')
from qmm_smallm_v3 import qmm_v3

HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;      // 0..3
    const uint n0   = (threadgroup_position_in_grid.x * 4 + sg) * 8;
    threadgroup half wt_all[4][8 * 16];
    threadgroup half* wt = wt_all[sg];
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint r = lane >> 2;            // row within the 8-row block
    const uint q = lane & 3;             // which 4-value quarter of the 16-value word
    const device uint* wrow = w + (size_t)(n0 + r) * words;
    const device T* srow = scales + (size_t)(n0 + r) * groups;
    const device T* brow = biases + (size_t)(n0 + r) * groups;
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
    for (uint word = 0; word < words; ++word) {
        const uint g = word >> 3;
        const float s = float(srow[g]);
        const float b = float(brow[g]);
        const uint p = wrow[word] >> (8u * q);
        threadgroup half* dst = wt + r * 16 + q * 4;
        dst[0] = half(float((p      ) & 3u) * s + b);
        dst[1] = half(float((p >> 2u) & 3u) * s + b);
        dst[2] = half(float((p >> 4u) & 3u) * s + b);
        dst[3] = half(float((p >> 6u) & 3u) * s + b);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 W0, W1, X0, X1;
        // W tiles: stored [n][k] with row stride 16; we need B = W^T (k x n) -> load transposed
        simdgroup_load(W0, wt, 16, ulong2(0, 0), true);
        simdgroup_load(W1, wt + 8, 16, ulong2(0, 0), true);
        simdgroup_load(X0, x + (size_t)word * 16, K);
        simdgroup_load(X1, x + (size_t)word * 16 + 8, K);
        simdgroup_multiply_accumulate(acc0, X0, W0, acc0);
        simdgroup_multiply_accumulate(acc1, X1, W1, acc1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // sum the two accumulators through threadgroup memory (no direct add for simdgroup_matrix of different origin)
    threadgroup float tmp_all[4][64];
    threadgroup float* tmp = tmp_all[sg];
    simdgroup_store(acc0, tmp, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float v0 = tmp[lane], v1 = tmp[lane + 32];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc1, tmp, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    v0 += tmp[lane]; v1 += tmp[lane + 32];
    // tmp is [m][n] row-major 8x8: index i -> m = i / 8, n = i % 8
    out[(size_t)(lane / 8) * N + n0 + (lane % 8)] = v0;
    out[(size_t)((lane + 32) / 8) * N + n0 + ((lane + 32) % 8)] = v1;
"""
_k = mx.fast.metal_kernel(name="qmm_smallm_v4", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC, header=HDR)

def qmm_v4(x2d, w, scales, biases):
    M, K = x2d.shape; N = w.shape[0]
    assert K % 128 == 0 and N % 32 == 0
    if M < 8:
        x2d = mx.concatenate([x2d, mx.zeros((8 - M, K), dtype=x2d.dtype)], axis=0)
    assert x2d.shape[0] == 8
    out = _k(inputs=[x2d, w, scales, biases], template=[("T", x2d.dtype), ("N", N), ("K", K)],
             grid=(N // 32 * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
    return out[:M].astype(x2d.dtype)

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
        for M in (4, 8):
            x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16)
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2)
            got = qmm_v4(x, w, s, b); mx.eval(ref, got)
            err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
            t_ref = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
            t_v3 = bench(lambda: qmm_v3(x, w, s, b)); t_v4 = bench(lambda: qmm_v4(x, w, s, b))
            print(f'{name:26s} M={M}: stock {t_ref:6.2f}  v3 {t_v3:6.2f}  v4 {t_v4:6.2f} ms  (v4 {t_ref/t_v4:4.2f}x stock)  max|Δ| {err:.4f}')
