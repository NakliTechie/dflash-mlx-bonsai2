"""v4b: v4 with two packed words (32 k) per barrier pair and 8 SIMD-groups per threadgroup."""
import sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, 'lab')
from qmm_smallm_v4 import qmm_v4

HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;      // 0..7
    const uint n0   = (threadgroup_position_in_grid.x * 8 + sg) * 8;
    threadgroup half wt_all[8][8 * 32];
    threadgroup half* wt = wt_all[sg];
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint r = lane >> 2;
    const uint q = lane & 3;
    const device uint* wrow = w + (size_t)(n0 + r) * words;
    const device T* srow = scales + (size_t)(n0 + r) * groups;
    const device T* brow = biases + (size_t)(n0 + r) * groups;
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);
    for (uint word = 0; word < words; word += 2) {
        const uint g = word >> 3;                 // both words share the 128-group (word even, 8 words per group)
        const float s = float(srow[g]);
        const float b = float(brow[g]);
        const uint p0 = wrow[word] >> (8u * q);
        const uint p1 = wrow[word + 1] >> (8u * q);
        threadgroup half* d0 = wt + r * 32 + q * 4;
        threadgroup half* d1 = d0 + 16;
        d0[0] = half(float((p0      ) & 3u) * s + b); d0[1] = half(float((p0 >> 2u) & 3u) * s + b);
        d0[2] = half(float((p0 >> 4u) & 3u) * s + b); d0[3] = half(float((p0 >> 6u) & 3u) * s + b);
        d1[0] = half(float((p1      ) & 3u) * s + b); d1[1] = half(float((p1 >> 2u) & 3u) * s + b);
        d1[2] = half(float((p1 >> 4u) & 3u) * s + b); d1[3] = half(float((p1 >> 6u) & 3u) * s + b);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 W0, W1, W2, W3, X0, X1, X2, X3;
        simdgroup_load(W0, wt,      32, ulong2(0, 0), true);
        simdgroup_load(W1, wt + 8,  32, ulong2(0, 0), true);
        simdgroup_load(W2, wt + 16, 32, ulong2(0, 0), true);
        simdgroup_load(W3, wt + 24, 32, ulong2(0, 0), true);
        const device T* xb = x + (size_t)word * 16;
        simdgroup_load(X0, xb, K); simdgroup_load(X1, xb + 8, K); simdgroup_load(X2, xb + 16, K); simdgroup_load(X3, xb + 24, K);
        simdgroup_multiply_accumulate(acc0, X0, W0, acc0);
        simdgroup_multiply_accumulate(acc1, X1, W1, acc1);
        simdgroup_multiply_accumulate(acc2, X2, W2, acc2);
        simdgroup_multiply_accumulate(acc3, X3, W3, acc3);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float tmp_all[8][64];
    threadgroup float* tmp = tmp_all[sg];
    float v0 = 0.0f, v1 = 0.0f;
    simdgroup_store(acc0, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v0 += tmp[lane]; v1 += tmp[lane + 32]; simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc1, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v0 += tmp[lane]; v1 += tmp[lane + 32]; simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc2, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v0 += tmp[lane]; v1 += tmp[lane + 32]; simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc3, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v0 += tmp[lane]; v1 += tmp[lane + 32];
    out[(size_t)(lane / 8) * N + n0 + (lane % 8)] = v0;
    out[(size_t)((lane + 32) / 8) * N + n0 + ((lane + 32) % 8)] = v1;
"""
_k = mx.fast.metal_kernel(name="qmm_smallm_v4b", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC, header=HDR)

def qmm_v4b(x2d, w, scales, biases):
    M, K = x2d.shape; N = w.shape[0]
    assert K % 128 == 0 and N % 64 == 0
    if M < 8:
        x2d = mx.concatenate([x2d, mx.zeros((8 - M, K), dtype=x2d.dtype)], axis=0)
    out = _k(inputs=[x2d, w, scales, biases], template=[("T", x2d.dtype), ("N", N), ("K", K)],
             grid=(N // 64 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
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
        x = (mx.random.normal((8, K)) * 0.5).astype(mx.float16)
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2)
        got = qmm_v4b(x, w, s, b); mx.eval(ref, got)
        err = float(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max())
        t_ref = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
        t_v4 = bench(lambda: qmm_v4(x, w, s, b)); t_v4b = bench(lambda: qmm_v4b(x, w, s, b))
        print(f'{name:26s} M=8: stock {t_ref:6.2f}  v4 {t_v4:6.2f}  v4b {t_v4b:6.2f} ms  (v4b {t_ref/t_v4b:4.2f}x stock, {t_v4/t_v4b:4.2f}x v4)  max|Δ| {err:.4f}')
