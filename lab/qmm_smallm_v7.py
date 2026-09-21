"""v7 = v6 + register tiling: each SIMD-group owns NT consecutive 8-row weight tiles (NT accumulator chains, one
X^T tile load feeds NT MMAs) + X pre-transposed to [K, 8] so B tiles are contiguous 8x8 blocks (stride 8).
Row sums R[m, g] are precomputed with one tiny op (x.reshape(8, K/128, 128).sum(-1)) and read per group."""
import sys, time, statistics as st
import mlx.core as mx
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint n0   = (threadgroup_position_in_grid.x * SGS + sg) * (8 * NT);
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint nl = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
    const uint cl = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
    const uint sh0 = cl * 2u, sh1 = (cl + 8u) * 2u;
    const device uint* wrow[NT]; const device T* srow[NT]; const device T* brow[NT];
    #pragma unroll
    for (uint t = 0; t < NT; ++t) { wrow[t] = w + (size_t)(n0 + t * 8 + nl) * words; srow[t] = scales + (size_t)(n0 + t * 8 + nl) * groups; brow[t] = biases + (size_t)(n0 + t * 8 + nl) * groups; }
    float v0[NT], v1[NT];
    #pragma unroll
    for (uint t = 0; t < NT; ++t) { v0[t] = 0.0f; v1[t] = 0.0f; }
    for (uint g = 0; g < groups; ++g) {
        simdgroup_float8x8 acc[NT];
        #pragma unroll
        for (uint t = 0; t < NT; ++t) acc[t] = simdgroup_float8x8(0.0f);
        const device T* xb = xt + (size_t)g * 128 * 8;           // X^T block for this group: [128 k x 8 m], stride 8
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            simdgroup_half8x8 B0, B1;
            simdgroup_load(B0, xb + (j * 16) * 8, 8);
            simdgroup_load(B1, xb + (j * 16 + 8) * 8, 8);
            #pragma unroll
            for (uint t = 0; t < NT; ++t) {
                const uint p = wrow[t][g * 8 + j];
                simdgroup_half8x8 A0, A1;
                thread auto& e0 = A0.thread_elements(); thread auto& e1 = A1.thread_elements();
                e0[0] = as_type<half>(ushort(0x4000u | (((p >> sh0) & 3u) << 8))); e0[1] = as_type<half>(ushort(0x4000u | (((p >> (sh0 + 2u)) & 3u) << 8)));
                e1[0] = as_type<half>(ushort(0x4000u | (((p >> sh1) & 3u) << 8))); e1[1] = as_type<half>(ushort(0x4000u | (((p >> (sh1 + 2u)) & 3u) << 8)));
                simdgroup_multiply_accumulate(acc[t], A0, B0, acc[t]);
                simdgroup_multiply_accumulate(acc[t], A1, B1, acc[t]);
            }
        }
        const float R0 = rs[(size_t)cl * groups + g], R1 = rs[(size_t)(cl + 1) * groups + g];
        #pragma unroll
        for (uint t = 0; t < NT; ++t) {
            thread auto& c = acc[t].thread_elements();
            const float s = float(srow[t][g]), b = float(brow[t][g]);
            v0[t] += 2.0f * s * c[0] + (b - 4.0f * s) * R0;
            v1[t] += 2.0f * s * c[1] + (b - 4.0f * s) * R1;
        }
    }
    #pragma unroll
    for (uint t = 0; t < NT; ++t) {
        out[(size_t)cl * N + n0 + t * 8 + nl] = v0[t];
        out[(size_t)(cl + 1) * N + n0 + t * 8 + nl] = v1[t];
    }
"""
_kernels = {}
def _kernel(nt):
    if nt not in _kernels:
        _kernels[nt] = mx.fast.metal_kernel(name=f"prism_qmm_m8_v7_nt{nt}", input_names=["xt", "rs", "w", "scales", "biases"], output_names=["out"], source=SRC, header=HDR)
    return _kernels[nt]
SGS = 4
def make(nt):
    def qmm(x2d, w, scales, biases):
        M, K = x2d.shape; N = w.shape[0]
        assert M == 8 and K % 128 == 0 and x2d.dtype == mx.float16 and N % (8 * nt * SGS) == 0, (N, nt)
        xt = mx.contiguous(x2d.T)                                     # [K, 8]
        rs = x2d.reshape(8, K // 128, 128).astype(mx.float32).sum(-1)   # [8, K/128]
        return _kernel(nt)(inputs=[xt, rs, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", SGS), ("NT", nt)],
                           grid=(N // (8 * nt * SGS) * 32 * SGS, 1, 1), threadgroup=(32 * SGS, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
    return qmm
qmm_v7nt1 = make(1); qmm_v7nt2 = make(2); qmm_v7nt4 = make(4)
qmm_v7 = qmm_v7nt2

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors'); p = 'language_model.model.layers.0.mlp.up_proj'
    w, s, b = W[p + '.weight'], W[p + '.scales'], W[p + '.biases']; mx.eval(w, s, b); K = w.shape[1] * 16
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.float16)
    ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2).astype(mx.float32)
    for nt, fn in ((1, qmm_v7nt1), (2, qmm_v7nt2), (4, qmm_v7nt4)):
        r = fn(x, w, s, b); mx.eval(r); print(f'v7 NT={nt}: max|Δ| {float(mx.abs(ref - r).max()):.4f}, argmax rows agree {int((ref.argmax(-1) == r.argmax(-1)).sum())}/8')
