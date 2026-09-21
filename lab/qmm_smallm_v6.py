"""v6: register-only 8-row verify GEMM. Orientation C[n x m] = W[n x k] * X^T[k x m]; each lane owns 2 consecutive
k of one n row in the A (weight) tile (probed layout: row = ((lane>>1)&3) | ((lane>>4)&1)<<2, col = (lane&1)*2 |
((lane>>3)&1)<<2), so it dequantizes its 2 weights straight into A.thread_elements() from one packed word — no
threadgroup memory, no barriers. Magic-number dequant (0x4000 | q<<8 = 2 + q/2); per-128-group accumulator
scaled in registers by 2*s[n,g]; bias + magic offset recovered from activation row sums:
out = sum_g 2 s P'_g + (b - 4 s) R_g."""
import sys, time, statistics as st
import mlx.core as mx
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint n0   = (threadgroup_position_in_grid.x * SGS + sg) * 8;
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint nl = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);   // this lane's row in the A/C tiles
    const uint cl = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);   // first of its 2 consecutive columns
    const device uint* wrow = w + (size_t)(n0 + nl) * words;
    const device T* srow = scales + (size_t)(n0 + nl) * groups;
    const device T* brow = biases + (size_t)(n0 + nl) * groups;
    // row-sum duty: lane sums 32 k of X row (lane>>2), quarter (lane&3)
    const uint rr = lane >> 2, rq = lane & 3;
    const device T* xs = x + (size_t)rr * K + rq * 32;
    const uint sh0 = cl * 2u, sh1 = (cl + 8u) * 2u;              // bit offsets of (k=cl, k=cl+1) in slab 2j / 2j+1
    float v0 = 0.0f, v1 = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float rs = 0.0f;
        const device T* xg = xs + (size_t)g * 128;
        #pragma unroll
        for (uint i = 0; i < 32; i += 4) { const vec<T,4> x4 = *(const device vec<T,4>*)(xg + i); rs += float(x4.x) + float(x4.y) + float(x4.z) + float(x4.w); }
        rs += simd_shuffle_xor(rs, 1); rs += simd_shuffle_xor(rs, 2);
        simdgroup_float8x8 acc(0.0f);
        const device uint* wp = wrow + g * 8;
        const device T* xb = x + (size_t)g * 128;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint p = wp[j];
            simdgroup_half8x8 A0, A1, B0, B1;
            thread auto& e0 = A0.thread_elements(); thread auto& e1 = A1.thread_elements();
            e0[0] = as_type<half>(ushort(0x4000u | (((p >> sh0) & 3u) << 8))); e0[1] = as_type<half>(ushort(0x4000u | (((p >> (sh0 + 2u)) & 3u) << 8)));
            e1[0] = as_type<half>(ushort(0x4000u | (((p >> sh1) & 3u) << 8))); e1[1] = as_type<half>(ushort(0x4000u | (((p >> (sh1 + 2u)) & 3u) << 8)));
            simdgroup_load(B0, xb + j * 16, K, ulong2(0, 0), true);
            simdgroup_load(B1, xb + j * 16 + 8, K, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, A0, B0, acc);
            simdgroup_multiply_accumulate(acc, A1, B1, acc);
        }
        thread auto& c = acc.thread_elements();                       // (nl, cl) and (nl, cl+1): m = cl, cl+1
        const float R0 = simd_shuffle(rs, cl * 4), R1 = simd_shuffle(rs, (cl + 1) * 4);
        const float s = float(srow[g]), b = float(brow[g]);
        v0 += 2.0f * s * c[0] + (b - 4.0f * s) * R0;
        v1 += 2.0f * s * c[1] + (b - 4.0f * s) * R1;
    }
    out[(size_t)cl * N + n0 + nl] = v0;
    out[(size_t)(cl + 1) * N + n0 + nl] = v1;
"""
_k = mx.fast.metal_kernel(name="prism_qmm_m8_v6", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC, header=HDR)
SGS = 4
def qmm_v6(x2d, w, scales, biases):
    M, K = x2d.shape; N = w.shape[0]
    assert M == 8 and K % 128 == 0 and N % (8 * SGS) == 0 and x2d.dtype == mx.float16
    return _k(inputs=[x2d, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", SGS)],
              grid=(N // (8 * SGS) * 32 * SGS, 1, 1), threadgroup=(32 * SGS, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]

if __name__ == "__main__":
    sys.path.insert(0, 'lab')
    from dflash_mlx.runtime.prism_qmm import qmm_m8 as qmm_v4b
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors')
    mod = lambda p: (W[p + '.weight'], W[p + '.scales'], W[p + '.biases'])
    for name, p in [('up_proj', 'language_model.model.layers.0.mlp.up_proj'), ('down_proj', 'language_model.model.layers.0.mlp.down_proj'), ('o_proj', 'language_model.model.layers.3.self_attn.o_proj'), ('lm_head', 'language_model.lm_head')]:
        w, s, b = mod(p); mx.eval(w, s, b); K = w.shape[1] * 16
        x = (mx.random.normal((8, K)) * 0.5).astype(mx.float16)
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2).astype(mx.float32)
        r6 = qmm_v6(x, w, s, b); r4 = qmm_v4b(x, w, s, b); mx.eval(ref, r6, r4)
        print(f'{name:10s} max|Δ| v6 {float(mx.abs(ref - r6).max()):.4f} (v4b {float(mx.abs(ref - r4).max()):.4f}), rel {float(mx.abs(ref - r6).max() / mx.abs(ref).max()):.2e}, argmax rows agree {int((ref.argmax(-1) == r6.argmax(-1)).sum())}/8')
