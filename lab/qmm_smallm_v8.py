"""v8 experiments on top of v7 (NT=1, X^T + row sums precomputed):
  v8a: nibble LUT dequant — the lane's 2 consecutive k are a 4-bit field; a 16-entry constant table gives both
       halves packed in one uint (fewer int ops per element).
  v8b: two independent accumulator chains per tile (even/odd slabs) for MMA ILP, summed in the epilogue.
  v8c: both.
Same math as v7; exactness checked against stock."""
import sys, time, statistics as st
import mlx.core as mx
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
LUT = "constant uint LUT16[16] = {" + ", ".join(
    f"0x{((0x4000 | ((n >> 2) << 8)) << 16) | (0x4000 | ((n & 3) << 8)):08x}u" for n in range(16)) + "};\n"
SRC_TMPL = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint n0   = (threadgroup_position_in_grid.x * SGS + sg) * 8;
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint nl = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
    const uint cl = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2);
    const uint sh0 = cl * 2u, sh1 = (cl + 8u) * 2u;
    const device uint* wrow = w + (size_t)(n0 + nl) * words;
    const device T* srow = scales + (size_t)(n0 + nl) * groups;
    const device T* brow = biases + (size_t)(n0 + nl) * groups;
    float v0 = 0.0f, v1 = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        simdgroup_float8x8 acc(0.0f);
        __CHAIN2_DECL__
        const device T* xb = xt + (size_t)g * 128 * 8;
        const device uint* wp = wrow + g * 8;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            simdgroup_half8x8 A0, A1, B0, B1;
            simdgroup_load(B0, xb + (j * 16) * 8, 8);
            simdgroup_load(B1, xb + (j * 16 + 8) * 8, 8);
            const uint p = wp[j];
            thread auto& e0 = A0.thread_elements(); thread auto& e1 = A1.thread_elements();
            __DEQUANT__
            simdgroup_multiply_accumulate(acc, A0, B0, acc);
            simdgroup_multiply_accumulate(__ACC2__, A1, B1, __ACC2__);
        }
        __CHAIN2_FOLD__
        thread auto& c = acc.thread_elements();
        const float R0 = rs[(size_t)cl * groups + g], R1 = rs[(size_t)(cl + 1) * groups + g];
        const float s = float(srow[g]), b = float(brow[g]);
        v0 += 2.0f * s * c[0] + (b - 4.0f * s) * R0;
        v1 += 2.0f * s * c[1] + (b - 4.0f * s) * R1;
    }
    out[(size_t)cl * N + n0 + nl] = v0;
    out[(size_t)(cl + 1) * N + n0 + nl] = v1;
"""
DEQ_SHIFT = r"""
            e0[0] = as_type<half>(ushort(0x4000u | (((p >> sh0) & 3u) << 8))); e0[1] = as_type<half>(ushort(0x4000u | (((p >> (sh0 + 2u)) & 3u) << 8)));
            e1[0] = as_type<half>(ushort(0x4000u | (((p >> sh1) & 3u) << 8))); e1[1] = as_type<half>(ushort(0x4000u | (((p >> (sh1 + 2u)) & 3u) << 8)));"""
DEQ_LUT = r"""
            { const half2 h0 = as_type<half2>(LUT16[(p >> sh0) & 15u]); const half2 h1 = as_type<half2>(LUT16[(p >> sh1) & 15u]);
              e0[0] = h0.x; e0[1] = h0.y; e1[0] = h1.x; e1[1] = h1.y; }"""
CHAIN2_DECL = "simdgroup_float8x8 acc2(0.0f);"
CHAIN2_FOLD = r"""{ thread auto& c2 = acc2.thread_elements(); thread auto& c1 = acc.thread_elements(); c1[0] += c2[0]; c1[1] += c2[1]; }"""
def build(name, lut, chain2):
    src = SRC_TMPL.replace("__DEQUANT__", DEQ_LUT if lut else DEQ_SHIFT)
    src = src.replace("__CHAIN2_DECL__", CHAIN2_DECL if chain2 else "").replace("__ACC2__", "acc2" if chain2 else "acc").replace("__CHAIN2_FOLD__", CHAIN2_FOLD if chain2 else "")
    return mx.fast.metal_kernel(name=name, input_names=["xt", "rs", "w", "scales", "biases"], output_names=["out"], source=src, header=HDR + (LUT if lut else ""))
SGS = 4
def make(name, lut, chain2):
    k = build(name, lut, chain2)
    def qmm(x2d, w, scales, biases):
        M, K = x2d.shape; N = w.shape[0]
        assert M == 8 and K % 128 == 0 and x2d.dtype == mx.float16 and N % (8 * SGS) == 0
        xt = mx.contiguous(x2d.T); rs = x2d.reshape(8, K // 128, 128).astype(mx.float32).sum(-1)
        return k(inputs=[xt, rs, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", SGS)],
                 grid=(N // (8 * SGS) * 32 * SGS, 1, 1), threadgroup=(32 * SGS, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
    return qmm
qmm_v8a = make("prism_qmm_m8_v8a", True, False)
qmm_v8b = make("prism_qmm_m8_v8b", False, True)
qmm_v8c = make("prism_qmm_m8_v8c", True, True)

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors'); p = 'language_model.model.layers.0.mlp.up_proj'
    w, s, b = W[p + '.weight'], W[p + '.scales'], W[p + '.biases']; mx.eval(w, s, b); K = w.shape[1] * 16
    x = (mx.random.normal((8, K)) * 0.5).astype(mx.float16)
    ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2).astype(mx.float32)
    for name, fn in (('v8a', qmm_v8a), ('v8b', qmm_v8b), ('v8c', qmm_v8c)):
        r = fn(x, w, s, b); mx.eval(r); print(f'{name}: max|Δ| {float(mx.abs(ref - r).max()):.4f}, argmax rows agree {int((ref.argmax(-1) == r.argmax(-1)).sum())}/8')
