"""v5: one 128-k group per barrier pair (8 packed words per lane), magic-number dequant (2-bit field -> half via
0x4000 | q<<8 = 2 + q/2, no convert/multiply/add), 16 MMAs per group into a fresh accumulator, then the group's
8x8 partial is stored once and scaled per output column by 2*s[n,g]; the bias and the magic offset are recovered
from per-group activation row sums: out = sum_g 2 s P'_g + (b - 4 s) R_g."""
import sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, 'lab')
from dflash_mlx.runtime.prism_qmm import qmm_m8 as qmm_v4b

HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;      // 0..7
    const uint n0   = (threadgroup_position_in_grid.x * 8 + sg) * 8;
    threadgroup half wt_all[8][8 * 128];                    // per SG: 8 rows x 128 k
    threadgroup float tmp_all[8][64];
    threadgroup half* wt = wt_all[sg];
    threadgroup float* tmp = tmp_all[sg];
    if (n0 >= N) return;
    const uint words = K / 16;
    const uint groups = K / 128;
    const uint r = lane >> 2;                               // dequant row 0..7
    const uint q = lane & 3;                                // byte within word -> 4 weights
    const device uint* wrow = w + (size_t)(n0 + r) * words;
    // output ownership (matches simdgroup_store row-major 8x8): v0 -> (m0, n0+c0), v1 -> (m1, n0+c1)
    const uint m0 = lane >> 3, c0 = lane & 7, m1 = (lane + 32) >> 3, c1 = (lane + 32) & 7;
    const device T* s0 = scales + (size_t)(n0 + c0) * groups;
    const device T* b0 = biases + (size_t)(n0 + c0) * groups;
    const device T* s1 = scales + (size_t)(n0 + c1) * groups;
    const device T* b1 = biases + (size_t)(n0 + c1) * groups;
    // row-sum ownership: lane sums 32 k of row rs = lane>>2, quarter q
    const device T* xs = x + (size_t)r * K + q * 32;
    float v0 = 0.0f, v1 = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        // dequant 8 words -> 32 halves at wt[r*128 + word*16 + q*4 .. +3]
        const device uint* wp = wrow + g * 8;
        threadgroup ushort* d = (threadgroup ushort*)(wt + r * 128 + q * 4);
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {
            const uint p = wp[i] >> (8u * q);
            threadgroup ushort* di = d + i * 16;
            di[0] = ushort(0x4000u | ((p       & 3u) << 8)); di[1] = ushort(0x4000u | (((p >> 2u) & 3u) << 8));
            di[2] = ushort(0x4000u | (((p >> 4u) & 3u) << 8)); di[3] = ushort(0x4000u | (((p >> 6u) & 3u) << 8));
        }
        // row sums for this group (each lane: 32 k of row r)
        float rs = 0.0f;
        const device T* xg = xs + (size_t)g * 128;
        #pragma unroll
        for (uint i = 0; i < 32; i += 4) { const vec<T,4> x4 = *(const device vec<T,4>*)(xg + i); rs += float(x4.x) + float(x4.y) + float(x4.z) + float(x4.w); }
        rs += simd_shuffle_xor(rs, 1); rs += simd_shuffle_xor(rs, 2);   // full 128-sum for row r, in all 4 lanes of the quad
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 acc(0.0f);
        const device T* xb = x + (size_t)g * 128;
        #pragma unroll
        for (uint slab = 0; slab < 16; ++slab) {
            simdgroup_half8x8 W, X;
            simdgroup_load(W, wt + slab * 8, 128, ulong2(0, 0), true);
            simdgroup_load(X, xb + slab * 8, K);
            simdgroup_multiply_accumulate(acc, X, W, acc);
        }
        simdgroup_store(acc, tmp, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const float R0 = simd_shuffle(rs, m0 * 4), R1 = simd_shuffle(rs, m1 * 4);
        const float sa = float(s0[g]), ba = float(b0[g]), sb = float(s1[g]), bb = float(b1[g]);
        v0 += 2.0f * sa * tmp[lane]      + (ba - 4.0f * sa) * R0;
        v1 += 2.0f * sb * tmp[lane + 32] + (bb - 4.0f * sb) * R1;
    }
    out[(size_t)m0 * N + n0 + c0] = v0;
    out[(size_t)m1 * N + n0 + c1] = v1;
"""
_k = mx.fast.metal_kernel(name="prism_qmm_m8_v5", input_names=["x", "w", "scales", "biases"], output_names=["out"], source=SRC, header=HDR)

def qmm_v5(x2d, w, scales, biases):
    M, K = x2d.shape; N = w.shape[0]
    assert M == 8 and K % 128 == 0 and N % 64 == 0 and x2d.dtype == mx.float16
    return _k(inputs=[x2d, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K)],
              grid=(N // 64 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    W = mx.load(PACK + '/model.safetensors')
    mod = lambda p: (W[p + '.weight'], W[p + '.scales'], W[p + '.biases'])
    tests = {'up_proj (17408x5120)': mod('language_model.model.layers.0.mlp.up_proj'),
             'down_proj (5120x17408)': mod('language_model.model.layers.0.mlp.down_proj'),
             'in_proj_qkv (10240x5120)': mod('language_model.model.layers.0.linear_attn.in_proj_qkv'),
             'o_proj (5120x6144)': mod('language_model.model.layers.3.self_attn.o_proj'),
             'lm_head (248320x5120)': mod('language_model.lm_head')}
    def bench(fn, reps=30):
        for _ in range(3): mx.eval(fn())
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
        return st.median(ts)
    tot4 = tot5 = 0.0
    for name, (w, s, b) in tests.items():
        mx.eval(w, s, b); N, K = w.shape[0], w.shape[1] * 16
        x = (mx.random.normal((8, K)) * 0.5).astype(mx.float16)
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2).astype(mx.float32)
        r4 = qmm_v4b(x, w, s, b); r5 = qmm_v5(x, w, s, b); mx.eval(ref, r4, r5)
        e4 = float(mx.abs(ref - r4).max()); e5 = float(mx.abs(ref - r5).max()); am = int((ref.argmax(-1) == r5.argmax(-1)).sum())
        t4 = bench(lambda: qmm_v4b(x, w, s, b)); t5 = bench(lambda: qmm_v5(x, w, s, b)); ts = bench(lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2))
        gf = N * K * 8 * 2 / 1e9
        print(f'{name:26s} stock16 {ts:6.3f}  v4b {t4:6.3f}  v5 {t5:6.3f} ms  v5/v4b {t4/t5:4.2f}x  {gf/t5:5.2f} TFLOPS  max|Δ| v4b {e4:.4f} v5 {e5:.4f}  argmax rows agree {am}/8')
