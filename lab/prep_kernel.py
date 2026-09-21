"""Fused verify-input prep for the v7 GEMM: one Metal kernel replaces (x * signs) -> hadamard_transform ->
astype(fp16) -> transpose -> contiguous -> reshape/sum for the row sums. Input x [8, K] (fp16 or fp32), signs [K]
(the pack's explicit Hadamard signs), block 1024; outputs xt [K, 8] fp16 (transformed, scaled 1/sqrt(block)) and
rs [8, K/128] fp32 (row sums of the TRANSFORMED fp16 values per 128-group, exactly what qmm_m8_v7 consumes).
One threadgroup (256 threads, 4 elements each) per (row m, 1024-block); 10 butterfly stages in threadgroup
memory; each SIMD-group covers exactly one 128-group so rs is a simd_sum."""
import sys, math, time, statistics as st
import mlx.core as mx
HDR = "#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;          // 0..255
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0..7 -> 128-group within the block
    const uint m = threadgroup_position_in_grid.y;              // row 0..7
    const uint blk = threadgroup_position_in_grid.x;            // 1024-block index
    const uint k0 = blk * 1024;
    threadgroup float buf[1024];
    // load 4 consecutive elements with signs applied
    #pragma unroll
    for (uint i = 0; i < 4; ++i) { const uint k = tid * 4 + i; buf[k] = float(x[(size_t)m * K + k0 + k]) * float(signs[k0 + k]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // in-place FWHT: stages h = 1, 2, ..., 512; each thread handles 2 butterflies per stage
    for (uint h = 1; h < 1024; h <<= 1) {
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            const uint p = tid + r * 256;                       // pair index 0..511
            const uint i = (p / h) * (2 * h) + (p % h);
            const float a = buf[i], b = buf[i + h];
            buf[i] = a + b; buf[i + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = 1.0f / sqrt(1024.0f);
    float local = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        const uint k = tid * 4 + i;
        const half hv = half(buf[k] * scale);
        xt[(size_t)(k0 + k) * 8 + m] = hv;
        local += float(hv);
    }
    const float rsum = simd_sum(local);                         // this SIMD-group == 128-group (blk*8 + sg)
    if (lane == 0) rs[(size_t)m * (K / 128) + blk * 8 + sg] = rsum;
"""
_k = None
def _kernel():
    global _k
    if _k is None:
        _k = mx.fast.metal_kernel(name="prism_verify_prep", input_names=["x", "signs"], output_names=["xt", "rs"], source=SRC, header=HDR)
    return _k

def prep(x2d, signs, block=1024):
    M, K = x2d.shape
    assert M == 8 and block == 1024 and K % 1024 == 0
    return _kernel()(inputs=[x2d, signs], template=[("K", K)], grid=(K // 1024 * 256, 8, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(K, 8), (8, K // 128)], output_dtypes=[mx.float16, mx.float32])

if __name__ == "__main__":
    PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
    sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
    sys.path.insert(0, 'lab'); from prism_pack_loader import load_text_model
    from dflash_mlx.runtime.prism_qmm import qmm_m8_v7, _get_kernel_v7, _V7_SGS
    model, _ = load_text_model(PACK)
    m = model.model.layers[0].mlp.up_proj; signs = m.signs; K = m.weight.shape[1] * 16
    for dt in (mx.float32, mx.float16):
        x = (mx.random.normal((8, K)) * 0.7).astype(dt)
        ref_t = prism_rt.fwht(x, 1024, signs).astype(mx.float16)                      # [8, K]
        ref_xt = mx.contiguous(ref_t.T); ref_rs = ref_t.reshape(8, K // 128, 128).astype(mx.float32).sum(-1)
        xt, rs = prep(x, signs); mx.eval(ref_xt, ref_rs, xt, rs)
        print(f'x {dt}: xt max|Δ| {float(mx.abs(ref_xt.astype(mx.float32) - xt.astype(mx.float32)).max()):.5f}  rs max|Δ| {float(mx.abs(ref_rs - rs).max()):.5f}  (|xt| max {float(mx.abs(ref_xt).max()):.2f})')
        # end-to-end through the GEMM: prep -> kernel vs the module's own patched path
        out_ref = m(x).astype(mx.float32).reshape(8, -1)
        N = m.weight.shape[0]
        out_new = _get_kernel_v7()(inputs=[xt, rs, m.weight, m.scales, m.biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", _V7_SGS)],
                                   grid=(N // (8 * _V7_SGS) * 32 * _V7_SGS, 1, 1), threadgroup=(32 * _V7_SGS, 1, 1), output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
        mx.eval(out_ref, out_new)
        print(f'   GEMM out max|Δ| {float(mx.abs(out_ref - out_new).max()):.4f} rel {float(mx.abs(out_ref - out_new).max() / mx.abs(out_ref).max()):.2e}, argmax rows agree {int((out_ref.argmax(-1) == out_new.argmax(-1)).sum())}/8')
    def bench(fn, reps=30):
        for _ in range(3): mx.eval(fn())
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1000)
        return min(ts), st.median(ts)
    x = (mx.random.normal((8, K)) * 0.7).astype(mx.float32)
    xs = [(mx.random.normal((8, K)) * 0.7).astype(mx.float32) for _ in range(16)]; mx.eval(*xs)
    t_old = bench(lambda: [(lambda t: (mx.contiguous(t.T), t.reshape(8, K // 128, 128).astype(mx.float32).sum(-1)))(prism_rt.fwht(xx, 1024, signs).astype(mx.float16)) for xx in xs])
    t_new = bench(lambda: [prep(xx, signs) for xx in xs])
    print(f'prep per call (16 batched): old ops {t_old[0]/16:.4f}/{t_old[1]/16:.4f} ms  fused {t_new[0]/16:.4f}/{t_new[1]/16:.4f} ms (min/median)')
