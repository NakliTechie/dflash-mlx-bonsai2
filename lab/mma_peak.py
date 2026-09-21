"""Peak simdgroup_matrix half MMA throughput on this GPU (register-only loop, no memory traffic)."""
import time, statistics as st, mlx.core as mx
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    simdgroup_half8x8 A, B; simdgroup_float8x8 c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
    threadgroup half seed[64]; seed[lane] = half(float(lane) * 0.001f + float(thread_position_in_grid.x & 7) * 0.01f); seed[lane + 32] = half(0.5f);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A, seed, 8); simdgroup_load(B, seed, 8);
    for (uint i = 0; i < ITERS; ++i) {
        simdgroup_multiply_accumulate(c0, A, B, c0); simdgroup_multiply_accumulate(c1, B, A, c1);
        simdgroup_multiply_accumulate(c2, A, A, c2); simdgroup_multiply_accumulate(c3, B, B, c3);
    }
    threadgroup float tmp[64]; simdgroup_store(c0, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup);
    float v = tmp[lane]; simdgroup_store(c1, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v += tmp[lane];
    simdgroup_store(c2, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v += tmp[lane]; simdgroup_store(c3, tmp, 8); simdgroup_barrier(mem_flags::mem_threadgroup); v += tmp[lane];
    out[thread_position_in_grid.x] = v;
"""
k = mx.fast.metal_kernel(name="mma_peak", input_names=["x"], output_names=["out"], source=SRC, header=HDR)
ITERS = 4096; SGS = 20 * 64 * 8   # plenty of SIMD-groups to fill 20 cores
def run(): return k(inputs=[mx.zeros((1,))], template=[("ITERS", ITERS)], grid=(SGS * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(SGS * 32,)], output_dtypes=[mx.float32])[0]
for _ in range(3): mx.eval(run())
ts = []
for _ in range(10):
    t0 = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t0)
t = st.median(ts); flops = SGS * ITERS * 4 * (8 * 8 * 8 * 2)
print(f"peak half MMA: {flops / t / 1e12:.2f} TFLOPS  ({t*1000:.1f} ms for {flops/1e9:.0f} GFLOP)")
