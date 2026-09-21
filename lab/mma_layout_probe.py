"""Which 2 elements of an 8x8 simdgroup matrix does each lane hold in thread_elements()? Load a tile whose
element (r,c) = r*8+c, then dump each lane's two values -> (row, col) pairs."""
import mlx.core as mx
HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    threadgroup half tile[64];
    tile[lane] = half(float(lane)); tile[lane + 32] = half(float(lane + 32));
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 A; simdgroup_load(A, tile, 8);
    thread auto& e = A.thread_elements();
    out[lane * 2] = float(e[0]); out[lane * 2 + 1] = float(e[1]); if (lane == 0) { out3[0] = float(sizeof(e)); out3[1] = float(e[2]); out3[2] = float(e[3]); }
    // also: accumulator of A * I (identity) has the same layout as A; check float8x8 via multiply
    simdgroup_float8x8 C(0.0f); simdgroup_half8x8 I; threadgroup half id[64]; id[lane] = half(((lane / 8) == (lane % 8)) ? 1.0f : 0.0f); id[lane+32] = half((((lane+32) / 8) == ((lane+32) % 8)) ? 1.0f : 0.0f);
    simdgroup_barrier(mem_flags::mem_threadgroup); simdgroup_load(I, id, 8); simdgroup_multiply_accumulate(C, A, I, C);
    thread auto& f = C.thread_elements();
    out2[lane * 2] = f[0]; out2[lane * 2 + 1] = f[1];
"""
k = mx.fast.metal_kernel(name="mma_layout_probe", input_names=["x"], output_names=["out", "out2", "out3"], source=SRC, header=HDR)
o, o2, o3 = k(inputs=[mx.zeros((1,))], template=[], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(64,), (64,), (4,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
print("sizeof(thread_elements) bytes:", o3[0].item(), "e[2], e[3]:", o3[1].item(), o3[2].item())
o = o.tolist(); o2 = o2.tolist()
for lane in range(32):
    a, b = int(o[2*lane]), int(o[2*lane+1]); c, d = int(o2[2*lane]), int(o2[2*lane+1])
    print(f"lane {lane:2d}: half A elems (r,c)=({a//8},{a%8}) ({b//8},{b%8})   float C elems ({c//8},{c%8}) ({d//8},{d%8})")
