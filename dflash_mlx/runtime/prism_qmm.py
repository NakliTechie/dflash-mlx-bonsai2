"""Verify-time matmul for PrismML `Packed` (Hadamard + 2-bit g128 affine) modules.

MLX's stock 2-bit `quantized_matmul` is bandwidth-tuned for M=1 and slow for the 8-row verify block of a
DFlash2 round, and slower still on fp32 activations. This module installs a class-level `Packed.__call__`
that, for exactly-8-row inputs, casts activations to fp16 and (mode `v4b`) runs a simdgroup_matrix 8x8 MMA
kernel: one SIMD-group owns 8 output rows, lanes dequantize two packed words per step into a threadgroup half
tile, four transposed simdgroup_loads + four MMAs against X tiles from device memory, float accumulators.
Measured on Ternary-Bonsai-2-27B (M4 Pro): verify_block(8) 438 ms -> 174 ms (fp16 cast) -> 136 ms (v4b),
argmax identical. See LocalMind/plan/2026-09-18-dflash-bonsai2-plan.md.
"""
from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

_HDR = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n"
_SRC = r"""
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
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(name="prism_qmm_m8_v4b", input_names=["x", "w", "scales", "biases"],
                                       output_names=["out"], source=_SRC, header=_HDR)
    return _kernel


def qmm_m8(x2d: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x2d: [8, K] fp16; w uint32 [N, K/16]; scales/biases fp16 [N, K/128]. Returns float32 [8, N]."""
    M, K = x2d.shape
    N = w.shape[0]
    if M != 8 or K % 128 != 0 or N % 64 != 0 or x2d.dtype != mx.float16:
        raise ValueError(f"qmm_m8 needs [8, K%128==0] fp16 input and N%64==0, got M={M} K={K} N={N} {x2d.dtype}")
    return _get_kernel()(inputs=[x2d, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K)],
                         grid=(N // 64 * 256, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]


def install_prism_verify_linears(packed_cls: Any, fwht: Any, mode: str | None = None) -> str:
    """Patch `Packed.__call__` for 8-row inputs. mode: 'v4b' (default), 'fp16' (cast only), 'off'."""
    mode = (mode or os.environ.get("DFLASH_PRISM_VERIFY", "v4b")).lower()
    if mode == "off" or getattr(packed_cls, "_dflash_verify_mode", None) == mode:
        return mode
    stock_call = getattr(packed_cls, "_dflash_stock_call", None) or packed_cls.__call__

    def verify_call(self, x):
        if self.embedding:
            return stock_call(self, x)
        shape = x.shape
        rows = 1
        for d in shape[:-1]:
            rows *= d
        if rows == 1:
            return stock_call(self, x)
        if mode == "fp16" or rows != 8 or self.weight.shape[0] % 64 != 0:
            # 2..7 rows (adaptive verify shortens blocks) and >8 rows: the fp16 cast alone is 2.5x
            return stock_call(self, x.astype(mx.float16)).astype(x.dtype)
        x2 = x.reshape(rows, shape[-1])
        if self.block:
            x2 = fwht(x2, self.block, self.signs)
        out = qmm_m8(x2.astype(mx.float16), self.weight, self.scales, self.biases)
        return out.astype(x.dtype).reshape(*shape[:-1], -1)

    packed_cls._dflash_stock_call = stock_call
    packed_cls.__call__ = verify_call
    packed_cls._dflash_verify_mode = mode
    return mode
