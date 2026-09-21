"""Verify-time matmul for PrismML `Packed` (Hadamard + 2-bit g128 affine) modules.

MLX's stock 2-bit `quantized_matmul` is bandwidth-tuned for M=1 and slow for the 8-row verify block of a
DFlash2 round, and slower still on fp32 activations. This module installs a class-level `Packed.__call__`
that, for exactly-8-row inputs, casts activations to fp16 and (mode `v4b`) runs a simdgroup_matrix 8x8 MMA
kernel: one SIMD-group owns 8 output rows, lanes dequantize two packed words per step into a threadgroup half
tile, four transposed simdgroup_loads + four MMAs against X tiles from device memory, float accumulators.
Measured on Ternary-Bonsai-2-27B (M4 Pro): verify_block(8) 438 ms -> 174 ms (fp16 cast) -> 136 ms (v4b),
argmax identical. Mode `v7` (default since 2026-09-21) is the register-only kernel below, 1.30x v4b per shape.
See LocalMind/plan/2026-09-18-dflash-bonsai2-plan.md.
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


# ---- v7: register-only, X pre-transposed, row sums precomputed (lab/qmm_smallm_v7.py, NT=1) ----
# C[n x m] = W[n x k] * X^T[k x m]. Probed simdgroup 8x8 layout: lane holds row ((lane>>1)&3)|((lane>>4)&1)<<2,
# cols (lane&1)*2|((lane>>3)&1)<<2 (+1), so each lane dequantizes its 2 consecutive k straight into
# A.thread_elements() from one packed word: no threadgroup memory, no barriers. Magic-number dequant
# (0x4000 | q<<8 = 2 + q/2); per-128-group accumulator scaled by 2*s[n,g] in registers; bias and offset from
# row sums: out = sum_g 2 s P'_g + (b - 4 s) R_g. Batched microbench (M4 Pro): 1.30x v4b on every shape.
_SRC_V7 = r"""
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
        const device T* xb = xt + (size_t)g * 128 * 8;
        const device uint* wp = wrow + g * 8;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            simdgroup_half8x8 A0, A1, B0, B1;
            simdgroup_load(B0, xb + (j * 16) * 8, 8);
            simdgroup_load(B1, xb + (j * 16 + 8) * 8, 8);
            const uint p = wp[j];
            thread auto& e0 = A0.thread_elements(); thread auto& e1 = A1.thread_elements();
            e0[0] = as_type<half>(ushort(0x4000u | (((p >> sh0) & 3u) << 8))); e0[1] = as_type<half>(ushort(0x4000u | (((p >> (sh0 + 2u)) & 3u) << 8)));
            e1[0] = as_type<half>(ushort(0x4000u | (((p >> sh1) & 3u) << 8))); e1[1] = as_type<half>(ushort(0x4000u | (((p >> (sh1 + 2u)) & 3u) << 8)));
            simdgroup_multiply_accumulate(acc, A0, B0, acc);
            simdgroup_multiply_accumulate(acc, A1, B1, acc);
        }
        thread auto& c = acc.thread_elements();
        const float R0 = rs[(size_t)cl * groups + g], R1 = rs[(size_t)(cl + 1) * groups + g];
        const float s = float(srow[g]), b = float(brow[g]);
        v0 += 2.0f * s * c[0] + (b - 4.0f * s) * R0;
        v1 += 2.0f * s * c[1] + (b - 4.0f * s) * R1;
    }
    out[(size_t)cl * N + n0 + nl] = v0;
    out[(size_t)(cl + 1) * N + n0 + nl] = v1;
"""
_kernel_v7 = None
_V7_SGS = 4


def _get_kernel_v7():
    global _kernel_v7
    if _kernel_v7 is None:
        _kernel_v7 = mx.fast.metal_kernel(name="prism_qmm_m8_v7", input_names=["xt", "rs", "w", "scales", "biases"],
                                          output_names=["out"], source=_SRC_V7, header=_HDR)
    return _kernel_v7


def qmm_m8_v7(x2d: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x2d: [8, K] fp16; w uint32 [N, K/16]; scales/biases fp16 [N, K/128]. Returns float32 [8, N]."""
    M, K = x2d.shape
    N = w.shape[0]
    if M != 8 or K % 128 != 0 or N % (8 * _V7_SGS) != 0 or x2d.dtype != mx.float16:
        raise ValueError(f"qmm_m8_v7 needs [8, K%128==0] fp16 input and N%32==0, got M={M} K={K} N={N} {x2d.dtype}")
    xt = mx.contiguous(x2d.T)
    rs = x2d.reshape(8, K // 128, 128).astype(mx.float32).sum(-1)
    return _get_kernel_v7()(inputs=[xt, rs, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", _V7_SGS)],
                            grid=(N // (8 * _V7_SGS) * 32 * _V7_SGS, 1, 1), threadgroup=(32 * _V7_SGS, 1, 1),
                            output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]


# ---- fused input prep for v7: (x * signs) -> Hadamard(1024) -> fp16 -> transpose [K, 8] + per-128-group row sums,
# one kernel instead of ~6 ops; bit-exact vs the op sequence (lab/prep_kernel.py), 2.5x faster per call.
_SRC_PREP = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint m = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;
    const uint k0 = blk * 1024;
    threadgroup float buf[1024];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) { const uint k = tid * 4 + i; buf[k] = float(x[(size_t)m * K + k0 + k]) * float(signs[k0 + k]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 1024; h <<= 1) {
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            const uint p = tid + r * 256;
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
    const float rsum = simd_sum(local);
    if (lane == 0) rs[(size_t)m * (K / 128) + blk * 8 + sg] = rsum;
"""
_kernel_prep = None


def _get_kernel_prep():
    global _kernel_prep
    if _kernel_prep is None:
        _kernel_prep = mx.fast.metal_kernel(name="prism_verify_prep", input_names=["x", "signs"], output_names=["xt", "rs"],
                                            source=_SRC_PREP, header="#include <metal_simdgroup>\nusing namespace metal;\n")
    return _kernel_prep


def verify_prep(x2d: mx.array, signs: mx.array) -> tuple[mx.array, mx.array]:
    """x2d [8, K] (fp16/fp32, K % 1024 == 0) -> (xt fp16 [K, 8], rs fp32 [8, K/128]) with the 1024-block Hadamard applied."""
    K = x2d.shape[1]
    return tuple(_get_kernel_prep()(inputs=[x2d, signs], template=[("K", K)], grid=(K // 1024 * 256, 8, 1), threadgroup=(256, 1, 1),
                                    output_shapes=[(K, 8), (8, K // 128)], output_dtypes=[mx.float16, mx.float32]))


def qmm_m8_v7_prepped(xt: mx.array, rs: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    K, N = xt.shape[0], w.shape[0]
    return _get_kernel_v7()(inputs=[xt, rs, w, scales, biases], template=[("T", mx.float16), ("N", N), ("K", K), ("SGS", _V7_SGS)],
                            grid=(N // (8 * _V7_SGS) * 32 * _V7_SGS, 1, 1), threadgroup=(32 * _V7_SGS, 1, 1),
                            output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]


def install_prism_verify_linears(packed_cls: Any, fwht: Any, mode: str | None = None) -> str:
    """Patch `Packed.__call__` for 8-row inputs. mode: 'v7' (default), 'v4b', 'fp16' (cast only), 'off'."""
    mode = (mode or os.environ.get("DFLASH_PRISM_VERIFY", "v7")).lower()
    kernel = qmm_m8_v7 if mode == "v7" else qmm_m8
    if getattr(packed_cls, "_dflash_verify_mode", None) == mode:
        return mode
    stock_call = getattr(packed_cls, "_dflash_stock_call", None) or packed_cls.__call__
    if mode == "off":                                   # restore the pack's own __call__ (fp32 activations, stock qmm)
        packed_cls.__call__ = stock_call
        packed_cls._dflash_stock_call = stock_call
        packed_cls._dflash_verify_mode = mode
        return mode
    stats = None
    if os.environ.get("DFLASH_PRISM_VERIFY_STATS"):          # rows-per-call histogram, printed at exit (diagnostic)
        import atexit, collections, sys
        stats = collections.Counter()
        atexit.register(lambda: print("[prism_qmm] rows-per-call histogram:", dict(sorted(stats.items())), "mode", mode, file=sys.stderr))

    def verify_call(self, x):
        if self.embedding:
            return stock_call(self, x)
        shape = x.shape
        rows = 1
        for d in shape[:-1]:
            rows *= d
        if stats is not None:
            stats[rows] += 1
        # The real DFlash2 verify is NOT 8 rows: the selector emits variable-length paths, so a "block 8" run
        # verifies 4-5 rows per cycle (rows-per-call histogram of a real benchmark: {1: decode, 4/5: verify,
        # 33: prefill}). The kernels cost the same for any M <= 8, so pad 2..7-row calls to 8 (zero rows) and
        # slice; before this, every real verify silently took the fp16 + stock path.
        pad_ok = mode == "v7" and 2 <= rows <= 8
        if mode == "fp16" or (rows != 8 and not pad_ok) or self.weight.shape[0] % 64 != 0:   # both kernels need N % 64 == 0 (v7: N % 32)
            # 1 row: 52 -> 46 ms per decode step; >8 rows: the fp16 cast alone is 2.5x on the verify block.
            return stock_call(self, x.astype(mx.float16)).astype(x.dtype)
        x2 = x.reshape(rows, shape[-1])
        if rows < 8:
            x2 = mx.concatenate([x2, mx.zeros((8 - rows, shape[-1]), dtype=x2.dtype)], axis=0)
        if stats is not None:
            stats[100 + rows] += 1   # 10x = went through the custom kernel
        if mode == "v7" and self.block == 1024 and self.signs is not None and shape[-1] % 1024 == 0:
            xt, rs = verify_prep(x2, self.signs)                       # fused sign * Hadamard * transpose + row sums
            out = qmm_m8_v7_prepped(xt, rs, self.weight, self.scales, self.biases)
            return out[:rows].astype(x.dtype).reshape(*shape[:-1], -1)
        if self.block:
            x2 = fwht(x2, self.block, self.signs)
        out = kernel(x2.astype(mx.float16), self.weight, self.scales, self.biases)
        return out[:rows].astype(x.dtype).reshape(*shape[:-1], -1)

    packed_cls._dflash_stock_call = stock_call
    packed_cls.__call__ = verify_call
    packed_cls._dflash_verify_mode = mode
    return mode
