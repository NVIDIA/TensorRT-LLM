# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3.1 decode indexer prologue (keys and queries) in one CuTe DSL kernel.

Per token (one CTA of 8 warps; lane ``l`` of a warp owns columns ``4l .. 4l+3`` of a 128-wide row):

- queries: each warp quantizes 4 of the 32 heads: ``fused_cat_fp8``'s rule (amax over the BF16
  row, floored at 1e-12, times the FP32 constant 1/448, UE8M0 round-up; ``rcp.approx.ftz``
  reciprocal; ``cvt.rn.satfinite`` E4M3), FP8 queries [rows, 32, 128] and the head weights
  ``(w * q_scale) * weight_scale`` in FP32;
- keys (warp 0, after its heads): LayerNorm in FP32 of the BF16 key (two-pass mean / variance,
  ``rsqrt.approx``, ``(x - mean) * rstd * w + b`` with the last two as one FMA), BF16 rounding,
  the same FP8 quantization, and the scatter into the indexer K cache with
  ``indexer_k_cache_scatter_op``'s flat-index unravel and slot rule (rows below ``n_write`` with
  both slots non-negative; the page view's inner dims are contiguous, so the unravel is one
  division by the page size).

The K3.1 indexer is position-free (no RoPE). The kernel waits for the earlier grids before its
first load and then triggers its dependents (the PDL logits kernel waits for this grid to finish
before it reads queries, weights or keys).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import dsl_user_op

from .rubin_ull_dsa_index_ptx import _asm

HEADS, DIM, WARPS = 32, 128, 8
THREADS = 32 * WARPS


@dsl_user_op
def _ld_bf16x4(addr, *, loc=None, ip=None):
    """Four BF16 at 8-byte aligned ``addr`` as FP32."""
    return _asm(
        "{ .reg .b32 a, b; ld.global.nc.v2.b32 {a, b}, [$4]; "
        "shl.b32 $0, a, 16; and.b32 $1, a, 0xffff0000; shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,l",
        (Uint64(addr),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_bf16(addr, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .b16 h; .reg .b32 w; ld.global.nc.b16 h, [$1]; cvt.u32.u16 w, h; shl.b32 $0, w, 16; }",
        "=f,l",
        (Uint64(addr),),
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_max_abs(a, b, c, d, *, loc=None, ip=None) -> Float32:
    """max(|a|, |b|, |c|, |d|) over the warp (non-negative floats order as their bits)."""
    return _asm(
        "{ .reg .f32 x, y, m; .reg .b32 u; abs.f32 x, $1; abs.f32 y, $2; max.f32 m, x, y; "
        "abs.f32 x, $3; abs.f32 y, $4; max.f32 x, x, y; max.f32 m, m, x; "
        "mov.b32 u, m; redux.sync.max.u32 u, u, -1; mov.b32 $0, u; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_sum(x, *, loc=None, ip=None) -> Float32:
    """Butterfly sum over the warp (every lane gets the same value)."""
    return _asm(
        "{ .reg .f32 t, s; mov.f32 s, $1; "
        "shfl.sync.bfly.b32 t, s, 16, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 8, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 4, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 2, 31, -1; add.f32 s, s, t; "
        "shfl.sync.bfly.b32 t, s, 1, 31, -1; add.f32 $0, s, t; }",
        "=f,f",
        (Float32(x),),
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _scale_inv(amax, *, loc=None, ip=None):
    """(UE8M0 scale, rcp.approx.ftz(scale)) of a row amax, ``fused_cat_fp8``'s rule."""
    return _asm(
        "{ .reg .f32 a, r; .reg .b32 bits, m, e; .reg .pred p; "
        "max.f32 a, $2, 0f2B8CBCCC; mul.f32 r, a, 0f3B124925; mov.b32 bits, r; "
        "and.b32 m, bits, 0x007FFFFF; and.b32 e, bits, 0x7F800000; setp.ne.u32 p, m, 0; "
        "@p add.u32 e, e, 0x00800000; mov.b32 $0, e; rcp.approx.ftz.f32 $1, $0; }",
        "=f,=f,f",
        (Float32(amax),),
        (Float32, Float32),
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fp8x4(a, b, c, d, inv, *, loc=None, ip=None) -> Uint32:
    """E4M3 codes of (a, b, c, d) * inv, RN satfinite, packed little-endian."""
    return _asm(
        "{ .reg .f32 x, y, z, w; .reg .b16 lo, hi; mul.f32 x, $1, $5; mul.f32 y, $2, $5; "
        "mul.f32 z, $3, $5; mul.f32 w, $4, $5; cvt.rn.satfinite.e4m3x2.f32 lo, y, x; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, w, z; mov.b32 $0, {lo, hi}; }",
        "=r,f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d), Float32(inv)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ln_bf16(x, mean, rstd, w, b, *, loc=None, ip=None) -> Float32:
    """BF16 rounding (RN) of ``fma((x - mean) * rstd, w, b)``, as FP32."""
    return _asm(
        "{ .reg .f32 t; .reg .b16 h; .reg .b32 u; sub.f32 t, $1, $2; mul.f32 t, t, $3; "
        "fma.rn.f32 t, t, $4, $5; cvt.rn.bf16.f32 h, t; cvt.u32.u16 u, h; shl.b32 u, u, 16; "
        "mov.b32 $0, u; }",
        "=f,f,f,f,f,f",
        (Float32(x), Float32(mean), Float32(rstd), Float32(w), Float32(b)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rstd(var, eps, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 t; add.f32 t, $1, $2; rsqrt.approx.f32 $0, t; }",
        "=f,f,f",
        (Float32(var), Float32(eps)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_u32(addr, value, *, loc=None, ip=None) -> None:
    _asm("st.global.b32 [$0], $1;", "l,r", (Uint64(addr), Uint32(value)), loc=loc, ip=ip)


@dsl_user_op
def _st_u32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.b32 [$1], $2; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_f32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.f32 [$1], $2; }",
        "r,l,f",
        (Uint32(cond), Uint64(addr), Float32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bits(x, *, loc=None, ip=None) -> Uint32:
    return _asm("mov.b32 $0, $1;", "=r,f", (Float32(x),), Uint32, effects=False, loc=loc, ip=ip)


@dsl_user_op
def _cache_offset(flat, inner, s0, *, loc=None, ip=None) -> Uint64:
    """(flat / inner) * s0 + flat % inner, int64 (non-negative operands)."""
    return _asm(
        "{ .reg .u64 q, r; div.u64 q, $1, $2; rem.u64 r, $1, $2; mad.lo.u64 $0, q, $3, r; }",
        "=l,l,l,l",
        (Uint64(flat), Uint64(inner), Uint64(s0)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_f32x4(addr, *, loc=None, ip=None):
    return _asm(
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        (Uint64(addr),),
        (Float32, Float32, Float32, Float32),
        loc=loc,
        ip=ip,
    )


_ARGS = (
    "k",
    "k_stride",
    "q",
    "q_stride_tok",
    "q_stride_head",
    "w",
    "w_stride",
    "ln_w",
    "ln_b",
    "q_out",
    "w_out",
    "cache",
    "slot_fp8",
    "slot_scale",
    "n_write",
    "d1",
    "d2",
    "d3",
    "s0",
    "s1",
    "s2",
    "s3",
    "eps_bits",
    "ws_bits",
    # d1 carries the page bytes d1*d2*d3; d2, d3, s1, s2, s3 are unused.
)


class IndexerPrologueKernel:
    """Operands are 64-bit scalars (addresses, strides in elements, cache dims/strides in bytes,
    FP32 bit patterns of eps and weight_scale), in the order of ``_ARGS``."""

    @cute.jit
    def __call__(
        self,
        anchor: cute.Tensor,
        k: Int64,
        k_stride: Int64,
        q: Int64,
        q_stride_tok: Int64,
        q_stride_head: Int64,
        w: Int64,
        w_stride: Int64,
        ln_w: Int64,
        ln_b: Int64,
        q_out: Int64,
        w_out: Int64,
        cache: Int64,
        slot_fp8: Int64,
        slot_scale: Int64,
        n_write: Int64,
        d1: Int64,
        d2: Int64,
        d3: Int64,
        s0: Int64,
        s1: Int64,
        s2: Int64,
        s3: Int64,
        eps_bits: Int64,
        ws_bits: Int64,
        rows: Int32,
        stream,
    ):
        self.kernel(
            anchor,
            k,
            k_stride,
            q,
            q_stride_tok,
            q_stride_head,
            w,
            w_stride,
            ln_w,
            ln_b,
            q_out,
            w_out,
            cache,
            slot_fp8,
            slot_scale,
            n_write,
            d1,
            d2,
            d3,
            s0,
            s1,
            s2,
            s3,
            eps_bits,
            ws_bits,
        ).launch(grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        anchor: cute.Tensor,
        k: Int64,
        k_stride: Int64,
        q: Int64,
        q_stride_tok: Int64,
        q_stride_head: Int64,
        w: Int64,
        w_stride: Int64,
        ln_w: Int64,
        ln_b: Int64,
        q_out: Int64,
        w_out: Int64,
        cache: Int64,
        slot_fp8: Int64,
        slot_scale: Int64,
        n_write: Int64,
        d1: Int64,
        d2: Int64,
        d3: Int64,
        s0: Int64,
        s1: Int64,
        s2: Int64,
        s3: Int64,
        eps_bits: Int64,
        ws_bits: Int64,
    ):
        (
            k,
            k_stride,
            q,
            q_stride_tok,
            q_stride_head,
            w,
            w_stride,
            ln_w,
            ln_b,
            q_out,
            w_out,
            cache,
            slot_fp8,
            slot_scale,
            n_write,
            d1,
            d2,
            d3,
            s0,
            s1,
            s2,
            s3,
            eps_bits,
            ws_bits,
        ) = (
            Uint64(k),
            Uint64(k_stride),
            Uint64(q),
            Uint64(q_stride_tok),
            Uint64(q_stride_head),
            Uint64(w),
            Uint64(w_stride),
            Uint64(ln_w),
            Uint64(ln_b),
            Uint64(q_out),
            Uint64(w_out),
            Uint64(cache),
            Uint64(slot_fp8),
            Uint64(slot_scale),
            Uint64(n_write),
            Uint64(d1),
            Uint64(d2),
            Uint64(d3),
            Uint64(s0),
            Uint64(s1),
            Uint64(s2),
            Uint64(s3),
            Uint64(eps_bits),
            Uint64(ws_bits),
        )
        tid_i, _, _ = cute.arch.thread_idx()
        tok_i, _, _ = cute.arch.block_idx()
        tid = Uint32(tid_i)
        tok = Uint64(Uint32(tok_i))
        lane = tid & Uint32(31)
        warp = tid >> Uint32(5)
        col = Uint64(lane * Uint32(4))
        is_lane0 = _is_zero(lane)

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Every load first (the asm loads keep program order, so this is what lets them overlap).
        q_base = q + tok * q_stride_tok * Uint64(2)
        w_base = w + tok * w_stride * Uint64(2)
        heads = []
        xs = []
        whs = []
        for i in cutlass.range_constexpr(HEADS // WARPS):
            head = Uint64(warp * Uint32(HEADS // WARPS) + Uint32(i))
            heads.append(head)
            xs.append(_ld_bf16x4(q_base + (head * q_stride_head + col) * Uint64(2)))
            whs.append(_ld_bf16(w_base + head * Uint64(2)))
        k0, k1, k2, k3 = _ld_bf16x4(k + (tok * k_stride + col) * Uint64(2))
        g0, g1, g2, g3 = _ld_f32x4(ln_w + col * Uint64(4))
        b0, b1, b2, b3 = _ld_f32x4(ln_b + col * Uint64(4))
        slot = _ld_s64_if(n_write, slot_fp8 + tok * Uint64(8))
        slot_s = _ld_s64_if(n_write, slot_scale + tok * Uint64(8))

        q_row = q_out + tok * Uint64(HEADS * DIM)
        w_row = w_out + tok * Uint64(HEADS * 4)
        ws = _f32_of(Uint32(ws_bits))
        for i in cutlass.range_constexpr(HEADS // WARPS):
            x0, x1, x2, x3 = xs[i]
            scale, inv = _scale_inv(_warp_max_abs(x0, x1, x2, x3))
            _st_u32(q_row + heads[i] * Uint64(DIM) + col, _fp8x4(x0, x1, x2, x3, inv))
            _st_f32_if(is_lane0, w_row + heads[i] * Uint64(4), _mul2(whs[i], scale, ws))

        # Keys: warp 0.
        if warp == Uint32(0):
            mean = _warp_sum(_add4(k0, k1, k2, k3)) * Float32(1.0 / DIM)
            c0, c1, c2, c3 = k0 - mean, k1 - mean, k2 - mean, k3 - mean
            var = _warp_sum(_sq4(c0, c1, c2, c3)) * Float32(1.0 / DIM)
            rstd = _rstd(var, _f32_of(Uint32(eps_bits)))
            v0 = _ln_bf16(k0, mean, rstd, g0, b0)
            v1 = _ln_bf16(k1, mean, rstd, g1, b1)
            v2 = _ln_bf16(k2, mean, rstd, g2, b2)
            v3 = _ln_bf16(k3, mean, rstd, g3, b3)
            scale, inv = _scale_inv(_warp_max_abs(v0, v1, v2, v3))
            codes = _fp8x4(v0, v1, v2, v3, inv)
            write = _write_ok(tok, n_write, slot, slot_s)
            # The view's inner dims [d1, d2, d3] are contiguous (checked by the host): the scatter
            # kernel's unravel is (flat / (d1 d2 d3)) * s0 + flat % (d1 d2 d3), and a token's 128
            # code bytes stay inside one page.
            _st_u32_if(write, cache + _cache_offset(slot, d1, s0) + col, codes)
            _st_u32_if(write & is_lane0, cache + _cache_offset(slot_s, d1, s0), _bits(scale))


@dsl_user_op
def _mul2(w, scale, ws, *, loc=None, ip=None) -> Float32:
    """``(w * scale) * ws``, two RN multiplies."""
    return _asm(
        "{ .reg .f32 t; mul.rn.f32 t, $1, $2; mul.rn.f32 $0, t, $3; }",
        "=f,f,f,f",
        (Float32(w), Float32(scale), Float32(ws)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _add4(a, b, c, d, *, loc=None, ip=None) -> Float32:
    return _asm(
        "{ .reg .f32 s, t; add.rn.f32 s, $1, $2; add.rn.f32 t, $3, $4; add.rn.f32 $0, s, t; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sq4(a, b, c, d, *, loc=None, ip=None) -> Float32:
    """Sum of squares without FMA contraction (mul then add, each RN)."""
    return _asm(
        "{ .reg .f32 x, y, s, t; mul.rn.f32 x, $1, $1; mul.rn.f32 y, $2, $2; add.rn.f32 s, x, y; "
        "mul.rn.f32 x, $3, $3; mul.rn.f32 y, $4, $4; add.rn.f32 t, x, y; add.rn.f32 $0, s, t; }",
        "=f,f,f,f,f",
        (Float32(a), Float32(b), Float32(c), Float32(d)),
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_s64_if(n, addr, *, loc=None, ip=None) -> Uint64:
    """``*addr`` (int64) if ``n`` > 0, else -1."""
    return _asm(
        "{ .reg .pred p; setp.ne.u64 p, $1, 0; mov.b64 $0, -1; @p ld.global.nc.b64 $0, [$2]; }",
        "=l,l,l",
        (Uint64(n), Uint64(addr)),
        Uint64,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f32_of(bits, *, loc=None, ip=None) -> Float32:
    return _asm("mov.b32 $0, $1;", "=f,r", (Uint32(bits),), Float32, effects=False, loc=loc, ip=ip)


@dsl_user_op
def _is_zero(x, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.eq.u32 p, $1, 0; selp.u32 $0, 1, 0, p; }",
        "=r,r",
        (Uint32(x),),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _write_ok(tok, n_write, slot, slot_s, *, loc=None, ip=None) -> Uint32:
    """tok < n_write and both slots non-negative."""
    return _asm(
        "{ .reg .pred p, q, r; setp.lt.s64 p, $1, $2; setp.ge.s64 q, $3, 0; setp.ge.s64 r, $4, 0; "
        "and.pred p, p, q; and.pred p, p, r; selp.u32 $0, 1, 0, p; }",
        "=r,l,l,l,l",
        (Uint64(tok), Uint64(n_write), Uint64(slot), Uint64(slot_s)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


def _architecture(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


@functools.lru_cache(maxsize=None)
def _compile(arch: str):
    anchor = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), stride_order=(0,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        IndexerPrologueKernel(),
        anchor,
        *([Int64(0)] * len(_ARGS)),
        Int32(1),
        stream,
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


def _f32_bits(x: float) -> int:
    import struct

    return struct.unpack("<i", struct.pack("<f", x))[0]


class IndexerPrologue:
    """Launches the prologue (``launch_indexer_prologue`` with RoPE off, as one CuTe DSL kernel)."""

    def __init__(self, device: torch.device):
        self.device = device
        with torch.cuda.device(device):
            self.compiled = _compile(_architecture(device))
        self.anchor = torch.zeros(1, dtype=torch.int32, device=device)

    def __call__(
        self,
        k: torch.Tensor,
        q: torch.Tensor,
        w: torch.Tensor,
        ln_w: torch.Tensor,
        ln_b: torch.Tensor,
        eps: float,
        weight_scale: float,
        q_out: torch.Tensor,
        w_out: torch.Tensor,
        cache: torch.Tensor | None,
        slot_fp8: torch.Tensor | None,
        slot_scale: torch.Tensor | None,
        n_write: int,
    ) -> None:
        """k [rows, 128], q [rows, 32, 128], w [rows, 32] BF16 (unit inner stride); LayerNorm
        weight/bias FP32 [128]; q_out FP8 [rows, 32, 128] and w_out FP32 [rows, 32] contiguous;
        the first ``n_write`` keys go to ``cache`` at the int64 slot mappings."""
        rows = k.shape[0]
        if rows == 0:
            return
        assert k.dtype == q.dtype == w.dtype == torch.bfloat16
        assert k.stride(-1) == 1 and q.stride(-1) == 1 and w.stride(-1) == 1
        assert tuple(q.shape[1:]) == (HEADS, DIM) and k.shape[1] == DIM and w.shape[1] == HEADS
        assert (
            ln_w.dtype == ln_b.dtype == torch.float32
            and ln_w.is_contiguous()
            and ln_b.is_contiguous()
        )
        assert q_out.dtype == torch.float8_e4m3fn and q_out.is_contiguous()
        assert w_out.dtype == torch.float32 and w_out.is_contiguous()
        if cache is not None and n_write > 0:
            c = cache.view(torch.uint8)
            dims, strides = c.shape, c.stride()
            # Inner dims contiguous (the pool page view): the scatter's unravel is one division.
            assert c.dim() == 4 and strides[3] == 1 and strides[2] == dims[3]
            assert strides[1] == dims[2] * dims[3] and strides[0] % 4 == 0 and dims[3] % 4 == 0
            assert slot_fp8.dtype == slot_scale.dtype == torch.int64
            cache_args = (
                c.data_ptr(),
                slot_fp8.data_ptr(),
                slot_scale.data_ptr(),
                int(n_write),
                dims[1] * dims[2] * dims[3],
                0,
                0,
                strides[0],
                0,
                0,
                0,
            )
        else:
            cache_args = (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0)
        self.compiled(
            self.anchor,
            k.data_ptr(),
            k.stride(0),
            q.data_ptr(),
            q.stride(0),
            q.stride(1),
            w.data_ptr(),
            w.stride(0),
            ln_w.data_ptr(),
            ln_b.data_ptr(),
            q_out.data_ptr(),
            w_out.data_ptr(),
            *cache_args,
            _f32_bits(eps),
            _f32_bits(weight_scale),
            rows,
        )


_PROLOGUES: dict = {}


def indexer_prologue(device: torch.device) -> IndexerPrologue:
    index = device.index if device.index is not None else torch.cuda.current_device()
    runner = _PROLOGUES.get(index)
    if runner is None:
        runner = _PROLOGUES[index] = IndexerPrologue(torch.device("cuda", index))
    return runner
