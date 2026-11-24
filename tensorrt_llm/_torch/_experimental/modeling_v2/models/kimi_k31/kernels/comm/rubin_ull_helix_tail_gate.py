# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3's Helix tail (V projection + exchange + combine) with the DSA output gate in its receive loop.

:class:`K3HelixTailGateKernel` is ``k3_helix_tail.K3HelixTailKernel`` (no split fold, no MXFP8
output) whose receivers store ``bf16(out * sigmoid(gate))`` instead of the BF16 combined ``out``.
The senders, the exchange and the combine are the parent's code; only ``receive`` differs, by
the gate after the combine:

- the BF16 gate row ``[T, H / P * 128]`` (any row stride that is a multiple of 8) is loaded
  before the receive polls, as the parent's MXFP8 path loads its gate;
- ``sigmoid`` is that of ``mla.output_gate``'s kernel, the two PyTorch BF16 kernels' bits:
  ``1 / (1 + expf(-g))`` in FP32 with the accurate ``expf`` (``cute.math.exp`` without fast
  math, libdevice's ``__nv_expf``), ``add.rn`` and ``rcp.rn`` (the correctly rounded quotient),
  rounded to BF16 (nearest even);
- the product is the FP32 product of the two BF16 values (``mul.rn``, denormals kept) rounded
  to BF16, the parent's ``_gate_mul_bf16x8``.

So the output equals ``sigmoid_gate_into(k3_helix_tail(...), gate)`` bit for bit, without the
second launch and without the combined rows' round trip through global memory, except for a
row that no rank holds a key for (every sender's max is -inf): its weights are 0 and its output
is zeros, where the ungated tail's would be NaN.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from . import rubin_ull_k3_helix_tail as _tail
from .rubin_ull_k3_helix_a2a_fused import (  # noqa: E501
    _NEG_INF_BITS,
    _all_valid2,
    _all_valid4,
    _as_f32,
    _asm,
    _bf16x2_fma,
    _corrected_exp,
    _fadd,
    _fmax,
    _fmul,
    _fv,
    _iv,
    _ld_nc_v4,
    _ld_vol_v2,
    _ld_vol_v4,
    _pack_bf16x2,
    _rcp,
    _shfl_bfly,
    _shfl_idx,
    _st_sentinel_v2,
    _st_sentinel_v4,
    _st_v4,
)
from ..gemm.rubin_ull_k3_fp8w_lowm_gemm import (  # noqa: E501
    _gate_mul_bf16x8,
)

__all__ = ["K3HelixTailGateKernel", "k3_helix_tail_gated"]

THREADS = _tail.THREADS


@dsl_user_op
def _bf16x2_to_f32(word, *, loc=None, ip=None):
    """The two BF16 values of ``word`` (low half first) as FP32 (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 l, h;\nshl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 $0, l;\nmov.b32 $1, h;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _rcp_one_plus(e, *, loc=None, ip=None):
    """``rcp.rn(1 + e)``: ``1 / (1 + e)`` correctly rounded, as ``output_gate``'s kernel."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(e, loc=loc, ip=ip)],
            "{\n.reg .f32 d;\nadd.rn.f32 d, $1, 0f3F800000;\nrcp.rn.f32 $0, d;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _sigmoid_bf16x2(word: Int32) -> Int32:
    """``bf16(sigmoid(g))`` of the two BF16 gate values of ``word`` (bitwise PyTorch's)."""
    lo, hi = _bf16x2_to_f32(word)
    s_lo = _rcp_one_plus(cute.math.exp(-lo, fastmath=False))
    s_hi = _rcp_one_plus(cute.math.exp(-hi, fastmath=False))
    return _pack_bf16x2(s_lo, s_hi)


class K3HelixTailGateKernel(_tail.K3HelixTailKernel):
    """The parent kernel (``fold`` off, no MXFP8 output) with the sigmoid gate in ``receive``."""

    def __init__(self, cp_size: int, num_heads: int, has_mask: bool, use_pdl: bool,
                 loopback: bool = False):  # fmt: skip
        super().__init__(cp_size, num_heads, False, has_mask, use_pdl, mx_out=False, debug=False,
                         timing=False, loopback=loopback)  # fmt: skip

    @cute.jit
    def receive(
        self,
        tidx: Int32,
        rcta: Int32,
        out_addr: Int64,
        recv_base: Int64,
        num_tokens: Int32,
        n_recv: Int32,
        gate_addr: Int64,
        gate_ld: Int32,
        xq_addr: Int64,
        xs_addr: Int64,
        ts: Int64,
    ):
        """The parent's receive (same polls, combine and re-arm), storing the gated output."""
        P, RV, DV, CPH = self.P, self.RV, self.DV, self.CPH
        lane = tidx % 32
        group = lane - lane % 16
        for t in cutlass.range(rcta, num_tokens, n_recv, unroll=1):
            rec0 = recv_base + Int64(t * (P * RV)) * 16
            for p in cutlass.range_constexpr(self.passes):
                c = tidx + Int32(p * THREADS)
                h = c // CPH
                col = c - h * CPH
                src_l = col % 16
                valid = Int32(c < DV)
                has_stats = valid * Int32(src_l < P)
                owns_stats = has_stats * Int32(col < 16)
                st_addr = rec0 + Int64(src_l * RV + DV) * 16 + Int64(h * 8)
                d_addr = rec0 + Int64(c) * 16
                stats = cute.make_rmem_tensor((2,), Int32)
                stats[0] = Int32(_NEG_INF_BITS)
                stats[1] = Int32(0)
                vals = cute.make_rmem_tensor((P * 4,), Int32)
                g = cute.make_rmem_tensor((4,), Int32)
                # The gate is ready before this grid starts: its load overlaps the polls.
                for e in cutlass.range_constexpr(4):
                    g[e] = Int32(0)
                if valid != 0:
                    gw = _ld_nc_v4(gate_addr + (Int64(t) * Int64(gate_ld) + Int64(c * 8)) * 2)
                    for e in cutlass.range_constexpr(4):
                        g[e] = gw[e]
                if valid != 0:
                    ready = Int32(0)
                    while ready == 0:
                        ready = Int32(1)
                        if has_stats != 0:
                            sv = _ld_vol_v2(st_addr)
                            stats[0] = sv[0]
                            stats[1] = sv[1]
                            ready = _all_valid2(sv[0], sv[1])
                        for s in cutlass.range_constexpr(P):
                            w = _ld_vol_v4(d_addr + Int64(s * RV * 16))
                            for e in cutlass.range_constexpr(4):
                                vals[s * 4 + e] = w[e]
                            ready = ready & _all_valid4(*w)
                mx = _as_f32(stats[0])
                gmax = mx
                for off in (1, 2, 4, 8):
                    gmax = _fmax(gmax, _shfl_bfly(gmax, off))
                corrected = _corrected_exp(mx, gmax, _as_f32(stats[1]))
                total = corrected
                for off in (1, 2, 4, 8):
                    total = _fadd(total, _shfl_bfly(total, off))
                weight = _fmul(corrected, _rcp(total))
                if gmax == Float32(-Float32.inf):
                    # No rank holds a key for this row (every sender's max is -inf, a padding
                    # row): -inf - -inf would make the weights NaN; the row is zeros instead.
                    weight = Float32(0.0)
                acc = [Float32(0.0)] * 8
                for s in cutlass.range_constexpr(P):
                    ws = _shfl_idx(weight, group + s)
                    for e in cutlass.range_constexpr(4):
                        acc[2 * e], acc[2 * e + 1] = _bf16x2_fma(
                            vals[s * 4 + e], ws, acc[2 * e], acc[2 * e + 1]
                        )
                o = [_pack_bf16x2(acc[2 * e], acc[2 * e + 1]) for e in range(4)]
                sg = [_sigmoid_bf16x2(g[e]) for e in range(4)]
                y = _gate_mul_bf16x8(o[0], o[1], o[2], o[3], sg[0], sg[1], sg[2], sg[3])
                if valid != 0:
                    _st_v4(out_addr + Int64(t * DV + c) * 16, y[0], y[1], y[2], y[3])
                    for s in cutlass.range_constexpr(P):
                        _st_sentinel_v4(d_addr + Int64(s * RV * 16))
                if owns_stats != 0:
                    _st_sentinel_v2(st_addr)


_KERNELS: dict = {}
_COMPILED: dict = {}


def k3_helix_tail_gated(
    rows: torch.Tensor,
    stats: torch.Tensor,
    w_uv: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    *,
    zero_kv_mask: torch.Tensor | None,
    cp_rank: int,
    cp_size: int,
    ws_addr: int,
    rank_stride: int,
    half_bytes: int,
    state: torch.Tensor,
    max_ctas: int,
    use_pdl: bool = True,
    loopback: bool = False,
) -> None:
    """``k3_helix_tail`` (``fold=None``) of ``T`` tokens storing ``bf16(out * sigmoid(gate))``.

    Args as ``k3_helix_tail``; ``gate`` is BF16 ``[T, H / P * 128]`` with unit column stride, a
    row stride that is a multiple of 8 elements and a 16-byte aligned base; ``out`` BF16
    contiguous ``[T, H / P * 128]``.
    """
    num_heads = w_uv.shape[0]
    num_tokens = rows.shape[0] // num_heads
    width = num_heads // cp_size * _tail.V_HEAD_DIM
    if num_tokens == 0:
        return
    if (
        gate.dtype != torch.bfloat16
        or gate.shape != (num_tokens, width)
        or gate.stride(1) != 1
        or gate.stride(0) % 8
        or gate.data_ptr() % 16
        or out.shape != (num_tokens, width)
        or not out.is_contiguous()
    ):
        raise ValueError("The gated Helix tail needs a BF16 gate and output of [T, H / P * 128]")
    if num_tokens * cp_size * _tail.record_bytes(num_heads // cp_size, _tail.V_HEAD_DIM) > half_bytes:
        raise ValueError("the Helix tail receive buffer is too small for this many tokens")
    has_mask = zero_kv_mask is not None
    flags = (cp_size, num_heads, has_mask, use_pdl, loopback)
    kernel = _KERNELS.get(flags)
    if kernel is None:
        kernel = _KERNELS[flags] = K3HelixTailGateKernel(*flags)
    n_tiles = _tail._ceil_div(num_tokens, _tail.ROWS)
    tiles_per_cta = _tail._ceil_div(num_heads * n_tiles, max(max_ctas, num_heads))
    n_ctas = num_heads * _tail._ceil_div(n_tiles, tiles_per_cta)
    # Never dereferenced without a fold; any valid pointers keep the parent's signature.
    ptrs = tuple(_tail._gmem(cutlass.Float32, stats, 4) for _ in range(3))
    ptrs += (_tail._gmem(cutlass.Int32, stats, 4),)
    ptrs += tuple(_tail._gmem(cutlass.Float32, stats, 4) for _ in range(2))
    args = (
        *ptrs,
        Int64(rows.data_ptr()),
        Int64(stats.data_ptr()),
        Int64(w_uv.data_ptr()),
        Int64(zero_kv_mask.data_ptr() if has_mask else 0),
        Int64(out.data_ptr()),
        Int64(ws_addr),
        Int64(state.data_ptr()),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(gate.data_ptr()),
        Int64(0),
        Int64(0),
        Int64(rank_stride),
        Int64(half_bytes),
        Float32(1.0),
        Float32(1.0),
        Int32(num_tokens),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(gate.stride(0)),
        Int32(cp_rank),
        Int32(tiles_per_cta),
        Int32(n_ctas),
        cuda.CUstream(torch.cuda.current_stream(rows.device).cuda_stream),
    )
    compiled = _COMPILED.get(flags)
    if compiled is None:
        compiled = _COMPILED[flags] = cute.compile(kernel, *args)
    compiled(*args)
