# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 strict (greedy) verification in one launch (CuTe DSL).

What ``ops/spec/dflash2_verify.py``'s ``dflash2_greedy_verify`` computes in torch: the first
maximum of every target row (``torch.argmax``: NaN above everything, ties to the lower index,
``-0 == +0``), ``tau`` = the leading drafts equal to those maxima and the bonus = the maximum of
row ``tau``. Integer results only, so bitwise for every input.

Grid: one CTA of 256 threads per (request, target row, slice of the row's columns); small
batches split each row over more slices (:func:`greedy_slice_plan`, a function of the shape).
A CTA reads its columns once in 16-byte vectors (float32, bfloat16 or float16 logits), keeps a
first maximum per vector lane and reduces them; the last CTA of a request (a per-request arrival
counter it resets, so every launch reuses the workspace and CUDA-graph replay is safe) combines
the slices of every row and writes ``tau`` and the bonus.
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .dflash2_block_verify import (
    _argmax_wins,
    _asm,
    _atom_add_global,
    _fence_gpu,
    _fv,
    _ilt,
    _iv,
    _ld_cg_f32,
    _ld_cg_i32,
    _ld_nc_i32,
    _ld_nc_i64,
    _ld_s_f32,
    _ld_s_i32,
    _lv,
    _self,
    _seli,
    _shfl_xor_f32,
    _shfl_xor_i32,
    _st_f32,
    _st_i32,
    _st_i64,
    _st_s_f32,
    _st_s_i32,
)

THREADS = 256
WARPS = THREADS // 32
#: Most drafted tokens (block width - 1) and most CTAs one row is split over.
MAX_DRAFTS = 15
MAX_SLICES = 32
_DTYPES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}


@dsl_user_op
def _ld_nc_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [j], loc=loc, ip=ip)) for j in range(4))


@dsl_user_op
def _ld_nc_b16(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 h;\nld.global.nc.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


def _unpack_halves(word, kind):
    """The two 16-bit values of ``word`` (low half first) as float32."""
    if kind == "bf16":
        text_lo = "shl.b32 $0, $1, 16;"
        text_hi = "and.b32 $0, $1, 0xffff0000;"
        return [Float32(_asm(T.f32(), [_iv(word)], t, "=f,r", False)) for t in (text_lo, text_hi)]
    text_lo = "{\n.reg .b16 a, b;\nmov.b32 {a, b}, $1;\ncvt.f32.f16 $0, a;\n}"
    text_hi = "{\n.reg .b16 a, b;\nmov.b32 {a, b}, $1;\ncvt.f32.f16 $0, b;\n}"
    return [Float32(_asm(T.f32(), [_iv(word)], t, "=f,r", False)) for t in (text_lo, text_hi)]


@dsl_user_op
def _bits_to_f32(bits, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_iv(bits, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", False, loc=loc, ip=ip)
    )


def _half_to_f32(bits16, kind):
    if kind == "bf16":
        return Float32(_asm(T.f32(), [_iv(bits16)], "shl.b32 $0, $1, 16;", "=f,r", False))
    return Float32(
        _asm(
            T.f32(),
            [_iv(bits16)],
            "{\n.reg .b16 a;\ncvt.u16.u32 a, $1;\ncvt.f32.f16 $0, a;\n}",
            "=f,r",
            False,
        )
    )


@dsl_user_op
def _argmax_step(v, i, bv, bi, *, loc=None, ip=None):
    """Fold column ``i`` (larger than every column folded so far) into the first maximum
    ``(bv, bi)``: it wins if larger, or NaN over a number."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [
            _fv(v, loc=loc, ip=ip),
            _iv(i, loc=loc, ip=ip),
            _fv(bv, loc=loc, ip=ip),
            _iv(bi, loc=loc, ip=ip),
        ],
        "{\n.reg .pred gt, nv, nb, t;\n"
        "setp.gt.f32 gt, $2, $4;\nsetp.nan.f32 nv, $2, $2;\nsetp.num.f32 nb, $4, $4;\n"
        "and.pred t, nv, nb;\nor.pred t, t, gt;\n"
        "selp.f32 $0, $2, $4, t;\nselp.b32 $1, $3, $5, t;\n}",
        "=f,=r,f,r,f,r",
        False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


class DFlash2GreedyVerifyKernel:
    """``(tau, bonus)`` of every request by strict acceptance; see the module docstring.

    Args:
        drafts: g, drafted tokens per request.
        vocab: V, columns per target row.
        kind: logits dtype, ``"f32"``, ``"bf16"`` or ``"f16"``.
        draft_i64: whether the drafted tokens are int64 (else int32).
    """

    def __init__(self, drafts: int, vocab: int, kind: str, draft_i64: bool):
        if not 1 <= drafts <= MAX_DRAFTS:
            raise ValueError(f"1..{MAX_DRAFTS} drafted tokens, got {drafts}")
        self.g = drafts
        self.vocab = vocab
        self.kind = kind
        self.esize = 4 if kind == "f32" else 2
        self.per_vec = 16 // self.esize  # columns per 16-byte vector
        self.vec = vocab % self.per_vec == 0
        self.draft_i64 = draft_i64

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        s_b: Int64,
        s_i: Int64,
        drafts: Int64,
        sd_b: Int64,
        sd_s: Int64,
        tau_t: cute.Tensor,
        bonus_out: Int64,
        sl_v: Int64,
        sl_x: Int64,
        counters: Int64,
        num_reqs: Int32,
        slices: Int32,
        per_slice: Int32,
        stream,
    ):
        @cute.struct
        class SharedStorage:
            red_v: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, WARPS], 16]
            red_i: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, WARPS], 16]
            misc_i: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            logits,
            s_b,
            s_i,
            drafts,
            sd_b,
            sd_s,
            tau_t,
            bonus_out,
            sl_v,
            sl_x,
            counters,
            slices,
            per_slice,
        ).launch(
            grid=(num_reqs, self.g + 1, slices),
            block=(THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        s_b: Int64,
        s_i: Int64,
        drafts: Int64,
        sd_b: Int64,
        sd_s: Int64,
        tau_t: cute.Tensor,
        bonus_out: Int64,
        sl_v: Int64,
        sl_x: Int64,
        counters: Int64,
        slices: Int32,
        per_slice: Int32,
    ):
        g, V, kind, esize, pv = self.g, self.vocab, self.kind, self.esize, self.per_vec
        draft_bytes = 8 if self.draft_i64 else 4
        tau_out = tau_t.iterator.toint()
        tid, _, _ = cute.arch.thread_idx()
        req, row, sl = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        red_v = storage.red_v.data_ptr().toint()
        red_i = storage.red_i.data_ptr().toint()
        misc_i = storage.misc_i.data_ptr().toint()
        b64 = Int64(req)
        rows_total = Int64(g + 1)
        row_base = logits + (b64 * s_b + Int64(row) * s_i) * esize

        col_begin = sl * per_slice
        col_end = col_begin + per_slice
        if col_end > V:
            col_end = Int32(V)

        # First maximum of this slice: one running maximum per vector lane (each sees its
        # columns in increasing order), then the CTA's.
        ninf = Float32(float("-inf"))
        big = Int32(2147483647)
        best_v = ninf
        best_i = big
        if cutlass.const_expr(self.vec):
            bvs = [ninf] * pv
            bis = [big] * pv
            b0, b1, b2, b3, b4, b5, b6, b7 = (bvs + [ninf] * 8)[:8]
            j0, j1, j2, j3, j4, j5, j6, j7 = (bis + [big] * 8)[:8]
            for col in cutlass.range(col_begin + tid * pv, col_end, THREADS * pv, unroll=1):
                words = _ld_nc_v4_b32(row_base + Int64(col) * esize)
                if cutlass.const_expr(kind == "f32"):
                    vals = [_bits_to_f32(wd) for wd in words]
                else:
                    vals = []
                    for q in cutlass.range_constexpr(4):
                        vals += _unpack_halves(words[q], kind)
                cur_v = [b0, b1, b2, b3, b4, b5, b6, b7][:pv]
                cur_i = [j0, j1, j2, j3, j4, j5, j6, j7][:pv]
                for c in cutlass.range_constexpr(pv):
                    cur_v[c], cur_i[c] = _argmax_step(vals[c], col + c, cur_v[c], cur_i[c])
                cur_v = cur_v + [ninf] * (8 - pv)
                cur_i = cur_i + [big] * (8 - pv)
                b0, b1, b2, b3, b4, b5, b6, b7 = cur_v
                j0, j1, j2, j3, j4, j5, j6, j7 = cur_i
            lanes_v = [b0, b1, b2, b3, b4, b5, b6, b7][:pv]
            lanes_i = [j0, j1, j2, j3, j4, j5, j6, j7][:pv]
            for c in cutlass.range_constexpr(pv):
                win = _argmax_wins(lanes_v[c], lanes_i[c], best_v, best_i)
                best_v = _self(win, lanes_v[c], best_v)
                best_i = _seli(win, lanes_i[c], best_i)
        else:
            for col in cutlass.range(col_begin + tid, col_end, THREADS, unroll=1):
                if cutlass.const_expr(kind == "f32"):
                    v = _bits_to_f32(_ld_nc_i32(row_base + Int64(col) * 4))
                else:
                    v = _half_to_f32(_ld_nc_b16(row_base + Int64(col) * 2), kind)
                best_v, best_i = _argmax_step(v, col, best_v, best_i)
        for m in cutlass.range_constexpr(5):
            ov = _shfl_xor_f32(best_v, 1 << m)
            oi = _shfl_xor_i32(best_i, 1 << m)
            win = _argmax_wins(ov, oi, best_v, best_i)
            best_v = _self(win, ov, best_v)
            best_i = _seli(win, oi, best_i)
        if lane == 0:
            _st_s_f32(red_v + 4 * warp, best_v)
            _st_s_i32(red_i + 4 * warp, best_i)
        cute.arch.barrier()
        if tid == 0:
            bv = _ld_s_f32(red_v)
            bi = _ld_s_i32(red_i)
            for wi in cutlass.range_constexpr(1, WARPS):
                ov = _ld_s_f32(red_v + 4 * wi)
                oi = _ld_s_i32(red_i + 4 * wi)
                win = _argmax_wins(ov, oi, bv, bi)
                bv = _self(win, ov, bv)
                bi = _seli(win, oi, bi)
            off = ((b64 * rows_total + Int64(row)) * MAX_SLICES + Int64(sl)) * 4
            _st_f32(sl_v + off, bv)
            _st_i32(sl_x + off, bi)

        # Arrival: the last CTA of the request combines the slices and accepts.
        cute.arch.barrier()
        if tid == 0:
            _fence_gpu()
            old = _atom_add_global(counters + b64 * 4, Int32(1))
            last = Int32(0)
            if old == (g + 1) * slices - 1:
                last = Int32(1)
                _fence_gpu()
            _st_s_i32(misc_i, last)
        cute.arch.barrier()
        if _ld_s_i32(misc_i) != 0:
            if warp == 0:
                in_sl = _ilt(lane, slices)
                lane_sl = _seli(in_sl, lane, Int32(0))
                bests = []
                for s in cutlass.range_constexpr(g + 1):
                    off = ((b64 * rows_total + Int64(s)) * MAX_SLICES + Int64(lane_sl)) * 4
                    bv = _self(in_sl, _ld_cg_f32(sl_v + off), ninf)
                    bi = _seli(in_sl, _ld_cg_i32(sl_x + off), big)
                    for m in cutlass.range_constexpr(5):
                        ov = _shfl_xor_f32(bv, 1 << m)
                        oi = _shfl_xor_i32(bi, 1 << m)
                        win = _argmax_wins(ov, oi, bv, bi)
                        bv = _self(win, ov, bv)
                        bi = _seli(win, oi, bi)
                    bests.append(bi)
                if lane == 0:
                    tau = Int32(0)
                    run = Int32(1)
                    for s in cutlass.range_constexpr(g):
                        addr = drafts + (b64 * sd_b + Int64(s) * sd_s) * draft_bytes
                        if cutlass.const_expr(self.draft_i64):
                            d = _ld_nc_i64(addr)
                        else:
                            d = Int64(_ld_nc_i32(addr))
                        same = Int32(0)
                        if d == Int64(bests[s]):
                            same = Int32(1)
                        run = run * same
                        tau = tau + run
                    bonus = bests[0]
                    for s in cutlass.range_constexpr(1, g + 1):
                        bonus = _seli(
                            _ilt(Int32(s - 1), tau) * _ilt(tau, Int32(s + 1)), bests[s], bonus
                        )
                    _st_i64(tau_out + b64 * 8, Int64(tau))
                    _st_i64(bonus_out + b64 * 8, Int64(bonus))
                    _st_i32(counters + b64 * 4, Int32(0))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled(drafts, vocab, kind, draft_i64):
    kernel = DFlash2GreedyVerifyKernel(drafts, vocab, kind, draft_i64)
    a = Int64(1 << 20)
    i = Int32(1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    tau_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),))
    return cute.compile(
        kernel,
        *([a] * 6),
        tau_fake,
        *([a] * 4),
        i,
        i,
        i,
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def greedy_slice_plan(num_reqs: int, drafts: int, vocab: int, per_vec: int, num_sms: int):
    """``(slices, columns per slice)``: about four CTAs per SM, every slice a whole number of
    the CTA's 16-byte sweeps. A function of the shape only."""

    def ceil_div(a, b):
        return -(-a // b)

    sweep = THREADS * per_vec
    rows = num_reqs * (drafts + 1)
    want = max(1, ceil_div(vocab * rows, 4 * num_sms))
    per_slice = sweep * min(ceil_div(vocab, sweep), ceil_div(want, sweep))
    if ceil_div(vocab, per_slice) > MAX_SLICES:
        per_slice = sweep * ceil_div(ceil_div(vocab, MAX_SLICES), sweep)
    return ceil_div(vocab, per_slice), per_slice


#: Launch scratch per (device, power-of-two request capacity, drafts), sliced to each call's requests: a bounded
#: set however the batch size varies. Entries are never replaced, so a captured graph keeps a live buffer.
_WORKSPACES: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _workspace(device, num_reqs, drafts):
    capacity = 1 << (num_reqs - 1).bit_length()
    key = (device, capacity, drafts)
    ws = _WORKSPACES.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dflash2_greedy_verify must run once eagerly for this shape before CUDA graph capture"
            )
        rows = capacity * (drafts + 1)
        ws = (
            torch.empty(rows * MAX_SLICES, dtype=torch.float32, device=device),
            torch.empty(rows * MAX_SLICES, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    values, indices, counters = ws
    rows = num_reqs * (drafts + 1)
    return values[: rows * MAX_SLICES], indices[: rows * MAX_SLICES], counters[:num_reqs]


def dflash2_greedy_verify_cute(
    target_logits: torch.Tensor, draft_tokens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(tau [n] int64, bonus [n] int64)``; arguments as ``dflash2_greedy_verify``.

    ``target_logits`` float32 / bfloat16 / float16 ``[n, g + 1, V]`` with a unit vocab stride
    (16-byte aligned rows when ``V`` is a multiple of a 16-byte vector), ``draft_tokens``
    int32/int64 ``[n, g]`` (any strides). Compiles once per (g, V, dtypes); the first call of a
    batch size allocates the workspace, so it must be eager.
    """
    n, rows, vocab = target_logits.shape
    g = rows - 1
    device = target_logits.device
    tau = torch.empty(n, dtype=torch.int64, device=device)
    bonus = torch.empty(n, dtype=torch.int64, device=device)
    if n == 0:
        return tau, bonus
    kind = _DTYPES[target_logits.dtype]
    compiled = _compiled(g, vocab, kind, draft_tokens.dtype == torch.int64)
    per_vec = 16 // target_logits.element_size()
    slices, per_slice = greedy_slice_plan(n, g, vocab, per_vec, _num_sms(device.index or 0))
    sl_v, sl_x, counters = _workspace(device, n, g)
    compiled(
        target_logits.data_ptr(),
        target_logits.stride(0),
        target_logits.stride(1),
        draft_tokens.data_ptr(),
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        tau,
        bonus.data_ptr(),
        sl_v.data_ptr(),
        sl_x.data_ptr(),
        counters.data_ptr(),
        n,
        slices,
        per_slice,
    )
    return tau, bonus


__all__ = ["DFlash2GreedyVerifyKernel", "dflash2_greedy_verify_cute", "greedy_slice_plan"]
