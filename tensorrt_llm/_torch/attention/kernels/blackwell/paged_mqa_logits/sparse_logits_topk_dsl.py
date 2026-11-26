# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact Top-K of BF16 logit rows that ranks 8-column groups first (CuTe DSL), ties lowest column first.

A row's Top-K lies in its ``top_k`` best groups by (smallest radix key, index), so one CTA per row
radix-selects those groups, then ranks only their ``8 * top_k`` logits.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import SmemAllocator

from tensorrt_llm._torch.kernels.blackwell.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

GROUP = 8
THREADS = 256
# Descending-order key of an excluded logit (above every BF16 key).
_NO_KEY = 0x10000


def _asm(operands, text, constraints, result=None, loc=None, ip=None):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_b32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r, l",
            result=T.i32(),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_v4(addr, *, loc=None, ip=None):
    out = _asm(
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        result=llvm.StructType.get_literal([T.i32()] * 4),
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r, r",
            result=T.i32(),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_b32(addr, value, *, loc=None, ip=None):
    _asm(
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r, r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_shared_add(addr, value, *, loc=None, ip=None):
    _asm(
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r, r",
        loc=loc,
        ip=ip,
    )


@cute.jit
def _ordered_key(value):
    """The radix Top-K's descending-order key of the BF16 in the low half of ``value``: a larger
    value has a smaller key, +0 ranks above -0."""
    bits = value & Int32(0xFFFF)
    key = (bits ^ Int32(0xFFFF)) & Int32(0x7FFF)
    if (bits & Int32(0x8000)) != Int32(0):
        key = bits
    return key


class SparseLogitsTopKKernel:
    """One CTA of 256 threads per row; see the module docstring."""

    def __init__(self, top_k: int, groups: int, pdl: bool):
        if groups % THREADS or top_k % THREADS or top_k > groups:
            raise ValueError("Sparse logits Top-K: groups and top_k must be multiples of 256")
        self.top_k = top_k
        self.groups = groups
        self.pdl = pdl
        self.warps = THREADS // 32

    @cute.jit
    def __call__(
        self,
        logits: cute.Pointer,
        counts: cute.Pointer,
        positions: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        logit_stride: Int64,
        pos_stride: Int64,
        out_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(logits, counts, positions, out, logit_stride, pos_stride, out_stride).launch(
            grid=[rows, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl
        )

    @cute.jit
    def scan(self, value, sums, slot, tid):
        """Exclusive prefix sum of ``value`` over the CTA's threads (one barrier; ``slot`` selects a
        private row of ``sums``)."""
        lane = tid % 32
        warp = tid // 32
        incl = value
        for x in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, 1 << x)
            if lane >= (1 << x):
                incl = incl + up
        if lane == 31:
            _st_shared_b32(sums + (slot * self.warps + warp) * 4, incl)
        cute.arch.sync_threads()
        before = incl - value
        for w in cutlass.range_constexpr(self.warps):
            v = _ld_shared_b32(sums + (slot * self.warps + w) * 4)
            if w < warp:
                before = before + v
        return before

    @cute.jit
    def bucket(self, bins, remaining, lane):
        """The bin of ``bins`` (256 counts) holding the ``remaining``-th smallest key and that key's
        rank within the bin, found by every warp on its own (lane l scans bins 8 l .. 8 l + 7)."""
        total = Int32(0)
        for i in cutlass.range_constexpr(8):
            total = total + _ld_shared_b32(bins + (8 * lane + i) * 4)
        incl = total
        for x in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, 1 << x)
            if lane >= (1 << x):
                incl = incl + up
        run = incl - total
        found = Int32(0)
        within = Int32(0)
        for i in cutlass.range_constexpr(8):
            c = _ld_shared_b32(bins + (8 * lane + i) * 4)
            if (run < remaining) & (remaining <= run + c):
                found = 8 * lane + i
                within = remaining - run
            run = run + c
        owner = (incl - total < remaining) & (remaining <= incl)
        src = Int32(cute.arch.vote_ballot_sync(owner))
        src_lane = Int32(cute.arch.popc(src ^ (src - 1))) - 1
        found = cute.arch.shuffle_sync(found, src_lane)
        within = cute.arch.shuffle_sync(within, src_lane)
        return found, within

    @cute.jit
    def radix(self, keys, count, k, hist, lane):
        """The k-th smallest of the CTA's keys below _NO_KEY and how many keys equal to it rank
        within the k smallest (``hist``: two zeroed 256-bin histograms; one barrier per pass)."""
        for i in cutlass.range_constexpr(count):
            key = keys[i]
            if key < Int32(_NO_KEY):
                _red_shared_add(hist + (key >> 8) * 4, Int32(1))
        cute.arch.sync_threads()
        high, remaining = self.bucket(hist, Int32(k), lane)
        for i in cutlass.range_constexpr(count):
            key = keys[i]
            if (key < Int32(_NO_KEY)) & ((key >> 8) == high):
                _red_shared_add(hist + (256 + (key & 255)) * 4, Int32(1))
        cute.arch.sync_threads()
        low, remaining = self.bucket(hist + 256 * 4, remaining, lane)
        return (high << 8) | low, remaining

    @cute.jit
    def flags(self, keys, count, thr, quota, sums, slot, tid):
        """Per key 1 if it ranks within the k smallest (``radix``'s threshold and quota, ties in
        thread then index order), else 0, written over ``keys``; the thread's first output slot."""
        less = Int32(0)
        ties = Int32(0)
        for i in cutlass.range_constexpr(count):
            if keys[i] < thr:
                less = less + 1
            if keys[i] == thr:
                ties = ties + 1
        before = self.scan(less | (ties << 16), sums, slot, tid)
        rank = before >> 16
        for i in cutlass.range_constexpr(count):
            t = Int32(0)
            if keys[i] < thr:
                t = Int32(1)
            if keys[i] == thr:
                if rank < quota:
                    t = Int32(1)
                rank = rank + 1
            keys[i] = t
        return (before & 0xFFFF) + cutlass.min(before >> 16, quota)

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        counts: cute.Pointer,
        positions: cute.Pointer,
        out: cute.Pointer,
        logit_stride: Int64,
        pos_stride: Int64,
        out_stride: Int64,
    ):
        top_k = cutlass.const_expr(self.top_k)
        bpt = cutlass.const_expr(self.groups // THREADS)
        gpt = cutlass.const_expr(top_k // THREADS)
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid % 32
        smem = SmemAllocator()
        # Stage A and stage B histograms, two passes each.
        hist_a = smem.allocate_array(Int32, 512, byte_alignment=16).toint()
        hist_b = smem.allocate_array(Int32, 512, byte_alignment=16).toint()
        sums = smem.allocate_array(Int32, 2 * self.warps, byte_alignment=16).toint()
        group_keys = smem.allocate_array(Int32, self.groups, byte_alignment=16).toint()
        chosen = smem.allocate_array(Int32, top_k, byte_alignment=16).toint()
        for i in cutlass.range_constexpr(2):
            _st_shared_b32(hist_a + (i * THREADS + tid) * 4, Int32(0))
            _st_shared_b32(hist_b + (i * THREADS + tid) * 4, Int32(0))
        if cutlass.const_expr(self.pdl):
            griddepcontrol_wait()
        n = _ld_global_b32(counts.toint() + Int64(row) * 4)
        pos_row = positions.toint() + Int64(row) * pos_stride * 4
        out_row = out.toint() + Int64(row) * out_stride * 4
        if n <= top_k:
            for i in cutlass.range(tid, top_k, THREADS, unroll=1):
                v = Int32(-1)
                if i < n:
                    v = _ld_global_b32(pos_row + Int64(i) * 4)
                _st_global_b32(out_row + Int64(i) * 4, v)
        else:
            nb = (n + GROUP - 1) // GROUP
            lrow = logits.toint() + Int64(row) * logit_stride * 2
            # Group keys: the smallest key of each group's valid logits (coalesced 16-byte loads of
            # groups tid, tid + 256, ..), then each thread takes bpt consecutive groups.
            words = cute.make_rmem_tensor((bpt * 4,), Int32)
            for i in cutlass.range_constexpr(bpt):
                w4 = _ld_global_v4(lrow + Int64(i * THREADS + tid) * (GROUP * 2))
                for v in cutlass.range_constexpr(4):
                    words[4 * i + v] = w4[v]
            for i in cutlass.range_constexpr(bpt):
                col = (i * THREADS + tid) * GROUP
                key = Int32(_NO_KEY)
                for j in cutlass.range_constexpr(GROUP):
                    if col + j < n:
                        key = cutlass.min(
                            key, _ordered_key(words[4 * i + j // 2] >> (16 * (j % 2)))
                        )
                _st_shared_b32(group_keys + (i * THREADS + tid) * 4, key)
            cute.arch.sync_threads()
            bk = cute.make_rmem_tensor((bpt,), Int32)
            for i in cutlass.range_constexpr(bpt):
                bk[i] = _ld_shared_b32(group_keys + (tid * bpt + i) * 4)
            # The top_k groups by (key, index), in group order.
            thr = Int32(_NO_KEY)
            quota = Int32(0)
            if nb > top_k:
                thr, quota = self.radix(bk, bpt, top_k, hist_a, lane)
            slot = self.flags(bk, bpt, thr, quota, sums, 0, tid)
            for i in cutlass.range_constexpr(bpt):
                if bk[i] != Int32(0):
                    _st_shared_b32(chosen + slot * 4, tid * bpt + i)
                    slot = slot + 1
            cute.arch.sync_threads()
            # Rank the chosen groups' logits in column order.
            ns = cutlass.min(nb, Int32(top_k))
            packed = cute.make_rmem_tensor((gpt * GROUP // 2,), Int32)
            pos = cute.make_rmem_tensor((gpt * GROUP,), Int32)
            cols = cute.make_rmem_tensor((gpt,), Int32)
            # Every load of the thread's groups (logits and positions) before any use.
            for g in cutlass.range_constexpr(gpt):
                idx = tid * gpt + g
                col = Int32(-1)
                for j in cutlass.range_constexpr(GROUP // 2):
                    packed[g * GROUP // 2 + j] = Int32(0)
                for j in cutlass.range_constexpr(GROUP):
                    pos[g * GROUP + j] = Int32(-1)
                if idx < ns:
                    col = _ld_shared_b32(chosen + idx * 4) * GROUP
                    w4 = _ld_global_v4(lrow + Int64(col) * 2)
                    for j in cutlass.range_constexpr(GROUP // 2):
                        packed[g * GROUP // 2 + j] = w4[j]
                    for h in cutlass.range_constexpr(2):
                        p4 = _ld_global_v4(pos_row + Int64(col + 4 * h) * 4)
                        for j in cutlass.range_constexpr(4):
                            pos[g * GROUP + 4 * h + j] = p4[j]
                cols[g] = col
            ek = cute.make_rmem_tensor((gpt * GROUP,), Int32)
            for g in cutlass.range_constexpr(gpt):
                for j in cutlass.range_constexpr(GROUP):
                    key = Int32(_NO_KEY)
                    if (cols[g] >= 0) & (cols[g] + j < n):
                        key = _ordered_key(packed[(g * GROUP + j) // 2] >> (16 * (j % 2)))
                    ek[g * GROUP + j] = key
            thr2, quota2 = self.radix(ek, gpt * GROUP, top_k, hist_b, lane)
            slot = self.flags(ek, gpt * GROUP, thr2, quota2, sums, 1, tid)
            for i in cutlass.range_constexpr(gpt * GROUP):
                if ek[i] != Int32(0):
                    _st_global_b32(out_row + Int64(slot) * 4, pos[i])
                    slot = slot + 1
        if cutlass.const_expr(self.pdl):
            griddepcontrol_launch_dependents()


_COMPILED: dict = {}


def _ptr(dtype, tensor, align):
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def sparse_logits_topk_supported(logits: torch.Tensor, top_k: int) -> bool:
    """Whether ``sparse_logits_topk_`` ranks these rows for ``top_k``."""
    width = logits.shape[-1]
    return (
        logits.dim() == 2
        and logits.dtype == torch.bfloat16
        and width % (GROUP * THREADS) == 0
        and 0 < top_k <= width // GROUP
        and top_k % THREADS == 0
        and logits.stride(1) == 1
        and logits.stride(0) % 8 == 0
        and logits.data_ptr() % 16 == 0
    )


def sparse_logits_topk_(
    logits: torch.Tensor,
    counts: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    pdl: bool = False,
) -> torch.Tensor:
    """Per row the ``output.shape[1]`` largest of its leading ``counts`` BF16 ``logits`` as
    ``positions[row, column]`` in increasing column order, ``-1`` padding last."""
    rows, width = logits.shape
    top_k = output.shape[1]
    if (
        not sparse_logits_topk_supported(logits, top_k)
        or counts.dtype != torch.int32
        or not counts.is_contiguous()
        or counts.numel() != rows
        or positions.dtype != torch.int32
        or positions.stride(1) != 1
        or positions.stride(0) % 4
        or positions.data_ptr() % 16
        or positions.shape[0] != rows
        or positions.shape[1] < width
        or output.dtype != torch.int32
        or output.stride(1) != 1
    ):
        raise ValueError("Sparse logits Top-K: unsupported tensor geometry")
    if rows == 0:
        return output
    args = (
        _ptr(cutlass.Uint8, logits, 16),
        _ptr(cutlass.Int32, counts, 4),
        _ptr(cutlass.Int32, positions, 16),
        _ptr(cutlass.Int32, output, 4),
        Int32(rows),
        Int64(logits.stride(0)),
        Int64(positions.stride(0)),
        Int64(output.stride(0)),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    key = (logits.device.index, top_k, width // GROUP, bool(pdl))
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compile the sparse logits Top-K before CUDA graph capture")
        kernel = SparseLogitsTopKKernel(top_k, width // GROUP, bool(pdl))
        fn = _COMPILED[key] = cute.compile(kernel, *args)
    fn(*args)
    return output
