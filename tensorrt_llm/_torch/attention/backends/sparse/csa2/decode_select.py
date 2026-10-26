# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode selections of the CSA2 indexer, each in one launch.

``fused_decode_select_`` replaces the decode chain MaskLogits -> TopK -> FinalizeSelection over
paged indexer logits for at most ``MAX_ROWS`` rows: row ``r`` selects the ``top_k`` largest logits
among its columns ``c < visible[r]`` whose index page ``valid[r, c // page_rows]`` is set, ties at
the k-th value lowest column first, and writes the selected columns in increasing order followed
by ``-1`` padding -- FinalizeSelection's output for that selection.

One cluster of CTAs per row, each CTA a contiguous column range, each thread a contiguous run of
columns held as order-preserving 32-bit keys in registers (0 for an excluded column). Up to four
8-bit radix passes find the k-th key, stopping early once a pass takes its threshold digit's keys
whole: every CTA counts its keys of the current prefix in shared memory (adding runs of one digit
at once; a thread whose keys are all equal adds them with one atomic), one warp sums the
cluster's counts over distributed shared memory and scans them from the top digit. The CTAs then
push their (greater, tied) totals to the CTAs after them, which gives every thread the output slot
of its first selected column.

``select_mapped_consumer_topk_`` is a candidate consumer's decode Top-K with MapSelection folded
in. One CTA per row runs the radix-select Top-K over the row's leading ``counts[q]`` candidate
columns (the single-CTA path of ``SinglePassMultiCTARadixTopKKernel``, ties at the k-th score
lowest column first), flags the selected columns in a shared bitmap instead of writing them, and
lists them in increasing column order as ``positions[q, column]``, ``-1`` padding last. That is the
output of the decode radix-select (``cute_dsl_indexer_topk_decode``) followed by
``map_selection_``, for rows whose leading ``counts[q]`` columns map to positions ``>= 0``
(candidate expansion publishes ``-1`` only past them).
"""

import functools

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm.math_utils import ceil_div, pad_up

#: Most rows one launch serves (a cluster per row).
MAX_ROWS = 16
_THREADS = 512
_BINS = 256
#: Most columns a thread holds.
_MAX_PER_THREAD = 48
# One bitmap word per thread in the consumer Top-K's column-order emission: 32 x 1024 columns.
_MAX_COLUMNS = 32 * 1024
# Positions per bitmap word loaded ahead of the emission's scan.
_PREFETCH = 4


def plan(cols: int) -> tuple[int, int] | None:
    """(CTAs per row, columns per thread, a multiple of 4) for ``cols`` logits columns, or None
    when a row does not fit in a 16-CTA cluster's registers."""
    for cluster in (8, 16):
        per_thread = pad_up(ceil_div(cols, cluster * _THREADS), 4)
        if per_thread <= _MAX_PER_THREAD:
            return cluster, per_thread
    return None


def fused_decode_select_supported(logits: torch.Tensor, top_k: int) -> bool:
    rows, cols = logits.shape
    return (
        0 < rows <= MAX_ROWS
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
        and logits.data_ptr() % 16 == 0
        and 0 < top_k <= 2048
        and torch.cuda.get_device_capability(logits.device)[0] >= 9
        and plan(cols) is not None
    )


@functools.lru_cache(maxsize=1)
def _kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, const_expr, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    def asm(res, operands, text, constraints, *, loc=None, ip=None):
        return llvm.inline_asm(
            res,
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
    def ld_global_v4(addr: Int64, *, loc=None, ip=None):
        out = asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
            "=f,=f,=f,=f,l",
            loc=loc,
            ip=ip,
        )
        return tuple(
            Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
        )

    @dsl_user_op
    def ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u8 $0, [$1];",
                "=r,l",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def ld_global_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.s32 $0, [$1];",
                "=r,l",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_global_s32(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "st.global.s32 [$0], $1;",
            "l,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ordered(value: Float32, *, loc=None, ip=None) -> Uint32:
        """Order-preserving key of an FP32 value: larger values, larger keys (never 0)."""
        return Uint32(
            asm(
                T.i32(),
                [Float32(value).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b32 b, m;\n\t.reg .pred n;\n\t"
                "mov.b32 b, $1;\n\t"
                "setp.lt.s32 n, b, 0;\n\t"
                "selp.b32 m, 0xffffffff, 0x80000000, n;\n\t"
                "xor.b32 $0, b, m;\n\t}",
                "=r,f",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def red_shared_add(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "red.shared.add.u32 [$0], $1;",
            "r,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_shared_s32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "st.shared.s32 [$0], $1;",
            "r,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_cluster_v4(addr: Int32, rank: Int32, *, loc=None, ip=None):
        """Load 4 consecutive 32-bit words from CTA ``rank``'s shared memory at ``addr``."""
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $4, $5;\n\t"
            "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [a];\n\t}",
            "=r,=r,=r,=r,r,r",
            loc=loc,
            ip=ip,
        )
        return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4))

    @dsl_user_op
    def shfl_up(value: Int32, delta: int, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int32(value).ir_value(loc=loc, ip=ip)],
                f"shfl.sync.up.b32 $0, $1, {delta}, 0, 0xffffffff;",
                "=r,r",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def st_cluster_s32(addr: Int32, rank: Int32, value: Int32, *, loc=None, ip=None) -> None:
        """Store to CTA ``rank``'s shared memory at this CTA's offset ``addr``."""
        asm(
            None,
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(rank).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $0, $1;\n\t"
            "st.shared::cluster.s32 [a], $2;\n\t}",
            "r,r,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def shfl_idx(value: Int32, lane: int, *, loc=None, ip=None) -> Int32:
        return Int32(
            asm(
                T.i32(),
                [Int32(value).ir_value(loc=loc, ip=ip)],
                f"shfl.sync.idx.b32 $0, $1, {lane}, 31, 0xffffffff;",
                "=r,r",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def count_key(
        key,
        mask,
        prefix,
        hist,
        run_digit,
        run_count,
        low,
        high,
        shift: int,
        track: bool,
        *,
        loc=None,
        ip=None,
    ):
        """Count ``key`` (when nonzero with ``key & mask == prefix``) into the run of its digit
        ``(key >> shift) & 255``, first adding the previous run to ``hist`` when the digit
        changes, and with ``track`` widen [low, high] to it. Returns the new (run digit, run
        count, low, high)."""
        out = asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [
                Uint32(key).ir_value(loc=loc, ip=ip),
                Uint32(mask).ir_value(loc=loc, ip=ip),
                Uint32(prefix).ir_value(loc=loc, ip=ip),
                Int32(hist).ir_value(loc=loc, ip=ip),
                Int32(run_digit).ir_value(loc=loc, ip=ip),
                Int32(run_count).ir_value(loc=loc, ip=ip),
                Uint32(low).ir_value(loc=loc, ip=ip),
                Uint32(high).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred m, d, f;\n\t.reg .b32 t, g;\n\t"
            "and.b32 t, $4, $5;\n\t"
            "setp.eq.u32 m, t, $6;\n\t"
            "setp.ne.and.u32 m, $4, 0, m;\n\t"
            f"bfe.u32 g, $4, {shift}, 8;\n\t"
            "setp.ne.and.u32 d, g, $0, m;\n\t"
            "setp.ne.and.u32 f, $1, 0, d;\n\t"
            "@f mad.lo.u32 t, $0, 4, $7;\n\t"
            "@f red.shared.add.u32 [t], $1;\n\t"
            "@d mov.b32 $0, g;\n\t"
            "@d mov.b32 $1, 0;\n\t"
            "@m add.u32 $1, $1, 1;\n\t"
            + ("@m min.u32 $2, $2, $4;\n\t@m max.u32 $3, $3, $4;\n\t" if track else "")
            + "}",
            "=r,=r,=r,=r,r,r,r,r,0,1,2,3",
            loc=loc,
            ip=ip,
        )
        return (
            Int32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
            Int32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), out, [3], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def reduce_warp(value: Uint32, op: str, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            asm(
                T.i32(),
                [Uint32(value).ir_value(loc=loc, ip=ip)],
                f"redux.sync.{op}.u32 $0, $1, 0xffffffff;",
                "=r,r",
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def red_shared_u32(addr: Int32, value: Uint32, op: str, *, loc=None, ip=None) -> None:
        asm(
            None,
            [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
            f"red.shared.{op}.u32 [$0], $1;",
            "r,r",
            loc=loc,
            ip=ip,
        )

    class FusedDecodeSelect:
        def __init__(
            self, cluster: int, per_thread: int, top_k: int, page_rows: int, use_pdl: bool
        ):
            self.cluster = cluster
            self.per_thread = per_thread
            self.top_k = top_k
            self.page_rows = page_rows
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            visible: cute.Pointer,
            out: cute.Pointer,
            rows: Int32,
            cols: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            out_stride: Int64,
            stream,
        ):
            self.kernel(
                logits, valid, visible, out, cols, logits_stride, valid_stride, out_stride
            ).launch(
                grid=[self.cluster, rows, 1],
                block=[_THREADS, 1, 1],
                cluster=[self.cluster, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.jit
        def _block_scan(self, tidx, value, sWarp):
            """Exclusive prefix over the threads in order, and the block total."""
            lane = tidx % 32
            warp = tidx // 32
            inclusive = value
            for delta in (1, 2, 4, 8, 16):
                up = shfl_up(inclusive, delta)
                if lane >= delta:
                    inclusive = inclusive + up
            if lane == 31:
                sWarp[warp] = inclusive
            cute.arch.sync_threads()
            before = Int32(0)
            total = Int32(0)
            for w in cutlass.range_constexpr(_THREADS // 32):
                count = sWarp[w]
                if w < warp:
                    before = before + count
                total = total + count
            return before + inclusive - value, total

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            valid: cute.Pointer,
            visible: cute.Pointer,
            out: cute.Pointer,
            cols: Int32,
            logits_stride: Int64,
            valid_stride: Int64,
            out_stride: Int64,
        ):
            per_thread: cutlass.Constexpr = self.per_thread
            cluster: cutlass.Constexpr = self.cluster
            top_k: cutlass.Constexpr = self.top_k
            tidx, _, _ = cute.arch.thread_idx()
            rank, row, _ = cute.arch.block_idx()
            lane = tidx % 32

            smem = SmemAllocator()
            sHist = smem.allocate_tensor(Int32, cute.make_layout((2 * _BINS,)), byte_alignment=16)
            sWarp = smem.allocate_tensor(Int32, cute.make_layout((_THREADS // 32,)), 16)
            sPick = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
            # Per pass parity: the [min, max] key of the bin being refined.
            sRange = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
            sCounts = smem.allocate_tensor(Int32, cute.make_layout((2 * cluster,)), 16)
            hist_base = Int32(sHist.iterator.toint())
            counts_base = Int32(sCounts.iterator.toint())
            range_base = Int32(sRange.iterator.toint())

            if tidx < _BINS:
                st_shared_s32(hist_base + tidx * 4, Int32(0))
            if const_expr(self.use_pdl):
                griddepcontrol_wait()
            first = (rank * _THREADS + tidx) * per_thread
            limit = ld_global_s32(Int64(visible.toint()) + Int64(row) * 4)
            if limit > cols:
                limit = cols
            row_logits = Int64(logits.toint()) + Int64(row) * logits_stride * 4
            row_valid = Int64(valid.toint()) + Int64(row) * valid_stride
            keys = cute.make_rmem_tensor(cute.make_layout((per_thread,)), Uint32)
            for v in cutlass.range_constexpr(per_thread // 4):
                c0 = first + 4 * v
                x0 = Float32(0.0)
                x1 = Float32(0.0)
                x2 = Float32(0.0)
                x3 = Float32(0.0)
                backed = Int32(0)
                if c0 < limit:
                    x0, x1, x2, x3 = ld_global_v4(row_logits + Int64(c0) * 4)
                    backed = ld_global_u8(row_valid + Int64(c0 // self.page_rows))
                for j, x in enumerate((x0, x1, x2, x3)):
                    key = Uint32(0)
                    if (c0 + j < limit) & (backed != 0):
                        key = ordered(x)
                    keys[4 * v + j] = key
            if const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            head = keys[0]
            same = keys[1] == head
            for e in cutlass.range_constexpr(2, per_thread):
                same = same & (keys[e] == head)
            cute.arch.sync_threads()

            # Up to four radix passes over the keys of the current prefix, top digit first.
            prefix = Uint32(0)
            prefix_mask = Uint32(0)
            need = Int32(0)
            done = Int32(0)
            for p in cutlass.range_constexpr(4):
                if done == Int32(0):
                    shift = 24 - 8 * p
                    hist = hist_base + (p % 2) * _BINS * 4
                    span = range_base + (p % 2) * 16
                    need_in = need
                    low = Uint32(0xFFFFFFFF)
                    high = Uint32(0)
                    if same:
                        if (head != Uint32(0)) & ((head & prefix_mask) == prefix):
                            digit = Int32((head >> Uint32(shift)) & Uint32(_BINS - 1))
                            red_shared_add(hist + digit * 4, Int32(per_thread))
                            low = head
                            high = head
                    else:
                        run_digit = Int32(-1)
                        run_count = Int32(0)
                        for g in cutlass.range_constexpr(per_thread // 4):
                            if const_expr(p == 0):
                                for j in cutlass.range_constexpr(4):
                                    run_digit, run_count, low, high = count_key(
                                        keys[4 * g + j],
                                        prefix_mask,
                                        prefix,
                                        hist,
                                        run_digit,
                                        run_count,
                                        low,
                                        high,
                                        shift,
                                        False,
                                    )
                            else:
                                hit = (keys[4 * g] & prefix_mask) == prefix
                                for j in cutlass.range_constexpr(1, 4):
                                    hit = hit | ((keys[4 * g + j] & prefix_mask) == prefix)
                                if hit:
                                    for j in cutlass.range_constexpr(4):
                                        run_digit, run_count, low, high = count_key(
                                            keys[4 * g + j],
                                            prefix_mask,
                                            prefix,
                                            hist,
                                            run_digit,
                                            run_count,
                                            low,
                                            high,
                                            shift,
                                            p != 3,
                                        )
                        if run_count != 0:
                            red_shared_add(hist + run_digit * 4, run_count)
                    if const_expr(p == 1 or p == 2):
                        low = reduce_warp(low, "min")
                        high = reduce_warp(high, "max")
                        if (lane == 0) & (high != Uint32(0)):
                            red_shared_u32(span, low, "min")
                            red_shared_u32(span + 4, high, "max")
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    if const_expr(p < 3):
                        # Every CTA read the other buffers in the previous pass, before this barrier.
                        if tidx < _BINS:
                            following = hist_base + ((p + 1) % 2) * _BINS * 4
                            st_shared_s32(following + tidx * 4, Int32(0))
                        if tidx == 0:
                            st_shared_s32(range_base + ((p + 1) % 2) * 16, Int32(-1))
                            st_shared_s32(range_base + ((p + 1) % 2) * 16 + 4, Int32(0))
                    if tidx < 32:
                        # The peers' [min, max] pairs first, so their loads overlap the counts'.
                        if const_expr(p == 1 or p == 2):
                            low = Uint32(0xFFFFFFFF)
                            high = Uint32(0)
                            if lane < cluster:
                                words = ld_cluster_v4(span, lane)
                                low = Uint32(words[0])
                                high = Uint32(words[1])
                        # Lane l sums digits 255 - 8l .. 248 - 8l over the cluster, top first.
                        counts = cute.make_rmem_tensor(cute.make_layout((8,)), Int32)
                        for j in cutlass.range_constexpr(8):
                            counts[j] = Int32(0)
                        for peer in cutlass.range_constexpr(cluster):
                            for half in cutlass.range_constexpr(2):
                                words = ld_cluster_v4(
                                    hist + (_BINS - 4 - 8 * lane - 4 * half) * 4, Int32(peer)
                                )
                                for j in cutlass.range_constexpr(4):
                                    counts[4 * half + j] = counts[4 * half + j] + words[3 - j]
                        local = Int32(0)
                        for j in cutlass.range_constexpr(8):
                            local = local + counts[j]
                        inclusive = local
                        for delta in (1, 2, 4, 8, 16):
                            up = shfl_up(inclusive, delta)
                            if lane >= delta:
                                inclusive = inclusive + up
                        above = inclusive - local
                        if const_expr(p == 0):
                            need = cutlass.min(shfl_idx(inclusive, 31), Int32(top_k))
                            if lane == 0:
                                sPick[2] = need
                        if (above < need) & (need <= inclusive):
                            remaining = need - above
                            for j in cutlass.range_constexpr(8):
                                if (remaining > 0) & (remaining <= counts[j]):
                                    sPick[0] = _BINS - 1 - 8 * lane - j
                                    sPick[1] = remaining
                                    sPick[3] = Int32(remaining == counts[j])
                                remaining = remaining - counts[j]
                        if const_expr(p == 1 or p == 2):
                            low = reduce_warp(low, "min")
                            high = reduce_warp(high, "max")
                            if lane == 0:
                                sPick[4] = Int32((low == high) & (high != Uint32(0)))
                                sPick[5] = Int32(low)
                    cute.arch.sync_threads()
                    if const_expr(p == 0):
                        need = sPick[2]
                    if need > 0:
                        prefix = prefix | (Uint32(sPick[0]) << Uint32(shift))
                        prefix_mask = prefix_mask | (Uint32(_BINS - 1) << Uint32(shift))
                        need = sPick[1]
                        if const_expr(p == 1 or p == 2):
                            if sPick[4] != 0:
                                # Every key of the bin being refined is equal: it is the k-th key.
                                prefix = Uint32(sPick[5])
                                need = need_in
                                done = Int32(1)
                        if const_expr(p < 3):
                            if (done == Int32(0)) & (sPick[3] != 0) & (prefix != Uint32(0)):
                                # The threshold digit's keys are all taken: every key from the prefix
                                # up is selected, so stop with the threshold just below it.
                                prefix = prefix - Uint32(1)
                                need = Int32(0)
                                done = Int32(1)
            # Every key above `prefix` is selected and, of the keys equal to it, the `need`
            # lowest columns.
            greater = Int32(0)
            ties = Int32(0)
            if same:
                if head > prefix:
                    greater = Int32(per_thread)
                if (head == prefix) & (head != Uint32(0)):
                    ties = Int32(per_thread)
            else:
                for e in cutlass.range_constexpr(per_thread):
                    key = keys[e]
                    if key > prefix:
                        greater = greater + 1
                    if (key == prefix) & (key != Uint32(0)):
                        ties = ties + 1
            before, total = self._block_scan(tidx, (greater << 16) | ties, sWarp)
            if (tidx > rank) & (tidx < cluster):
                st_cluster_s32(counts_base + rank * 8, tidx, total >> 16)
                st_cluster_s32(counts_base + rank * 8 + 4, tidx, total & 0xFFFF)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            greater_before = Int32(0)
            ties_before = Int32(0)
            for peer in cutlass.range_constexpr(cluster):
                if peer < rank:
                    greater_before = greater_before + sCounts[2 * peer]
                    ties_before = ties_before + sCounts[2 * peer + 1]
            tie_rank = ties_before + (before & 0xFFFF)
            taken = need - tie_rank
            if taken < 0:
                taken = Int32(0)
            if taken > ties:
                taken = ties
            slot = greater_before + (before >> 16) + cutlass.min(tie_rank, need)
            row_out = Int64(out.toint()) + Int64(row) * out_stride * 4
            if greater + taken > 0:
                if same:
                    # All columns (greater) or the lowest `taken` (tied).
                    for e in cutlass.range(greater + taken):
                        st_global_s32(row_out + Int64(slot + e) * 4, first + e)
                else:
                    for e in cutlass.range_constexpr(per_thread):
                        key = keys[e]
                        take = key > prefix
                        if (key == prefix) & (key != Uint32(0)):
                            take = tie_rank < need
                            tie_rank = tie_rank + 1
                        if take:
                            st_global_s32(row_out + Int64(slot) * 4, first + e)
                            slot = slot + 1
            if rank == cluster - 1:
                selected = greater_before + (total >> 16) + need
                for i in cutlass.range(tidx, top_k, _THREADS):
                    if i >= selected:
                        st_global_s32(row_out + Int64(i) * 4, Int32(-1))

    return FusedDecodeSelect


_COMPILED: dict = {}


def fused_decode_select_(
    logits: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    output: torch.Tensor,
    page_rows: int,
) -> torch.Tensor:
    """``output [rows, top_k]`` (int32): per row the ``top_k`` largest ``logits [rows, cols]``
    (FP32) among its columns ``c < visible[r]`` whose ``valid[r, c // page_rows]`` (bool) is set,
    ties lowest column first, as increasing columns followed by ``-1``."""
    import cuda.bindings.driver as cuda
    import cutlass

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import make_ptr

    rows, cols = logits.shape
    top_k = output.shape[1]
    if not fused_decode_select_supported(logits, top_k):
        raise ValueError("CSA2 fused decode selection: unsupported logits geometry")
    if (
        output.dtype != torch.int32
        or output.shape[0] != rows
        or output.stride(1) != 1
        or valid.dtype != torch.bool
        or valid.shape[0] != rows
        or valid.stride(1) != 1
        or visible.dtype != torch.int32
        or visible.numel() != rows
        or not visible.is_contiguous()
        or page_rows % 4 != 0
    ):
        raise ValueError("CSA2 fused decode selection: int32 output/visible, bool page rows")
    cluster, per_thread = plan(cols)
    use_pdl = get_env_enable_pdl()

    def pointer(tensor, dtype, align):
        return make_ptr(
            dtype, tensor.data_ptr(), cutlass.cute.AddressSpace.gmem, assumed_align=align
        )

    args = (
        pointer(logits, cutlass.Float32, 16),
        pointer(valid, cutlass.Uint8, 1),
        pointer(visible, cutlass.Int32, 4),
        pointer(output, cutlass.Int32, 4),
        cutlass.Int32(rows),
        cutlass.Int32(cols),
        cutlass.Int64(logits.stride(0)),
        cutlass.Int64(valid.stride(0)),
        cutlass.Int64(output.stride(0)),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    key = (logits.device.index, cluster, per_thread, top_k, page_rows, use_pdl)
    compiled = _COMPILED.get(key)
    if compiled is None:
        import cutlass.cute as cute

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the CSA2 fused decode selection before CUDA Graph capture")
        kernel = _kernel_type()(cluster, per_thread, top_k, page_rows, use_pdl)
        compiled = _COMPILED[key] = cute.compile(kernel, *args)
    compiled(*args)
    return output


_CONSUMER_COMPILED: dict = {}


def _consumer_kernel_class():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Int32
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.block_scan import warp_scan
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk import (
        SinglePassMultiCTARadixTopKKernel,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
        TRTLLM_ENABLE_PDL,
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    class MappedRadixTopKKernel(SinglePassMultiCTARadixTopKKernel):
        """Single-CTA radix-select per row whose selection is listed as positions in column
        order (see the module docstring)."""

        def __init__(self, dtype, chunk_size: int, top_k: int, num_sms: int):
            super().__init__(
                dtype,
                chunk_size,
                top_k,
                next_n=1,
                num_copy_bits=256,
                ctas_per_group=1,
                num_sms=num_sms,
            )
            if chunk_size % 32 or chunk_size > _MAX_COLUMNS:
                raise ValueError("CSA2 fused consumer Top-K needs a multiple of 32 columns")
            self.words = chunk_size // 32

        @cute.jit
        def __call__(
            self,
            input_data: cute.Tensor,
            seqlen: cute.Tensor,
            positions: cute.Tensor,
            output_indices: cute.Tensor,
            stream,
        ):
            num_rows = input_data.shape[0]
            num_groups = min(self.num_sms, num_rows)
            self.mapped_kernel(input_data, seqlen, positions, output_indices).launch(
                grid=(num_groups, 1, 1),
                block=(self.num_threads, 1, 1),
                stream=stream,
                use_pdl=TRTLLM_ENABLE_PDL,
            )

        @cute.kernel
        def mapped_kernel(
            self,
            input_data: cute.Tensor,
            seqlen: cute.Tensor,
            positions: cute.Tensor,
            output_indices: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            grid_size, _, _ = cute.arch.grid_dim()
            top_k = cutlass.const_expr(self.top_k)
            chunk_size = cutlass.const_expr(self.chunk_size)
            num_threads = cutlass.const_expr(self.num_threads)
            num_warps = cutlass.const_expr(self.num_warps)
            words = cutlass.const_expr(self.words)
            num_rows = input_data.shape[0]
            warp = tidx // 32
            lane = tidx % 32

            griddepcontrol_wait()

            smem = SmemAllocator()
            local_histogram = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((self.radix,), order=(0,)),
                byte_alignment=128,
            )
            prefix_buf = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((self.radix,), order=(0,)),
                byte_alignment=128,
            )
            s_scalars = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((4,), order=(0,)),
                byte_alignment=16,
            )
            num_warps_scan = cutlass.const_expr(min(self.radix, 256) // 32)
            s_warp_sums = smem.allocate_tensor(
                element_type=Int32,
                layout=cute.make_ordered_layout((num_warps_scan,), order=(0,)),
                byte_alignment=128,
            )
            shared_ordered = smem.allocate_tensor(
                element_type=self.ordered_type,
                layout=cute.make_ordered_layout((self.chunk_size,), order=(0,)),
                byte_alignment=128,
            )
            flags = smem.allocate_tensor(cutlass.Uint32, cute.make_layout(words), 16)
            warp_totals = smem.allocate_tensor(Int32, cute.make_layout(num_warps), 16)

            row_idx = bidx
            while row_idx < num_rows:
                length = seqlen[row_idx]
                if length > chunk_size:
                    length = Int32(chunk_size)
                input_row = input_data[row_idx, None]
                out_row = output_indices[row_idx, None]
                pos_row = positions[row_idx, None]
                if top_k >= length:
                    # Every valid column is selected, in column order.
                    for i in range(tidx, top_k, num_threads):
                        value = Int32(-1)
                        if i < length:
                            value = Int32(pos_row[i])
                        out_row[i] = value
                else:
                    if tidx < words:
                        flags[tidx] = cutlass.Uint32(0)
                    prologue_elems, aligned_size, left_size = self.load_chunk_to_smem(
                        input_row, shared_ordered, Int32(0), length, tidx
                    )
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        prefix = cutlass.Uint32(0)
                    else:
                        prefix = cutlass.Uint16(0)
                    remaining_k = Int32(top_k)
                    for r in cutlass.range_constexpr(self.num_rounds):
                        shift = cutlass.const_expr(self.ordered_bits - (r + 1) * self.radix_bits)
                        prefix, remaining_k = self._radix_round_single_cta(
                            r,
                            shift,
                            shared_ordered,
                            length,
                            prefix,
                            remaining_k,
                            local_histogram,
                            prefix_buf,
                            s_scalars,
                            s_warp_sums,
                            num_threads,
                            tidx,
                        )
                    # Flag the elements above the pivot and the first ties, lowest column first.
                    self.collect_output_single_cta(
                        shared_ordered,
                        length,
                        Int32(0),
                        prologue_elems,
                        aligned_size,
                        left_size,
                        prefix,
                        top_k,
                        local_histogram,
                        out_row,
                        None,
                        tidx,
                        prefix_buf,
                        mark_flags=flags,
                    )
                    # Column order: one bitmap word per thread, an exclusive scan over the
                    # threads' counts, then each thread lists its word's columns. The positions
                    # of a word's first columns are loaded before the scan's barrier.
                    bits = cutlass.Uint32(0)
                    if tidx < words:
                        bits = flags[tidx]
                    count = Int32(cute.arch.popc(bits))
                    early = cute.make_rmem_tensor((_PREFETCH,), Int32)
                    pending = bits
                    for i in cutlass.range_constexpr(_PREFETCH):
                        early[i] = Int32(-1)
                        if pending != cutlass.Uint32(0):
                            lowest = pending & (cutlass.Uint32(0) - pending)
                            column = tidx * 32 + Int32(cute.arch.popc(lowest - cutlass.Uint32(1)))
                            early[i] = Int32(pos_row[column])
                            pending = pending & (pending - cutlass.Uint32(1))
                    inclusive = warp_scan(count, tidx, lane, 32)
                    if lane == 31:
                        warp_totals[warp] = inclusive
                    cute.arch.barrier()
                    slot = inclusive - count
                    total = Int32(0)
                    for w in cutlass.range_constexpr(num_warps):
                        if w < warp:
                            slot = slot + warp_totals[w]
                        total = total + warp_totals[w]
                    for i in cutlass.range_constexpr(_PREFETCH):
                        if bits != cutlass.Uint32(0):
                            out_row[slot] = early[i]
                            slot = slot + 1
                            bits = bits & (bits - cutlass.Uint32(1))
                    while bits != cutlass.Uint32(0):
                        lowest = bits & (cutlass.Uint32(0) - bits)
                        column = tidx * 32 + Int32(cute.arch.popc(lowest - cutlass.Uint32(1)))
                        out_row[slot] = Int32(pos_row[column])
                        slot = slot + 1
                        bits = bits & (bits - cutlass.Uint32(1))
                    for k in range(total + tidx, top_k, num_threads):
                        out_row[k] = Int32(-1)
                    # The next row reloads shared_ordered and the bitmap.
                    cute.arch.barrier()
                row_idx = row_idx + grid_size

            griddepcontrol_launch_dependents()

    return MappedRadixTopKKernel


def fused_consumer_topk_supported(scores: torch.Tensor, top_k: int) -> bool:
    """Whether ``select_mapped_consumer_topk_`` reproduces the decode radix-select followed by
    ``map_selection_`` for these scores: the dispatcher's radix-select single-CTA path (BF16 or
    FP16 rows that one CTA holds), with deterministic tie admission, on aligned rows."""
    if not (scores.is_cuda and scores.dim() == 2):
        return False
    rows, width = scores.shape
    if rows == 0 or width == 0 or width % 32 or width > _MAX_COLUMNS:
        return False
    if scores.dtype not in (torch.bfloat16, torch.float16):
        return False
    vector = 32 // scores.element_size()
    if scores.stride(1) != 1 or scores.stride(0) % vector or scores.data_ptr() % 32:
        return False
    try:
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    except ImportError:
        return False
    if not top_k or top_k > 16384:
        return False
    if not ops._radix_select_preferred(scores.dtype, width, rows):
        return False
    runner = ops.CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner
    chunk, ctas, _ = runner._get_chunk_config(
        ops._TORCH_TO_CUTLASS_DTYPE[scores.dtype], width, None, 256, rows
    )
    return ctas == 1 and chunk is not None and chunk >= width


def select_mapped_consumer_topk_(
    scores: torch.Tensor,
    counts: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Top-k of each row's leading ``counts[q]`` scores (ties lowest column first), written to
    ``output`` as ``positions[q, column]`` in increasing column order, ``-1`` padding last."""
    rows, width = scores.shape
    top_k = output.shape[1]
    if rows == 0:
        return output
    if not (
        output.dtype == torch.int32
        and output.is_contiguous()
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and positions.shape[0] == rows
        and positions.shape[1] >= width
        and counts.dtype == torch.int32
    ):
        raise ValueError("CSA2 fused consumer Top-K: contiguous int32 output, positions, counts")
    counts = counts.reshape(-1)
    if not counts.is_contiguous():
        counts = counts.contiguous()
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    num_sms = ops._get_num_sms()
    dtype = ops._TORCH_TO_CUTLASS_DTYPE[scores.dtype]
    key = (scores.dtype, width, top_k, num_sms, scores.device.index)
    compiled = _CONSUMER_COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the CSA2 fused consumer Top-K before CUDA Graph capture")
        n_rows, n_cols, n_pos = cute.sym_int(), cute.sym_int(), cute.sym_int()
        kernel = _consumer_kernel_class()(dtype, width, top_k, num_sms)
        compiled = _CONSUMER_COMPILED[key] = cute.compile(
            kernel,
            cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, n_cols), stride_order=(1, 0), assumed_align=32
            ),
            cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_rows,), stride_order=(0,)),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, n_pos), stride_order=(1, 0)
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k), stride_order=(1, 0)
            ),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    compiled(scores, counts, positions, output)
    return output
