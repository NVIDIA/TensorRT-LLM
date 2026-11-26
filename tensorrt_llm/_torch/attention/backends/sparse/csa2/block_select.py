# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact CSA2 decode Top-K from the scorer's 32-position block maxima: the top_k 32-position blocks, the
top_k 8-position blocks within them, then positions; lowest-index ties at each level give the flat Top-K."""

import functools

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: First-level keys a CTA holds: a row's 32-position blocks, or its 128-position blocks above that.
MAX_KEYS = 8192
#: Positions per candidate block of a publication (the selection's 8-position level).
PUBLISH_BLOCK = 8
# The last level selects among at most this many keys per thread when that many pass its floor,
# packed in index order.
_PACKED = 4
_BINS = 256
_COARSE = 128
_PAGE = 64


def block_select_threads(publish: bool) -> int:
    """Threads per row: 512 for a publication's wider levels, else 256."""
    return 512 if publish else 256


def _first_tiers(width: int, threads: int) -> tuple[int, ...]:
    """First-level keys per thread of the tiers a row picks by its visible length (3072, 4096, 6144 blocks,
    then ``MAX_KEYS``); they stop once 128-position keys cover the width (fewer registers, more CTAs)."""
    blocks = -(-width // 32)
    tiers = []
    for cap in (3072, 4096, 6144, MAX_KEYS):
        per = min(MAX_KEYS // threads, -(-min(cap, blocks) // (4 * threads)) * 4)
        if not tiers or per > tiers[-1]:
            tiers.append(per)
        if 4 * tiers[-1] * threads >= blocks:
            break
    return tuple(tiers)


def block_select_supported(width: int, top_k: int, publish_k: int = 0) -> bool:
    """Whether ``block_select_`` serves rows of ``width`` logits columns."""
    return (
        0 < width <= MAX_KEYS * _COARSE
        and (top_k == 512 or (top_k == 256 and publish_k == 0))
        and publish_k in (0, 512, 1024, 2048)
        and (publish_k == 0 or publish_k > top_k)
        # The first level dispatches three tiers.
        and len(_first_tiers(width, block_select_threads(publish_k > 0))) <= 3
    )


@functools.lru_cache(maxsize=1)
def _kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import Float32, Int32, Int64, Uint32
    from cutlass.cutlass_dsl import T, const_expr, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    from .select_ops import (
        asm,
        count_key,
        ld_global_s32,
        ld_global_u8,
        ld_global_v4,
        red_shared_add,
        red_shared_u32,
        reduce_warp,
        shfl_idx,
        shfl_up,
        st_global_s32,
        st_global_v4_s32,
        warp_sum,
    )

    @dsl_user_op
    def ordered(value: Float32, *, loc=None, ip=None) -> Uint32:
        """Order-preserving key of an FP32 value: larger values, larger keys (never 0), -0 as +0."""
        return Uint32(
            asm(
                T.i32(),
                [Float32(value).ir_value(loc=loc, ip=ip)],
                "{\n\t.reg .b32 b, m;\n\t.reg .pred n, z;\n\t"
                "mov.b32 b, $1;\n\t"
                "setp.eq.b32 z, b, 0x80000000;\n\t"
                "selp.b32 b, 0, b, z;\n\t"
                "setp.lt.s32 n, b, 0;\n\t"
                "selp.b32 m, 0xffffffff, 0x80000000, n;\n\t"
                "xor.b32 $0, b, m;\n\t}",
                "=r,f",
                loc=loc,
                ip=ip,
            )
        )

    class BlockSelect:
        def __init__(
            self,
            top_k: int,
            publish_k: int,
            threads: int,
            first: tuple,
            use_pdl: bool,
        ):
            self.top_k = top_k
            self.publish_k = publish_k
            self.threads = threads
            self.warps = threads // 32
            self.first = first
            self.sel_cap = max(top_k, publish_k)
            self.use_pdl = use_pdl

        @cute.jit
        def __call__(
            self,
            logits: cute.Pointer,
            bm32: cute.Pointer,
            valid: cute.Pointer,
            visible: cute.Pointer,
            out: cute.Pointer,
            published: cute.Pointer,
            sparse_ids: cute.Pointer,
            sparse_counts: cute.Pointer,
            rows: Int32,
            width: Int32,
            logits_stride: Int64,
            bm32_stride: Int64,
            valid_stride: Int64,
            out_stride: Int64,
            published_stride: Int64,
            sparse_stride: Int64,
            published_width: Int32,
            stream,
        ):
            self.kernel(
                logits,
                bm32,
                valid,
                visible,
                out,
                published,
                sparse_ids,
                sparse_counts,
                width,
                logits_stride,
                bm32_stride,
                valid_stride,
                out_stride,
                published_stride,
                sparse_stride,
                published_width,
            ).launch(
                grid=[rows, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

        # ------------------------------------------------------------------ keys

        @cute.jit
        def _key32(self, value, b, backed, n32, pin32):
            """Key of 32-position block ``b`` whose visible maximum is ``value``."""
            key = Uint32(0)
            if b < n32:
                if backed != 0:
                    key = ordered(value)
                if b == pin32:
                    key = Uint32(0xFFFFFFFF)
            return key

        @cute.jit
        def _load_first(
            self, keys, per: cutlass.Constexpr, coarse, n32, n128, pin32, row_bm32, row_valid, tidx
        ):
            """This thread's run of first-level keys: 32-position blocks, or 128-position ones."""
            if coarse:
                for g in cutlass.range_constexpr(per):
                    b128 = per * tidx + g
                    best = Uint32(0)
                    if b128 < n128:
                        x = ld_global_v4(row_bm32 + Int64(4 * b128) * 4)
                        lo = ld_global_u8(row_valid + Int64(2 * b128))
                        hi = Int32(0)
                        if 4 * b128 + 2 < n32:
                            hi = ld_global_u8(row_valid + Int64(2 * b128 + 1))
                        for j in cutlass.range_constexpr(4):
                            backed = lo
                            if const_expr(j >= 2):
                                backed = hi
                            key = self._key32(x[j], 4 * b128 + j, backed, n32, pin32)
                            if key > best:
                                best = key
                    keys[g] = best
            else:
                for v in cutlass.range_constexpr(per // 4):
                    first = per * tidx + 4 * v
                    for j in cutlass.range_constexpr(4):
                        keys[4 * v + j] = Uint32(0)
                    if first < n32:
                        x = ld_global_v4(row_bm32 + Int64(first) * 4)
                        lo = ld_global_u8(row_valid + Int64(first // 2))
                        hi = Int32(0)
                        if first + 2 < n32:
                            hi = ld_global_u8(row_valid + Int64(first // 2 + 1))
                        for j in cutlass.range_constexpr(4):
                            backed = lo
                            if const_expr(j >= 2):
                                backed = hi
                            keys[4 * v + j] = self._key32(x[j], first + j, backed, n32, pin32)

        @cute.jit
        def _load_children32(
            self,
            keys,
            per: cutlass.Constexpr,
            count,
            parents,
            n32,
            pin32,
            floor,
            row_bm32,
            row_valid,
            tidx,
        ):
            """Keys of the 32-position blocks of the listed 128-position ones (four per parent)."""
            for q in cutlass.range_constexpr(per // 4):
                p = (per // 4) * tidx + q
                for j in cutlass.range_constexpr(4):
                    keys[4 * q + j] = Uint32(0)
                if p < count:
                    first = 4 * parents[p]
                    x = ld_global_v4(row_bm32 + Int64(first) * 4)
                    lo = ld_global_u8(row_valid + Int64(first // 2))
                    hi = Int32(0)
                    if first + 2 < n32:
                        hi = ld_global_u8(row_valid + Int64(first // 2 + 1))
                    for j in cutlass.range_constexpr(4):
                        backed = lo
                        if const_expr(j >= 2):
                            backed = hi
                        key = self._key32(x[j], first + j, backed, n32, pin32)
                        if key < floor:
                            key = Uint32(0)
                        keys[4 * q + j] = key

        @cute.jit
        def _key8(self, b, vis, row_logits):
            """Key of visible 8-position block ``b``: the maximum of its visible logits."""
            m = Float32(float("-inf"))
            a = ld_global_v4(row_logits + Int64(8 * b) * 4)
            c = ld_global_v4(row_logits + Int64(8 * b + 4) * 4)
            for i in cutlass.range_constexpr(4):
                if 8 * b + i < vis:
                    m = cute.arch.fmax(m, a[i])
                if 8 * b + 4 + i < vis:
                    m = cute.arch.fmax(m, c[i])
            return ordered(m)

        @cute.jit
        def _load_blocks8(
            self,
            keys,
            per: cutlass.Constexpr,
            count,
            parents,
            n8,
            floor,
            pin8,
            row_valid,
            tidx,
            row_logits,
            vis,
        ):
            """Keys of the 8-position blocks of the listed 32-position ones; with a pin (publication)
            a listed block may be unbacked and only the pin survives."""
            for q in cutlass.range_constexpr(per // 4):
                p = (per // 4) * tidx + q
                for j in cutlass.range_constexpr(4):
                    keys[4 * q + j] = Uint32(0)
                if p < count:
                    b32 = parents[p]
                    backed = Int32(1)
                    if pin8 >= 0:
                        backed = ld_global_u8(row_valid + Int64(b32 // 2))
                    for j in cutlass.range_constexpr(4):
                        b = 4 * b32 + j
                        key = Uint32(0)
                        if (b < n8) & (backed != 0):
                            key = self._key8(b, vis, row_logits)
                        if key < floor:
                            key = Uint32(0)
                        if b == pin8:
                            key = Uint32(0xFFFFFFFF)
                        keys[4 * q + j] = key

        @cute.jit
        def _load_listed8(
            self,
            keys,
            per: cutlass.Constexpr,
            count,
            parents,
            listed_keys,
            pin8,
            row_valid,
            tidx,
            row_logits,
            vis,
        ):
            """Keys of the listed (published) 8-position blocks, as they were selected except for
            the pinned block's own."""
            for q in cutlass.range_constexpr(per):
                p = per * tidx + q
                keys[q] = Uint32(0)
                if p < count:
                    b = parents[p]
                    key = Uint32(listed_keys[p])
                    if b == pin8:
                        key = self._key8(b, vis, row_logits)
                        if ld_global_u8(row_valid + Int64(b // 8)) == 0:
                            key = Uint32(0)
                    keys[q] = key

        @cute.jit
        def _load_positions(
            self, keys, per: cutlass.Constexpr, count, parents, vis, floor, row_logits, tidx
        ):
            """Keys of the eight positions of each listed 8-position block."""
            for q in cutlass.range_constexpr(per // 8):
                p = (per // 8) * tidx + q
                for j in cutlass.range_constexpr(8):
                    keys[8 * q + j] = Uint32(0)
                if p < count:
                    first = 8 * parents[p]
                    for h in cutlass.range_constexpr(2):
                        x = ld_global_v4(row_logits + Int64(first + 4 * h) * 4)
                        for j in cutlass.range_constexpr(4):
                            key = Uint32(0)
                            if first + 4 * h + j < vis:
                                key = ordered(x[j])
                            if key < floor:
                                key = Uint32(0)
                            keys[8 * q + 4 * h + j] = key

        # ---------------------------------------------------------------- select

        @cute.jit
        def _block_scan(self, tidx, value, sWarp):
            """Exclusive prefix of ``value`` over the threads in order, and the block total."""
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
            for w in cutlass.range_constexpr(self.warps):
                count = sWarp[w]
                if w < warp:
                    before = before + count
                total = total + count
            cute.arch.sync_threads()
            return before + inclusive - value, total

        @cute.jit
        def _select(
            self,
            keys,
            per: cutlass.Constexpr,
            k: cutlass.Constexpr,
            sHist,
            sSpan,
            sCtl,
            sWarp,
            tidx,
        ):
            """(threshold, ties, count) of the top ``k`` nonzero register keys: those above the
            threshold and its ``ties`` lowest-index copies."""
            lane = tidx % 32
            warp = tidx // 32
            hist_base = Int32(sHist.iterator.toint())
            span_base = Int32(sSpan.iterator.toint())
            head = keys[0]
            same = cutlass.Boolean(True)
            # This thread's nonzero key range, compared as unsigned keys.
            kmin = Uint32(0xFFFFFFFF)
            kmax = Uint32(0)
            for e in cutlass.range_constexpr(per):
                if const_expr(e > 0):
                    same = same & (keys[e] == head)
                if (keys[e] != Uint32(0)) & (keys[e] < kmin):
                    kmin = keys[e]
                if keys[e] > kmax:
                    kmax = keys[e]
            prefix = Uint32(0)
            prefix_mask = Uint32(0)
            need = Int32(0)
            done = Int32(0)
            thr = Uint32(0)
            ties = Int32(0)
            count = Int32(0)
            for p in cutlass.range_constexpr(4):
                if done == Int32(0):
                    shift = 24 - 8 * p
                    hist = hist_base + (p % 2) * _BINS * 4
                    span = span_base + (p % 2) * 8
                    low = Uint32(0xFFFFFFFF)
                    high = Uint32(0)
                    # Runs of one digit in a thread's keys reach the histogram as one add.
                    run_digit = Int32(-1)
                    run_count = Int32(0)
                    if same:
                        if (head != Uint32(0)) & ((head & prefix_mask) == prefix):
                            run_digit = Int32((head >> Uint32(shift)) & Uint32(_BINS - 1))
                            run_count = Int32(per)
                            low = head
                            high = head
                    elif (kmax >= prefix) & (kmin <= (prefix | (prefix_mask ^ Uint32(0xFFFFFFFF)))):
                        # Only threads holding keys of the bucket being refined count.
                        for e in cutlass.range_constexpr(per):
                            run_digit, run_count, low, high = count_key(
                                keys[e],
                                prefix_mask,
                                prefix,
                                hist,
                                run_digit,
                                run_count,
                                low,
                                high,
                                shift,
                                0 < p < 3,
                            )
                    if run_count != 0:
                        red_shared_add(hist + run_digit * 4, run_count)
                    if const_expr(0 < p < 3):
                        low = reduce_warp(low, "min")
                        high = reduce_warp(high, "max")
                        if (lane == 0) & (high != Uint32(0)):
                            red_shared_u32(span, low, "min")
                            red_shared_u32(span + 4, high, "max")
                    cute.arch.sync_threads()
                    # The other buffers were last read in the previous pass, before this barrier.
                    if tidx < _BINS:
                        sHist[((p + 1) % 2) * _BINS + tidx] = Int32(0)
                    if tidx == 0:
                        sSpan[((p + 1) % 2) * 2] = Int32(-1)
                        sSpan[((p + 1) % 2) * 2 + 1] = Int32(0)
                    if warp == 0:
                        # Lane l sums digits 255 - 8l .. 248 - 8l, top first.
                        counts = cute.make_rmem_tensor(cute.make_layout((8,)), Int32)
                        local = Int32(0)
                        for j in cutlass.range_constexpr(8):
                            counts[j] = sHist[(p % 2) * _BINS + _BINS - 1 - 8 * lane - j]
                            local = local + counts[j]
                        inclusive = local
                        for delta in (1, 2, 4, 8, 16):
                            up = shfl_up(inclusive, delta)
                            if lane >= delta:
                                inclusive = inclusive + up
                        if const_expr(p == 0):
                            total = shfl_idx(inclusive, 31)
                            need = cutlass.min(total, Int32(k))
                            if lane == 0:
                                sCtl[2] = need
                                sCtl[6] = Int32(total <= Int32(k))
                        above = inclusive - local
                        if (above < need) & (need <= inclusive):
                            remaining = need - above
                            for j in cutlass.range_constexpr(8):
                                if (remaining > 0) & (remaining <= counts[j]):
                                    sCtl[0] = _BINS - 1 - 8 * lane - j
                                    sCtl[1] = remaining
                                    sCtl[3] = Int32(remaining == counts[j])
                                remaining = remaining - counts[j]
                        if const_expr(0 < p < 3):
                            lo_key = Uint32(sSpan[(p % 2) * 2])
                            hi_key = Uint32(sSpan[(p % 2) * 2 + 1])
                            if lane == 0:
                                sCtl[4] = Int32((lo_key == hi_key) & (hi_key != Uint32(0)))
                                sCtl[5] = Int32(lo_key)
                    cute.arch.sync_threads()
                    if const_expr(p == 0):
                        need = sCtl[2]
                        count = need
                        if sCtl[6] != Int32(0):
                            # At most k keys: every nonzero key is selected.
                            done = Int32(1)
                    if done == Int32(0):
                        need_in = need
                        prefix = prefix | (Uint32(sCtl[0]) << Uint32(shift))
                        prefix_mask = prefix_mask | (Uint32(_BINS - 1) << Uint32(shift))
                        need = sCtl[1]
                        if const_expr(0 < p < 3):
                            if sCtl[4] != Int32(0):
                                # Every key of the bin being refined is equal: it is the k-th key.
                                thr = Uint32(sCtl[5])
                                ties = need_in
                                done = Int32(1)
                        if done == Int32(0):
                            if const_expr(p < 3):
                                if (sCtl[3] != Int32(0)) & (prefix != Uint32(0)):
                                    # The threshold digit's keys are all taken.
                                    thr = prefix - Uint32(1)
                                    done = Int32(1)
                            else:
                                thr = prefix
                                ties = need
            # A selection stopping after an even pass leaves its counts in the first histogram.
            if tidx < _BINS:
                sHist[tidx] = Int32(0)
            return thr, ties, count

        @cute.jit
        def _emit(
            self,
            keys,
            per: cutlass.Constexpr,
            thr,
            ties,
            sWarp,
            tidx,
            fan: cutlass.Constexpr,
            parents,
            target,
            to_global,
            index_map=None,
            target_keys=None,
        ):
            """Store each selected entry ``i = per * tidx + e`` (or ``index_map`` of it) in index order as ``i``
            (``fan`` 0), ``parents[i]`` (1) or ``fan * parents[i // fan] + i % fan``; its key to ``target_keys``."""
            greater = Int32(0)
            equal = Int32(0)
            for e in cutlass.range_constexpr(per):
                key = keys[e]
                if key > thr:
                    greater = greater + 1
                if (key == thr) & (key != Uint32(0)):
                    equal = equal + 1
            before, total = self._block_scan(tidx, (greater << 16) | equal, sWarp)
            tie_rank = before & 0xFFFF
            slot = (before >> 16) + cutlass.min(tie_rank, ties)
            if greater + equal > 0:
                for e in cutlass.range_constexpr(per):
                    key = keys[e]
                    take = key > thr
                    if (key == thr) & (key != Uint32(0)):
                        take = tie_rank < ties
                        tie_rank = tie_rank + 1
                    if take:
                        i = per * tidx + e
                        if const_expr(index_map is not None):
                            i = index_map[i]
                        value = i
                        if const_expr(fan == 1):
                            value = parents[i]
                        elif const_expr(fan > 1):
                            value = fan * parents[i // fan] + i % fan
                        if const_expr(to_global is None):
                            target[slot] = value
                        else:
                            st_global_s32(to_global + Int64(slot) * 4, value)
                        if const_expr(target_keys is not None):
                            target_keys[slot] = Int32(key)
                        slot = slot + 1
            cute.arch.sync_threads()

        @cute.jit
        def _packed_select(
            self,
            keys,
            per: cutlass.Constexpr,
            k: cutlass.Constexpr,
            fan: cutlass.Constexpr,
            parents,
            target,
            to_global,
            sPackKey,
            sPackIdx,
            sHist,
            sSpan,
            sCtl,
            sWarp,
            tidx,
        ):
            """``_select`` and ``_emit`` of the register keys; when at most _PACKED nonzero keys per
            thread remain, among those packed in index order instead. Returns the count."""
            live = Int32(0)
            for e in cutlass.range_constexpr(per):
                live = live + Int32(keys[e] != Uint32(0))
            before, total = self._block_scan(tidx, live, sWarp)
            thr = Uint32(0)
            ties = Int32(0)
            c = Int32(0)
            if total <= _PACKED * self.threads:
                slot = before
                for e in cutlass.range_constexpr(per):
                    if keys[e] != Uint32(0):
                        sPackKey[slot] = Int32(keys[e])
                        sPackIdx[slot] = per * tidx + e
                        slot = slot + 1
                cute.arch.sync_threads()
                packed = cute.make_rmem_tensor(cute.make_layout((_PACKED,)), Uint32)
                for j in cutlass.range_constexpr(_PACKED):
                    packed[j] = Uint32(0)
                    if _PACKED * tidx + j < total:
                        packed[j] = Uint32(sPackKey[_PACKED * tidx + j])
                thr, ties, c = self._select(packed, _PACKED, k, sHist, sSpan, sCtl, sWarp, tidx)
                self._emit(
                    packed,
                    _PACKED,
                    thr,
                    ties,
                    sWarp,
                    tidx,
                    fan,
                    parents,
                    target,
                    to_global,
                    sPackIdx,
                )
            else:
                thr, ties, c = self._select(keys, per, k, sHist, sSpan, sCtl, sWarp, tidx)
                self._emit(keys, per, thr, ties, sWarp, tidx, fan, parents, target, to_global)
            return c

        @cute.jit
        def _first_tier(
            self,
            per: cutlass.Constexpr,
            k: cutlass.Constexpr,
            coarse,
            n32,
            n128,
            pin32,
            row_bm32,
            row_valid,
            sSelA,
            sSelB,
            sHist,
            sSpan,
            sCtl,
            sWarp,
            tidx,
        ):
            """Level 1 with ``per`` keys per thread (sSelA lists the k blocks, increasing)."""
            first = cute.make_rmem_tensor(cute.make_layout((per,)), Uint32)
            self._load_first(first, per, coarse, n32, n128, pin32, row_bm32, row_valid, tidx)
            thr, ties, c = self._select(first, per, k, sHist, sSpan, sCtl, sWarp, tidx)
            if coarse:
                self._emit(first, per, thr, ties, sWarp, tidx, 0, None, sSelB, None)
                per4: cutlass.Constexpr = 4 * k // self.threads
                children = cute.make_rmem_tensor(cute.make_layout((per4,)), Uint32)
                self._load_children32(
                    children, per4, c, sSelB, n32, pin32, thr, row_bm32, row_valid, tidx
                )
                thr, ties, c = self._select(children, per4, k, sHist, sSpan, sCtl, sWarp, tidx)
                self._emit(children, per4, thr, ties, sWarp, tidx, 4, sSelB, sSelA, None)
            else:
                self._emit(first, per, thr, ties, sWarp, tidx, 0, None, sSelA, None)
            return c, thr

        @cute.jit
        def _first_blocks32(
            self,
            k: cutlass.Constexpr,
            coarse,
            n32,
            n128,
            pin32,
            row_bm32,
            row_valid,
            sSelA,
            sSelB,
            sHist,
            sSpan,
            sCtl,
            sWarp,
            tidx,
        ):
            """List the row's k 32-position blocks with the largest keys in sSelA (increasing), on the smallest
            tier that holds its blocks; returns (count, threshold)."""
            args = (k, coarse, n32, n128, pin32, row_bm32, row_valid)
            args += (sSelA, sSelB, sHist, sSpan, sCtl, sWarp, tidx)
            tiers: cutlass.Constexpr = self.first
            c = Int32(0)
            thr = Uint32(0)
            if const_expr(len(tiers) == 1):
                c, thr = self._first_tier(tiers[0], *args)
            elif const_expr(len(tiers) == 2):
                if n32 <= tiers[0] * self.threads:
                    c, thr = self._first_tier(tiers[0], *args)
                else:
                    c, thr = self._first_tier(tiers[1], *args)
            else:
                if n32 <= tiers[0] * self.threads:
                    c, thr = self._first_tier(tiers[0], *args)
                elif n32 <= tiers[1] * self.threads:
                    c, thr = self._first_tier(tiers[1], *args)
                else:
                    c, thr = self._first_tier(tiers[2], *args)
            return c, thr

        @cute.jit
        def _publish(
            self,
            sSel,
            sWarp,
            c,
            vis,
            published,
            sparse_ids,
            sparse_counts,
            row,
            published_stride,
            sparse_stride,
            published_width,
            tidx,
        ):
            """Expand the c listed 8-position blocks (increasing) to their visible positions, their
            sparse ids (padding repeats the last) and the count of visible positions."""
            lane = tidx % 32
            warp = tidx // 32
            row_pub = Int64(published.toint()) + Int64(row) * published_stride * 4
            # Four consecutive columns (half an 8-position block) per vector store.
            for quad in cutlass.range(tidx, published_width // 4, self.threads):
                k = quad // 2
                first = Int32(-1)
                if k < c:
                    first = 8 * sSel[k] + 4 * (quad % 2)
                values = cute.make_rmem_tensor(cute.make_layout((4,)), Int32)
                for j in cutlass.range_constexpr(4):
                    values[j] = Int32(-1)
                    if (first >= 0) & (first + j < vis):
                        values[j] = first + j
                st_global_v4_s32(row_pub + Int64(quad) * 16, *(values[j] for j in range(4)))
            visible_count = Int32(0)
            for k in cutlass.range(tidx, c, self.threads):
                visible_count = visible_count + cutlass.max(
                    cutlass.min(vis - 8 * sSel[k], Int32(8)), Int32(0)
                )
            visible_count = warp_sum(visible_count)
            if lane == 0:
                sWarp[warp] = visible_count
            last = Int32(0)
            if c > 0:
                last = sSel[c - 1]
            row_sparse = Int64(sparse_ids.toint()) + Int64(row) * sparse_stride * 4
            for quad in cutlass.range(tidx, published_width // 32, self.threads):
                blocks = cute.make_rmem_tensor(cute.make_layout((4,)), Int32)
                for j in cutlass.range_constexpr(4):
                    blocks[j] = last
                    if 4 * quad + j < c:
                        blocks[j] = sSel[4 * quad + j]
                st_global_v4_s32(row_sparse + Int64(quad) * 16, *(blocks[j] for j in range(4)))
            cute.arch.sync_threads()
            if tidx == 0:
                total = Int32(0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + sWarp[w]
                st_global_s32(Int64(sparse_counts.toint()) + Int64(row) * 4, total)
            cute.arch.sync_threads()

        # ---------------------------------------------------------------- kernel

        @cute.kernel
        def kernel(
            self,
            logits: cute.Pointer,
            bm32: cute.Pointer,
            valid: cute.Pointer,
            visible: cute.Pointer,
            out: cute.Pointer,
            published: cute.Pointer,
            sparse_ids: cute.Pointer,
            sparse_counts: cute.Pointer,
            width: Int32,
            logits_stride: Int64,
            bm32_stride: Int64,
            valid_stride: Int64,
            out_stride: Int64,
            published_stride: Int64,
            sparse_stride: Int64,
            published_width: Int32,
        ):
            top_k: cutlass.Constexpr = self.top_k
            publish_k: cutlass.Constexpr = self.publish_k
            threads: cutlass.Constexpr = self.threads
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()

            smem = SmemAllocator()
            sSelA = smem.allocate_tensor(Int32, cute.make_layout((self.sel_cap,)), 16)
            sSelB = smem.allocate_tensor(Int32, cute.make_layout((self.sel_cap,)), 16)
            sHist = smem.allocate_tensor(Int32, cute.make_layout((2 * _BINS,)), 16)
            sSpan = smem.allocate_tensor(Int32, cute.make_layout((4,)), 16)
            sWarp = smem.allocate_tensor(Int32, cute.make_layout((self.warps,)), 16)
            sCtl = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
            # Also the publication's block keys between its second level and its top_k blocks.
            pack_cap: cutlass.Constexpr = max(_PACKED * threads, publish_k)
            sPackKey = smem.allocate_tensor(Int32, cute.make_layout((pack_cap,)), 16)
            sPackIdx = smem.allocate_tensor(Int32, cute.make_layout((_PACKED * threads,)), 16)
            for i in cutlass.range(tidx, 2 * _BINS, threads):
                sHist[i] = Int32(0)
            if tidx == 0:
                sSpan[0] = Int32(-1)
                sSpan[1] = Int32(0)
                sSpan[2] = Int32(-1)
                sSpan[3] = Int32(0)

            if const_expr(self.use_pdl):
                griddepcontrol_wait()
            vis = ld_global_s32(Int64(visible.toint()) + Int64(row) * 4)
            if vis > width:
                vis = width
            if vis < 0:
                vis = Int32(0)
            n8 = (vis + 7) // 8
            n32 = (vis + 31) // 32
            n128 = (vis + _COARSE - 1) // _COARSE
            row_logits = Int64(logits.toint()) + Int64(row) * logits_stride * 4
            row_bm32 = Int64(bm32.toint()) + Int64(row) * bm32_stride * 4
            row_valid = Int64(valid.toint()) + Int64(row) * valid_stride
            coarse = n32 > self.first[-1] * threads
            cute.arch.sync_threads()

            if const_expr(publish_k > 0):
                # The newest visible block is always published.
                pin8 = Int32(-1)
                pin32 = Int32(-1)
                if vis > 0:
                    pin8 = (vis - 1) // 8
                    pin32 = (vis - 1) // 32
                c, thr = self._first_blocks32(
                    publish_k,
                    coarse,
                    n32,
                    n128,
                    pin32,
                    row_bm32,
                    row_valid,
                    sSelA,
                    sSelB,
                    sHist,
                    sSpan,
                    sCtl,
                    sWarp,
                    tidx,
                )
                per_pub: cutlass.Constexpr = 4 * publish_k // threads
                blocks8 = cute.make_rmem_tensor(cute.make_layout((per_pub,)), Uint32)
                self._load_blocks8(
                    blocks8,
                    per_pub,
                    c,
                    sSelA,
                    n8,
                    thr,
                    pin8,
                    row_valid,
                    tidx,
                    row_logits,
                    vis,
                )
                thr, ties, c = self._select(
                    blocks8, per_pub, publish_k, sHist, sSpan, sCtl, sWarp, tidx
                )
                self._emit(
                    blocks8, per_pub, thr, ties, sWarp, tidx, 4, sSelA, sSelB, None, None, sPackKey
                )
                self._publish(
                    sSelB,
                    sWarp,
                    c,
                    vis,
                    published,
                    sparse_ids,
                    sparse_counts,
                    row,
                    published_stride,
                    sparse_stride,
                    published_width,
                    tidx,
                )
                # The row's top_k 8-position blocks are among the published ones (unpinned).
                per_listed: cutlass.Constexpr = publish_k // threads
                listed = cute.make_rmem_tensor(cute.make_layout((per_listed,)), Uint32)
                self._load_listed8(
                    listed, per_listed, c, sSelB, sPackKey, pin8, row_valid, tidx, row_logits, vis
                )
                thr, ties, c = self._select(
                    listed, per_listed, top_k, sHist, sSpan, sCtl, sWarp, tidx
                )
                self._emit(listed, per_listed, thr, ties, sWarp, tidx, 1, sSelB, sSelA, None)
                level3 = sSelA
            else:
                c, thr = self._first_blocks32(
                    top_k,
                    coarse,
                    n32,
                    n128,
                    Int32(-1),
                    row_bm32,
                    row_valid,
                    sSelA,
                    sSelB,
                    sHist,
                    sSpan,
                    sCtl,
                    sWarp,
                    tidx,
                )
                per8: cutlass.Constexpr = 4 * top_k // threads
                blocks8 = cute.make_rmem_tensor(cute.make_layout((per8,)), Uint32)
                self._load_blocks8(
                    blocks8,
                    per8,
                    c,
                    sSelA,
                    n8,
                    thr,
                    Int32(-1),
                    row_valid,
                    tidx,
                    row_logits,
                    vis,
                )
                thr, ties, c = self._select(blocks8, per8, top_k, sHist, sSpan, sCtl, sWarp, tidx)
                self._emit(blocks8, per8, thr, ties, sWarp, tidx, 4, sSelA, sSelB, None)
                level3 = sSelB
            # The top_k positions of the top_k 8-position blocks (increasing).
            per1: cutlass.Constexpr = 8 * top_k // threads
            positions = cute.make_rmem_tensor(cute.make_layout((per1,)), Uint32)
            self._load_positions(positions, per1, c, level3, vis, thr, row_logits, tidx)
            row_out = Int64(out.toint()) + Int64(row) * out_stride * 4
            c = self._packed_select(
                positions,
                per1,
                top_k,
                8,
                level3,
                None,
                row_out,
                sPackKey,
                sPackIdx,
                sHist,
                sSpan,
                sCtl,
                sWarp,
                tidx,
            )
            for i in cutlass.range(c + tidx, top_k, threads):
                st_global_s32(row_out + Int64(i) * 4, Int32(-1))
            if const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()

    return BlockSelect


_COMPILED: dict = {}


def _check_maxima(block_max32, valid, visible, rows, width):
    if (
        block_max32.dtype != torch.float32
        or block_max32.shape[0] != rows
        or block_max32.shape[1] < -(-width // _COARSE) * 4
        or block_max32.stride(1) != 1
        or block_max32.stride(0) % 4
        or block_max32.data_ptr() % 16
        or valid.dtype != torch.bool
        or valid.shape[0] != rows
        or valid.stride(1) != 1
        or valid.shape[1] < -(-width // _PAGE)
        or visible.dtype != torch.int32
        or visible.numel() != rows
        or not visible.is_contiguous()
    ):
        raise ValueError("CSA2 block selection: unsupported maxima geometry")


def _check_publication(publication, rows, width, publish_k):
    positions, sparse_ids, counts = publication
    if (
        positions.dtype != torch.int32
        or positions.shape[0] != rows
        or positions.stride(1) != 1
        or positions.stride(0) % 4
        or positions.data_ptr() % 16
        or sparse_ids.stride(0) % 4
        or sparse_ids.data_ptr() % 16
        or positions.shape[1] % 32
        or positions.shape[1] < 8 * min(publish_k, -(-width // 8))
        or sparse_ids.dtype != torch.int32
        or sparse_ids.shape[0] != rows
        or sparse_ids.stride(1) != 1
        or sparse_ids.shape[1] < positions.shape[1] // 8
        or counts.dtype != torch.int32
        or counts.numel() != rows
        or not counts.is_contiguous()
    ):
        raise ValueError("CSA2 block selection: unsupported publication geometry")


def _check_logits(logits, rows, columns):
    if (
        logits.dtype != torch.float32
        or logits.shape[0] != rows
        or logits.stride(1) != 1
        or logits.stride(0) % 4
        or logits.data_ptr() % 16
        or logits.stride(0) < columns
    ):
        raise ValueError("CSA2 block selection: unsupported logits geometry")


def _launch(width, top_k, publish_k, tensors, device):
    import cuda.bindings.driver as cuda
    import cutlass

    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

    (logits, bm32, valid, visible, output, publication) = tensors
    rows = visible.numel()
    positions, sparse_ids, counts = publication if publication is not None else (visible,) * 3
    use_pdl = get_env_enable_pdl()
    threads = block_select_threads(publish_k > 0)
    first = _first_tiers(width, threads)

    def pointer(tensor, dtype, align):
        return make_ptr(
            dtype, tensor.data_ptr(), cutlass.cute.AddressSpace.gmem, assumed_align=align
        )

    def stride(tensor):
        return cutlass.Int64(tensor.stride(0) if tensor.dim() > 1 else 0)

    args = (
        pointer(logits, cutlass.Float32, 16),
        pointer(bm32, cutlass.Float32, 16),
        pointer(valid, cutlass.Uint8, 1),
        pointer(visible, cutlass.Int32, 4),
        pointer(output, cutlass.Int32, 4),
        pointer(positions, cutlass.Int32, 4),
        pointer(sparse_ids, cutlass.Int32, 4),
        pointer(counts, cutlass.Int32, 4),
        cutlass.Int32(rows),
        cutlass.Int32(width),
        stride(logits),
        stride(bm32),
        stride(valid),
        stride(output),
        stride(positions),
        stride(sparse_ids),
        cutlass.Int32(positions.shape[1] if publication is not None else 0),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    key = (device.index, top_k, publish_k, threads, first, use_pdl)
    compiled = _COMPILED.get(key)
    if compiled is None:
        import cutlass.cute as cute

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the CSA2 block selection before CUDA Graph capture")
        kernel = _kernel_type()(top_k, publish_k, threads, first, use_pdl)
        compiled = _COMPILED[key] = cute.compile(kernel, *args)
    compiled(*args)


def block_select_(
    logits: torch.Tensor,
    block_max32: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    output: torch.Tensor,
    publication: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    publish_blocks: int = 0,
) -> torch.Tensor:
    """Per row the ``top_k`` largest visible, backed ``logits`` columns, increasing, then ``-1`` (and with
    ``publication`` its ``publish_blocks`` largest 8-position blocks), from ``block_max32``'s 32-column maxima."""
    rows, width = logits.shape
    top_k = output.shape[1]
    publish_k = publish_blocks if publication is not None else 0
    if not block_select_supported(width, top_k, publish_k):
        raise ValueError("CSA2 block selection: unsupported logits width or selection size")
    _check_logits(logits, rows, -(-width // 32) * 32)
    _check_maxima(block_max32, valid, visible, rows, width)
    if output.dtype != torch.int32 or output.shape[0] != rows or output.stride(1) != 1:
        raise ValueError("CSA2 block selection: unsupported output geometry")
    if publication is not None:
        _check_publication(publication, rows, width, publish_k)
    tensors = (logits, block_max32, valid, visible, output, publication)
    _launch(width, top_k, publish_k, tensors, logits.device)
    return output
