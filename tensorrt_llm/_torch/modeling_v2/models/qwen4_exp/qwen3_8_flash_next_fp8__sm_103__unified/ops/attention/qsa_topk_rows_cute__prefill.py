# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA's block top-k and its expansion for a context chunk's rows over a wide score table.

Same contract and bits as the family's `qsa_topk_expand_decode` (the `block_topk` largest of
each row's visible scores, ties to the higher block, in block order, -1 padded, then the token
expansion with the causal tail). One CTA a row reads the window twice from global, never staged:
a per-warp bound `lo` with at least `block_topk` keys at or above it, a candidate list of those
keys, radix passes that find the threshold's bin, and bitmaps whose scan orders the selected
blocks. Scores are non-negative fp32 read as int32 bit patterns; -0.0 ranks as zero.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Consecutive scores a thread holds per slab (one 16-byte load).
KPT = 4
#: Threads a row (a CTA).
THREADS = 512
#: Slabs whose loads are issued together.
UNROLL = 4
#: Candidate list entries: about 4 % of a window survives `lo`, so ~100k-block windows fit.
CAPC = 3072
_HIST = 2048
_I32_MAX = 0x7FFFFFFF
#: CTAs an SM the register allocation is held to.
MIN_BLOCKS = 3
#: Score columns this kernel serves: wider than one family CTA's registers, up to a
#: 262144-token table with MTP's extra pages.
MIN_WIDTH = 4096 + KPT
MAX_WIDTH = 1 << 17


class _TopkRowsKernel:
    def __init__(self, *, width, key_stride, block_topk, compress, token_topk, pdl):
        self.nt = THREADS
        self.nw = THREADS // 32
        self.slab = KPT * THREADS
        self.width, self.key_stride, self.k, self.cr, self.token_topk, self.pdl = (
            width,
            key_stride,
            block_topk,
            compress,
            token_topk,
            pdl,
        )
        self.final = token_topk + compress - 1
        self.words = -(-width // 32)
        self.wpt = -(-self.words // THREADS)
        self.per_warp = -(-block_topk // self.nw)
        # Values a lane keeps for the bound: each warp needs `per_warp` of them at or above
        # its value, from at most 32 * vpl.
        self.vpl = max(2, -(-(3 * self.per_warp) // 64))
        self.union_words = max(2 * self.words, _HIST)
        assert self.per_warp <= 32 * self.vpl, "too few lane values for the per-warp bound"

    def smem_bytes(self) -> int:
        return (self.union_words + 2 * CAPC + 64 + 32 + 16) * 4 + 6 * 128

    @cute.jit
    def __call__(
        self,
        keys_t: cute.Tensor,
        end_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        len_ptr: cute.Pointer,
        topk_ptr: cute.Pointer,
        sel_ptr: cute.Pointer,
        rows: cutlass.Int32,
        stream,
    ):
        mKeys = cute.make_tensor(
            keys_t.iterator, cute.make_layout((rows, self.width), stride=(self.key_stride, 1))
        )
        per_row = cute.make_layout((rows,), stride=(1,))
        mEnd = cute.make_tensor(end_ptr, per_row)
        mPos = cute.make_tensor(pos_ptr, per_row)
        mLen = cute.make_tensor(len_ptr, per_row)
        mTopk = cute.make_tensor(topk_ptr, cute.make_layout((rows, self.k), stride=(self.k, 1)))
        mSel = cute.make_tensor(
            sel_ptr, cute.make_layout((rows, self.final), stride=(self.final, 1))
        )
        ld_keys = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), cutlass.Int32, num_bits_per_copy=32 * KPT
        )
        self.kernel(mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys).launch(
            grid=[rows, 1, 1],
            block=[self.nt, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=MIN_BLOCKS,
        )

    @cute.jit
    def _load(self, ld, gK, row, s, tidx, n, kr):
        """Slab `s`'s four keys of thread `tidx` into `kr` (a negative as zero, -1 past the
        window)."""
        col4 = s * self.nt + tidx
        e0 = col4 * KPT
        kr.fill(-1)
        if e0 < n:
            cute.copy(ld, gK[row, (None, col4)], kr)
        for c in cutlass.range_constexpr(KPT):
            b = cutlass.max(kr[c], cutlass.Int32(0))
            if e0 + c >= n:
                b = cutlass.Int32(-1)
            kr[c] = b

    @cute.jit
    def _insert(self, mv, b):
        """`b` into the thread's descending `vpl` largest (`mv[0]` the largest)."""
        for j in cutlass.range_constexpr(self.vpl - 1, 0, -1):
            mv[j] = cutlass.max(mv[j], cutlass.min(mv[j - 1], b))
        mv[0] = cutlass.max(mv[0], b)

    @cute.jit
    def _load_full(self, ld, gK, row, s, tidx, kr):
        """`_load` of a slab wholly inside the window (no bounds)."""
        cute.copy(ld, gK[row, (None, s * self.nt + tidx)], kr)
        for c in cutlass.range_constexpr(KPT):
            kr[c] = cutlass.max(kr[c], cutlass.Int32(0))

    @cute.jit
    def _compact(self, kr, s0, tidx, lane, lo, state, ckey, cidx):
        """The UNROLL slabs' keys at or above `lo` appended to the list (a warp scan of the
        counts, one shared atomic a warp)."""
        cnt = cutlass.Int32(0)
        for u in cutlass.range_constexpr(UNROLL):
            for c in cutlass.range_constexpr(KPT):
                if kr[u][c] >= lo:
                    cnt = cnt + 1
        incl = cnt
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        base = cutlass.Int32(0)
        if lane == 31:
            if incl > 0:
                base = cute.arch.atomic_add(state.iterator + 5, incl, scope="cta")
        base = cute.arch.shuffle_sync(base, offset=31)
        pos = base + incl - cnt
        if cnt > 0:
            for u in cutlass.range_constexpr(UNROLL):
                e0 = ((s0 + u) * self.nt + tidx) * KPT
                for c in cutlass.range_constexpr(KPT):
                    if kr[u][c] >= lo:
                        if pos < CAPC:
                            ckey[pos] = kr[u][c]
                            cidx[pos] = e0 + c
                        pos = pos + 1

    @cute.jit
    def _excl_scan(self, v, warp, lane, ws, half):
        """Block-wide exclusive prefix of `v` and the total (one barrier; `ws[half..half+15]`)."""
        incl = v
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            ws[half + warp] = incl
        cute.arch.sync_threads()
        above = cutlass.Int32(0)
        if lane < warp:
            above = ws[half + lane]
        total = cutlass.Int32(0)
        if lane < self.nw:
            total = ws[half + lane]
        pre = cute.arch.warp_reduction_sum(above) + incl - v
        total = cute.arch.warp_reduction_sum(total)
        return pre, total

    @cute.jit
    def _narrow(self, tidx, warp, lane, hist, ws, state, prefix, mask, need, shift):
        """One pass's histogram read (and cleared) from the top bin down: the bin holding the
        `need`-th key becomes the threshold's (prefix, mask, need left, keys in the bin); the
        passes stop once that bin holds exactly the keys still needed or no bits are left."""
        bpt: cutlass.Constexpr = _HIST // self.nt
        h = cute.make_rmem_tensor(cute.make_layout((bpt,), stride=(1,)), cutlass.Int32)
        h.fill(0)
        own = cutlass.Int32(0)
        for j in cutlass.range_constexpr(bpt):
            h[j] = hist[_HIST - 1 - bpt * tidx - j]
            hist[_HIST - 1 - bpt * tidx - j] = cutlass.Int32(0)
            own = own + h[j]
        before, _ = self._excl_scan(own, warp, lane, ws, 0)
        if before < need:
            if need <= before + own:
                run = before
                for j in cutlass.range_constexpr(bpt):
                    if run < need:
                        if need <= run + h[j]:
                            digit = cutlass.Int32(_HIST - 1 - bpt * tidx - j)
                            state[0] = prefix | (digit << shift)
                            state[1] = mask | (cutlass.Int32(_HIST - 1) << shift)
                            state[2] = need - run
                            state[3] = h[j]
                            state[4] = cutlass.Int32(
                                cutlass.select_((h[j] == need - run) | (shift == 0), 1, 0)
                            )
                            state[6] = cutlass.max(shift - 11, cutlass.Int32(0))
                    run = run + h[j]
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys: cute.CopyAtom):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        nt: cutlass.Constexpr = self.nt
        k: cutlass.Constexpr = self.k
        cr: cutlass.Constexpr = self.cr
        words: cutlass.Constexpr = self.words
        wpt: cutlass.Constexpr = self.wpt
        smem = cutlass.utils.SmemAllocator()
        # The radix histogram and the two bitmaps share one region: the passes clear every bin
        # before the bitmaps are set.
        region = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.union_words,), stride=(1,)), byte_alignment=128
        )
        hist = cute.make_tensor(region.iterator, cute.make_layout((_HIST,), stride=(1,)))
        selw = cute.make_tensor(region.iterator, cute.make_layout((words,), stride=(1,)))
        tiew = cute.make_tensor(region.iterator + words, cute.make_layout((words,), stride=(1,)))
        ckey = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAPC,), stride=(1,)), byte_alignment=128
        )
        cidx = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAPC,), stride=(1,)), byte_alignment=128
        )
        ws = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((64,), stride=(1,)), byte_alignment=128
        )
        wv = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        state = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((16,), stride=(1,)), byte_alignment=128
        )

        # 0. Shared state and the row's host-filled metadata, before the wait.
        for b in cutlass.range_constexpr(-(-self.union_words // nt)):
            if tidx + b * nt < self.union_words:
                region[tidx + b * nt] = cutlass.Int32(0)
        if tidx == 0:
            # A window of at most k blocks takes every block (mask 0, prefix 0, nothing skipped).
            state[0] = cutlass.Int32(0)
            state[1] = cutlass.Int32(0)
            state[2] = cutlass.Int32(0)
            state[3] = cutlass.Int32(0)
            state[4] = cutlass.Int32(1)
            state[5] = cutlass.Int32(0)  # candidates compacted
            state[6] = cutlass.Int32(0)
        n = mEnd[row]
        seqlen = mLen[row]
        visible = cutlass.Int32(mPos[row]) + 1
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        gK = cute.logical_divide(mKeys, (None, KPT))
        slab: cutlass.Constexpr = self.slab
        nslabs = (n + (slab - 1)) // slab
        # Slabs in whole UNROLL groups wholly inside the window (no per-key bounds).
        nfull = (n // (slab * UNROLL)) * UNROLL
        kr = [
            cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
            for _ in range(UNROLL)
        ]

        # 1. Each thread's `vpl` largest keys; the per-warp bound; `lo`.
        lo = cutlass.Int32(0)
        hb = cutlass.Int32(-1)
        if n > k:
            mv = cute.make_rmem_tensor(cute.make_layout((self.vpl,), stride=(1,)), cutlass.Int32)
            mv.fill(-1)
            for s0 in cutlass.range(0, nfull, UNROLL, unroll=1):
                for u in cutlass.range_constexpr(UNROLL):
                    self._load_full(ld_keys, gK, row, s0 + u, tidx, kr[u])
                for u in cutlass.range_constexpr(UNROLL):
                    for c in cutlass.range_constexpr(KPT):
                        self._insert(mv, kr[u][c])
            for s0 in cutlass.range(nfull, nslabs, UNROLL, unroll=1):
                for u in cutlass.range_constexpr(UNROLL):
                    self._load(ld_keys, gK, row, s0 + u, tidx, n, kr[u])
                for u in cutlass.range_constexpr(UNROLL):
                    for c in cutlass.range_constexpr(KPT):
                        self._insert(mv, kr[u][c])
            m1 = mv[0]
            have_t = cutlass.Int32(0)
            for j in cutlass.range_constexpr(self.vpl):
                have_t = have_t + cutlass.Int32(cutlass.select_(mv[j] >= 0, 1, 0))
            have = cute.arch.warp_redux_sync(have_t, "add")
            v = cutlass.Int32(-1)
            if have >= self.per_warp:
                t = cutlass.Int32(0)
                # The top 16 bits (exponent and 7 mantissa bits): a key at or below the
                # exact per-warp value is still a valid bound, within 1/128 of it.
                for bit in cutlass.range_constexpr(30, 14, -1):
                    c = t | cutlass.Int32(1 << bit)
                    cnt_t = cutlass.Int32(0)
                    for j in cutlass.range_constexpr(self.vpl):
                        cnt_t = cnt_t + cutlass.Int32(cutlass.select_(mv[j] >= c, 1, 0))
                    cnt = cute.arch.warp_redux_sync(cnt_t, "add")
                    if cnt >= self.per_warp:
                        t = c
                v = t
            wmax = cute.arch.warp_redux_sync(m1, "max")
            if lane == 0:
                wv[warp] = v
                wv[16 + warp] = wmax
            cute.arch.sync_threads()
            bound = cutlass.Int32(_I32_MAX)
            gmax = cutlass.Int32(-1)
            if lane < self.nw:
                bound = wv[lane]
                gmax = wv[16 + lane]
            bound = cute.arch.warp_reduction(bound, lambda x, y: cutlass.min(x, y))
            gmax = cute.arch.warp_reduction_max(gmax)
            lo = cutlass.max(bound, cutlass.Int32(0))
            # Every candidate shares the bits above the highest bit where `gmax` and `lo`
            # differ; the radix passes walk down from there, 11 bits at a time.
            diff = gmax ^ lo
            if diff != 0:
                hb = 31 - cutlass.Int32(cute.arch.clz(diff))
            low_bits = (cutlass.Int64(1) << cutlass.Int64(hb + 1)) - 1
            mask0 = cutlass.Int32(cutlass.Int64(_I32_MAX) & (cutlass.Int64(-1) ^ low_bits))
            if tidx == 0:
                state[0] = gmax & mask0  # threshold prefix
                state[1] = mask0  # its mask
                state[2] = cutlass.Int32(k)  # still needed at the threshold
                state[3] = cutlass.Int32(0)  # keys in the threshold's bin
                state[4] = cutlass.Int32(cutlass.select_(hb >= 0, 0, 1))  # narrowing done
                state[6] = cutlass.max(hb - 10, cutlass.Int32(0))  # the next pass's shift

        # 2. The candidates (keys at or above `lo`), unordered, with their blocks.
        for s0 in cutlass.range(0, nfull, UNROLL, unroll=1):
            for u in cutlass.range_constexpr(UNROLL):
                self._load_full(ld_keys, gK, row, s0 + u, tidx, kr[u])
            self._compact(kr, s0, tidx, lane, lo, state, ckey, cidx)
        for s0 in cutlass.range(nfull, nslabs, UNROLL, unroll=1):
            for u in cutlass.range_constexpr(UNROLL):
                self._load(ld_keys, gK, row, s0 + u, tidx, n, kr[u])
            self._compact(kr, s0, tidx, lane, lo, state, ckey, cidx)
        cute.arch.sync_threads()
        m = state[5]
        listed = m <= CAPC

        # 3. The threshold's bin, narrowed by radix passes over the list (or the window)
        # until it holds exactly the keys still needed or every bit is fixed.
        for _p in cutlass.range_constexpr(3):
            if state[4] == 0:
                prefix = state[0]
                mask = state[1]
                need = state[2]
                shift = state[6]
                if listed:
                    for i in cutlass.range(tidx, m, nt, unroll=1):
                        key = ckey[i]
                        if (key & mask) == prefix:
                            cute.arch.atomic_add(
                                hist.iterator + ((key >> shift) & (_HIST - 1)),
                                cutlass.Int32(1),
                                scope="cta",
                            )
                else:
                    for s0 in cutlass.range(0, nslabs, UNROLL, unroll=1):
                        for u in cutlass.range_constexpr(UNROLL):
                            self._load(ld_keys, gK, row, s0 + u, tidx, n, kr[u])
                        for u in cutlass.range_constexpr(UNROLL):
                            for c in cutlass.range_constexpr(KPT):
                                key = kr[u][c]
                                if key >= 0:
                                    if (key & mask) == prefix:
                                        cute.arch.atomic_add(
                                            hist.iterator + ((key >> shift) & (_HIST - 1)),
                                            cutlass.Int32(1),
                                            scope="cta",
                                        )
                cute.arch.sync_threads()
                self._narrow(tidx, warp, lane, hist, ws, state, prefix, mask, need, shift)
        prefix = state[0]
        mask = state[1]
        need = state[2]
        at_thr = state[3]
        if n > k:
            if hb < 0:
                # Every candidate equals the largest: the bin is all of them.
                at_thr = m

        # 4. Bitmaps of the blocks above the threshold's bin and in it.
        if listed:
            for i in cutlass.range(tidx, m, nt, unroll=1):
                mk = ckey[i] & mask
                e = cidx[i]
                bit = cutlass.Int32(1) << (e & 31)
                if mk > prefix:
                    cute.arch.atomic_or(selw.iterator + (e >> 5), bit, scope="cta")
                elif mk == prefix:
                    cute.arch.atomic_or(tiew.iterator + (e >> 5), bit, scope="cta")
        else:
            for s0 in cutlass.range(0, nslabs, UNROLL, unroll=1):
                for u in cutlass.range_constexpr(UNROLL):
                    self._load(ld_keys, gK, row, s0 + u, tidx, n, kr[u])
                for u in cutlass.range_constexpr(UNROLL):
                    e0 = ((s0 + u) * nt + tidx) * KPT
                    for c in cutlass.range_constexpr(KPT):
                        key = kr[u][c]
                        e = e0 + c
                        bit = cutlass.Int32(1) << (e & 31)
                        if key >= 0:
                            mk = key & mask
                            if mk > prefix:
                                cute.arch.atomic_or(selw.iterator + (e >> 5), bit, scope="cta")
                            elif mk == prefix:
                                cute.arch.atomic_or(tiew.iterator + (e >> 5), bit, scope="cta")
        cute.arch.sync_threads()

        # 5. Block order: each thread folds its `wpt` words' kept tie blocks (the lowest `skip`
        # dropped) into the selected words; a scan gives each word its first output position.
        skip = cutlass.max(at_thr - need, cutlass.Int32(0))
        w0 = tidx * wpt
        own_q = cutlass.Int32(0)
        for j in cutlass.range_constexpr(wpt):
            if w0 + j < words:
                own_q = own_q + cute.arch.popc(tiew[w0 + j])
        tied_before, _ = self._excl_scan(own_q, warp, lane, ws, 0)
        fw = cute.make_rmem_tensor(cute.make_layout((wpt,), stride=(1,)), cutlass.Int32)
        fw.fill(0)
        own_t = cutlass.Int32(0)
        for j in cutlass.range_constexpr(wpt):
            if w0 + j < words:
                tw = tiew[w0 + j]
                nq = cute.arch.popc(tw)
                drop = cutlass.min(cutlass.max(skip - tied_before, cutlass.Int32(0)), nq)
                # clear the lowest `drop` set bits of the word's tie class
                for _d in cutlass.range(drop, unroll=1):
                    tw = tw & (tw - 1)
                tied_before = tied_before + nq
                f = selw[w0 + j] | tw
                fw[j] = f
                own_t = own_t + cute.arch.popc(f)
        first, _ = self._excl_scan(own_t, warp, lane, ws, 32)
        for j in cutlass.range_constexpr(wpt):
            if w0 + j < words:
                selw[w0 + j] = fw[j]
                tiew[w0 + j] = first
                first = first + cute.arch.popc(fw[j])
        cute.arch.sync_threads()
        # Output position `o` (one a thread a round): its word by binary search over the words'
        # first positions, its block by the in-word rank of the set bit.
        total = n
        if n > k:
            total = cutlass.Int32(k)
        for c in cutlass.range_constexpr(-(-k // nt)):
            o = tidx + c * nt
            if o < total:
                lo_w = cutlass.Int32(0)
                hi_w = cutlass.Int32(words - 1)
                while lo_w < hi_w:
                    mid = (lo_w + hi_w + 1) >> 1
                    if tiew[mid] <= o:
                        lo_w = mid
                    else:
                        hi_w = mid - 1
                x = selw[lo_w]
                r = o - tiew[lo_w]
                bit = cutlass.Int32(0)
                for h in (16, 8, 4, 2, 1):
                    cnt_h = cute.arch.popc((x >> bit) & cutlass.Int32((1 << h) - 1))
                    if r >= cnt_h:
                        r = r - cnt_h
                        bit = bit + h
                e = lo_w * 32 + bit
                mTopk[row, o] = e
                for jj in cutlass.range_constexpr(cr):
                    tok = e * cr + jj
                    val = cutlass.Int32(-1)
                    if tok < seqlen:
                        val = tok
                    mSel[row, o * cr + jj] = val

        # 6. Padding of the block list; the causal tail after the selected tokens, then -1.
        for c in cutlass.range_constexpr(-(-k // nt)):
            slot = tidx + c * nt
            if slot < k:
                if slot >= total:
                    mTopk[row, slot] = cutlass.Int32(-1)
        tail_start = (visible // cr) * cr
        tail_count = visible - tail_start
        body = total * cr
        if body > self.token_topk:
            body = cutlass.Int32(self.token_topk)
        for c in cutlass.range_constexpr(-(-self.final // nt)):
            col = tidx + c * nt
            if col < self.final:
                if col >= body:
                    off = col - body
                    val = cutlass.Int32(-1)
                    if off < cr - 1:
                        if off < tail_count:
                            if tail_start + off < seqlen:
                                val = tail_start + off
                    mSel[row, col] = val


_COMPILED: dict = {}
_INDEX_TYPES = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


def supports(scores, topk_indices) -> bool:
    """Whether this kernel serves a call over this score table and `block_topk`."""
    width = scores.shape[1]
    return (
        scores.dtype == torch.float32
        and scores.stride(1) == 1
        and width % KPT == 0
        and scores.stride(0) % KPT == 0
        and MIN_WIDTH <= width <= MAX_WIDTH
        and -(-topk_indices.shape[1] // (THREADS // 32)) <= 96
    )


def qsa_topk_rows(
    scores,
    row_ends,
    query_positions,
    sequence_lengths,
    topk_indices,
    compress_ratio,
    token_topk,
    out=None,
):
    """`qsa_topk_expand_decode`'s contract, bit for bit, for wide tables (`supports`): writes
    `topk_indices` and returns the token selection `[rows, token_topk + compress_ratio - 1]`
    int32 (`out` when given)."""
    rows, width = scores.shape
    block_topk = topk_indices.shape[1]
    sel = out
    if sel is None:
        sel = torch.empty(
            (rows, token_topk + compress_ratio - 1), dtype=torch.int32, device=scores.device
        )
    if rows == 0:
        return sel
    pdl = bool(get_env_enable_pdl())
    keys = scores.view(torch.int32)
    key = (
        width,
        scores.stride(0),
        block_topk,
        compress_ratio,
        token_topk,
        query_positions.dtype,
        sequence_lengths.dtype,
        pdl,
    )
    ptrs = (
        row_ends.data_ptr(),
        query_positions.data_ptr(),
        sequence_lengths.data_ptr(),
        topk_indices.data_ptr(),
        sel.data_ptr(),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        gm = cute.AddressSpace.gmem
        kernel = _TopkRowsKernel(
            width=width,
            key_stride=scores.stride(0),
            block_topk=block_topk,
            compress=compress_ratio,
            token_topk=token_topk,
            pdl=pdl,
        )
        fn = _COMPILED[key] = cute.compile(
            kernel,
            from_dlpack(keys, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.Int32, ptrs[0], gm, assumed_align=4),
            make_ptr(
                _INDEX_TYPES[query_positions.dtype],
                ptrs[1],
                gm,
                assumed_align=query_positions.element_size(),
            ),
            make_ptr(
                _INDEX_TYPES[sequence_lengths.dtype],
                ptrs[2],
                gm,
                assumed_align=sequence_lengths.element_size(),
            ),
            make_ptr(cutlass.Int32, ptrs[3], gm, assumed_align=4),
            make_ptr(cutlass.Int32, ptrs[4], gm, assumed_align=4),
            cutlass.Int32(rows),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    fn(keys, *ptrs, rows)
    return sel
