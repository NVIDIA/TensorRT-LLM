# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA's block top-k and its expansion to token indices, one launch per batch of rows.

A CuTe DSL kernel written for the qwen4_exp tp1 target and shared with the unified target
(promoted from their copies): an in-tree callable. The decode path ran `qsa_topk_select`
(the CUDA indexer top-k, one CTA a row) and then `qsa_expand_block_indices` (one
more launch) per attention layer. Here one CTA per row holds four consecutive scores a
thread and narrows to the k-th largest by radix passes over the scores' bit patterns
(shared-memory histograms of 11 bits), starting below the bits every visible score
shares. It stops as soon as the bin holding the threshold has at most `CAP` keys and
ranks those exactly by pairwise comparison (ties to the higher block, the canonical
selection's rule). Every selected block is then placed in block order by one block-wide
scan, and expanded to token indices with the causal tail, as the expansion entry does.

Scores are a sum of ReLUs times a positive scale, so non-negative: their fp32 bit
patterns, read as int32, order like their values (a negative zero is taken as zero, a
NaN ranks above everything).

A table wider than one CTA's four keys a thread (`MAX_WIDTH`, e.g. the 65536 columns of a
262144-token `max_seq_len`) takes `_TopkExpandWideKernel`, still one CTA a row and one
launch: the row's window is staged in shared memory, a bound every top-k key clears is
read off the lanes' maxima, the keys above it are compacted in block order, and the same
selection runs over that short list (or over the window, slab by slab, when the list
cannot hold them).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_expand_block_indices import (
    qsa_expand_block_indices as _expand,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .._op import Arch, Cell, OpWrapper

#: Consecutive scores a thread holds (one 16-byte load).
KPT = 4
#: Score columns a row may have: `KPT` a thread, at most 1024 threads a CTA.
MAX_WIDTH = KPT * 1024
#: CTAs an SM the register allocation is held to. At the 2592-wide score table a row is 672
#: threads, which the per-subpartition register file holds once an SM at the compiler's own
#: 48 registers a thread; at three (32 a thread) rows overlap each other's barriers and
#: shared-memory round trips (GB300, a mixed step's 16256 context rows: 426 -> 236 us on
#: half-zero rows, 486 -> 284 on relu-sparse ones; 128 decode rows 8.2 us both ways).
MIN_BLOCKS = 3
#: Radix passes at most (11 bits each; 31 magnitude bits).
_PASSES = 3
_HIST = 2048
_I32_MAX = 0x7FFFFFFF
#: At most this many keys in the threshold's bin are ranked pairwise instead of
#: narrowed by another pass.
CAP = 128
#: Threads that own the 2048 bins of a scan, four each.
_SCAN_THREADS = _HIST // 4
#: Consecutive threads that rank one candidate of the threshold's bin, each over every
#: `_RANK_LANES`-th candidate (`_SCAN_THREADS` threads cover `CAP` candidates).
_RANK_LANES = _SCAN_THREADS // CAP

#: Tables wider than `MAX_WIDTH` take the wide kernel: one CTA a row (1024 threads by
#: default) walks its visible window in slabs of `KPT` columns a thread.
_WIDE_THREADS = 1024
#: The widest table the wide kernel serves (the row count times the row stride stays an
#: Int32 product for 8192 rows).
MAX_WIDE_WIDTH = 1 << 17
#: Shared memory of an SM on sm_100/sm_103 (228 KiB), the most one CTA may hold (227 KiB),
#: and what the driver reserves a CTA (1 KiB).
_SMEM_SM = 233472
_SMEM_LIMIT = 232448
_SMEM_RESERVED = 1024
#: The wide kernel's fixed shared state in 4-byte words: the histogram, the candidates,
#: three sets of warp totals, the scan's double-buffered warp totals and the state words;
#: and the alignment padding of its allocations.
_WIDE_FIXED_WORDS = _HIST + 2 * CAP + 3 * 32 + 64 + 16
_WIDE_PAD = 10 * 128


def _threads(width: int) -> int:
    return max(_SCAN_THREADS, -(-width // (KPT * 32)) * 32)


class _TopkExpandKernel:
    def __init__(self, *, width, key_stride, block_topk, compress, token_topk, pdl):
        self.width, self.key_stride, self.k, self.cr, self.token_topk, self.pdl = (
            width,
            key_stride,
            block_topk,
            compress,
            token_topk,
            pdl,
        )
        self.final = token_topk + compress - 1
        self.threads = _threads(width)
        self.warps = self.threads // 32
        self.flag_words = -(-width // 32)

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
        # The keys arrive as a tensor, as the TVM-FFI entry takes its stream from a GPU
        # tensor; their layout is rebuilt with the static width and row stride the 16-byte
        # loads need. The row count is the call's.
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
            block=[self.threads, 1, 1],
            smem=(_HIST + self.flag_words + 2 * CAP + 64 + 16) * 4 + 128,
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=MIN_BLOCKS,
        )

    @cute.kernel
    def kernel(self, mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys: cute.CopyAtom):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        nt: cutlass.Constexpr = self.threads
        nw: cutlass.Constexpr = self.warps
        k: cutlass.Constexpr = self.k
        cr: cutlass.Constexpr = self.cr
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((_HIST,), stride=(1,)), byte_alignment=16
        )
        flags = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((self.flag_words,), stride=(1,)), byte_alignment=16
        )
        cand_key = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAP,), stride=(1,)), byte_alignment=16
        )
        cand_idx = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAP,), stride=(1,)), byte_alignment=16
        )
        wa = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=16
        )
        wb = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=16
        )
        state = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((16,), stride=(1,)), byte_alignment=16
        )

        # 0. Shared state, before the wait (nothing global is read here).
        for b in cutlass.range_constexpr(-(-_HIST // nt)):
            if tidx + b * nt < _HIST:
                hist[tidx + b * nt] = cutlass.Int32(0)
        for b in cutlass.range_constexpr(-(-self.flag_words // nt)):
            if tidx + b * nt < self.flag_words:
                flags[tidx + b * nt] = cutlass.Int32(0)
        if tidx == 0:
            state[5] = cutlass.Int32(0)  # candidates compacted
        # The row's metadata, host-filled: read before the wait too.
        n = mEnd[row]
        seqlen = mLen[row]
        visible = cutlass.Int32(mPos[row]) + 1
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # 1. Four consecutive keys a thread (a column past the window is no candidate:
        # -1); the largest and smallest key in the window.
        e0 = tidx * KPT
        keys = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        keys.fill(-1)
        if e0 < self.width:
            cute.copy(ld_keys, cute.logical_divide(mKeys, (None, KPT))[row, (None, tidx)], keys)
        kmax = cutlass.Int32(-1)
        kmin = cutlass.Int32(_I32_MAX)
        for c in cutlass.range_constexpr(KPT):
            b = keys[c]
            if b < 0:  # -0.0 (or a negative): rank as zero
                b = cutlass.Int32(0)
            if e0 + c >= n:
                b = cutlass.Int32(-1)
            keys[c] = b
            kmax = cutlass.max(kmax, b)
            if b >= 0:
                kmin = cutlass.min(kmin, b)
        kmax = cute.arch.warp_reduction_max(kmax)
        kmin = cute.arch.warp_reduction(kmin, lambda x, y: cutlass.min(x, y))
        if lane == 0:
            wa[warp] = kmax
            wb[warp] = kmin
        cute.arch.sync_threads()
        gmax = cutlass.Int32(-1)
        gmin = cutlass.Int32(_I32_MAX)
        if lane < nw:
            gmax = wa[lane]
            gmin = wb[lane]
        gmax = cute.arch.warp_reduction_max(gmax)
        gmin = cute.arch.warp_reduction(gmin, lambda x, y: cutlass.min(x, y))

        # 2. Every key in the window shares the bits above the highest bit where the
        # largest and the smallest differ (`hb`); radix passes walk down from there, 11 bits
        # at a time, until the threshold's bin holds at most CAP keys or no bits are left.
        diff = gmax ^ gmin
        hb = cutlass.Int32(-1)
        if diff != 0:
            hb = 31 - cutlass.Int32(cute.arch.clz(diff))
        low = (cutlass.Int64(1) << cutlass.Int64(hb + 1)) - 1
        mask0 = cutlass.Int32(cutlass.Int64(_I32_MAX) & (cutlass.Int64(-1) ^ low))
        if tidx == 0:
            state[0] = gmax & mask0  # threshold prefix
            state[1] = mask0  # its mask
            state[2] = cutlass.Int32(k)  # still needed at the threshold
            state[3] = n  # keys in the threshold's bin
            state[4] = cutlass.Int32(1)  # narrowing done
            state[6] = cutlass.max(hb - 10, 0)  # the next pass's shift
            if n > k:
                if hb >= 0:
                    state[4] = cutlass.Int32(0)
        cute.arch.sync_threads()
        for _p in cutlass.range_constexpr(_PASSES):
            if state[4] == 0:
                prefix = state[0]
                mask = state[1]
                need = state[2]
                shift = state[6]
                for c in cutlass.range_constexpr(KPT):
                    if keys[c] >= 0:
                        if (keys[c] & mask) == prefix:
                            digit = (keys[c] >> shift) & (_HIST - 1)
                            cute.arch.atomic_add(
                                hist.iterator + digit, cutlass.Int32(1), scope="cta"
                            )
                cute.arch.sync_threads()
                # Thread t < 512 owns bins 2047-4t .. 2044-4t, counted from the top; each is
                # cleared for the next pass once read.
                h = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
                h.fill(0)
                own = cutlass.Int32(0)
                if tidx < _SCAN_THREADS:
                    for j in cutlass.range_constexpr(4):
                        h[j] = hist[_HIST - 1 - 4 * tidx - j]
                        hist[_HIST - 1 - 4 * tidx - j] = cutlass.Int32(0)
                        own = own + h[j]
                incl = own
                for off in (1, 2, 4, 8, 16):
                    up = cute.arch.shuffle_sync_up(incl, offset=off)
                    if lane >= off:
                        incl = incl + up
                if lane == 31:
                    wa[warp] = incl
                cute.arch.sync_threads()
                above = cutlass.Int32(0)
                if lane < warp:
                    above = wa[lane]
                before = cute.arch.warp_reduction_sum(above) + incl - own  # keys in the bins above
                if tidx < _SCAN_THREADS:
                    if before < need:
                        if need <= before + own:
                            run = before
                            for j in cutlass.range_constexpr(4):
                                if run < need:
                                    if need <= run + h[j]:
                                        state[0] = prefix | ((_HIST - 1 - 4 * tidx - j) << shift)
                                        state[1] = mask | ((_HIST - 1) << shift)
                                        state[2] = need - run
                                        state[3] = h[j]
                                        state[4] = cutlass.Int32(
                                            cutlass.select_((h[j] <= CAP) | (shift == 0), 1, 0)
                                        )
                                        state[6] = cutlass.max(shift - 11, 0)
                                run = run + h[j]
                cute.arch.sync_threads()

        prefix = state[0]
        mask = state[1]
        need = state[2]
        at_thr = state[3]
        # 3. At most CAP keys in the threshold's bin: rank them exactly, ties to the
        # higher block, and flag the `need` best.
        if n > k:
            if at_thr <= CAP:
                for c in cutlass.range_constexpr(KPT):
                    if keys[c] >= 0:
                        if (keys[c] & mask) == prefix:
                            slot = cute.arch.atomic_add(
                                state.iterator + 5, cutlass.Int32(1), scope="cta"
                            )
                            cand_key[slot] = keys[c]
                            cand_idx[slot] = e0 + c
                cute.arch.sync_threads()
                # Candidate `ci` is ranked by `_RANK_LANES` consecutive lanes, lane `sub`
                # counting the better ones among candidates sub, sub + _RANK_LANES, ...;
                # butterfly shuffles add the counts up.
                ci = tidx // _RANK_LANES
                sub = tidx % _RANK_LANES
                rank = cutlass.Int32(0)
                it = cutlass.Int32(0)
                if ci < at_thr:
                    kt = cand_key[ci]
                    it = cand_idx[ci]
                    for j in cutlass.range(sub, at_thr, _RANK_LANES, unroll=4):
                        kj = cand_key[j]
                        ij = cand_idx[j]
                        better = (kj > kt) | ((kj == kt) & (ij > it))
                        rank = rank + cutlass.Int32(cutlass.select_(better, 1, 0))
                for off in cutlass.range_constexpr(_RANK_LANES.bit_length() - 1):
                    rank = rank + cute.arch.shuffle_sync_bfly(rank, offset=1 << off)
                if ci < at_thr:
                    if sub == 0:
                        if rank < need:
                            cute.arch.atomic_or(
                                flags.iterator + (it >> 5),
                                cutlass.Int32(1) << (it & 31),
                                scope="cta",
                            )
                cute.arch.sync_threads()

        # 4. Block order: one scan of (selected, tied at the threshold) counts. A tied key
        # is left only when the passes ran out of bits with more than CAP keys tied; the
        # last `need` of those in block order are taken.
        take = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        tied = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        own = cutlass.Int32(0)
        for c in cutlass.range_constexpr(KPT):
            t_c = cutlass.Int32(0)
            q_c = cutlass.Int32(0)
            if keys[c] >= 0:
                if n <= k:
                    t_c = cutlass.Int32(1)
                else:
                    mk = keys[c] & mask
                    if mk > prefix:
                        t_c = cutlass.Int32(1)
                    if mk == prefix:
                        if at_thr > CAP:
                            q_c = cutlass.Int32(1)
                        else:
                            e = e0 + c
                            t_c = (flags[e >> 5] >> (e & 31)) & 1
            take[c] = t_c
            tied[c] = q_c
            own = own + t_c + (q_c << 16)
        incl = own
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            wb[warp] = incl
        cute.arch.sync_threads()
        above = cutlass.Int32(0)
        if lane < warp:
            above = wb[lane]
        pre = cute.arch.warp_reduction_sum(above) + incl - own
        taken_before = pre & 0xFFFF
        tied_before = pre >> 16
        # The tied keys left out, the lowest-index ones (none unless the threshold's bin
        # holds more keys than are still needed).
        skip = cutlass.max(at_thr - need, 0)
        for c in cutlass.range_constexpr(KPT):
            e = e0 + c
            sel = cutlass.Int32(0)
            pos = cutlass.Int32(0)
            if tied[c] == 1:
                if tied_before >= skip:
                    sel = cutlass.Int32(1)
                    pos = taken_before + tied_before - skip
                tied_before = tied_before + 1
            if take[c] == 1:
                sel = cutlass.Int32(1)
                pos = taken_before + cutlass.max(tied_before - skip, 0)
                taken_before = taken_before + 1
            if sel == 1:
                mTopk[row, pos] = e
                for j in cutlass.range_constexpr(cr):
                    tok = e * cr + j
                    val = cutlass.Int32(-1)
                    if tok < seqlen:
                        val = tok
                    mSel[row, pos * cr + j] = val

        # 5. Padding of the block list; the causal tail after the selected tokens, then -1.
        total = n
        if n > k:
            total = cutlass.Int32(k)
        for c in cutlass.range_constexpr(-(-k // nt)):
            s = tidx + c * nt
            if s < k:
                if s >= total:
                    mTopk[row, s] = cutlass.Int32(-1)
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


def _cache_slabs(width: int, threads: int, ctas: int) -> int:
    """Slabs of the wide kernel's window cache: as many of the table's as the shared memory
    left after the fixed state and the candidate list hold with `ctas` CTAs an SM (11 of
    4096 columns at one CTA of 1024 threads: a window of up to 45056 blocks, 180224 tokens,
    is read from global once)."""
    slab = KPT * threads
    per_cta = min(_SMEM_LIMIT, _SMEM_SM // ctas - _SMEM_RESERVED)
    room = per_cta - (_WIDE_FIXED_WORDS + 2 * slab) * 4 - _WIDE_PAD
    return max(0, min(-(-width // slab), room // (slab * 4)))


class _TopkExpandWideKernel:
    """`_TopkExpandKernel`'s selection over a table wider than one CTA's registers. The row's
    visible window (`row_ends`, not the table width) is walked in slabs, `KPT` consecutive
    keys a thread a slab; its first `cache_slabs` slabs are read from global once (cp.async
    into shared memory) and every walk reads them from there, a later slab from global (L2).

    Prefilter: every lane's largest key is a distinct key of the window, so with `per_warp`
    the `k / warps` rounded up, the smallest over the warps of the `per_warp`-th largest lane
    maximum of each warp (`lo`) has at least `k` keys at or above it. The `k`-th largest key,
    every key tied with it and everything above are among those, so the keys below `lo` are
    no candidates and the selection is unchanged.

    Two walks of the window: the lane maxima, then the candidates (keys at or above `lo`)
    compacted in block order into a shared list of up to `KPT` a thread (a window of ~21k
    relu-sum scores keeps about a thousand). The narrow kernel's radix passes, pairwise rank
    of the threshold's bin and block-order scan then run once over that list, `KPT`
    consecutive entries a thread. A window with more candidates than the list holds (a
    flat or degenerate row) takes the same steps over the whole window instead, slab by
    slab, the threshold bin's `need` best kept as a cut (the worst of them, key and block)
    and the block-order scan's counts carried from slab to slab."""

    def __init__(
        self,
        *,
        width,
        key_stride,
        block_topk,
        compress,
        token_topk,
        pdl,
        threads=_WIDE_THREADS,
        ctas=1,
        prefilter=True,
    ):
        self.width, self.key_stride, self.k, self.cr, self.token_topk, self.pdl = (
            width,
            key_stride,
            block_topk,
            compress,
            token_topk,
            pdl,
        )
        assert threads % 32 == 0 and _SCAN_THREADS <= threads <= 1024, threads
        self.final = token_topk + compress - 1
        self.threads = threads
        self.warps = threads // 32
        self.slab = KPT * threads
        self.ctas = ctas
        self.cache_slabs = _cache_slabs(width, threads, ctas)
        self.per_warp = -(-block_topk // self.warps)
        self.prefilter = bool(prefilter) and self.per_warp <= 32

    def smem_bytes(self) -> int:
        return (self.cache_slabs * self.slab + 2 * self.slab + _WIDE_FIXED_WORDS) * 4 + _WIDE_PAD

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
        g2s = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int32, num_bits_per_copy=32 * KPT)
        self.kernel(mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys, g2s).launch(
            grid=[rows, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=self.ctas,
        )

    @cute.jit
    def _slab(self, keys, gK, sK, row, s, tidx, n, lo, ld_keys: cute.CopyAtom):
        """Slab `s`'s `KPT` keys of thread `tidx` into `keys`: a negative (-0.0) as zero, -1
        past the window and below `lo`."""
        col4 = s * self.threads + tidx
        e0 = col4 * KPT
        if const_expr(self.cache_slabs > 0):
            if s < self.cache_slabs:
                cute.autovec_copy(sK[(None, col4)], keys)
            else:
                keys.fill(-1)
                if e0 < n:
                    cute.copy(ld_keys, gK[row, (None, col4)], keys)
        else:
            keys.fill(-1)
            if e0 < n:
                cute.copy(ld_keys, gK[row, (None, col4)], keys)
        for c in cutlass.range_constexpr(KPT):
            b = cutlass.max(keys[c], cutlass.Int32(0))
            if (b < lo) | (e0 + c >= n):
                b = cutlass.Int32(-1)
            keys[c] = b

    @cute.jit
    def _passes(self, keys, tidx, warp, lane, hist, wsum, state):
        """The narrow kernel's radix passes over `keys` (this thread's share; -1 is none):
        from the state's prefix, mask and shift, 11 bits a pass, until the threshold's bin
        holds at most CAP keys or no bits are left."""
        for _p in cutlass.range_constexpr(_PASSES):
            if state[4] == 0:
                prefix = state[0]
                mask = state[1]
                need = state[2]
                shift = state[6]
                for c in cutlass.range_constexpr(KPT):
                    if keys[c] >= 0:
                        if (keys[c] & mask) == prefix:
                            digit = (keys[c] >> shift) & (_HIST - 1)
                            cute.arch.atomic_add(
                                hist.iterator + digit, cutlass.Int32(1), scope="cta"
                            )
                cute.arch.sync_threads()
                self._narrow(tidx, warp, lane, hist, wsum, state, prefix, mask, need, shift)

    @cute.jit
    def _narrow(self, tidx, warp, lane, hist, wsum, state, prefix, mask, need, shift):
        """One pass's histogram read (and cleared) from the top bin down: the bin holding
        the `need`-th key becomes the threshold's."""
        h = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Int32)
        h.fill(0)
        own = cutlass.Int32(0)
        if tidx < _SCAN_THREADS:
            for j in cutlass.range_constexpr(4):
                h[j] = hist[_HIST - 1 - 4 * tidx - j]
                hist[_HIST - 1 - 4 * tidx - j] = cutlass.Int32(0)
                own = own + h[j]
        incl = own
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        above = cutlass.Int32(0)
        if lane < warp:
            above = wsum[lane]
        before = cute.arch.warp_reduction_sum(above) + incl - own
        if tidx < _SCAN_THREADS:
            if before < need:
                if need <= before + own:
                    run = before
                    for j in cutlass.range_constexpr(4):
                        if run < need:
                            if need <= run + h[j]:
                                state[0] = prefix | ((_HIST - 1 - 4 * tidx - j) << shift)
                                state[1] = mask | ((_HIST - 1) << shift)
                                state[2] = need - run
                                state[3] = h[j]
                                state[4] = cutlass.Int32(
                                    cutlass.select_((h[j] <= CAP) | (shift == 0), 1, 0)
                                )
                                state[6] = cutlass.max(shift - 11, 0)
                        run = run + h[j]
        cute.arch.sync_threads()

    @cute.jit
    def _cut(self, tidx, state, cand_key, cand_idx, at_thr, need):
        """The threshold bin's `at_thr` <= CAP keys (gathered into `cand_key`/`cand_idx`)
        ranked pairwise, ties to the higher block: the `need`-th best is the cut, kept in
        `state[7]` (key) and `state[8]` (block)."""
        ci = tidx // _RANK_LANES
        sub = tidx % _RANK_LANES
        rank = cutlass.Int32(0)
        kt = cutlass.Int32(0)
        it = cutlass.Int32(0)
        if ci < at_thr:
            kt = cand_key[ci]
            it = cand_idx[ci]
            for j in cutlass.range(sub, at_thr, _RANK_LANES, unroll=4):
                kj = cand_key[j]
                ij = cand_idx[j]
                better = (kj > kt) | ((kj == kt) & (ij > it))
                rank = rank + cutlass.Int32(cutlass.select_(better, 1, 0))
        for off in cutlass.range_constexpr(_RANK_LANES.bit_length() - 1):
            rank = rank + cute.arch.shuffle_sync_bfly(rank, offset=1 << off)
        if ci < at_thr:
            if sub == 0:
                if rank == need - 1:
                    state[7] = kt
                    state[8] = it
        cute.arch.sync_threads()

    @cute.jit
    def _mark(self, keys, idx, take, tied, everything, prefix, mask, at_thr, cut_key, cut_idx):
        """Each of this thread's keys taken (above the threshold's bin, or in it at or above
        the cut) or tied (in it, when the passes ran out of bits with more than CAP keys);
        `everything` takes every key. Returns the packed counts, taken + (tied << 16)."""
        own = cutlass.Int32(0)
        for c in cutlass.range_constexpr(KPT):
            t_c = cutlass.Int32(0)
            q_c = cutlass.Int32(0)
            if keys[c] >= 0:
                if everything:
                    t_c = cutlass.Int32(1)
                else:
                    mk = keys[c] & mask
                    if mk > prefix:
                        t_c = cutlass.Int32(1)
                    if mk == prefix:
                        if at_thr > CAP:
                            q_c = cutlass.Int32(1)
                        elif (keys[c] > cut_key) | ((keys[c] == cut_key) & (idx[c] >= cut_idx)):
                            t_c = cutlass.Int32(1)
            take[c] = t_c
            tied[c] = q_c
            own = own + t_c + (q_c << 16)
        return own

    @cute.jit
    def _emit(self, row, idx, take, tied, taken_before, tied_before, skip, seqlen, mTopk, mSel):
        """This thread's selected blocks at their block-order positions, and their tokens
        below the sequence length (-1 past it). `tied_before - skip` of the tied keys before
        these are taken (the lowest-index `skip` of the tie class are left)."""
        for c in cutlass.range_constexpr(KPT):
            e = idx[c]
            sel = cutlass.Int32(0)
            pos = cutlass.Int32(0)
            if tied[c] == 1:
                if tied_before >= skip:
                    sel = cutlass.Int32(1)
                    pos = taken_before + tied_before - skip
                tied_before = tied_before + 1
            if take[c] == 1:
                sel = cutlass.Int32(1)
                pos = taken_before + cutlass.max(tied_before - skip, 0)
                taken_before = taken_before + 1
            if sel == 1:
                mTopk[row, pos] = e
                for j in cutlass.range_constexpr(self.cr):
                    tok = e * self.cr + j
                    val = cutlass.Int32(-1)
                    if tok < seqlen:
                        val = tok
                    mSel[row, pos * self.cr + j] = val

    @cute.jit
    def _select_listed(
        self,
        ckey,
        cidx,
        keys,
        row,
        tidx,
        warp,
        lane,
        m,
        seqlen,
        hist,
        cand_key,
        cand_idx,
        wsum,
        state,
        mTopk,
        mSel,
    ):
        """The narrow kernel's selection over the candidate list: `KPT` consecutive entries
        a thread, so one block-wide scan puts them in block order."""
        idx = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        take = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        tied = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        cute.autovec_copy(ckey[(None, tidx)], keys)
        cute.autovec_copy(cidx[(None, tidx)], idx)
        for c in cutlass.range_constexpr(KPT):
            if tidx * KPT + c >= m:
                keys[c] = cutlass.Int32(-1)
        self._passes(keys, tidx, warp, lane, hist, wsum, state)
        prefix = state[0]
        mask = state[1]
        need = state[2]
        at_thr = state[3]
        if m > self.k:
            if at_thr <= CAP:
                for c in cutlass.range_constexpr(KPT):
                    if keys[c] >= 0:
                        if (keys[c] & mask) == prefix:
                            slot = cute.arch.atomic_add(
                                state.iterator + 5, cutlass.Int32(1), scope="cta"
                            )
                            cand_key[slot] = keys[c]
                            cand_idx[slot] = idx[c]
                cute.arch.sync_threads()
                self._cut(tidx, state, cand_key, cand_idx, at_thr, need)
        skip = cutlass.max(at_thr - need, 0)
        own = self._mark(
            keys, idx, take, tied, m <= self.k, prefix, mask, at_thr, state[7], state[8]
        )
        incl = own
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        above = cutlass.Int32(0)
        if lane < warp:
            above = wsum[lane]
        pre = cute.arch.warp_reduction_sum(above) + incl - own
        self._emit(row, idx, take, tied, pre & 0xFFFF, pre >> 16, skip, seqlen, mTopk, mSel)

    @cute.jit
    def _select_window(
        self,
        keys,
        gK,
        sK,
        row,
        tidx,
        warp,
        lane,
        n,
        nslabs,
        lo,
        m,
        seqlen,
        hist,
        cand_key,
        cand_idx,
        wsum,
        wsc,
        state,
        mTopk,
        mSel,
        ld_keys: cute.CopyAtom,
    ):
        """The same selection when the list cannot hold every candidate: each step walks
        the window slab by slab, the block-order scan's counts carried from slab to slab (a
        slab's fit 16 bits each; the warp totals alternate between two halves of `wsc`)."""
        nt = self.threads
        nw = self.warps
        for _p in cutlass.range_constexpr(_PASSES):
            if state[4] == 0:
                prefix = state[0]
                mask = state[1]
                need = state[2]
                shift = state[6]
                for s in cutlass.range(nslabs):
                    self._slab(keys, gK, sK, row, s, tidx, n, lo, ld_keys)
                    for c in cutlass.range_constexpr(KPT):
                        if keys[c] >= 0:
                            if (keys[c] & mask) == prefix:
                                digit = (keys[c] >> shift) & (_HIST - 1)
                                cute.arch.atomic_add(
                                    hist.iterator + digit, cutlass.Int32(1), scope="cta"
                                )
                cute.arch.sync_threads()
                self._narrow(tidx, warp, lane, hist, wsum, state, prefix, mask, need, shift)
        prefix = state[0]
        mask = state[1]
        need = state[2]
        at_thr = state[3]
        if m > self.k:
            if at_thr <= CAP:
                for s in cutlass.range(nslabs):
                    self._slab(keys, gK, sK, row, s, tidx, n, lo, ld_keys)
                    e0 = (s * nt + tidx) * KPT
                    for c in cutlass.range_constexpr(KPT):
                        if keys[c] >= 0:
                            if (keys[c] & mask) == prefix:
                                slot = cute.arch.atomic_add(
                                    state.iterator + 5, cutlass.Int32(1), scope="cta"
                                )
                                cand_key[slot] = keys[c]
                                cand_idx[slot] = e0 + c
                cute.arch.sync_threads()
                self._cut(tidx, state, cand_key, cand_idx, at_thr, need)
        skip = cutlass.max(at_thr - need, 0)
        cut_key = state[7]
        cut_idx = state[8]
        idx = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        take = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        tied = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        carry_taken = cutlass.Int32(0)
        carry_tied = cutlass.Int32(0)
        for s in cutlass.range(nslabs):
            self._slab(keys, gK, sK, row, s, tidx, n, lo, ld_keys)
            e0 = (s * nt + tidx) * KPT
            for c in cutlass.range_constexpr(KPT):
                idx[c] = e0 + c
            own = self._mark(
                keys, idx, take, tied, m <= self.k, prefix, mask, at_thr, cut_key, cut_idx
            )
            incl = own
            for off in (1, 2, 4, 8, 16):
                up = cute.arch.shuffle_sync_up(incl, offset=off)
                if lane >= off:
                    incl = incl + up
            half = (s & 1) * 32
            if lane == 31:
                wsc[half + warp] = incl
            cute.arch.sync_threads()
            above = cutlass.Int32(0)
            if lane < warp:
                above = wsc[half + lane]
            pre = cute.arch.warp_reduction_sum(above) + incl - own
            slab_total = cutlass.Int32(0)
            if lane < nw:
                slab_total = wsc[half + lane]
            slab_total = cute.arch.warp_reduction_sum(slab_total)
            self._emit(
                row,
                idx,
                take,
                tied,
                carry_taken + (pre & 0xFFFF),
                carry_tied + (pre >> 16),
                skip,
                seqlen,
                mTopk,
                mSel,
            )
            carry_taken = carry_taken + (slab_total & 0xFFFF)
            carry_tied = carry_tied + (slab_total >> 16)

    @cute.kernel
    def kernel(
        self, mKeys, mEnd, mPos, mLen, mTopk, mSel, ld_keys: cute.CopyAtom, g2s: cute.CopyAtom
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        nt: cutlass.Constexpr = self.threads
        nw: cutlass.Constexpr = self.warps
        slab: cutlass.Constexpr = self.slab
        k: cutlass.Constexpr = self.k
        cr: cutlass.Constexpr = self.cr
        smem = cutlass.utils.SmemAllocator()
        sK = None
        if const_expr(self.cache_slabs > 0):
            # `KPT` consecutive keys (mode 0) of each thread-slab column (mode 1).
            sK = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout((KPT, self.cache_slabs * slab // KPT), stride=(1, KPT)),
                byte_alignment=128,
            )
        hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((_HIST,), stride=(1,)), byte_alignment=128
        )
        cand_key = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAP,), stride=(1,)), byte_alignment=128
        )
        cand_idx = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((CAP,), stride=(1,)), byte_alignment=128
        )
        wa = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        wb = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        wc = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), byte_alignment=128
        )
        wsc = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((64,), stride=(1,)), byte_alignment=128
        )
        state = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((16,), stride=(1,)), byte_alignment=128
        )
        # The candidates in block order: keys and blocks, `KPT` consecutive a thread.
        ckey = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((KPT, nt), stride=(1, KPT)), byte_alignment=128
        )
        cidx = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((KPT, nt), stride=(1, KPT)), byte_alignment=128
        )

        # 0. Shared state and the row's host-filled metadata, before the wait.
        for b in cutlass.range_constexpr(-(-_HIST // nt)):
            if tidx + b * nt < _HIST:
                hist[tidx + b * nt] = cutlass.Int32(0)
        if tidx == 0:
            state[5] = cutlass.Int32(0)  # candidates compacted
        n = mEnd[row]
        seqlen = mLen[row]
        visible = cutlass.Int32(mPos[row]) + 1
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # 1. The window's first slabs into shared memory, every load in flight at once.
        gK = cute.logical_divide(mKeys, (None, KPT))
        nslabs = (n + (slab - 1)) // slab
        if const_expr(self.cache_slabs > 0):
            ncached = cutlass.min(nslabs, cutlass.Int32(self.cache_slabs))
            for s in cutlass.range(ncached):
                col4 = s * nt + tidx
                if col4 * KPT < n:
                    cute.copy(g2s, gK[row, (None, col4)], sK[(None, col4)])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 2. The window's largest key, and the prefilter's bound `lo`: each warp's
        # `per_warp`-th largest lane maximum (ranked by every lane against all 32, ties to
        # the lower lane, so exactly one lane holds each rank), then their minimum.
        keys = cute.make_rmem_tensor(cute.make_layout((KPT,), stride=(1,)), cutlass.Int32)
        tmax = cutlass.Int32(-1)
        for s in cutlass.range(nslabs):
            self._slab(keys, gK, sK, row, s, tidx, n, cutlass.Int32(0), ld_keys)
            for c in cutlass.range_constexpr(KPT):
                tmax = cutlass.max(tmax, keys[c])
        kmax = cute.arch.warp_reduction_max(tmax)
        if lane == 0:
            wa[warp] = kmax
        if const_expr(self.prefilter):
            ahead = cutlass.Int32(0)
            for i in cutlass.range_constexpr(32):
                v = cute.arch.shuffle_sync(tmax, offset=i)
                if (v > tmax) | ((v == tmax) & (lane > i)):
                    ahead = ahead + 1
            if ahead == self.per_warp - 1:
                wc[warp] = tmax
        cute.arch.sync_threads()
        gmax = cutlass.Int32(-1)
        if lane < nw:
            gmax = wa[lane]
        gmax = cute.arch.warp_reduction_max(gmax)
        # Keys below `lo` are no candidates; all are when the window holds at most k.
        lo = cutlass.Int32(0)
        if const_expr(self.prefilter):
            if n > k:
                bound = cutlass.Int32(_I32_MAX)
                if lane < nw:
                    bound = wc[lane]
                bound = cute.arch.warp_reduction(bound, lambda x, y: cutlass.min(x, y))
                lo = cutlass.max(bound, cutlass.Int32(0))

        # 3. The candidates, compacted in block order: a scan of each slab's counts (warp
        # totals alternating between two halves of `wsc`, one barrier a slab), carried
        # from slab to slab. `m` counts them all, kept or not.
        cflat_key = cute.make_tensor(ckey.iterator, cute.make_layout((KPT * nt,), stride=(1,)))
        cflat_idx = cute.make_tensor(cidx.iterator, cute.make_layout((KPT * nt,), stride=(1,)))
        carry = cutlass.Int32(0)
        for s in cutlass.range(nslabs):
            self._slab(keys, gK, sK, row, s, tidx, n, lo, ld_keys)
            e0 = (s * nt + tidx) * KPT
            cnt = cutlass.Int32(0)
            for c in cutlass.range_constexpr(KPT):
                if keys[c] >= 0:
                    cnt = cnt + 1
            incl = cnt
            for off in (1, 2, 4, 8, 16):
                up = cute.arch.shuffle_sync_up(incl, offset=off)
                if lane >= off:
                    incl = incl + up
            half = (s & 1) * 32
            if lane == 31:
                wsc[half + warp] = incl
            cute.arch.sync_threads()
            above = cutlass.Int32(0)
            if lane < warp:
                above = wsc[half + lane]
            pos = carry + cute.arch.warp_reduction_sum(above) + incl - cnt
            slab_total = cutlass.Int32(0)
            if lane < nw:
                slab_total = wsc[half + lane]
            slab_total = cute.arch.warp_reduction_sum(slab_total)
            for c in cutlass.range_constexpr(KPT):
                if keys[c] >= 0:
                    if pos < KPT * nt:
                        cflat_key[pos] = keys[c]
                        cflat_idx[pos] = e0 + c
                    pos = pos + 1
            carry = carry + slab_total
        m = carry
        cute.arch.sync_threads()

        # 4. The narrow kernel's selection over the list, or over the window when the list
        # cannot hold every candidate.
        diff = gmax ^ lo
        hb = cutlass.Int32(-1)
        if diff != 0:
            hb = 31 - cutlass.Int32(cute.arch.clz(diff))
        low = (cutlass.Int64(1) << cutlass.Int64(hb + 1)) - 1
        mask0 = cutlass.Int32(cutlass.Int64(_I32_MAX) & (cutlass.Int64(-1) ^ low))
        if tidx == 0:
            state[0] = gmax & mask0  # threshold prefix
            state[1] = mask0  # its mask
            state[2] = cutlass.Int32(k)  # still needed at the threshold
            state[3] = m  # keys in the threshold's bin
            state[4] = cutlass.Int32(1)  # narrowing done
            state[6] = cutlass.max(hb - 10, 0)  # the next pass's shift
            if m > k:
                if hb >= 0:
                    state[4] = cutlass.Int32(0)
        cute.arch.sync_threads()
        if m <= KPT * nt:
            self._select_listed(
                ckey,
                cidx,
                keys,
                row,
                tidx,
                warp,
                lane,
                m,
                seqlen,
                hist,
                cand_key,
                cand_idx,
                wb,
                state,
                mTopk,
                mSel,
            )
        else:
            self._select_window(
                keys,
                gK,
                sK,
                row,
                tidx,
                warp,
                lane,
                n,
                nslabs,
                lo,
                m,
                seqlen,
                hist,
                cand_key,
                cand_idx,
                wa,
                wsc,
                state,
                mTopk,
                mSel,
                ld_keys,
            )

        # 5. Padding of the block list; the causal tail after the selected tokens, then -1.
        total = n
        if n > k:
            total = cutlass.Int32(k)
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


#: One compiled launch per score-table geometry, k, compress ratio, token budget, index
#: dtypes, PDL and wide-kernel configuration; the row count is a runtime argument.
_COMPILED: dict = {}


#: SMs of each device, read once.
_SMS: dict = {}


def _wide_config(rows: int, device) -> tuple:
    """The wide kernel's `(threads, CTAs an SM, prefilter)` for a call of `rows` rows: one
    1024-thread CTA an SM (the largest window cache) while the rows fit one wave, two once
    they do not, so one CTA's walks overlap the other's loads (GB300, 65536 columns: 8160
    rows of ~30k-block windows 790 -> 640 us; 4 and 16 decode rows 16.5 -> 18.4 and 24.6 ->
    28.7 us the other way)."""
    sms = _SMS.get(device.index)
    if sms is None:
        sms = _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return (_WIDE_THREADS, 1 if rows <= sms else 2, True)


_INDEX_TYPES = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


def launch(
    scores,
    row_ends,
    query_positions,
    sequence_lengths,
    topk_indices,
    compress_ratio,
    token_topk,
    pdl=True,
    out=None,
):
    """The launch behind `QsaTopkExpandDecode.__call__`: writes `topk_indices` and returns the
    token selection `[rows, token_topk + compress_ratio - 1]` int32 (`out` when given)."""
    rows, width = scores.shape
    block_topk = topk_indices.shape[1]
    sel = out
    if sel is None:
        sel = torch.empty(
            (rows, token_topk + compress_ratio - 1), dtype=torch.int32, device=scores.device
        )
    if rows == 0:
        return sel
    pdl = bool(pdl and get_env_enable_pdl())
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
        None if width <= MAX_WIDTH else _wide_config(rows, scores.device),
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
        geometry = dict(
            width=width,
            key_stride=scores.stride(0),
            block_topk=block_topk,
            compress=compress_ratio,
            token_topk=token_topk,
            pdl=pdl,
        )
        if width <= MAX_WIDTH:
            kernel = _TopkExpandKernel(**geometry)
        else:
            threads, ctas, prefilter = key[-1]
            kernel = _TopkExpandWideKernel(
                **geometry, threads=threads, ctas=ctas, prefilter=prefilter
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


def supports(scores) -> bool:
    """Whether a score table this wide is one this entry serves (see `is_valid`)."""
    return scores.shape[1] % KPT == 0 and scores.shape[1] <= MAX_WIDE_WIDTH


#: The unified target's score-table widths, `block_table.shape[1] * 32 // 4`: core's `max_seq_len`
#: 9216 (288 pages) and MTP's 16384 (512 pages, `MAX_WIDTH` itself), served by the narrow
#: kernel; the tune's `max_seq_len` 262144 with MTP's extra pages (8196 pages, 65568 columns)
#: and without (65536), served by the wide one.
_CORE_WIDTH = 2304
_MTP_WIDTH = 4096
_TUNE_WIDTH = 65568
_TUNE_WIDTH_NO_MTP = 65536


def _spread(edges: tuple, count: int, first: int, last: int) -> tuple:
    """`edges`, then `count - len(edges)` values spread evenly over `[first, last]`."""
    fill = count - len(edges)
    return tuple(edges) + tuple(first + (r * (last - first)) // (fill - 1) for r in range(fill))


#: 128 one-row decode requests at core's width. The edges are windows of 0 and 1 blocks,
#: k - 1, k and k + 1 blocks (512, the `n <= k` copy-through and the first narrowing),
#: and the full 2304-block window.
_CORE_DECODE_128 = _spread(
    (1, 2, 3, 6, 7, 2046, 2047, 2048, 2051, 2052, 9212, 9214, 9215), 128, 8, 9215
)
#: MTP7 verify at MTP's width: 128 requests x 8 consecutive rows, one starting where its
#: rows cross k blocks and one ending at the full 4096-block window.
_MTP7_VERIFY_1024 = tuple((start, 8) for start in _spread((1, 2041, 2044, 16376), 128, 9, 16375))
#: Decode rows at the tune's 65536-wide table, by window: 0 and 1 blocks, k - 1, k and k + 1,
#: one slab (4096 columns) - 1, itself and + 1, the 1024-thread CTA's window cache (11 slabs,
#: 45056 blocks) - 1, itself and + 1, the whole table, and four between.
_WIDE_EDGES_16 = (
    2,
    3,
    2043,
    2047,
    2051,
    16380,
    16384,
    16388,
    180220,
    180224,
    180228,
    262143,
    50000,
    99999,
    150001,
    220000,
)


class QsaTopkExpandDecode(OpWrapper):
    """`qsa_topk_select` (zero row starts, `row_ends` the visible block counts),
    `qsa_canonical_selection` and `qsa_expand_block_indices`, in one launch: one CTA a row, for
    decode rows and a context chunk's rows alike.

    Per row: the `block_topk` largest of `scores[row, :row_ends[row]]` (non-negative fp32),
    ties at the threshold to the higher block, written to `topk_indices[row]` in block order
    and padded with -1 -- the composition's selection after `qsa_canonical_selection`; the
    returned token selection is the expansion entry's: each selected group's members below
    `sequence_lengths[row]`, packed at the front, then the incomplete group at the causal
    frontier (`query_positions[row] + 1` visible tokens), then -1. `topk_indices` is written
    in place, and the selection into `out` when given (a contiguous int32
    `[rows, token_topk + compress_ratio - 1]`, e.g. a row range of the step's whole
    selection); nothing else is written.

    Fusion boundary: the selection and the expansion. Scoring is the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `width` is the score table's column count (and row stride): a compile key, and the
    # thread count (`_threads`: 640 at 2560, 576 at 2304, 1024 at 4096).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, 4 rows across the budget, one below it",
            spec=dict(rows=4, pos=(8700, 4095, 1500, 9900), width=2560),
        ),
        # The target writes a context chunk's rows through `out` into its rows of the
        # step's selection; relu-sparse scores put the zero class (negative zeros among it)
        # across the budget, the ties the canonical rule decides.
        Cell(
            why="context chunk: 2048 rows of one request (positions 6144..8191), relu-sparse "
            "scores, written through `out` into rows of a step-sized selection",
            spec=dict(rows=2048, chunk=(6144, 2048), relu_sparse=True, out_margin=64, width=2560),
        ),
        Cell(
            why="decode, 1 row of scores within 5% of each other",
            spec=dict(rows=1, pos=(8700,), narrow=True, width=2560),
        ),
        # Serving's decode rows at 8K: positive, log-normally spread over ~2 octaves, a
        # zero or two; the first pass leaves tens to ~CAP candidates at the threshold.
        Cell(
            why="decode, 16 rows of log-normal scores with a few zeros: threshold bins of "
            "up to CAP candidates ranked pairwise, ties included",
            spec=dict(
                rows=16, pos=tuple(8192 + 37 * r for r in range(16)), lognormal=True, width=2560
            ),
        ),
        # The unified target's call sites, at its own widths (`_CORE_WIDTH`, `_MTP_WIDTH`).
        Cell(
            why="core decode, 1 row at the 2304-wide table, its full window, relu-sparse",
            spec=dict(rows=1, pos=(9215,), relu_sparse=True, width=_CORE_WIDTH),
        ),
        Cell(
            why="core decode batch, 128 rows at the 2304-wide table (max_batch_size): windows "
            "of 0, 1, k-1, k, k+1 and 2304 blocks and 115 spread between, relu-sparse",
            spec=dict(rows=128, pos=_CORE_DECODE_128, relu_sparse=True, width=_CORE_WIDTH),
        ),
        Cell(
            why="MTP7 verify, 1024 rows (128 requests x 8) at the 4096-wide table, up to its "
            "full window, relu-sparse",
            spec=dict(rows=1024, requests=_MTP7_VERIFY_1024, relu_sparse=True, width=_MTP_WIDTH),
        ),
        Cell(
            why="core context chunk, 2048 rows (positions 6144..8191) at the 2304-wide table, "
            "relu-sparse, written through `out`",
            spec=dict(
                rows=2048, chunk=(6144, 2048), relu_sparse=True, out_margin=64, width=_CORE_WIDTH
            ),
        ),
        Cell(
            why="envelope's largest row: an 8192-row chunk (max_num_tokens, positions "
            "8192..16383) at the 4096-wide table, relu-sparse, written through `out`",
            spec=dict(
                rows=8192, chunk=(8192, 8192), relu_sparse=True, out_margin=64, width=_MTP_WIDTH
            ),
        ),
        # The tune's tables (`max_seq_len` 262144): 65568 columns with MTP's extra pages,
        # 65536 without, past `MAX_WIDTH`, so the wide kernel's. Relu-sum scores are the
        # indexer's form (four heads' ReLUs summed): the prefilter keeps ~5% of a window.
        Cell(
            why="wide: MTP3 verify, 4 rows (positions 84000..84003) at the 65568-wide table, "
            "relu-sum scores",
            spec=dict(rows=4, requests=((84000, 4),), relu_sum=True, width=_TUNE_WIDTH, graph=True),
        ),
        Cell(
            why="wide: 16 decode rows at the 65536-wide table, windows at the edges of k, a "
            "slab, the window cache and the table, relu-sum scores",
            spec=dict(
                rows=16, pos=_WIDE_EDGES_16, relu_sum=True, width=_TUNE_WIDTH_NO_MTP, graph=True
            ),
        ),
        Cell(
            why="wide: 4 rows of log-normal scores at 65568: threshold bins ranked pairwise, "
            "ties included",
            spec=dict(rows=4, pos=(100000, 30000, 4095, 2047), lognormal=True, width=_TUNE_WIDTH),
        ),
        Cell(
            why="wide: 2 rows of relu-sparse scores at 65536 (~1.4% positive): the prefilter "
            "keeps the zeros, more candidates than the list holds, the whole-window path; at "
            "30000 blocks the zero class (negative zeros among it) straddles the budget",
            spec=dict(rows=2, pos=(262143, 120000), relu_sparse=True, width=_TUNE_WIDTH_NO_MTP),
        ),
        Cell(
            why="wide: 3 rows of equal scores at 65568 (windows 25000, 513 and 512 blocks): "
            "the tie class is the window, the highest blocks taken",
            spec=dict(rows=3, pos=(100000, 2051, 2047), equal=True, width=_TUNE_WIDTH),
        ),
        Cell(
            why="wide: ht_dep4's context chunk, 8160 rows (positions 57000..65159) at the "
            "65536-wide table, relu-sum, written through `out`",
            spec=dict(
                rows=8160,
                chunk=(57000, 8160),
                relu_sum=True,
                out_margin=64,
                width=_TUNE_WIDTH_NO_MTP,
            ),
        ),
    )

    note = """
    Metadata consumed: `qsa_visible_blocks` (`row_ends`), the rows' logical positions
    and sequence lengths. Preconditions: zero row starts; `scores` fp32 `[rows, W]`, unit
    column stride, 16-byte aligned rows, `W % 4 == 0` and `W <= 131072` (up to 4096 the
    narrow kernel, four keys a thread and at most 1024 threads; wider tables the wide
    kernel, `_TopkExpandWideKernel`); `topk_indices` int32 `[rows, K]` contiguous; `row_ends` int32,
    the positions and lengths int32 or int64, each a contiguous `[rows]`. Ties: the higher
    block at the threshold, as `qsa_canonical_selection` rewrites the CUDA top-k's pick.

    Compiled once per score-table width and row stride (the block table's, fixed for an
    engine), k, compress ratio, token budget, index dtypes and PDL: the row count is a
    runtime argument, so an eager step's context rows, any count, reuse the kernel (read
    from the launch, not pinned by a test).

    JIT-compiled: `cute.compile` (`--enable-tvm-ffi`) on a key's first call, for the device
    that makes it. The cells ran that path on sm_103 (GB300) at widths 2560, 2304 and 4096.

    Certified, every cell bitwise against `reference` (block list and token selection):
    - tp1's cells at 2560 columns: decode 1 row of near-equal scores, 4 rows, 16 rows of
      log-normal scores (threshold bins ranked pairwise), and a 2048-row context chunk.
    - The unified target's cells at its own widths (`_CORE_WIDTH` 2304, `_MTP_WIDTH` 4096), with
      relu-sparse scores: the zero class, negative zeros among it, straddles the 512-block
      budget and is resolved by the tie rule. They are:
      - decode, 1 and 128 rows at 2304, with windows of 0, 1, 511, 512, 513 and 2304 blocks among them;
      - MTP7 verify, 1024 rows (128 requests x 8) at 4096;
      - a 2048-row context chunk at 2304, through `out`;
      - an 8192-row chunk (`max_num_tokens`) at 4096, through `out`.
    - The tune's tables (`max_seq_len` 262144: 65568 columns with MTP's pages, 65536
      without), the wide kernel: a 4-row MTP3 verify at ~84k tokens and 16 decode rows at
      the edges of k, a 4096-column slab, the 45056-block window cache and the whole table
      (relu-sum scores); 4 rows of log-normal scores (threshold bins ranked pairwise); 2
      relu-sparse rows (more candidates than the list holds: the whole-window path, the
      zero class straddling the budget); 3 rows of equal scores; and ht_dep4's 8160-row
      context chunk through `out` (two CTAs an SM past one wave of rows).
    - A CUDA-graph replay at 1x2560, 1x2304, 128x2304, 1024x4096, 4x65568, 16x65536 and
      2x65536. The scores are refilled before each replay, and the selection, read after
      the wait, must equal the reference.
    The widths are derived from `max_seq_len` (288, 512 and 8192 (+4) pages of 32 tokens),
    not read off an engine. 4096 is `MAX_WIDTH`: 1024 threads, the largest CTA; wider
    tables take the wide kernel, compiled per row-count class (one CTA an SM up to the
    SM count, two past it).

    `is_valid` refuses `token_topk != k * compress_ratio`. It sizes the selection's rows
    while `k * compress_ratio` places the tokens, so a smaller budget writes past each row.
    Measured unguarded, with `out` inside a larger buffer: the rows after it were overwritten.

    Not certified:
    - threshold bins ranked pairwise at 2304 and 4096 columns (the log-normal cell is at 2560);
    - decode batches other than 1, 4, 16 and 128 rows;
    - a NaN inside a window (it ranks above every finite score: the caller keeps the window finite);
    - index dtypes other than int64 positions and int32 lengths.

    Index limits, read from the source, not probed. The row count is an Int32 argument and
    the row layouts are built from it, so these products must stay below 2^31:
    - rows x row stride: 524288 rows at 4096 columns, 16383 at 131072 (`MAX_WIDE_WIDTH`);
    - rows x (token_topk + compress_ratio - 1): about 1.05 M rows at 2051.
    Positions are read as Int32. The envelope's largest call is 8192 rows.
    """

    def __call__(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        return launch(
            scores,
            row_ends,
            query_positions,
            sequence_lengths,
            topk_indices,
            compress_ratio,
            token_topk,
            out=out,
        )

    def reference(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        """`(topk_indices_after, selection)`: a stable descending sort per row (ties to the
        higher block), the k first in block order, then the expansion entry's reference."""
        rows, _ = scores.shape
        k = topk_indices.shape[1]
        out = torch.full_like(topk_indices, -1)
        for r in range(rows):
            n = int(row_ends[r])
            s = scores[r, :n].float().clamp(min=0.0)
            # A stable sort of the reversed row puts the higher block first among equals.
            order = (n - 1) - torch.sort(torch.flip(s, [0]), descending=True, stable=True).indices[
                :k
            ]
            chosen = torch.sort(order).values
            out[r, : chosen.numel()] = chosen.to(torch.int32)
        sel = _expand.reference(
            out,
            query_positions,
            sequence_lengths,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
        )
        return out, sel

    def is_valid(
        self,
        scores,
        row_ends,
        query_positions,
        sequence_lengths,
        topk_indices,
        compress_ratio,
        token_topk,
        out=None,
    ):
        assert scores.dtype == torch.float32 and scores.stride(1) == 1, (
            "fp32 scores, unit column stride"
        )
        assert scores.shape[1] % KPT == 0 and scores.shape[1] <= MAX_WIDE_WIDTH, (
            "W % 4 == 0, W <= 131072"
        )
        assert scores.stride(0) % KPT == 0 and scores.data_ptr() % 16 == 0, "16-byte aligned rows"
        assert topk_indices.dtype == torch.int32 and topk_indices.is_contiguous(), (
            "int32 topk_indices"
        )
        rows = scores.shape[0]
        assert row_ends.dtype == torch.int32 and row_ends.shape == (rows,), "int32 row_ends [rows]"
        for t in (query_positions, sequence_lengths):
            assert t.dtype in _INDEX_TYPES and t.shape == (rows,), "int32/int64 [rows]"
        assert all(t.is_contiguous() for t in (row_ends, query_positions, sequence_lengths)), (
            "contiguous row_ends, positions and lengths"
        )
        assert token_topk == topk_indices.shape[1] * compress_ratio, (
            "token_topk = k * compress_ratio"
        )
        assert out is None or (
            out.dtype == torch.int32
            and out.is_contiguous()
            and out.shape == (scores.shape[0], token_topk + compress_ratio - 1)
        ), "out: contiguous int32 [rows, token_topk + compress_ratio - 1]"


qsa_topk_expand_decode = QsaTopkExpandDecode()
