# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA's block top-k and its expansion to token indices, one launch per batch of rows.

Tier 5 (CuTe DSL written for this target). The decode path ran `qsa_topk_select`
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
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .qsa_expand_block_indices import qsa_expand_block_indices as _expand

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
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # 1. The row's metadata and four consecutive keys a thread (a column past the
        # window is no candidate: -1); the largest and smallest key in the window.
        n = mEnd[row]
        seqlen = mLen[row]
        visible = cutlass.Int32(mPos[row]) + 1
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
                if tidx < at_thr:
                    kt = cand_key[tidx]
                    it = cand_idx[tidx]
                    rank = cutlass.Int32(0)
                    for j in cutlass.range(0, at_thr, 1, unroll=8):
                        kj = cand_key[j]
                        ij = cand_idx[j]
                        better = (kj > kt) | ((kj == kt) & (ij > it))
                        rank = rank + cutlass.Int32(cutlass.select_(better, 1, 0))
                    if rank < need:
                        cute.arch.atomic_or(
                            flags.iterator + (it >> 5), cutlass.Int32(1) << (it & 31), scope="cta"
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


#: One compiled launch per score-table geometry, k, compress ratio, token budget, index
#: dtypes and PDL; the row count is a runtime argument.
_COMPILED: dict = {}
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
        fn = _COMPILED[key] = cute.compile(
            _TopkExpandKernel(
                width=width,
                key_stride=scores.stride(0),
                block_topk=block_topk,
                compress=compress_ratio,
                token_topk=token_topk,
                pdl=pdl,
            ),
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
    return scores.shape[1] % KPT == 0 and scores.shape[1] <= MAX_WIDTH


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

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, 4 rows across the budget, one below it",
            spec=dict(rows=4, pos=(8700, 4095, 1500, 9900)),
        ),
        # The target writes a context chunk's rows through `out` into its rows of the
        # step's selection; relu-sparse scores put the zero class (negative zeros among it)
        # across the budget, the ties the canonical rule decides.
        Cell(
            why="context chunk: 2048 rows of one request (positions 6144..8191), relu-sparse "
            "scores, written through `out` into rows of a step-sized selection",
            spec=dict(rows=2048, chunk=(6144, 2048), relu_sparse=True, out_margin=64),
        ),
        Cell(
            why="decode, 1 row of scores within 5% of each other",
            spec=dict(rows=1, pos=(8700,), narrow=True),
        ),
    )

    note = """
    Metadata consumed: `qsa_visible_blocks` (`row_ends`), the rows' logical positions
    and sequence lengths. Preconditions: zero row starts; `scores` fp32 `[rows, W]`, unit
    column stride, 16-byte aligned rows, `W % 4 == 0` and `W <= 4096` (four keys a thread,
    at most 1024 threads); `topk_indices` int32 `[rows, K]` contiguous; `row_ends` int32,
    the positions and lengths int32 or int64, each a contiguous `[rows]`. Ties: the higher
    block at the threshold, as `qsa_canonical_selection` rewrites the CUDA top-k's pick.

    Compiled once per score-table width and row stride (the block table's, fixed for an
    engine), k, compress ratio, token budget, index dtypes and PDL: the row count is a
    runtime argument, so an eager step's context rows, any count, reuse the kernel (read
    from the launch, not pinned by a test).

    Certified: decode batches of 1 and 4 rows over a 2560-wide score table, a 2048-row
    context chunk written through `out`, and a replay under a CUDA graph that reads the
    scores after its wait.
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
        assert scores.shape[1] % KPT == 0 and scores.shape[1] <= MAX_WIDTH, "W % 4 == 0, W <= 4096"
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
