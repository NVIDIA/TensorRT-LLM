# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL top-32 of the DFlash2 drafter logits, in ``torch.topk``'s exact output order.

The selector's candidates are each block slot's global top-32 over the 163,840-token vocabulary.
Under vocab tensor parallelism every rank takes the top-32 of its shard, the ranks all-gather the
interleaved ``(global id, value)`` pairs, and every rank takes the top-32 of the gathered values.
Both selections are ``torch.topk(x, 32)`` in the torch path, and the walk is sensitive to the
order of the candidates (its inverse CDF runs over them in order), so these kernels return the
candidates in torch's order, ties included:

* the set: the 32 largest under torch's radix order of the values (``-0 < +0``, every NaN above
  ``+inf``), the lowest index first among values equal to the 32nd;
* gathered (``gatherTopK``): the values strictly above the 32nd in index order, then the values
  equal to it in index order;
* sorted by the 32-element bitonic network of ``sortKeyValueInplace`` (16 threads, two elements
  each, ``GTOp`` comparisons: ``-0 == +0``, NaN above everything, ties swapped by the network);
  except where torch 2.14 sorts the gathered values stably instead (a contiguous row of exactly
  256 FP32 values and at least 152 rows; see :func:`torch_topk_sorts_stably`; no DFlash2 call
  is such a shape), which is the order of ``(radix key descending, index ascending)``.

One CTA of 256 threads selects among up to 8,192 elements. Logits are read coalesced, staged in
shared memory and taken back as one contiguous run a thread (radix keys in registers, BF16 keyed
two values to a word). The 32nd largest of the threads' maxima is a floor (32 threads hold a key at
least that large); the few keys at or above it are compacted in index order into shared memory and
ranked against each other (two barriers); past 1,024 of them (ties at the floor) a radix select
over the registers takes over. The 32 selected land in shared memory in the gathered order, where
one warp sorts them with the network. A wider row is split into chunks: each chunk's top-32 (by
the same rule, in index order) goes to a scratch row, and a second launch radix-selects from the
chunks' candidates.

``lamport`` mode (vocab-parallel pairs only) makes the exchange of the pairs over the Lamport
all-gather, which carries the word ``0x80000000`` (FP32 ``-0.0``, its arrival sentinel) as ``+0.0``,
lossless, and reproduces the gathered-logits path whose BF16 vocab gather is the Lamport kernel:
the selection keys a BF16 word ``0x80000000`` (an even ``+0.0`` and an odd ``-0.0``, which that
gather delivers as two ``+0.0``) as two ``+0.0``, a selected ``-0.0`` travels as ``+0.0`` with its
id word ``-(id + 1)`` (ids are non-negative, so the sign is free), and the merge restores it.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TOP_K = 32
THREADS = 256
#: Elements one CTA selects from: 32 per thread.
CHUNK = THREADS * 32
#: Survivors of the thread-maximum floor compacted into shared memory, four a thread.
CAND_PER_THREAD = 4
CAP = THREADS * CAND_PER_THREAD
#: CTAs an SM keeps resident for BF16 logits (2,048 threads, 32 registers each; half that for
#: FP32): the chunks are latency-bound.
MIN_BLOCKS = 8
_HIST = 256

#: Source and output layouts.
SRC_BITS16 = "bits16"  # BF16 logits, read as int16
SRC_BITS32 = "bits32"  # FP32 logits, read as int32
SRC_SCRATCH = "scratch"  # chunk candidates: (key, index, bits) int32 rows
SRC_PAIRS = "pairs"  # all-gathered interleaved (id, value) FP32 pairs, read as int32
OUT_VALUES_IDS = "values_ids"
OUT_PAIRS = "pairs"
OUT_SCRATCH = "scratch"
#: Interleaved FP32 ``(global id, value)`` pairs of the selection in index order (unsorted): the
#: operand of a vocab-parallel exchange whose merge reproduces the full-row selection's order.
OUT_PAIRS_INDEX = "pairs_index"


def torch_topk_sorts_stably(rows: int, width: int, dtype: torch.dtype, contiguous: bool) -> bool:
    """Whether ``torch.topk(x, 32)`` of ``[rows, width]`` returns ties in index order rather than
    in its bitonic network's order. Measured on torch 2.14: a contiguous FP32 input of width
    exactly 256 from 152 rows on; every other FP32 / BF16 width and row count of the DFlash2 cells,
    and a strided input of any shape (the value column of the all-gathered pairs), sort by the
    network."""
    return contiguous and dtype == torch.float32 and width == 256 and rows >= 152


def _i32(value: int) -> int:
    """A 32-bit pattern as the signed int32 the DSL takes."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _network_stages():
    """``(size, stride)`` of the 32-element bitonic network, in order; size 32 is the final
    merge (direction ``false`` for every pair)."""
    stages = []
    size = 2
    while size <= TOP_K:
        stride = size // 2
        while stride > 0:
            stages.append((size, stride))
            stride //= 2
        size *= 2
    return tuple(stages)


_STAGES = _network_stages()


@dsl_user_op
def _max_u16x2(a, b, *, loc=None, ip=None):
    """Lane-wise maximum of two pairs of unsigned 16-bit halves."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.u16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return Int32(llvm.bitcast(T.i32(), cutlass.Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


class Top32Kernel:
    """One CTA per (row, chunk): the chunk's top-32 in torch's order (``emit`` final) or in index
    order (``emit`` scratch).

    Args:
        src: the source layout (``SRC_*``).
        key16: the values are BF16 (16-bit radix keys) rather than FP32.
        per_thread: elements per thread (a run of consecutive elements).
        out: ``OUT_VALUES_IDS`` (values in the source dtype's bits, int64 ids),
            ``OUT_PAIRS`` (interleaved FP32 ``(id, value)``), ``OUT_SCRATCH`` or
            ``OUT_PAIRS_INDEX`` (the pairs in index order, one chunk).
        stable: sort the selected values stably (value descending, index ascending) instead of
            by the network.
        lamport: the Lamport exchange mode (module doc): BF16 logits key a word ``0x80000000`` as
            two ``+0.0``, index-ordered pairs carry a ``-0.0`` value as ``(-(id + 1), +0.0)`` and
            gathered pairs decode it.
    """

    def __init__(
        self, src: str, key16: bool, per_thread: int, out: str, stable: bool, lamport: bool = False
    ):
        self.src = src
        self.lamport = lamport
        self.key16 = key16
        self.per_thread = per_thread
        self.out = out
        self.stable = stable
        self.passes = 2 if key16 else 4
        self.logits_src = src in (SRC_BITS16, SRC_BITS32)
        # BF16 logits are read as 32-bit words of two values and keyed two at a time.
        self.packed = src == SRC_BITS16
        # FP32 keys take 32 registers a thread; packed BF16 ones 16.
        self.min_blocks = MIN_BLOCKS if self.packed else MIN_BLOCKS // 2
        # Narrow the field by the threads' maxima where a thread holds many elements.
        self.prefilter = self.logits_src and per_thread >= 8
        self.key_bits = 16 if key16 else 32
        # Unsigned radix keys live in int32 registers; XOR with FLIP orders them as signed.
        self.flip = 0 if key16 else _i32(1 << 31)
        # Radix keys of -0 and +0 (equal under the network's comparison).
        self.neg_zero = 0x7FFF if key16 else 0x7FFFFFFF
        self.pos_zero = 0x8000 if key16 else _i32(1 << 31)
        if src in (SRC_BITS16, SRC_BITS32):
            self.vec = 8 if src == SRC_BITS16 else 4
            if per_thread % self.vec:
                raise ValueError(f"{per_thread} elements per thread: not whole 16-byte vectors")

    @cute.jit
    def __call__(
        self,
        mSrc: cute.Tensor,  # [rows, width] bits; [rows, 2 m] pairs; [rows, m] scratch keys
        mSrcIdx: cute.Tensor,  # [rows, m] scratch indices (SRC_SCRATCH)
        mSrcBits: cute.Tensor,  # [rows, m] scratch value bits (SRC_SCRATCH)
        mOutA: cute.Tensor,  # values [rows, 32] | pairs [rows, 64] | scratch keys [rows, c * 32]
        mOutB: cute.Tensor,  # ids [rows, 32] int64 | - | scratch indices [rows, c * 32]
        mOutC: cute.Tensor,  # - | - | scratch value bits [rows, c * 32]
        width: Int32,  # elements per row of the source
        id_offset: Int32,  # added to the ids written (a shard's first global id)
        stream,
    ):
        rows = mOutA.shape[0]
        chunks = 1
        if cutlass.const_expr(self.out == OUT_SCRATCH):
            chunks = mOutA.shape[1] // TOP_K
        self.kernel(mSrc, mSrcIdx, mSrcBits, mOutA, mOutB, mOutC, width, id_offset).launch(
            grid=[chunks, rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.jit
    def skey(self, ukey):
        """Signed-comparable form of an unsigned radix key."""
        return ukey ^ Int32(self.flip)

    @cute.jit
    def net_key(self, ukey):
        """The network's comparison key: the radix key with -0 folded onto +0, signed form."""
        k = ukey
        if ukey == Int32(self.neg_zero):
            k = Int32(self.pos_zero)
        return k ^ Int32(self.flip)

    @cute.jit
    def reload_bits(self, mSrc: cute.Tensor, row, col):
        """The value bits of a logit, read again (the selected ones only)."""
        bits = Int32(0)
        if cutlass.const_expr(self.packed):
            bits = (self.canon_word(Int32(mSrc[row, col >> 1])) >> ((col & 1) * 16)) & 0xFFFF
        else:
            bits = Int32(mSrc[row, col])
        return bits

    @cute.jit
    def canon_word(self, w):
        """A BF16 word as the Lamport vocab gather delivers it: ``0x80000000`` (``+0.0`` then
        ``-0.0``) as ``0``; every other word unchanged (``lamport`` only)."""
        out = w
        if cutlass.const_expr(self.lamport):
            out = cutlass.select_(w == Int32(_i32(1 << 31)), Int32(0), w)
        return out

    @cute.jit
    def radix_key(self, bits):
        """torch's TopKTypeConfig radix key of the value bits (NaN -> all ones), branch-free."""
        key = Int32(0)
        if cutlass.const_expr(self.key16):
            negative = bits >> 15  # 0 / 1 (bits are zero-extended)
            key = bits ^ (Int32(0x8000) | ((Int32(0) - negative) & 0x7FFF))
            key = cutlass.select_((bits & 0x7FFF) > 0x7F80, Int32(0xFFFF), key)
        else:
            key = bits ^ (Int32(_i32(1 << 31)) | (bits >> 31))
            key = cutlass.select_((bits & 0x7FFFFFFF) > 0x7F800000, Int32(-1), key)
        return key

    @cute.jit
    def packed_key(self, w):
        """The radix keys of the two BF16 values of a 32-bit word (low half: the lower index)."""
        sign = (w >> 15) & 0x00010001
        key = w ^ (Int32(_i32(0x80008000)) | (sign * 0x7FFF))
        nan = (((w & 0x7FFF7FFF) + 0x007F007F) >> 15) & 0x00010001
        return key | (nan * 0xFFFF)

    @cute.jit
    def radix_select(
        self,
        keys: cute.Tensor,
        count: cutlass.Constexpr[int],
        aggregate: cutlass.Constexpr[bool],
        hist: cute.Tensor,
        state: cute.Tensor,
        warp: Int32,
        lane: Int32,
        live: Int32,
        key_bits: cutlass.Constexpr[int] = 0,
    ):
        """The 32nd largest of the block's keys into ``state`` (prefix, mask, equal ones still
        needed), 8 bits a pass from the top. With ``aggregate`` every thread holds one key
        (0 counts as the smallest); otherwise keys of 0 (no element) are skipped and a thread
        with ``live == 0`` holds none. ``state`` starts at ``(0, 0, 32)``
        and ``hist`` at zeros; ``hist`` is zero again on return."""
        bits: cutlass.Constexpr = key_bits if key_bits else self.key_bits
        for ps in cutlass.range_constexpr(bits // 8):
            shift: cutlass.Constexpr = bits - 8 * (ps + 1)
            prefix = state[0]
            pmask = state[1]
            if cutlass.const_expr(aggregate):
                # One key a thread, every lane present: one atomic per distinct bin of the warp
                # (the threads' maxima mostly share their high bits, and 32 lanes adding to one
                # address serialize).
                k = keys[0]
                digit = Int32(_HIST)
                if (k & pmask) == prefix:
                    digit = (k >> shift) & (_HIST - 1)
                peers = Int32(cute.arch.match_sync(-1, digit, "any"))
                if digit != _HIST:
                    if (peers & ((Int32(1) << lane) - 1)) == 0:
                        cute.arch.atomic_add(
                            hist.iterator + digit, cute.arch.popc(peers), scope="cta"
                        )
            else:
                if live != 0:
                    for j in cutlass.range_constexpr(count):
                        k = keys[j]
                        if k != 0:
                            if (k & pmask) == prefix:
                                cute.arch.atomic_add(
                                    hist.iterator + ((k >> shift) & (_HIST - 1)),
                                    Int32(1),
                                    scope="cta",
                                )
            cute.arch.sync_threads()
            if warp == 0:
                # Lane l owns bins 255 - 8 l down to 248 - 8 l.
                h = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Int32)
                own = Int32(0)
                for j in cutlass.range_constexpr(8):
                    h[j] = hist[_HIST - 1 - 8 * lane - j]
                    hist[_HIST - 1 - 8 * lane - j] = Int32(0)
                    own = own + h[j]
                incl = own
                for off in (1, 2, 4, 8, 16):
                    up = cute.arch.shuffle_sync_up(incl, offset=off)
                    if lane >= off:
                        incl = incl + up
                before = incl - own
                need = state[2]
                if before < need:
                    if need <= incl:
                        run = before
                        for j in cutlass.range_constexpr(8):
                            if run < need:
                                if need <= run + h[j]:
                                    chosen = _HIST - 1 - 8 * lane - j
                                    state[0] = prefix | (chosen << shift)
                                    state[1] = pmask | Int32(_i32((_HIST - 1) << shift))
                                    state[2] = need - run
                            run = run + h[j]
            cute.arch.sync_threads()

    @cute.jit
    def elem_idx(self, idxs, j: cutlass.Constexpr[int], base, from_base: cutlass.Constexpr[bool]):
        """Element ``j``'s index: ``base + j`` for a thread's run of logits, else ``idxs[j]``."""
        gi = base + j
        if cutlass.const_expr(not from_base):
            gi = idxs[j]
        return gi

    @cute.jit
    def finish(
        self,
        keys: cute.Tensor,
        idxs: cute.Tensor,
        vbits: cute.Tensor,
        pays: cute.Tensor,
        count: cutlass.Constexpr[int],
        from_base: cutlass.Constexpr[bool],
        base: Int32,
        live: Int32,
        mSrc: cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
        mOutC: cute.Tensor,
        hist: cute.Tensor,
        wsum: cute.Tensor,
        state: cute.Tensor,
        s_key: cute.Tensor,
        s_idx: cute.Tensor,
        s_bits: cute.Tensor,
        s_pay: cute.Tensor,
        row: Int32,
        chunk: Int32,
        warp: Int32,
        lane: Int32,
        id_offset: Int32,
    ):
        """Select the top 32 of the block's elements (``count`` a thread, in index order across
        the block: thread by thread, then ``j``) and write them out. ``keys`` are radix keys (0:
        no element); indices are ``base + j`` (``from_base``) or ``idxs``; value bits and payloads
        come from ``vbits`` / ``pays`` for candidate rows and are re-read from ``mSrc`` for
        logits. ``state`` starts at ``(0, 0, 32)``."""
        self.radix_select(keys, count, False, hist, state, warp, lane, live)

        kth = state[0]
        need = state[2]  # values equal to the 32nd taken
        kth_s = self.skey(kth)

        # 3. One block scan of (above, equal) counts orders the selection.
        own = Int32(0)
        k = Int32(0)
        sel = Int32(0)
        slot = Int32(0)
        col = Int32(0)
        if live != 0:
            for j in cutlass.range_constexpr(count):
                k = keys[j]
                if k != 0:
                    if self.skey(k) > kth_s:
                        own = own + (1 << 16)
                    if k == kth:
                        own = own + 1
        live = own
        incl = own
        for off in (1, 2, 4, 8, 16):
            up = cute.arch.shuffle_sync_up(incl, offset=off)
            if lane >= off:
                incl = incl + up
        if lane == 31:
            wsum[warp] = incl
        cute.arch.sync_threads()
        above = Int32(0)
        if lane < warp:
            above = wsum[lane]
        pre = cute.arch.warp_reduction_sum(above) + incl - own
        gt_before = pre >> 16
        eq_before = pre & 0xFFFF
        gt_total = TOP_K - need

        if cutlass.const_expr(self.out in (OUT_SCRATCH, OUT_PAIRS_INDEX)):
            # Index order: every value above the 32nd and the first `need` equal to it.
            slot = gt_before + cutlass.min(eq_before, need)
            eq_rank = eq_before
            if live != 0:
                for j in cutlass.range_constexpr(count):
                    k = keys[j]
                    sel = Int32(0)
                    if k != 0:
                        if self.skey(k) > kth_s:
                            sel = Int32(1)
                        if k == kth:
                            if eq_rank < need:
                                sel = Int32(1)
                            eq_rank = eq_rank + 1
                    if sel != 0:
                        col = chunk * TOP_K + slot
                        if cutlass.const_expr(self.out == OUT_PAIRS_INDEX):
                            if cutlass.const_expr(self.logits_src):
                                gi = self.elem_idx(idxs, j, base, from_base)
                                self.write_index_pair(
                                    mOutA, row, slot, gi, self.reload_bits(mSrc, row, gi), id_offset
                                )
                            else:
                                self.write_index_pair(
                                    mOutA, row, slot, idxs[j], vbits[j], id_offset
                                )
                        else:
                            mOutA[row, col] = k
                            if cutlass.const_expr(self.logits_src):
                                mOutB[row, col] = self.elem_idx(idxs, j, base, from_base)
                                mOutC[row, col] = self.reload_bits(
                                    mSrc, row, self.elem_idx(idxs, j, base, from_base)
                                )
                            else:
                                mOutB[row, col] = idxs[j]
                                mOutC[row, col] = vbits[j]
                        slot = slot + 1
        else:
            # Gathered order: above the 32nd by index, then equal to it by index.
            gt_slot = gt_before
            eq_rank = eq_before
            if live != 0:
                for j in cutlass.range_constexpr(count):
                    k = keys[j]
                    slot = Int32(-1)
                    if k != 0:
                        if self.skey(k) > kth_s:
                            slot = gt_slot
                            gt_slot = gt_slot + 1
                        if k == kth:
                            if eq_rank < need:
                                slot = gt_total + eq_rank
                            eq_rank = eq_rank + 1
                    if slot >= 0:
                        s_key[slot] = k
                        if cutlass.const_expr(self.logits_src):
                            gi = self.elem_idx(idxs, j, base, from_base)
                            s_idx[slot] = gi
                            s_bits[slot] = self.reload_bits(mSrc, row, gi)
                            s_pay[slot] = gi
                        else:
                            s_idx[slot] = idxs[j]
                            s_bits[slot] = vbits[j]
                            s_pay[slot] = pays[j]
            cute.arch.sync_threads()

            self.sort_and_write(
                mOutA, mOutB, s_key, s_idx, s_bits, s_pay, row, warp, lane, id_offset
            )

    @cute.jit
    def write_index_pair(self, mOutA: cute.Tensor, row, slot, gi, vbits, id_offset: Int32):
        """Pair ``slot`` of the index-ordered output: ``(gi + id_offset)`` as FP32 and the value's
        FP32 bits (a BF16 value widened exactly)."""
        vb = vbits
        if cutlass.const_expr(self.key16):
            vb = vbits << 16
        gid = gi + id_offset
        if cutlass.const_expr(self.lamport):
            # -0.0 is the Lamport sentinel: send (-(id + 1), +0.0), which the merge decodes.
            if vb == Int32(_i32(1 << 31)):
                gid = Int32(-1) - gid
                vb = Int32(0)
        mOutA[row, 2 * slot] = _f32_bits(cutlass.Float32(gid))
        mOutA[row, 2 * slot + 1] = vb

    @cute.jit
    def sort_and_write(
        self,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
        s_key: cute.Tensor,
        s_idx: cute.Tensor,
        s_bits: cute.Tensor,
        s_pay: cute.Tensor,
        row: Int32,
        warp: Int32,
        lane: Int32,
        id_offset: Int32,
    ):
        """Warp 0 sorts the 32 selected elements, in the gathered order in ``s_*``, as torch
        does and writes the outputs."""
        if warp == 0:
            k = s_key[lane]
            i = s_idx[lane]
            b = s_bits[lane]
            py = s_pay[lane]
            nk = self.net_key(k)
            if cutlass.const_expr(self.stable):
                # Rank by (radix key descending, index ascending): +0 above -0, NaNs tied.
                rk = self.skey(k)
                rank = Int32(0)
                for j in cutlass.range_constexpr(TOP_K):
                    rj = cute.arch.shuffle_sync(rk, offset=j)
                    ij = cute.arch.shuffle_sync(i, offset=j)
                    if rj > rk:
                        rank = rank + 1
                    if rj == rk:
                        if ij < i:
                            rank = rank + 1
                cute.arch.sync_warp()
                s_key[rank] = nk
                s_idx[rank] = i
                s_bits[rank] = b
                s_pay[rank] = py
                cute.arch.sync_warp()
                nk = s_key[lane]
                i = s_idx[lane]
                b = s_bits[lane]
                py = s_pay[lane]
                if cutlass.const_expr(not self.key16):
                    # That sort writes every NaN as 0x7FFFFFFF.
                    if nk == Int32(_i32(0x7FFFFFFF)):
                        b = Int32(_i32(0x7FFFFFFF))
            else:
                # sortKeyValueInplace's 32-element bitonic network, descending (GTOp).
                for st in cutlass.range_constexpr(len(_STAGES)):
                    size, stride = _STAGES[st]
                    o_nk = cute.arch.shuffle_sync_bfly(nk, offset=stride)
                    o_i = cute.arch.shuffle_sync_bfly(i, offset=stride)
                    o_b = cute.arch.shuffle_sync_bfly(b, offset=stride)
                    o_py = cute.arch.shuffle_sync_bfly(py, offset=stride)
                    a_nk = o_nk
                    b_nk = nk
                    if (lane & stride) == 0:
                        a_nk = nk
                        b_nk = o_nk
                    # The network thread of this pair, and its direction.
                    pos = lane & (31 ^ stride)
                    t = (pos & (stride - 1)) | ((pos >> 1) & (31 ^ (stride - 1)))
                    flag = Int32(0)
                    if cutlass.const_expr(size < TOP_K):
                        if (t & (size // 2)) != 0:
                            flag = Int32(1)
                    gt = Int32(0)
                    if a_nk > b_nk:
                        gt = Int32(1)
                    if gt == flag:
                        nk = o_nk
                        i = o_i
                        b = o_b
                        py = o_py
            if cutlass.const_expr(self.out == OUT_VALUES_IDS):
                if cutlass.const_expr(self.key16):
                    mOutA[row, lane] = cutlass.Int16(b)
                else:
                    mOutA[row, lane] = b
                gid = Int64(0)
                if cutlass.const_expr(self.src == SRC_PAIRS):
                    gid = Int64(_bits_f32(py))
                else:
                    gid = Int64(py + id_offset)
                mOutB[row, lane] = gid
            else:
                vb = b
                if cutlass.const_expr(self.key16):
                    vb = b << 16
                mOutA[row, 2 * lane] = _f32_bits(cutlass.Float32(py + id_offset))
                mOutA[row, 2 * lane + 1] = vb

    @cute.jit
    def finish_ranked(
        self,
        c_key: cute.Tensor,
        c_idx: cute.Tensor,
        total: Int32,
        mSrc: cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
        mOutC: cute.Tensor,
        state: cute.Tensor,
        s_key: cute.Tensor,
        s_idx: cute.Tensor,
        s_bits: cute.Tensor,
        s_pay: cute.Tensor,
        row: Int32,
        chunk: Int32,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        id_offset: Int32,
    ):
        """The top 32 of ``total`` (32..CAP) logit candidates in shared memory, in index order:
        each candidate's rank under (key descending, position ascending) from one pass over the
        others, the 32nd's key broadcast, then each selected one's slot (the gathered order, or
        index order for scratch) from a second pass. Two barriers instead of a radix select's
        two a pass."""
        for r in cutlass.range_constexpr(CAND_PER_THREAD):
            p = tidx + r * THREADS
            if p < total:
                kp = self.skey(c_key[p])
                rank = Int32(0)
                for q in cutlass.range(0, total, 1, unroll=4):
                    kq = self.skey(c_key[q])
                    rank = rank + cutlass.select_(
                        (kq > kp) | ((kq == kp) & (q < p)), Int32(1), Int32(0)
                    )
                if rank == TOP_K - 1:
                    state[0] = c_key[p]
        cute.arch.sync_threads()
        kth = state[0]
        kth_s = self.skey(kth)
        for r in cutlass.range_constexpr(CAND_PER_THREAD):
            p = tidx + r * THREADS
            if p < total:
                kp = c_key[p]
                kp_s = self.skey(kp)
                if kp_s >= kth_s:
                    gt_before = Int32(0)
                    eq_before = Int32(0)
                    gt_all = Int32(0)
                    for q in cutlass.range(0, total, 1, unroll=4):
                        kq_s = self.skey(c_key[q])
                        before = q < p
                        gt_all = gt_all + cutlass.select_(kq_s > kth_s, Int32(1), Int32(0))
                        gt_before = gt_before + cutlass.select_(
                            (kq_s > kth_s) & before, Int32(1), Int32(0)
                        )
                        eq_before = eq_before + cutlass.select_(
                            (kq_s == kth_s) & before, Int32(1), Int32(0)
                        )
                    need = TOP_K - gt_all
                    slot = Int32(-1)
                    if cutlass.const_expr(self.out in (OUT_SCRATCH, OUT_PAIRS_INDEX)):
                        if (kp_s > kth_s) | (eq_before < need):
                            slot = gt_before + cutlass.min(eq_before, need)
                    else:
                        if kp_s > kth_s:
                            slot = gt_before
                        else:
                            if eq_before < need:
                                slot = gt_all + eq_before
                    if slot >= 0:
                        gi = c_idx[p]
                        if cutlass.const_expr(self.out == OUT_PAIRS_INDEX):
                            self.write_index_pair(
                                mOutA, row, slot, gi, self.reload_bits(mSrc, row, gi), id_offset
                            )
                        elif cutlass.const_expr(self.out == OUT_SCRATCH):
                            col = chunk * TOP_K + slot
                            mOutA[row, col] = kp
                            mOutB[row, col] = gi
                            mOutC[row, col] = self.reload_bits(mSrc, row, gi)
                        else:
                            s_key[slot] = kp
                            s_idx[slot] = gi
                            s_bits[slot] = self.reload_bits(mSrc, row, gi)
                            s_pay[slot] = gi
        if cutlass.const_expr(self.out not in (OUT_SCRATCH, OUT_PAIRS_INDEX)):
            cute.arch.sync_threads()
            self.sort_and_write(
                mOutA, mOutB, s_key, s_idx, s_bits, s_pay, row, warp, lane, id_offset
            )

    @cute.kernel
    def kernel(
        self,
        mSrc: cute.Tensor,
        mSrcIdx: cute.Tensor,
        mSrcBits: cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
        mOutC: cute.Tensor,
        width: Int32,
        id_offset: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk, row, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        E: cutlass.Constexpr = self.per_thread

        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(Int32, cute.make_layout((_HIST,), stride=(1,)), 16)
        wsum = smem.allocate_tensor(Int32, cute.make_layout((32,), stride=(1,)), 16)
        state = smem.allocate_tensor(Int32, cute.make_layout((8,), stride=(1,)), 16)
        s_key = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        s_idx = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        s_bits = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        s_pay = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,), stride=(1,)), 16)
        if cutlass.const_expr(self.logits_src):
            # The staged chunk is consumed before the candidates are written: they share it.
            stage_slots: cutlass.Constexpr = THREADS * (E // (8 if self.packed else 4) + 1)
            stage = smem.allocate_tensor(
                Int32, cute.make_layout((4 * stage_slots,), stride=(1,)), 16
            )
            s_stage = cute.make_tensor(
                stage.iterator, cute.make_layout((4, stage_slots), stride=(1, 4))
            )
            c_key = cute.make_tensor(stage.iterator, cute.make_layout((CAP,), stride=(1,)))
            c_idx = cute.make_tensor(stage.iterator + CAP, cute.make_layout((CAP,), stride=(1,)))
        else:
            c_key = smem.allocate_tensor(Int32, cute.make_layout((CAP,), stride=(1,)), 16)
            c_idx = smem.allocate_tensor(Int32, cute.make_layout((CAP,), stride=(1,)), 16)

        hist[tidx] = Int32(0)
        if tidx == 0:
            state[0] = Int32(0)  # key prefix
            state[1] = Int32(0)  # prefix mask
            state[2] = Int32(TOP_K)  # still needed at or below the prefix

        # 1. This thread's run of elements: radix key (0: no element); from logits, the index
        # is the position and the value bits are re-read for the 32 selected only, which keeps
        # a thread's registers to its keys; from candidate rows, index, value bits and payload
        # (the id carried to the output) are held too.
        ukey = cute.make_rmem_tensor(cute.make_layout((E,), stride=(1,)), Int32)
        base = chunk * (THREADS * E) + tidx * E
        ukey2 = cute.make_rmem_tensor(cute.make_layout((max(E // 2, 1),), stride=(1,)), Int32)
        if cutlass.const_expr(self.logits_src):
            # 16-byte vectors: eight BF16 values (four words) or four FP32 ones. The chunk is
            # read coalesced (vector v * 256 + t by thread t) and staged in shared memory, a
            # 16-byte pad after each thread's run, from where every thread takes its own run.
            per_vec: cutlass.Constexpr = 8 if self.packed else 4
            nv: cutlass.Constexpr = E // per_vec
            gW = cute.logical_divide(mSrc, (None, 4))
            row_vecs = (width + per_vec - 1) // per_vec
            first = chunk * (THREADS * nv)
            staged = cute.make_rmem_tensor(cute.make_layout((4, nv), stride=(1, 4)), Int32)
            for v in cutlass.range_constexpr(nv):
                if first + v * THREADS + tidx < row_vecs:
                    cute.autovec_copy(gW[row, (None, first + v * THREADS + tidx)], staged[None, v])
                else:
                    for wj in cutlass.range_constexpr(4):
                        staged[wj, v] = Int32(0)
            for v in cutlass.range_constexpr(nv):
                g = v * THREADS + tidx
                cute.autovec_copy(staged[None, v], s_stage[None, (g // nv) * (nv + 1) + g % nv])
            cute.arch.sync_threads()
            words = cute.make_rmem_tensor(cute.make_layout((4, nv), stride=(1, 4)), Int32)
            for k in cutlass.range_constexpr(nv):
                cute.autovec_copy(s_stage[None, tidx * (nv + 1) + k], words[None, k])
            if cutlass.const_expr(self.packed):
                for i in cutlass.range_constexpr(E // 2):
                    # A zero word keys to 0x80008000: no element must stay 0.
                    ukey2[i] = cutlass.select_(
                        base + 2 * i < width,
                        self.packed_key(self.canon_word(words[i % 4, i // 4])),
                        Int32(0),
                    )
            else:
                for j in cutlass.range_constexpr(E):
                    ukey[j] = cutlass.select_(
                        base + j < width, self.radix_key(words[j % 4, j // 4]), Int32(0)
                    )
        else:
            idx = cute.make_rmem_tensor(cute.make_layout((E,), stride=(1,)), Int32)
            vbits = cute.make_rmem_tensor(cute.make_layout((E,), stride=(1,)), Int32)
            pay = cute.make_rmem_tensor(cute.make_layout((E,), stride=(1,)), Int32)
            for j in cutlass.range_constexpr(E):
                p = base + j
                ukey[j] = Int32(0)
                idx[j] = Int32(0)
                vbits[j] = Int32(0)
                pay[j] = Int32(0)
                if p < width:
                    if cutlass.const_expr(self.src == SRC_SCRATCH):
                        ukey[j] = Int32(mSrc[row, p])
                        idx[j] = Int32(mSrcIdx[row, p])
                        vbits[j] = Int32(mSrcBits[row, p])
                        pay[j] = idx[j]
                    else:
                        b = Int32(mSrc[row, 2 * p + 1])
                        py = Int32(mSrc[row, 2 * p])
                        if cutlass.const_expr(self.lamport):
                            # An id word below zero carries a -0.0 value: (-(id + 1), +0.0).
                            if py < 0:
                                py = _f32_bits(cutlass.Float32(Int32(-1) - Int32(_bits_f32(py))))
                                b = Int32(_i32(1 << 31))
                        ukey[j] = self.radix_key(b)
                        idx[j] = p
                        vbits[j] = b
                        pay[j] = py
        cute.arch.sync_threads()

        # 2. Radix select of the 32nd largest key. Logits first narrow the field: the 32nd
        # largest of the threads' maxima is at most the 32nd largest key (32 threads hold a key
        # at least that large), so keys below it never matter; a few dozen keys of the chunk are
        # left, and the threads holding none skip the passes.
        live = Int32(1)
        smask = Int32(0)
        floor1 = Int32(1)
        if cutlass.const_expr(self.prefilter):
            tkey = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), Int32)
            best = Int32(0)
            if cutlass.const_expr(self.packed):
                m2 = ukey2[0]
                for i in cutlass.range_constexpr(1, E // 2):
                    m2 = _max_u16x2(m2, ukey2[i])
                best = cutlass.max(m2 & 0xFFFF, (m2 >> 16) & 0xFFFF)
            else:
                # Keys of no element (0) are the smallest signed keys too.
                best_s = self.skey(ukey[0])
                for j in cutlass.range_constexpr(1, E):
                    best_s = cutlass.max(best_s, self.skey(ukey[j]))
                best = best_s ^ Int32(self.flip)
            floor_s = Int32(0)
            if cutlass.const_expr(self.packed):
                tkey[0] = best
                self.radix_select(tkey, 1, True, hist, state, warp, lane, live)
                floor_s = self.skey(state[0])
            else:
                # The top 16 bits of the FP32 keys suffice: truncating only lowers the floor.
                tkey[0] = (best >> 16) & 0xFFFF
                self.radix_select(tkey, 1, True, hist, state, warp, lane, live, 16)
                floor_s = self.skey(state[0] << 16)
            cute.arch.sync_threads()
            if tidx == 0:
                state[0] = Int32(0)
                state[1] = Int32(0)
                state[2] = Int32(TOP_K)
            live = Int32(0)
            if cutlass.const_expr(self.packed):
                # 16-bit keys compare unsigned; a key of 0 is no element.
                floor1 = cutlass.max(floor_s, Int32(1))
                for i in cutlass.range_constexpr(E // 2):
                    kw = ukey2[i]
                    smask = smask | cutlass.select_(
                        (kw & 0xFFFF) >= floor1, Int32(_i32(1 << (2 * i))), Int32(0)
                    )
                    smask = smask | cutlass.select_(
                        ((kw >> 16) & 0xFFFF) >= floor1, Int32(_i32(1 << (2 * i + 1))), Int32(0)
                    )
                live = smask
            else:
                # A key of 0 (no element) is the smallest signed key; valid ones are above it.
                floor1 = cutlass.max(floor_s, Int32(_i32((1 << 31) + 1)))
                for j in cutlass.range_constexpr(E):
                    smask = smask | cutlass.select_(
                        self.skey(ukey[j]) >= floor1, Int32(_i32(1 << j)), Int32(0)
                    )
                live = smask
            cute.arch.sync_threads()
        if cutlass.const_expr(self.prefilter):
            # The survivors (a few dozen unless the values tie) are compacted, in index order,
            # into shared memory, and the selection runs over them, four a thread; past CAP of
            # them (ties at the floor) it runs over the threads' runs in registers instead.
            mine = cute.arch.popc(smask)
            incl = mine
            for off in (1, 2, 4, 8, 16):
                up = cute.arch.shuffle_sync_up(incl, offset=off)
                if lane >= off:
                    incl = incl + up
            if lane == 31:
                wsum[warp] = incl
            cute.arch.sync_threads()
            above = Int32(0)
            total = Int32(0)
            if lane < THREADS // 32:
                total = wsum[lane]
                if lane < warp:
                    above = wsum[lane]
            off0 = cute.arch.warp_reduction_sum(above) + incl - mine
            total = cute.arch.warp_reduction_sum(total)
            cute.arch.sync_threads()
            w = Int32(0)
            p = Int32(0)
            m = smask
            kj = Int32(0)
            gi = Int32(0)
            if total <= CAP:
                if mine != 0:
                    w = off0
                    # One pass per survivor (its key is re-derived from the logit).
                    while m != 0:
                        gi = base + cute.arch.popc((m & (Int32(0) - m)) - 1)
                        c_key[w] = self.radix_key(self.reload_bits(mSrc, row, gi))
                        c_idx[w] = gi
                        w = w + 1
                        m = m & (m - 1)
                cute.arch.sync_threads()
                self.finish_ranked(
                    c_key,
                    c_idx,
                    total,
                    mSrc,
                    mOutA,
                    mOutB,
                    mOutC,
                    state,
                    s_key,
                    s_idx,
                    s_bits,
                    s_pay,
                    row,
                    chunk,
                    tidx,
                    warp,
                    lane,
                    id_offset,
                )
            else:
                for j in cutlass.range_constexpr(E):
                    if cutlass.const_expr(self.packed):
                        kj = (ukey2[j // 2] >> (16 * (j % 2))) & 0xFFFF
                        ukey[j] = cutlass.select_(kj >= floor1, kj, Int32(0))
                    else:
                        kj = ukey[j]
                        ukey[j] = cutlass.select_(self.skey(kj) >= floor1, kj, Int32(0))
                self.finish(
                    ukey,
                    ukey,
                    ukey,
                    ukey,
                    E,
                    True,
                    base,
                    live,
                    mSrc,
                    mOutA,
                    mOutB,
                    mOutC,
                    hist,
                    wsum,
                    state,
                    s_key,
                    s_idx,
                    s_bits,
                    s_pay,
                    row,
                    chunk,
                    warp,
                    lane,
                    id_offset,
                )
        else:
            if cutlass.const_expr(self.logits_src):
                self.finish(
                    ukey,
                    ukey,
                    ukey,
                    ukey,
                    E,
                    True,
                    base,
                    live,
                    mSrc,
                    mOutA,
                    mOutB,
                    mOutC,
                    hist,
                    wsum,
                    state,
                    s_key,
                    s_idx,
                    s_bits,
                    s_pay,
                    row,
                    chunk,
                    warp,
                    lane,
                    id_offset,
                )
            else:
                self.finish(
                    ukey,
                    idx,
                    vbits,
                    pay,
                    E,
                    False,
                    base,
                    live,
                    mSrc,
                    mOutA,
                    mOutB,
                    mOutC,
                    hist,
                    wsum,
                    state,
                    s_key,
                    s_idx,
                    s_bits,
                    s_pay,
                    row,
                    chunk,
                    warp,
                    lane,
                    id_offset,
                )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_OPTIONS = "--opt-level 2 --enable-tvm-ffi"
_TORCH_TO_CUTLASS = {torch.int16: cutlass.Int16, torch.int32: Int32, torch.int64: Int64}


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
    )


@functools.cache
def _compiled(src, key16, per_thread, out, stable, src_cols, out_a_dtype, out_cols, lamport=False):
    kernel = Top32Kernel(src, key16, per_thread, out, stable, lamport)
    rows = cute.sym_int()
    src_t = Int32
    return cute.compile(
        kernel,
        _row_major(src_t, (rows, src_cols), 16),
        _row_major(Int32, (rows, src_cols) if src == SRC_SCRATCH else (rows, 1)),
        _row_major(Int32, (rows, src_cols) if src == SRC_SCRATCH else (rows, 1)),
        _row_major(_TORCH_TO_CUTLASS[out_a_dtype], (rows, out_cols)),
        _row_major(
            Int64 if out == OUT_VALUES_IDS else Int32,
            (rows, out_cols if out not in (OUT_PAIRS, OUT_PAIRS_INDEX) else 1),
        ),
        _row_major(Int32, (rows, out_cols if out == OUT_SCRATCH else 1)),
        Int32(1),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


def _chunk_for(rows: int, width: int) -> int:
    """Elements a CTA of a split row takes."""
    return CHUNK


def _per_thread(width: int, vec: int) -> int:
    per = -(-width // THREADS)
    return -(-per // vec) * vec


def _bits_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def top32(
    logits: torch.Tensor,
    *,
    id_offset: int = 0,
    pairs: bool = False,
    stable: bool = None,
    chunk: int = None,
    index_order: bool = False,
    lamport: bool = False,
):
    """``torch.topk(logits, 32, dim=-1)`` of contiguous BF16 / FP32 ``[rows, width]`` logits, in
    torch's order; ids get ``id_offset`` added.

    Returns ``(values [rows, 32] in the logits dtype, ids [rows, 32] int64)``, or with ``pairs``
    the interleaved FP32 ``[rows, 64]`` ``(id, value)`` of the vocab-parallel exchange.
    ``stable`` overrides :func:`torch_topk_sorts_stably`. With ``index_order`` (pairs only) the
    same 32 pairs come in index order, unsorted: every rank's index-ordered pairs, all-gathered in
    rank order, are the full row's selection in index order, which :func:`merge_top32` takes into
    exactly the order of ``top32`` over the full row. ``lamport`` (index-ordered pairs only) is the
    Lamport exchange mode of the module doc, for :func:`merge_top32` with ``lamport``.
    """
    if index_order and not pairs:
        raise ValueError("index_order applies to the pairs output")
    if lamport and not index_order:
        raise ValueError("lamport applies to the index-ordered pairs")
    rows, width = logits.shape
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"top32 takes BF16 or FP32 logits, got {logits.dtype}")
    if width < TOP_K or width % 8:
        raise ValueError(f"top32 needs a width >= 32 and a multiple of 8, got {width}")
    key16 = logits.dtype == torch.bfloat16
    if stable is None:
        stable = torch_topk_sorts_stably(rows, width, logits.dtype, contiguous=True)
    device = logits.device
    if pairs:
        out_a = torch.empty((rows, 2 * TOP_K), dtype=torch.float32, device=device)
        out_b = torch.empty((rows, 1), dtype=torch.int32, device=device)
        result = (out_a,)
        out_kind = OUT_PAIRS_INDEX if index_order else OUT_PAIRS
        out_a_bits, out_cols = out_a.view(torch.int32), 2 * TOP_K
        if index_order:
            stable = False  # nothing is sorted
    else:
        values = torch.empty((rows, TOP_K), dtype=logits.dtype, device=device)
        out_b = torch.empty((rows, TOP_K), dtype=torch.int64, device=device)
        result = (values, out_b)
        out_kind, out_a_bits, out_cols = OUT_VALUES_IDS, _bits_view(values), TOP_K
    if rows == 0:
        return result if not pairs else result[0]
    src = SRC_BITS16 if key16 else SRC_BITS32
    vec = 8 if key16 else 4
    # BF16 logits go in as 32-bit words (two values each).
    bits = logits.view(torch.int32)
    src_cols = width // 2 if key16 else width
    dummy = torch.empty((rows, 1), dtype=torch.int32, device=device)
    out_c = torch.empty((rows, 1), dtype=torch.int32, device=device)
    chunk = _chunk_for(rows, width) if chunk is None else chunk
    if width <= CHUNK:
        _compiled(
            src,
            key16,
            _per_thread(width, vec),
            out_kind,
            stable,
            src_cols,
            out_a_bits.dtype,
            out_cols,
            lamport,
        )(bits, dummy, dummy, out_a_bits, out_b, out_c, width, id_offset)
    else:
        chunks = -(-width // chunk)
        if width - (chunks - 1) * chunk < TOP_K:
            raise ValueError(f"width {width}: the last {chunk}-element chunk holds fewer than 32")
        m = chunks * TOP_K
        s_key = torch.empty((rows, m), dtype=torch.int32, device=device)
        s_idx = torch.empty_like(s_key)
        s_bits = torch.empty_like(s_key)
        _compiled(
            src, key16, chunk // THREADS, OUT_SCRATCH, False, src_cols, torch.int32, m, lamport
        )(bits, dummy, dummy, s_key, s_idx, s_bits, width, 0)
        _compiled(
            SRC_SCRATCH,
            key16,
            -(-m // THREADS),
            out_kind,
            stable,
            m,
            out_a_bits.dtype,
            out_cols,
            lamport,
        )(s_key, s_idx, s_bits, out_a_bits, out_b, out_c, m, id_offset)
    return result if not pairs else result[0]


def merge_top32(gathered: torch.Tensor, stable: bool = None, lamport: bool = False):
    """The global top-32 of all-gathered ``(id, value)`` FP32 pairs ``[rows, 2 m]``: ``torch.topk``
    of the values ``gathered[:, 1::2]`` in torch's order, and the ids at the selected positions.
    With ``lamport`` a pair ``(-(id + 1), +0.0)`` is first read as ``(id, -0.0)`` (the Lamport
    exchange mode of the module doc).

    Returns ``(values [rows, 32] FP32, ids [rows, 32] int64)``.
    """
    rows, cols = gathered.shape
    m = cols // 2
    if m < TOP_K or m > CHUNK:
        raise ValueError(f"merge_top32 takes 32..{CHUNK} pairs per row, got {m}")
    if stable is None:
        # The torch path selects from the strided view gathered[..., 1::2].
        stable = torch_topk_sorts_stably(rows, m, torch.float32, contiguous=False)
    device = gathered.device
    values = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
    ids = torch.empty((rows, TOP_K), dtype=torch.int64, device=device)
    if rows == 0:
        return values, ids
    dummy = torch.empty((rows, 1), dtype=torch.int32, device=device)
    _compiled(
        SRC_PAIRS,
        False,
        -(-m // THREADS),
        OUT_VALUES_IDS,
        stable,
        cols,
        torch.int32,
        TOP_K,
        lamport,
    )(gathered.view(torch.int32), dummy, dummy, values.view(torch.int32), ids, dummy, m, 0)
    return values, ids


__all__ = ["TOP_K", "Top32Kernel", "merge_top32", "top32", "torch_topk_sorts_stably"]
