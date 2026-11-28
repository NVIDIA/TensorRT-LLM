# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernels of the DFlash2 sampled selector draft: the scoring operands and the walk.

The torch path (``dflash2_score_edges``, ``dflash2_uniforms``, ``dflash2_selector_walk`` and
``DFlash2ProposalStore.store``) is two dozen small launches plus a dozen per block slot. Here it is
two CuTe DSL kernels around the one product that stays in torch:

* :class:`SelectorOperandsKernel` (one CTA per request and slot) gathers the codebook rows of the
  slot's 32 predecessors (the anchor for slot 0) and 32 candidates and writes the two fp32 operands
  of the low-rank match, ``predecessor_codebook[a].float() * gate.float()`` and
  ``successor_codebook[b].float()``, contiguous ``[n, K, top_k, rank]`` as the torch expression
  builds them.
* the match itself stays ``torch.einsum("blpr,blcr->blpc", ...)``: torch's fp32 matmul is TF32 by
  default, its rounding is cuBLAS's, and the walk's draws are only bitwise when the scores are.
* :class:`SelectorWalkKernel` (one warp per request) adds the unary logits, derives the request's
  SplitMix64 uniforms, walks the slots and writes the drafted tokens and, optionally, the rows of
  the proposal store (candidate ids, proposal distributions, tokens).

The walk reproduces torch's float arithmetic operation by operation, so its draws are bitwise
those of the torch path for the same scores, seed and offset: ``softmax`` as torch's warp softmax
(lane ``c`` holds candidate ``c``; butterfly max, ``expf``, butterfly sum, IEEE division),
``cumsum`` as torch's scan (the Sklansky network of ``scan_innermost_dim`` for several requests,
a sequential scan for one request, whose single row torch hands to CUB), the inverse CDF as a
ballot count, and ``argmax`` as torch's (the first maximum, NaN above everything).
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cutlass import Float32, Int32, Int64, Uint64, cute

#: SplitMix64 constants (unsigned 64-bit).
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
MIX_1 = 0xBF58476D1CE4E5B9
MIX_2 = 0x94D049BB133111EB
#: Temperature floor of the walk: softmax(S / max(T, MIN_WALK_TEMPERATURE)).
MIN_WALK_TEMPERATURE = 1e-5
#: The kernels are written for this many candidates per slot: one per lane.
TOP_K = 32
#: Elements per 16-byte BF16 vector.
VEC = 8
OPERAND_THREADS = 256

_TORCH_TO_CUTLASS = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
}


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


@cute.jit
def _mix64(z: Uint64) -> Uint64:
    """SplitMix64's output function (wrapping unsigned arithmetic, logical shifts)."""
    z = (z ^ (z >> Uint64(30))) * Uint64(MIX_1)
    z = (z ^ (z >> Uint64(27))) * Uint64(MIX_2)
    return z ^ (z >> Uint64(31))


# ---------------------------------------------------------------------------
# Scoring operands
# ---------------------------------------------------------------------------


class SelectorOperandsKernel:
    """``pred[b, e, p] = predecessor_codebook[a(b, e, p)].float() * gate[b, e].float()`` and
    ``succ[b, e, c] = successor_codebook[candidate_ids[b, e, c]].float()``.

    ``a(b, 0, p)`` is the anchor of request ``b``; ``a(b, e, p) = candidate_ids[b, e - 1, p]``.
    One CTA per (request, slot) row of ``top_k`` predecessors and candidates; thread ``t`` covers
    columns ``[8 (t % 32), 8 (t % 32) + 8)`` of rows ``t // 32 + 8 i``. Products are single fp32
    multiplies, bitwise the torch expression's.
    """

    def __init__(self, slots: int, rank: int):
        if rank % (VEC * 32):
            raise ValueError(f"selector rank {rank} is not a multiple of {VEC * 32}")
        self.slots = slots
        self.rank = rank
        self.col_groups = rank // VEC  # vectors per row
        self.row_groups = OPERAND_THREADS // self.col_groups
        if TOP_K % self.row_groups:
            raise ValueError(f"rank {rank}: {self.row_groups} row groups do not divide {TOP_K}")
        self.rows_per_thread = TOP_K // self.row_groups

    @cute.jit
    def __call__(
        self,
        mPredCb: cute.Tensor,  # [vocab, rank] codebook dtype
        mSuccCb: cute.Tensor,  # [vocab, rank]
        mCand: cute.Tensor,  # [n * K * top_k] int32 / int64
        mAnchor: cute.Tensor,  # [n] int32 / int64
        mGate: cute.Tensor,  # [n * K, rank] gate dtype
        mPred: cute.Tensor,  # [n * K * top_k, rank] fp32
        mSucc: cute.Tensor,  # [n * K * top_k, rank] fp32
        stream,
    ):
        num_rows = mGate.shape[0]
        gPredCb = cute.zipped_divide(mPredCb, (1, VEC))
        gSuccCb = cute.zipped_divide(mSuccCb, (1, VEC))
        gGate = cute.zipped_divide(mGate, (1, VEC))
        gPred = cute.zipped_divide(mPred, (1, VEC))
        gSucc = cute.zipped_divide(mSucc, (1, VEC))
        self.kernel(gPredCb, gSuccCb, mCand, mAnchor, gGate, gPred, gSucc).launch(
            grid=[num_rows, 1, 1], block=[OPERAND_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        gPredCb: cute.Tensor,
        gSuccCb: cute.Tensor,
        mCand: cute.Tensor,
        mAnchor: cute.Tensor,
        gGate: cute.Tensor,
        gPred: cute.Tensor,
        gSucc: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()  # = b * K + e
        v = tid % self.col_groups
        r0 = tid // self.col_groups
        b = row // self.slots
        e = row - b * self.slots
        gate = gGate[(None, (row, v))].load().to(Float32)
        anchor = Int32(mAnchor[b])
        for i in cutlass.range_constexpr(self.rows_per_thread):
            p = r0 + i * self.row_groups
            out_row = row * TOP_K + p
            succ_id = Int32(mCand[out_row])
            pred_id = anchor
            if e > 0:
                pred_id = Int32(mCand[out_row - TOP_K])
            succ = gSuccCb[(None, (succ_id, v))].load().to(Float32)
            pred = gPredCb[(None, (pred_id, v))].load().to(Float32)
            gSucc[(None, (out_row, v))].store(succ)
            gPred[(None, (out_row, v))].store(pred * gate)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


class SelectorWalkKernel:
    """One sampled (or argmax) walk of the selector lattice per request; one warp per request.

    Args:
        slots: block slots ``K``.
        sequential_scan: the batch has one request (torch's cumsum hands a single row to CUB,
            which scans it sequentially); otherwise the Sklansky scan of ``scan_innermost_dim``.
        write_store: write the proposal store rows ``slot_ids[b]`` (candidate ids, q, tokens).
        realized: the greedy selector path instead (``dflash2_walk_candidate_paths`` and the
            scatter of ``select_candidate_path``): every request walks by argmax and slot ``e``'s
            score row under the predecessor taken is written, in ``block``'s dtype, at
            ``block[b, e, candidate_ids[b, e, c]]``; nothing else is written.
        realized_tokens: with ``realized``, write no score rows but the token
            ``torch.argmax`` would read off each slot's row of a ``-inf`` block holding them:
            ``tokens32[b, e]``, the candidate whose score (rounded to ``block_bf16``'s dtype)
            is largest, the lowest id among equal ones (NaN above everything), or 0 when every
            score is ``-inf`` (an all ``-inf`` row's argmax).
        block_bf16: the block that ``realized_tokens`` stands for is BF16 (else FP32).

    The score tensor is the bilinear term only and ``unary[b, e, c]`` is added
    (``unary.float() + term``, one fp32 add, as ``dflash2_score_edges``). Greedy rows are
    ``temperature <= greedy_threshold``.
    """

    def __init__(
        self,
        slots: int,
        sequential_scan: bool,
        write_store: bool,
        realized: bool = False,
        realized_tokens: bool = False,
        block_bf16: bool = False,
    ):
        self.slots = slots
        self.realized = realized
        self.realized_tokens = realized_tokens
        self.block_bf16 = block_bf16
        self.sequential_scan = sequential_scan
        self.write_store = write_store

    @cute.jit
    def __call__(
        self,
        mScores: cute.Tensor,  # [n, K, top_k, top_k] fp32
        mUnary: cute.Tensor,  # [n, K, top_k]
        mCand: cute.Tensor,  # [n, K, top_k] int32 / int64
        mSeed: cute.Tensor,  # [n] int64
        mOffset: cute.Tensor,  # [n] int64
        mTemp: cute.Tensor,  # [n] fp32
        mTok32: cute.Tensor,  # [n, K] int32
        mSlot: cute.Tensor,  # [n] int32 / int64 (write_store)
        mStoreCand: cute.Tensor,  # [rows, K_max, top_k] int64 (write_store)
        mStoreQ: cute.Tensor,  # [rows, K_max, top_k] fp32 (write_store)
        mStoreTok: cute.Tensor,  # [rows, K_max] int64 (write_store)
        mBlock: cute.Tensor,  # [n, K, vocab] (realized)
        greedy_threshold: Float32,
        stream,
    ):
        n = mCand.shape[0]
        self.kernel(
            mScores,
            mUnary,
            mCand,
            mSeed,
            mOffset,
            mTemp,
            mTok32,
            mSlot,
            mStoreCand,
            mStoreQ,
            mStoreTok,
            mBlock,
            greedy_threshold,
        ).launch(grid=[n, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.jit
    def argmax_lane(self, row: Float32, lane: Int32) -> Int32:
        """torch.argmax of the warp's row (lane c holds candidate c): the first maximum, NaN
        above everything. A total order, so the butterfly leaves every lane the same winner."""
        best_v = row
        best_i = lane
        for off in cutlass.range_constexpr(5):
            ov = cute.arch.shuffle_sync_bfly(best_v, offset=16 >> off)
            oi = cute.arch.shuffle_sync_bfly(best_i, offset=16 >> off)
            o_nan = Int32(ov != ov)
            c_nan = Int32(best_v != best_v)
            lower = Int32(oi < best_i)
            take = o_nan * (1 - c_nan + c_nan * lower) + (1 - o_nan) * (1 - c_nan) * (
                Int32(ov > best_v) + Int32(ov == best_v) * lower
            )
            if take != 0:
                best_v = ov
                best_i = oi
        return best_i

    @cute.jit
    def argmax_vocab(self, row: Float32, cand: Int32) -> Int32:
        """torch.argmax over the vocabulary of a ``-inf`` row holding ``row`` at ``cand`` (lane c
        holds candidate c, distinct ids): the lowest id among the largest (NaN above everything),
        0 when every value is ``-inf``."""
        v = row
        if cutlass.const_expr(self.block_bf16):
            v = Float32(cutlass.BFloat16(row))
        best_v = v
        best_i = cand
        for off in cutlass.range_constexpr(5):
            ov = cute.arch.shuffle_sync_bfly(best_v, offset=16 >> off)
            oi = cute.arch.shuffle_sync_bfly(best_i, offset=16 >> off)
            o_nan = Int32(ov != ov)
            c_nan = Int32(best_v != best_v)
            lower = Int32(oi < best_i)
            take = o_nan * (1 - c_nan + c_nan * lower) + (1 - o_nan) * (1 - c_nan) * (
                Int32(ov > best_v) + Int32(ov == best_v) * lower
            )
            if take != 0:
                best_v = ov
                best_i = oi
        if best_v == Float32(float("-inf")):
            best_i = Int32(0)
        return best_i

    @cute.kernel
    def kernel(
        self,
        mScores: cute.Tensor,
        mUnary: cute.Tensor,
        mCand: cute.Tensor,
        mSeed: cute.Tensor,
        mOffset: cute.Tensor,
        mTemp: cute.Tensor,
        mTok32: cute.Tensor,
        mSlot: cute.Tensor,
        mStoreCand: cute.Tensor,
        mStoreQ: cute.Tensor,
        mStoreTok: cute.Tensor,
        mBlock: cute.Tensor,
        greedy_threshold: Float32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()

        # temperatures.float().clamp(min=MIN_WALK_TEMPERATURE) (NaN stays NaN).
        temp = Float32(mTemp[b])
        greedy = Int32(0)
        if temp <= greedy_threshold:
            greedy = Int32(1)
        # clamp(min=...): a NaN temperature stays NaN.
        if temp < Float32(MIN_WALK_TEMPERATURE):
            temp = Float32(MIN_WALK_TEMPERATURE)

        # SplitMix64 key mix(mix(seed) ^ offset).
        key = _mix64(_mix64(Uint64(Int64(mSeed[b]))) ^ Uint64(Int64(mOffset[b])))

        slot_row = Int64(0)
        if cutlass.const_expr(self.write_store):
            slot_row = Int64(mSlot[b])

        prev = Int32(0)
        for e in cutlass.range_constexpr(self.slots):
            row = Float32(mScores[b, e, prev, lane])
            row = Float32(mUnary[b, e, lane]) + row
            cand = Int64(mCand[b, e, lane])

            if cutlass.const_expr(self.realized):
                if cutlass.const_expr(self.realized_tokens):
                    token = self.argmax_vocab(row, Int32(cand))
                    if lane == 0:
                        mTok32[b, e] = token
                else:
                    mBlock[b, e, cand] = mBlock.element_type(row)
                prev = self.argmax_lane(row, lane)
            else:
                # softmax(row / temp) as torch's warp softmax.
                x = row / temp
                mx = x
                for off in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_bfly(mx, offset=16 >> off)
                    # torch's Max functor: a < b ? b : a.
                    if mx < other:
                        mx = other
                ex = cute.math.exp(x - mx)
                total = ex
                for off in cutlass.range_constexpr(5):
                    total = total + cute.arch.shuffle_sync_bfly(total, offset=16 >> off)
                prob = ex / total
                if total == Float32(0.0):
                    prob = Float32(float("nan"))

                # cumsum(prob) as torch's scan.
                cum = prob
                if cutlass.const_expr(self.sequential_scan):
                    cum = cute.arch.shuffle_sync(prob, offset=0)
                    for k in cutlass.range_constexpr(1, TOP_K):
                        pk = cute.arch.shuffle_sync(prob, offset=k)
                        if lane >= k:
                            cum = cum + pk
                else:
                    for m in cutlass.range_constexpr(5):
                        s = 1 << m
                        src = (lane & (31 ^ (2 * s - 1))) | (s - 1)
                        other = cute.arch.shuffle_sync(cum, offset=src)
                        if (lane & s) != 0:
                            cum = cum + other

                # Draw e of the request's stream: mix(key + (e + 1) * gamma) >> 40, times 2**-24.
                bits = _mix64(key + Uint64((e + 1) * GOLDEN_GAMMA % (1 << 64))) >> Uint64(40)
                u = Float32(Int32(bits)) * Float32(1.0 / (1 << 24))
                ballot = cute.arch.vote_ballot_sync(u >= cum)
                drawn = cute.arch.popc(ballot)
                if drawn > TOP_K - 1:
                    drawn = Int32(TOP_K - 1)

                best_i = self.argmax_lane(row, lane)

                index = drawn
                q = prob
                if greedy != 0:
                    index = best_i
                    q = Float32(0.0)
                    if lane == best_i:
                        q = Float32(1.0)
                token = cute.arch.shuffle_sync(cand, offset=index)

                if lane == 0:
                    mTok32[b, e] = Int32(token)
                if cutlass.const_expr(self.write_store):
                    mStoreCand[slot_row, e, lane] = cand
                    mStoreQ[slot_row, e, lane] = q
                    if lane == 0:
                        mStoreTok[slot_row, e] = token
                prev = index


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _fake_stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


@functools.cache
def _compiled_operands(slots, rank, cb_dtype, gate_dtype, cand_dtype, anchor_dtype):
    kernel = SelectorOperandsKernel(slots, rank)
    vocab = cute.sym_int()
    rows = cute.sym_int()
    flat = cute.sym_int()
    nreq = cute.sym_int()
    cb = _TORCH_TO_CUTLASS[cb_dtype]
    return cute.compile(
        kernel,
        _row_major(cb, (vocab, rank), 16),
        _row_major(cb, (vocab, rank), 16),
        _row_major(_TORCH_TO_CUTLASS[cand_dtype], (flat,)),
        _row_major(_TORCH_TO_CUTLASS[anchor_dtype], (nreq,)),
        _row_major(_TORCH_TO_CUTLASS[gate_dtype], (rows, rank), 16),
        _row_major(cutlass.Float32, (flat, rank), 16),
        _row_major(cutlass.Float32, (flat, rank), 16),
        _fake_stream(),
        options=_OPTIONS,
    )


def selector_operands(
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    gate: torch.Tensor,
    anchor_token_ids: torch.Tensor,
):
    """``(predecessors * gate, successors)``, the fp32 ``[n, K, top_k, rank]`` operands of
    :func:`dflash2_score_edges`'s einsum."""
    n, slots, top_k = candidate_ids.shape
    rank = predecessor_codebook.shape[1]
    if top_k != TOP_K:
        raise ValueError(f"the selector kernels take top_k {TOP_K}, got {top_k}")
    pred = torch.empty((n, slots, top_k, rank), dtype=torch.float32, device=gate.device)
    succ = torch.empty_like(pred)
    if n == 0:
        return pred, succ
    _compiled_operands(
        slots,
        rank,
        predecessor_codebook.dtype,
        gate.dtype,
        candidate_ids.dtype,
        anchor_token_ids.dtype,
    )(
        predecessor_codebook,
        successor_codebook,
        candidate_ids.reshape(-1),
        anchor_token_ids,
        gate.reshape(n * slots, rank),
        pred.view(n * slots * top_k, rank),
        succ.view(n * slots * top_k, rank),
    )
    return pred, succ


@functools.cache
def _compiled_walk(
    slots,
    sequential_scan,
    unary_dtype,
    cand_dtype,
    write_store,
    slot_dtype,
    block_dtype=None,
    tokens_dtype=None,
):
    kernel = SelectorWalkKernel(
        slots,
        sequential_scan,
        write_store,
        realized=block_dtype is not None or tokens_dtype is not None,
        realized_tokens=tokens_dtype is not None,
        block_bf16=tokens_dtype == torch.bfloat16,
    )
    vocab = cute.sym_int()
    n = cute.sym_int()
    rows = cute.sym_int()
    kmax = cute.sym_int()
    f32 = cutlass.Float32
    i64 = cutlass.Int64
    return cute.compile(
        kernel,
        _row_major(f32, (n, slots, TOP_K, TOP_K)),
        _row_major(_TORCH_TO_CUTLASS[unary_dtype], (n, slots, TOP_K)),
        _row_major(_TORCH_TO_CUTLASS[cand_dtype], (n, slots, TOP_K)),
        _row_major(i64, (n,)),
        _row_major(i64, (n,)),
        _row_major(f32, (n,)),
        _row_major(cutlass.Int32, (n, slots)),
        _row_major(_TORCH_TO_CUTLASS[slot_dtype], (n,)),
        _row_major(i64, (rows, kmax, TOP_K)),
        _row_major(f32, (rows, kmax, TOP_K)),
        _row_major(i64, (rows, kmax)),
        _row_major(_TORCH_TO_CUTLASS[block_dtype or torch.float32], (n, slots, vocab)),
        Float32(0.0),
        _fake_stream(),
        options=_OPTIONS,
    )


def selector_walk(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    temperatures: torch.Tensor,
    *,
    unary_logits: torch.Tensor,
    greedy_threshold: float = 0.0,
    store=None,
    slot_ids: torch.Tensor = None,
    block_logits: torch.Tensor = None,
    tokens_block_dtype: torch.dtype = None,
):
    """Launch the walk; see :class:`SelectorWalkKernel`.

    ``scores``: ``[n, K, top_k, top_k]`` fp32, the bilinear term of the edge scores, to which
    ``unary_logits`` ``[n, K, top_k]`` is added. Greedy rows are ``temperatures <=
    greedy_threshold``. ``store``: ``(candidate_ids, q, tokens)`` of a proposal store, written at
    rows ``slot_ids``.

    Returns ``tokens32 [n, K] int32``. With ``block_logits`` (``[n, K, vocab]``) the greedy
    selector path runs instead and writes only the realized score rows into it (see
    ``realized``); with ``tokens_block_dtype`` it runs and writes only ``tokens32``, the argmax
    tokens of such a block of that dtype (see ``realized_tokens``).
    """
    n, slots, top_k = candidate_ids.shape
    if top_k != TOP_K:
        raise ValueError(f"the selector kernels take top_k {TOP_K}, got {top_k}")
    device = candidate_ids.device
    tok32 = torch.empty((n, slots), dtype=torch.int32, device=device)
    if n == 0:
        return tok32
    if store is not None:
        store_cand, store_q, store_tok = store
        slots_t = slot_ids
    else:
        store_cand = torch.empty((1, 1, top_k), dtype=torch.int64, device=device)
        store_q = torch.empty((1, 1, top_k), dtype=torch.float32, device=device)
        store_tok = torch.empty((1, 1), dtype=torch.int64, device=device)
        slots_t = torch.empty((n,), dtype=torch.int64, device=device)
    block = block_logits
    if block is None:
        block = torch.empty((n, slots, 1), dtype=torch.float32, device=device)
    _compiled_walk(
        slots,
        n == 1,
        unary_logits.dtype,
        candidate_ids.dtype,
        store is not None,
        slots_t.dtype,
        None if block_logits is None else block_logits.dtype,
        tokens_block_dtype,
    )(
        scores,
        unary_logits,
        candidate_ids,
        seed,
        offset,
        temperatures,
        tok32,
        slots_t,
        store_cand,
        store_q,
        store_tok,
        block,
        greedy_threshold,
    )
    return tok32


def selector_greedy_tokens(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    block_dtype: torch.dtype,
    *,
    unary_logits: torch.Tensor,
) -> torch.Tensor:
    """``torch.argmax(selector_greedy_path(scores, candidate_ids, block, ...), -1)`` as int32
    ``[n, K]`` for a ``block`` of ``block_dtype`` (BF16 / FP32), without the block: the walk
    writes each slot's token. ``scores`` as in :func:`selector_walk`."""
    if block_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"the greedy tokens stand for a BF16 or FP32 block, got {block_dtype}")
    n = candidate_ids.shape[0]
    device = candidate_ids.device
    if n == 0:
        return torch.empty((0, candidate_ids.shape[1]), dtype=torch.int32, device=device)
    # The greedy walk reads no seed, offset or temperature (their values are never used).
    unused = torch.empty((n,), dtype=torch.int64, device=device)
    return selector_walk(
        scores,
        candidate_ids,
        unused,
        unused,
        torch.empty((n,), dtype=torch.float32, device=device),
        unary_logits=unary_logits,
        tokens_block_dtype=block_dtype,
    )


def selector_greedy_path(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    block_logits: torch.Tensor,
    *,
    unary_logits: torch.Tensor,
) -> torch.Tensor:
    """``select_candidate_path``'s walk and scatter: ``block_logits`` filled with ``-inf`` except
    at each slot's candidates, which get the score row of the greedy walk's predecessor (in
    ``block_logits``' dtype). ``scores`` as in :func:`selector_walk`. Returns ``block_logits``."""
    n = candidate_ids.shape[0]
    block_logits.fill_(float("-inf"))
    if n == 0:
        return block_logits
    # The greedy walk reads no seed, offset or temperature (their values are never used).
    unused = torch.empty((n,), dtype=torch.int64, device=candidate_ids.device)
    selector_walk(
        scores,
        candidate_ids,
        unused,
        unused,
        torch.empty((n,), dtype=torch.float32, device=candidate_ids.device),
        unary_logits=unary_logits,
        block_logits=block_logits,
    )
    return block_logits


__all__ = [
    "MIN_WALK_TEMPERATURE",
    "TOP_K",
    "SelectorOperandsKernel",
    "SelectorWalkKernel",
    "selector_greedy_path",
    "selector_greedy_tokens",
    "selector_operands",
    "selector_walk",
]
