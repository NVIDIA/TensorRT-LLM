# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL launches of the per-request integer bookkeeping around a sampled DFlash2 decode step.

Each replaces a run of one-CTA torch launches with one launch of one CTA (a thread a request), the same integer
arithmetic in the same types:

* ``AcceptCommitKernel``: the verified block's tokens and counts. Per request, the greedy rule's or the block
  verification's ``(tau, bonus)`` by the request's temperature, the emitted tokens (the drafts, the last draft
  repeated, the bonus at ``tau``) and ``tau + 1``, then the forced acceptance count where it is set (a constant,
  or the fixed-seed pool draw of a counter bumped once per launch).
* ``AcceptRecordKernel``: what follows acceptance before the drafter: the KDA replay state's accepted-draft counts
  (``max(count - 1, 0)`` into each request's state slot, a dummy request's slot rewritten with its own value: every
  slot read before any is written), the deferred KV-length rewind ``1 - count`` and the drafter's ``kv_lens + 1``.
* ``DraftTailKernel``: what follows the drafter: the rewind (``max(kv_lens - rewind, 0)``) and the next step's input
  tokens (the last accepted token, then the drafts).
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

THREADS = 128
_INTS = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64, torch.uint8: cutlass.Uint8}

#: Forced acceptance modes of ``AcceptCommitKernel``.
FORCE_OFF = 0
FORCE_CONSTANT = 1
FORCE_POOL = 2


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
    )


class AcceptCommitKernel:
    """``accepted [n, K + 1] int32`` and ``num_accepted [n] int32`` of ``n`` generation requests (see the module
    doc); ``force`` is one of ``FORCE_*``."""

    def __init__(self, slots: int, force: int):
        self.slots = slots
        self.force = force

    @cute.jit
    def __call__(
        self,
        mTau: cute.Tensor,  # [n] int64, block verification
        mBonus: cute.Tensor,  # [n] int64
        mGTau: cute.Tensor,  # [n] int64, the greedy rule
        mGBonus: cute.Tensor,  # [n] int64
        mTemp: cute.Tensor,  # [n] FP32 request temperatures
        mDraft: cute.Tensor,  # [n, K] int32 / int64
        mAcc: cute.Tensor,  # [n, K + 1] int32
        mNum: cute.Tensor,  # [n] int32
        mCounter: cute.Tensor,  # [1] int64 (FORCE_POOL), bumped in place
        mPool: cute.Tensor,  # [P] FP32 (FORCE_POOL)
        threshold: Float32,
        base_total: Int32,
        max_total: Int32,
        frac: Float32,
        counter_stride: Int64,
        slot_stride: Int64,
        pool_mask: Int64,
        stream,
    ):
        self.kernel(
            mTau,
            mBonus,
            mGTau,
            mGBonus,
            mTemp,
            mDraft,
            mAcc,
            mNum,
            mCounter,
            mPool,
            threshold,
            base_total,
            max_total,
            frac,
            counter_stride,
            slot_stride,
            pool_mask,
        ).launch(grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mTau: cute.Tensor,
        mBonus: cute.Tensor,
        mGTau: cute.Tensor,
        mGBonus: cute.Tensor,
        mTemp: cute.Tensor,
        mDraft: cute.Tensor,
        mAcc: cute.Tensor,
        mNum: cute.Tensor,
        mCounter: cute.Tensor,
        mPool: cute.Tensor,
        threshold: Float32,
        base_total: Int32,
        max_total: Int32,
        frac: Float32,
        counter_stride: Int64,
        slot_stride: Int64,
        pool_mask: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n = mNum.shape[0]
        K: cutlass.Constexpr = self.slots
        counter = Int64(0)
        if cutlass.const_expr(self.force == FORCE_POOL):
            # The draw of this launch uses the counter after its bump; every thread reads it before the bump lands.
            counter = Int64(mCounter[0]) + Int64(1)
            cute.arch.sync_threads()
            if tidx == 0:
                mCounter[0] = counter
        for r in cutlass.range(tidx, n, THREADS):
            # torch.where(request_temperatures <= threshold, greedy, block).
            tau = Int64(mTau[r])
            bonus = Int64(mBonus[r])
            if Float32(mTemp[r]) <= threshold:
                tau = Int64(mGTau[r])
                bonus = Int64(mGBonus[r])
            # cat(drafts, last draft).scatter(1, tau, bonus).to(int32).
            for c in cutlass.range_constexpr(K + 1):
                tok = Int64(mDraft[r, min(c, K - 1)])
                if Int64(c) == tau:
                    tok = bonus
                mAcc[r, c] = Int32(tok)
            num = Int32(tau + Int64(1))
            if cutlass.const_expr(self.force == FORCE_CONSTANT):
                num = base_total
            if cutlass.const_expr(self.force == FORCE_POOL):
                index = (counter * counter_stride + Int64(r) * slot_stride) & pool_mask
                extra = Int32(0)
                if Float32(mPool[Int32(index)]) < frac:
                    extra = Int32(1)
                num = cutlass.min(base_total + extra, max_total)
            mNum[r] = num


class AcceptRecordKernel:
    """The KDA replay counts, the KV-length rewind and the drafter's ``kv_lens + 1`` of ``n`` generation requests
    (see the module doc)."""

    @cute.jit
    def __call__(
        self,
        mNum: cute.Tensor,  # [n] int32 accepted counts
        mState: cute.Tensor,  # [n] int32 / int64 state slots
        mDummy: cute.Tensor,  # [n] uint8 (bool) dummy requests
        mPrev: cute.Tensor,  # [S] int32 replay counts (any slot stride), updated in place
        mKvLens: cute.Tensor,  # [n] int32, updated in place
        mRewind: cute.Tensor,  # [n] int32
        stream,
    ):
        self.kernel(mNum, mState, mDummy, mPrev, mKvLens, mRewind).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mNum: cute.Tensor,
        mState: cute.Tensor,
        mDummy: cute.Tensor,
        mPrev: cute.Tensor,
        mKvLens: cute.Tensor,
        mRewind: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n = mNum.shape[0]
        # current = prev[slots] for every request before any write (torch's gather, then index_put).
        vals = cute.make_rmem_tensor(cute.make_layout((8,), stride=(1,)), Int32)
        for k in cutlass.range_constexpr(8):
            r = tidx + k * THREADS
            if r < n:
                num = Int32(mNum[r])
                accepted = cutlass.max(num - Int32(1), Int32(0))
                if Int32(mDummy[r]) != Int32(0):
                    accepted = Int32(mPrev[Int32(mState[r])])
                vals[k] = accepted
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(8):
            r = tidx + k * THREADS
            if r < n:
                mPrev[Int32(mState[r])] = vals[k]
                mRewind[r] = Int32(1) - Int32(mNum[r])
                mKvLens[r] = Int32(mKvLens[r]) + Int32(1)


class DraftTailKernel:
    """The KV-length rewind of ``n`` generation requests (``rewind``) and the next input tokens of ``b``
    requests (see the module doc)."""

    def __init__(self, width: int, drafts: int, rewind: bool):
        self.width = width
        self.drafts = drafts
        self.rewind = rewind

    @cute.jit
    def __call__(
        self,
        mKvLens: cute.Tensor,  # [n] int32, updated in place
        mRewind: cute.Tensor,  # [n] int32
        mAcc: cute.Tensor,  # [rows, width] int32 accepted tokens
        mBatch: cute.Tensor,  # [b] int32 / int64 rows of mAcc
        mNum: cute.Tensor,  # [b] int32 accepted counts
        mDraft: cute.Tensor,  # [b, drafts] int32 next draft tokens
        mOut: cute.Tensor,  # [b, 1 + drafts] int32
        stream,
    ):
        self.kernel(mKvLens, mRewind, mAcc, mBatch, mNum, mDraft, mOut).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mKvLens: cute.Tensor,
        mRewind: cute.Tensor,
        mAcc: cute.Tensor,
        mBatch: cute.Tensor,
        mNum: cute.Tensor,
        mDraft: cute.Tensor,
        mOut: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.rewind):
            for r in cutlass.range(tidx, mKvLens.shape[0], THREADS):
                mKvLens[r] = cutlass.max(Int32(mKvLens[r]) - Int32(mRewind[r]), Int32(0))
        for r in cutlass.range(tidx, mOut.shape[0], THREADS):
            # accepted[batch[r], num[r] - 1], a negative column counted from the end as torch indexes it.
            col = Int32(mNum[r]) - Int32(1)
            if col < Int32(0):
                col = col + Int32(self.width)
            mOut[r, 0] = Int32(mAcc[Int32(mBatch[r]), col])
            for c in cutlass.range_constexpr(self.drafts):
                mOut[r, 1 + c] = Int32(mDraft[r, c])


_OPTIONS = "--opt-level 2 --enable-tvm-ffi"


@functools.cache
def _compiled_commit(slots: int, force: int, draft_dtype: torch.dtype):
    n = cute.sym_int()
    pool = cute.sym_int()
    return cute.compile(
        AcceptCommitKernel(slots, force),
        _row_major(Int64, (n,)),
        _row_major(Int64, (n,)),
        _row_major(Int64, (n,)),
        _row_major(Int64, (n,)),
        _row_major(Float32, (n,)),
        _row_major(_INTS[draft_dtype], (n, slots)),
        _row_major(Int32, (n, slots + 1)),
        _row_major(Int32, (n,)),
        _row_major(Int64, (1,)),
        _row_major(Float32, (pool,)),
        Float32(0.0),
        Int32(0),
        Int32(0),
        Float32(0.0),
        Int64(0),
        Int64(0),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


@functools.cache
def _compiled_record(state_dtype: torch.dtype):
    n = cute.sym_int()
    s = cute.sym_int()
    return cute.compile(
        AcceptRecordKernel(),
        _row_major(Int32, (n,)),
        _row_major(_INTS[state_dtype], (n,)),
        _row_major(cutlass.Uint8, (n,)),
        # The replay counts live in the KV cache pool, one entry per slot at the pool's slot pitch.
        cute.runtime.make_fake_tensor(Int32, (s,), stride=(cute.sym_int64(),)),
        _row_major(Int32, (n,)),
        _row_major(Int32, (n,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


@functools.cache
def _compiled_tail(width: int, drafts: int, rewind: bool, batch_dtype: torch.dtype):
    n = cute.sym_int()
    b = cute.sym_int()
    rows = cute.sym_int()
    return cute.compile(
        DraftTailKernel(width, drafts, rewind),
        _row_major(Int32, (n,)),
        _row_major(Int32, (n,)),
        _row_major(Int32, (rows, width)),
        _row_major(_INTS[batch_dtype], (b,)),
        _row_major(Int32, (b,)),
        _row_major(Int32, (b, drafts)),
        _row_major(Int32, (b, 1 + drafts)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=_OPTIONS,
    )


#: Most requests a launch takes (eight a thread in ``AcceptRecordKernel``).
MAX_REQUESTS = 8 * THREADS


def accept_commit(
    tau: torch.Tensor,
    bonus: torch.Tensor,
    greedy_tau: torch.Tensor,
    greedy_bonus: torch.Tensor,
    temperatures: torch.Tensor,
    threshold: float,
    draft_tokens: torch.Tensor,
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    force: int,
    base_total: int,
    max_total: int,
    frac: float,
    counter: torch.Tensor,
    pool: torch.Tensor,
    counter_stride: int,
    slot_stride: int,
) -> None:
    """See :class:`AcceptCommitKernel`; writes ``accepted`` and ``num_accepted``."""
    n, slots = draft_tokens.shape
    if n == 0:
        return
    _compiled_commit(slots, force, draft_tokens.dtype)(
        tau,
        bonus,
        greedy_tau,
        greedy_bonus,
        temperatures,
        draft_tokens,
        accepted,
        num_accepted,
        counter,
        pool,
        threshold,
        base_total,
        max_total,
        frac,
        counter_stride,
        slot_stride,
        pool.shape[0] - 1,
    )


def accept_record(
    num_accepted: torch.Tensor,
    state_indices: torch.Tensor,
    is_dummy: torch.Tensor,
    prev_num_accepted: torch.Tensor,
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
) -> None:
    """See :class:`AcceptRecordKernel`; updates ``prev_num_accepted`` and ``kv_lens``, writes ``rewind``."""
    if num_accepted.shape[0] == 0:
        return
    _compiled_record(state_indices.dtype)(
        num_accepted, state_indices, is_dummy.view(torch.uint8), prev_num_accepted, kv_lens, rewind
    )


def draft_tail(
    kv_lens: torch.Tensor,
    rewind: torch.Tensor,
    accepted: torch.Tensor,
    batch_rows: torch.Tensor,
    num_accepted: torch.Tensor,
    next_drafts: torch.Tensor,
    out: torch.Tensor,
    apply_rewind: bool,
) -> None:
    """See :class:`DraftTailKernel`; updates ``kv_lens`` (``apply_rewind``), writes ``out``."""
    _compiled_tail(accepted.shape[1], next_drafts.shape[1], apply_rewind, batch_rows.dtype)(
        kv_lens, rewind, accepted, batch_rows, num_accepted, next_drafts, out
    )


__all__ = [
    "FORCE_CONSTANT",
    "FORCE_OFF",
    "FORCE_POOL",
    "MAX_REQUESTS",
    "THREADS",
    "AcceptCommitKernel",
    "AcceptRecordKernel",
    "DraftTailKernel",
    "accept_commit",
    "accept_record",
    "draft_tail",
]
