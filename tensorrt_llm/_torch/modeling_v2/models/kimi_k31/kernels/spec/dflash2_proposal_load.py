# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL read of the DFlash2 proposal store for verification (``DFlash2ProposalStore.load``).

The drafting step's walk wrote each request's candidate ids, proposal distribution ``q`` and
drafted tokens into the store row of its sequence slot (``SelectorWalkKernel``'s epilogue). The
verifying step reads them back for the tokens it verifies: a row whose stored tokens equal the
verified ones gives its candidates and ``q``; any other row gives the point mass on each verified
token (every lane the token, lane 0 the mass). One warp per request, lane ``c`` candidate ``c``:
pure copies and one ballot, bitwise the torch method's.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

TOP_K = 32
_IDS = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


def _row_major(dtype, shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape))))
    )


class ProposalLoadKernel:
    """``(candidate_ids [n, K, 32] int64, q [n, K, 32] FP32)`` of the store rows ``slot_ids`` for
    the ``draft_tokens [n, K]`` being verified."""

    def __init__(self, slots: int):
        if slots > 32:
            raise ValueError(f"{slots} slots: one lane per slot compares the stored tokens")
        self.slots = slots

    @cute.jit
    def __call__(
        self,
        mStoreCand: cute.Tensor,  # [rows, K_max, 32] int64
        mStoreQ: cute.Tensor,  # [rows, K_max, 32] FP32
        mStoreTok: cute.Tensor,  # [rows, K_max] int64
        mSlot: cute.Tensor,  # [n] int32 / int64
        mDraft: cute.Tensor,  # [n, K] int32 / int64
        mCand: cute.Tensor,  # [n, K, 32] int64
        mQ: cute.Tensor,  # [n, K, 32] FP32
        stream,
    ):
        n = mDraft.shape[0]
        self.kernel(mStoreCand, mStoreQ, mStoreTok, mSlot, mDraft, mCand, mQ).launch(
            grid=[n, 1, 1], block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mStoreCand: cute.Tensor,
        mStoreQ: cute.Tensor,
        mStoreTok: cute.Tensor,
        mSlot: cute.Tensor,
        mDraft: cute.Tensor,
        mCand: cute.Tensor,
        mQ: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        slot = Int64(mSlot[b])
        # Walked: every stored token of the row equals the verified one.
        same = Int32(1)
        if lane < self.slots:
            if Int64(mStoreTok[slot, lane]) != Int64(mDraft[b, lane]):
                same = Int32(0)
        walked = cute.arch.vote_ballot_sync(same != 0) == -1
        point_q = Float32(0.0)
        if lane == 0:
            point_q = Float32(1.0)
        for e in cutlass.range_constexpr(self.slots):
            if walked:
                mCand[b, e, lane] = Int64(mStoreCand[slot, e, lane])
                mQ[b, e, lane] = Float32(mStoreQ[slot, e, lane])
            else:
                mCand[b, e, lane] = Int64(mDraft[b, e])
                mQ[b, e, lane] = point_q


@functools.cache
def _compiled(slots, slot_dtype, draft_dtype):
    n = cute.sym_int()
    rows = cute.sym_int()
    kmax = cute.sym_int()
    return cute.compile(
        ProposalLoadKernel(slots),
        _row_major(Int64, (rows, kmax, TOP_K)),
        _row_major(Float32, (rows, kmax, TOP_K)),
        _row_major(Int64, (rows, kmax)),
        _row_major(_IDS[slot_dtype], (n,)),
        _row_major(_IDS[draft_dtype], (n, slots)),
        _row_major(Int64, (n, slots, TOP_K)),
        _row_major(Float32, (n, slots, TOP_K)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def proposal_load(
    store_candidate_ids: torch.Tensor,
    store_q: torch.Tensor,
    store_tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
):
    """See :class:`ProposalLoadKernel`."""
    n, slots = draft_tokens.shape
    device = draft_tokens.device
    candidate_ids = torch.empty((n, slots, TOP_K), dtype=torch.int64, device=device)
    q = torch.empty((n, slots, TOP_K), dtype=torch.float32, device=device)
    if n == 0:
        return candidate_ids, q
    _compiled(slots, slot_ids.dtype, draft_tokens.dtype)(
        store_candidate_ids, store_q, store_tokens, slot_ids, draft_tokens, candidate_ids, q
    )
    return candidate_ids, q


__all__ = ["ProposalLoadKernel", "proposal_load"]
