# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``draft_path_walk`` with the DFlash step's last bookkeeping in the same launch.

After the draft head's path walk (``draft_path_walk``, fast path ``draft_head``) the
built-in DFlash worker launches seven more kernels before the step ends
(``speculative/dflash.py``, ``speculative/interface.py``):

* ``_apply_kv_rewind_after_draft``: ``kv_lens_cuda[:g] -= 1 - num_accepted``,
  then ``clamp_(min=0)``;
* ``_prepare_next_new_tokens``: ``accepted_tokens[batch_indices[:g],
  num_accepted - 1]`` (two index casts, an ``index``) concatenated with the
  draft tokens -- the next step's input tokens.

Each program of the walk owns one request; once it has walked its K positions
it also writes that request's next input tokens (its last accepted token, then
the K draft tokens it just produced) and rewinds that request's KV length. The
walk itself is ``draft_path_walk``'s statement for statement, so its draft
tokens are the same; the tail is integer arithmetic on values that exist
before the walk starts. Bitwise the built-in results.

Generation-only steps (no context rows), one program per request.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .draft_path_walk import DraftPathWalk

#: The walk entry whose reference this one extends.
_walk = DraftPathWalk()


@triton.jit
def _draft_path_walk_tail_kernel(
    edge_ptr,
    unary_ptr,
    ids_ptr,
    tokens_ptr,
    num_positions,
    accepted_ptr,
    accepted_row_stride,
    num_accepted_ptr,
    batch_indices_ptr,
    next_tokens_ptr,
    kv_lens_ptr,
    rewind_ptr,
    TOP_K: tl.constexpr,
    KV_REWIND: tl.constexpr,
):
    b = tl.program_id(0)
    cand = tl.arange(0, TOP_K)
    next_row = next_tokens_ptr + b.to(tl.int64) * (num_positions + 1)
    # --- draft_path_walk, unchanged ---
    predecessor = b * 0
    for t in range(num_positions):
        row = (b * num_positions + t).to(tl.int64)
        edge = tl.load(edge_ptr + (row * TOP_K + predecessor) * TOP_K + cand)
        unary = tl.load(unary_ptr + row * TOP_K + cand)
        ids = tl.load(ids_ptr + row * TOP_K + cand)
        score = unary + edge
        is_nan = score != score
        nan_count = tl.sum(is_nan.to(tl.int32), axis=0)
        best = tl.max(tl.where(is_nan, float("-inf"), score), axis=0)
        hit = tl.where(nan_count > 0, is_nan, score == best)
        predecessor = tl.min(tl.where(hit, cand, TOP_K), axis=0)
        token = tl.min(tl.where(hit, ids, 9223372036854775807), axis=0)
        token = tl.where((nan_count == 0) & (best == float("-inf")), 0, token)
        tl.store(tokens_ptr + row, token.to(tl.int32))
        # --- the next step's input tokens, column 1 + t: this draft token ---
        tl.store(next_row + 1 + t, token.to(tl.int32))
    # --- column 0: accepted_tokens[batch_indices[b], num_accepted[b] - 1] ---
    source_row = tl.load(batch_indices_ptr + b).to(tl.int64)
    column = tl.load(num_accepted_ptr + b) - 1
    column = tl.where(column < 0, column + num_positions + 1, column)
    tl.store(next_row, tl.load(accepted_ptr + source_row * accepted_row_stride + column))
    # --- the KV-length rewind of this request ---
    if KV_REWIND:
        rewound = tl.load(kv_lens_ptr + b) - tl.load(rewind_ptr + b)
        tl.store(kv_lens_ptr + b, tl.maximum(rewound, 0))


class DraftPathWalkTail(OpWrapper):
    """``draft_path_walk``, plus each request's next input tokens and KV-length rewind.

        edges, unary, candidate_ids   as ``draft_path_walk`` ([B, K, top_k(, top_k)])
        accepted_tokens   [B, K + 1] int32, this step's accepted tokens (row stride free)
        num_accepted      [B] int32
        batch_indices     [>= B] int32, the rows of ``accepted_tokens`` (``spec_metadata.batch_indices_cuda``)
        kv_lens           [>= B] int32 or None: ``kv_lens[:B] = max(kv_lens[:B] - rewind, 0)`` in place
        rewind            [B] int32 or None (``1 - num_accepted``)

    Returns ``(tokens, next_tokens)``: the ``[B, K]`` int32 draft tokens and the
    ``[B, K + 1]`` int32 next input tokens.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: ``spec_metadata.batch_indices_cuda`` and the attention
    metadata's ``kv_lens_cuda`` (read-modify-write). One program per request,
    one warp, as ``draft_path_walk``. No programmatic-dependent-launch
    attributes.

    Call site: the DFlash worker's draft head under the ``path_walk_tail``
    fast path (``verify_fold/fold.py``), in place of ``draft_path_walk`` on
    generation-only steps outside the eager CUDA-graph warmup; the worker's
    ``_apply_kv_rewind_after_draft`` and ``_prepare_next_new_tokens`` then
    hand back what this launch wrote.

    Numerics: the walk as ``draft_path_walk`` (exact); the tail is integer
    arithmetic, bitwise the built-in ops.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        rewind: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requests, positions, top_k = unary.shape
        tokens = torch.empty((requests, positions), dtype=torch.int32, device=unary.device)
        next_tokens = torch.empty((requests, positions + 1), dtype=torch.int32, device=unary.device)
        rewinds = kv_lens is not None and rewind is not None
        _draft_path_walk_tail_kernel[(requests,)](
            edges,
            unary,
            candidate_ids,
            tokens,
            positions,
            accepted_tokens,
            accepted_tokens.stride(0),
            num_accepted,
            batch_indices,
            next_tokens,
            kv_lens if rewinds else tokens,
            rewind if rewinds else tokens,
            TOP_K=top_k,
            KV_REWIND=rewinds,
            num_warps=1,
        )
        return tokens, next_tokens

    def reference(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        rewind: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``draft_path_walk``'s reference, then the built-in rewind and next-token lines (no context rows).

        Rewinds ``kv_lens`` in place like the op.
        """
        batch_size = unary.shape[0]
        tokens = _walk.reference(edges, unary, candidate_ids)
        if kv_lens is not None and rewind is not None:
            kv_lens[0:batch_size] -= rewind
            kv_lens[0:batch_size].clamp_(min=0)
        next_new_tokens = accepted_tokens[batch_indices[:batch_size], num_accepted - 1].unsqueeze(1)
        next_new_tokens = torch.concat([next_new_tokens, tokens], dim=1)
        return tokens, next_new_tokens

    def is_valid(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        rewind: Optional[torch.Tensor],
    ) -> None:
        requests, positions, top_k = unary.shape
        if tuple(edges.shape) != (requests, positions, top_k, top_k):
            raise ValueError(f"edges {tuple(edges.shape)} != [B, K, top_k, top_k]")
        if tuple(candidate_ids.shape) != tuple(unary.shape) or candidate_ids.dtype != torch.int64:
            raise ValueError("candidate_ids must be int64 [B, K, top_k]")
        if edges.dtype != torch.float32 or unary.dtype != torch.float32:
            raise ValueError("edges and unary must be fp32")
        if not (edges.is_contiguous() and unary.is_contiguous() and candidate_ids.is_contiguous()):
            raise ValueError("edges, unary and candidate_ids must be contiguous")
        if top_k & (top_k - 1):
            raise ValueError(f"top_k {top_k} must be a power of two")
        if accepted_tokens.dtype != torch.int32 or accepted_tokens.stride(-1) != 1:
            raise ValueError("accepted_tokens must be int32 with unit column stride")
        if accepted_tokens.shape[1] != positions + 1:
            raise ValueError("accepted_tokens must have K + 1 columns")
        if num_accepted.dtype != torch.int32 or num_accepted.numel() != requests:
            raise ValueError("num_accepted must be int32 [B]")
        if batch_indices.dtype != torch.int32 or batch_indices.numel() < requests:
            raise ValueError("batch_indices must be int32 [>= B]")
        if (kv_lens is None) != (rewind is None):
            raise ValueError("kv_lens and rewind come together")
        if kv_lens is not None and (kv_lens.dtype != torch.int32 or rewind.dtype != torch.int32):
            raise ValueError("kv_lens and rewind must be int32")


draft_path_walk_tail = DraftPathWalkTail()
