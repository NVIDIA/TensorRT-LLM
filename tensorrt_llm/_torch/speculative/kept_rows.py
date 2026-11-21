# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rows an MTP-Eagle draft loop keeps from its first pass, known before that pass.

Pass 0 runs the draft layer over every row of the step (a context request's chunk, a generation request's 1 + k
verified tokens); the loop then keeps one row per request for the LM head and the next pass: a context request's last
row, a generation request's last accepted row. The other rows matter only through the K / V their attention appends.
A draft layer handed :class:`DraftKeptRows` runs attention on every row as before and its MLP (and what follows it)
on the kept rows only.
"""

from typing import List, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl


class DraftKeptRows(NamedTuple):
    """``ids``: int64 [b], the row each request keeps (the loop's step-0 gather ids). ``all_rank_counts``: under
    attention DP, every rank's kept-row count (the MoE's collectives take them); None with TP attention.
    ``compact``: the layer returns only the kept rows, in ``ids`` order, instead of scattering them back."""

    ids: torch.Tensor
    all_rank_counts: Optional[List[int]]
    compact: bool


def generation_only_on_every_rank(
    num_contexts: int,
    all_rank_num_seqs: Optional[List[int]],
    all_rank_num_gens: Optional[List[int]],
) -> bool:
    """Whether pass 0 may take kept rows: every rank's MoE collectives then run on every rank's kept-row count, so
    under attention DP (per-rank counts given) no rank may hold a context request; else this rank's batch decides."""
    if all_rank_num_seqs is None:
        return num_contexts == 0
    return all_rank_num_gens is not None and list(all_rank_num_gens) == list(all_rank_num_seqs)


# Batch counts are not specialized: an eager step at a new batch size reuses the kernel compiled for its BLOCK (one
# compile per power of two of the batch, from 16; one tile holds the whole batch for the cumsum).
@triton.jit(do_not_specialize=["num_contexts", "num_gens", "num_ctx_tokens"])
def _kept_row_ids_kernel(
    seq_lens_ptr,
    batch_idx_ptr,
    num_accepted_ptr,
    out_ptr,
    num_contexts,
    num_gens,
    num_ctx_tokens,
    K1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    is_ctx = i < num_contexts
    g = i - num_contexts
    is_gen = (i >= num_contexts) & (g < num_gens)
    lens = tl.load(seq_lens_ptr + i, mask=is_ctx, other=0).to(tl.int64)
    last_ctx = tl.cumsum(lens, axis=0) - 1
    bi = tl.load(batch_idx_ptr + g, mask=is_gen, other=0).to(tl.int64)
    acc = tl.load(num_accepted_ptr + i, mask=is_gen, other=0).to(tl.int64)
    ids = tl.where(is_ctx, last_ctx, bi * K1 + acc - 1 + num_ctx_tokens)
    tl.store(out_ptr + i, ids, mask=is_ctx | is_gen)


def kept_row_ids(
    seq_lens: torch.Tensor,
    batch_indices: torch.Tensor,
    num_accepted: torch.Tensor,
    num_contexts: int,
    num_gens: int,
    draft_len: int,
    num_ctx_tokens: int,
) -> torch.Tensor:
    """The draft loop's step-0 gather ids in one launch: contexts ``cumsum(seq_lens)[i] - 1``, generation requests
    ``batch_indices[g] * (draft_len + 1) + num_accepted[i] - 1 + num_ctx_tokens`` (int64)."""
    n = num_contexts + num_gens
    out = torch.empty((n,), dtype=torch.int64, device=num_accepted.device)
    if n:
        _kept_row_ids_kernel[(1,)](
            seq_lens,
            batch_indices,
            num_accepted,
            out,
            num_contexts,
            num_gens,
            num_ctx_tokens,
            K1=draft_len + 1,
            BLOCK=max(16, triton.next_power_of_2(n)),
            num_warps=1,
        )
    return out


@triton.jit(do_not_specialize=["a_stride", "b_stride"])
def _gather_rows2_kernel(
    a_ptr,
    b_ptr,
    ids_ptr,
    a_out_ptr,
    b_out_ptr,
    a_stride,
    b_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = cols < H
    src = tl.load(ids_ptr + r).to(tl.int64)
    tl.store(
        a_out_ptr + r.to(tl.int64) * H + cols,
        tl.load(a_ptr + src * a_stride + cols, mask=ok),
        mask=ok,
    )
    tl.store(
        b_out_ptr + r.to(tl.int64) * H + cols,
        tl.load(b_ptr + src * b_stride + cols, mask=ok),
        mask=ok,
    )


def gather_rows2(
    a: torch.Tensor, b: torch.Tensor, ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(a.index_select(0, ids), b.index_select(0, ids))``, in one launch for 2-D tensors of one width and dtype
    with unit column stride."""
    if (
        a.dim() != 2
        or b.dim() != 2
        or a.shape[1] != b.shape[1]
        or a.dtype != b.dtype
        or a.stride(1) != 1
        or b.stride(1) != 1
        or ids.dim() != 1
    ):
        return a.index_select(0, ids), b.index_select(0, ids)
    n, h = ids.numel(), a.shape[1]
    a_out = torch.empty((n, h), dtype=a.dtype, device=a.device)
    b_out = torch.empty((n, h), dtype=b.dtype, device=b.device)
    if n:
        block = min(1024, triton.next_power_of_2(h))
        _gather_rows2_kernel[(n, triton.cdiv(h, block))](
            a, b, ids, a_out, b_out, a.stride(0), b.stride(0), H=h, BLOCK=block, num_warps=4
        )
    return a_out, b_out


__all__ = ["DraftKeptRows", "gather_rows2", "generation_only_on_every_rank", "kept_row_ids"]
