# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA's device-side KV-length correction in one launch (MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens).

The torch formulation is ~23 one-CTA kernels (min, cumsum, gathers, div / clamp / remainder, the block-table lookup,
the valid-block counts) and runs at the head of every captured speculative decode step, twice. The
``trtllm::minimax_m3_msa_kv_update`` kernel computes the same int32 results in one launch:

    kv_true[b]  = min(kv_lens[b], staged[b])               -> seq_lens_out[b], cu_kv[b + 1] = prefix sum (cu_kv[0] = 0)
    pos[t]      = kv_true[r] - qo[r] + intra[t]            (r = q_batch_row[t])
    out_loc[t]  = table[r, min(max(pos, 0) // page, ncols - 1)] * page + max(pos, 0) % page
    n_valid[t]  = (max(pos + 1, 0) + page - 1) // page

``msa_kv_update`` takes a step when the kernel is built in and every operand is one it reads correctly, and returns
False, having written nothing, otherwise: the metadata then runs its torch formulation. ``reference`` is that
formulation as a pure function, for tests.
"""

from typing import Optional, Tuple

import torch

_OP_NAME = "minimax_m3_msa_kv_update"


def kernel_available() -> bool:
    """Whether this build registers the fused kernel (``thop/minimaxM3/minimaxM3MsaKvUpdateOp.cpp``)."""
    return hasattr(torch.ops.trtllm, _OP_NAME)


def msa_kv_update(
    kv_lens: torch.Tensor,
    kv_lens_staged: torch.Tensor,
    seq_lens: torch.Tensor,
    q_batch_row: torch.Tensor,
    q_intra: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    batch: int,
    total_q: int,
    msa_seq_lens: torch.Tensor,
    cu_kv_lens: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    n_valid_blocks: Optional[torch.Tensor],
) -> bool:
    """The fused correction (see module doc); False, with nothing written, when it cannot take the step.

    ``seq_lens`` are the per-request query lengths (``qo``). The kernel reads every operand but the block table as a
    dense int32 run from its first element and does not bounds-check the request row or block-table column it
    derives; the metadata's own staging keeps both in range.
    """
    if not kernel_available():
        return False
    ts = [
        kv_lens,
        kv_lens_staged,
        seq_lens,
        q_batch_row,
        q_intra,
        block_table,
        msa_seq_lens,
        out_cache_loc,
    ]
    ts += [t for t in (cu_kv_lens, n_valid_blocks) if t is not None]
    if (
        any(t.dtype != torch.int32 or not t.is_cuda for t in ts)
        or block_table.dim() != 2
        or block_table.stride(1) != 1
    ):
        return False
    if any(not t.is_contiguous() for t in ts if t is not block_table):
        return False
    if (
        kv_lens.numel() < batch
        or kv_lens_staged.numel() < batch
        or seq_lens.numel() < batch
        or msa_seq_lens.numel() < batch
        or q_batch_row.numel() < total_q
        or q_intra.numel() < total_q
        or out_cache_loc.numel() < total_q
        or (cu_kv_lens is not None and cu_kv_lens.numel() < batch + 1)
        or (n_valid_blocks is not None and n_valid_blocks.numel() < total_q)
    ):
        return False
    getattr(torch.ops.trtllm, _OP_NAME)(
        kv_lens,
        kv_lens_staged,
        seq_lens,
        q_batch_row,
        q_intra,
        block_table,
        tokens_per_block,
        batch,
        total_q,
        msa_seq_lens,
        cu_kv_lens,
        out_cache_loc,
        n_valid_blocks,
    )
    return True


def reference(
    kv_lens: torch.Tensor,
    kv_lens_staged: torch.Tensor,
    seq_lens: torch.Tensor,
    q_batch_row: torch.Tensor,
    q_intra: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    batch: int,
    total_q: int,
    cu_kv: bool,
    n_valid: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """``(msa_seq_lens[:batch], cu_kv_lens[:batch + 1], out_cache_loc[:total_q], n_valid_blocks[:total_q])`` as the
    kernel writes them: the metadata's torch formulation, returning new tensors. ``cu_kv`` / ``n_valid`` say which
    optional outputs exist (None otherwise)."""
    page = int(tokens_per_block)
    kv_true = torch.minimum(kv_lens[:batch], kv_lens_staged[:batch])
    cu = None
    if cu_kv:
        cu = torch.zeros(batch + 1, dtype=torch.int32, device=kv_lens.device)
        cu[1:] = torch.cumsum(kv_true, 0)
    qbr = q_batch_row[:total_q].to(torch.long)
    qo = seq_lens[:batch]
    pos = kv_true[qbr] - qo[qbr] + q_intra[:total_q]
    pos_slot = pos.clamp_min(0)
    block_col = torch.div(pos_slot, page, rounding_mode="floor").clamp(
        max=int(block_table.shape[1]) - 1
    )
    slots = block_table[qbr, block_col.to(torch.long)] * page + torch.remainder(pos_slot, page)
    counts = None
    if n_valid:
        counts = torch.div((pos + 1).clamp_min(0) + (page - 1), page, rounding_mode="floor").to(
            torch.int32
        )
    return kv_true.to(torch.int32), cu, slots.to(torch.int32), counts
