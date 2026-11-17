# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One kernel for MSA's device-side KV-length correction (MsaAttentionMetadata.on_update_kv_lens).

The torch formulation is ~23 one-CTA kernels (min, cumsum, gathers, div / clamp / remainder, the block-table lookup,
the valid-block counts) and runs at the head of every captured decode step, twice. This kernel computes the same
int32 results in one launch:

    kv_true[b]  = min(kv_lens[b], staged[b])               -> seq_lens_out[b], cu_kv[b + 1] = prefix sum (cu_kv[0] = 0)
    pos[t]      = kv_true[r] - qo[r] + intra[t]            (r = q_batch_row[t])
    out_loc[t]  = table[r, min(max(pos, 0) // page, ncols - 1)] * page + max(pos, 0) % page
    n_valid[t]  = (max(pos + 1, 0) + page - 1) // page

``msa_kv_update`` is the target's ``MsaKvUpdate`` (``tensorrt_llm._torch.pyexecutor.runtime_hooks``): the MSA
metadata calls it with its own buffers and falls back to the torch formulation when it returns False. The kernel is
``torch.ops.trtllm.minimax_m3_msa_kv_update``, called through this target's ``msa_kv_update`` entry.
"""

from typing import Optional

import torch

from ..ops.attention.msa_kv_update import msa_kv_update as _kv_update_op


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
    """The fused correction (see module doc); False, with nothing written, when an operand is outside what the
    kernel covers. Arguments in ``MsaKvUpdate`` order; ``seq_lens`` are the per-request query lengths (``qo``)."""
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
    _kv_update_op(
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
