# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The constant operands of ``glm_draft_ehproj_table``, built once from the checkpoint.

eh_proj is row-parallel over ``cat([enorm(embed(t)), hnorm(h)], -1)``, so its output splits into
``W[:, :H] @ enorm(embed(t))``, which depends on the draft token only, and ``W[:, H:] @ hnorm(h)``. Each rank
tabulates the first half for the tokens of its LM-head vocabulary shard and keeps its output rows of the second.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul

from ._draft_glue_rmsnorm import draft_glue_rmsnorm_cat

#: Table rows normalized and multiplied per block, and output columns per block (bounds the FP32 copies).
_ROW_BLOCK = 2048
_COL_BLOCK = 1536


def build_draft_token_table(
    eh_proj_weight: torch.Tensor,
    embed_table: torch.Tensor,
    enorm_weight: torch.Tensor,
    eps: float,
    tp_rank: int,
    tp_size: int,
    vocab_shard: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(token_table, hidden_rows)`` of rank ``tp_rank``.

    ``eh_proj_weight`` is the whole ``[H, 2H]`` BF16 eh_proj weight and ``embed_table`` the whole ``[V, H]`` BF16
    embedding. ``token_table[t] = bf16(W[:, :H] @ enorm(embed_table[tp_rank * vocab_shard + t]))``: enorm by the
    draft-step glue kernel (the arithmetic of draft step 0's eh_proj input), the product in FP32 without TF32,
    one rounding; a token past the table gives a zero row. ``hidden_rows`` is this rank's ``[H / tp_size, H]``
    slice ``W[tp_rank * H / tp : (tp_rank + 1) * H / tp, H:]``, the weight of the hidden-half GEMM.
    """
    hidden = embed_table.shape[1]
    share = hidden // tp_size
    device = embed_table.device
    hidden_rows = eh_proj_weight[tp_rank * share : (tp_rank + 1) * share, hidden:].contiguous()
    token_table = torch.empty((vocab_shard, hidden), dtype=torch.bfloat16, device=device)
    first = tp_rank * vocab_shard
    ids = torch.arange(first, first + vocab_shard, dtype=torch.int64, device=device)
    # The glue kernel's hidden-row operand is not read for a window inside the embedding half.
    unused_rows = torch.empty((_ROW_BLOCK, hidden), dtype=torch.bfloat16, device=device)
    with true_fp32_matmul():
        for row in range(0, vocab_shard, _ROW_BLOCK):
            n = min(_ROW_BLOCK, vocab_shard - row)
            normed = draft_glue_rmsnorm_cat(
                embed_table,
                ids[row : row + n],
                unused_rows[:n],
                enorm_weight,
                enorm_weight,
                eps,
                0,
                hidden,
                False,
            ).float()
            for lo in range(0, hidden, _COL_BLOCK):
                weight_t = eh_proj_weight[lo : lo + _COL_BLOCK, :hidden].float().t()
                token_table[row : row + n, lo : lo + _COL_BLOCK] = (normed @ weight_t).to(
                    torch.bfloat16
                )
    return token_table, hidden_rows
