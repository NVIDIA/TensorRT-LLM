# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA top-k block selection from per-KV-head block scores (the built-in selector)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class M3SelectBlocks(OpWrapper):
    """Per query row the ``topk`` highest-scoring KV blocks of ``max_score_kv`` [kv_heads, n_blocks, total_q] (fp32),
    with the first ``init_blocks`` and the last ``local_blocks`` valid blocks forced in and blocks at or beyond the
    row's ``n_valid_blocks`` excluded; [total_q, kv_heads, topk] int32 ascending ids, -1 padded. ``head_major_output``
    backs the result head-major so ``permute(1, 0, 2)`` is contiguous.

    The decode dispatch runs it where the long-row selector does not apply: requests of 17..128 KV blocks.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="eight decode rows of 17..128 blocks",
            spec=dict(total_q=8, n_blocks=128, mixed=True),
        ),
        Cell(
            why="one decode row at the 128-block edge",
            spec=dict(total_q=1, n_blocks=128, mixed=False),
        ),
        Cell(why="a 4-row verify at 40 blocks", spec=dict(total_q=4, n_blocks=40, mixed=False)),
    )

    note = """
    Exact: ties between equal scores break toward the lower block id; `reference` sorts with the same rule.
    """

    def __call__(
        self,
        max_score_kv: torch.Tensor,
        topk: int,
        n_valid_blocks: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        head_major_output: bool,
    ) -> torch.Tensor:
        nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
        return torch.ops.trtllm.minimax_m3_select_blocks(
            max_score_kv, nvb, topk, init_blocks, local_blocks, head_major_output
        )

    def reference(
        self,
        max_score_kv: torch.Tensor,
        topk: int,
        n_valid_blocks: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        head_major_output: bool,
    ) -> torch.Tensor:
        """Plain torch selection; ``head_major_output`` changes the backing only."""
        heads, n_blocks, total_q = max_score_kv.shape
        out = torch.full((total_q, heads, topk), -1, dtype=torch.int32, device=max_score_kv.device)
        blocks = torch.arange(n_blocks, device=max_score_kv.device)
        for t in range(total_q):
            nv = int(n_valid_blocks[t])
            for h in range(heads):
                s = max_score_kv[h, :, t].clone().double()
                s[blocks >= nv] = float("-inf")
                forced = (blocks < min(init_blocks, nv)) | (
                    (blocks >= nv - local_blocks) & (blocks < nv)
                )
                s[forced] = float("inf")
                order = sorted(range(nv), key=lambda b: (-float(s[b]), b))[: min(topk, nv)]
                ids = torch.tensor(sorted(order), dtype=torch.int32, device=out.device)
                out[t, h, : ids.numel()] = ids
        return out


m3_select_blocks = M3SelectBlocks()
