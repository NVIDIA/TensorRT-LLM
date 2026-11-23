# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map request-local KV positions to rows of the primary latent pool (viewed as `[-1, 576]`)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Primary-pool pages of 32 tokens; 2048 selected keys on a sparse layer.
_HY4 = dict(block_size=32, topk=2048)


class ConvertReqIndexToGlobal(OpWrapper):
    """`block_table[req, p // block_size] * stride_factor + layer_id * block_size + p % block_size`, or -1.

    -1 when `p < 0`, `p // block_size >= block_table.size(1)`, or the block-table entry is negative.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one sequence, one token, a context longer than top-k",
            spec=dict(**_HY4, seq_rows=[1], kv_lens=[5120], pool_layers=42, layer_id=0),
        ),
        Cell(
            why="the full decode batch: 64 sequences, one token, the last pool layer",
            spec=dict(
                **_HY4,
                seq_rows=[1] * 64,
                kv_lens=[300 + 73 * i for i in range(64)],
                pool_layers=42,
                layer_id=41,
            ),
        ),
        Cell(
            why="windowed layers: 512 columns per row, a decode batch past the window",
            spec=dict(
                _HY4, topk=512, seq_rows=[1] * 8, kv_lens=[5120] * 8, pool_layers=42, layer_id=2
            ),
        ),
        Cell(
            why="contexts shorter than top-k: most columns are -1 padding",
            spec=dict(**_HY4, seq_rows=[2, 2], kv_lens=[33, 1000], pool_layers=42, layer_id=5),
        ),
        Cell(
            why="prefill chunk: 4096 rows of one request, causal windows",
            spec=dict(**_HY4, seq_rows=[4096], kv_lens=[4096], pool_layers=42, layer_id=13),
        ),
        Cell(
            why="padded block table and positions past it: both must come back -1",
            spec=dict(
                **_HY4,
                seq_rows=[1, 1],
                kv_lens=[5000, 64],
                pool_layers=42,
                layer_id=1,
                pad_block_table=True,
            ),
        ),
    )

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.convert_req_index_to_global(
            req_id,
            block_table,
            token_indices,
            block_size,
            num_topk_tokens,
            stride_factor,
            layer_id,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        """int64, so it cannot wrap; covers every column of `token_indices`."""
        tok = token_indices.long()
        max_blocks = block_table.shape[1]
        block = torch.div(tok, block_size, rounding_mode="floor")
        in_block = tok - block * block_size + layer_id * block_size
        valid = (tok >= 0) & (block < max_blocks)
        rows = req_id.long().unsqueeze(1).expand_as(tok)
        base = block_table.long()[rows, block.clamp(0, max_blocks - 1)]
        valid &= base >= 0
        out = torch.where(valid, base * stride_factor + in_block, torch.full_like(tok, -1))
        return out.to(torch.int32)

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        # The kernel walks `num_topk_tokens` columns at row stride `token_indices.size(1)`; a mismatch corrupts rows.
        assert num_topk_tokens == token_indices.shape[1], (
            f"num_topk_tokens must equal token_indices.size(1); got {num_topk_tokens} "
            f"against a {tuple(token_indices.shape)} operand"
        )
        # The kernel's row count is `req_id.size(0)`.
        assert req_id.shape[0] == token_indices.shape[0], (
            f"req_id must hold exactly one entry per row of token_indices; got "
            f"{req_id.shape[0]} for {token_indices.shape[0]} rows"
        )
        # int32 arithmetic on the device: a pool row past 2**31 wraps silently.
        if block_table.numel():
            top = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
            assert top < 2**31, f"pool row {top} does not fit int32; the kernel wraps it silently"


convert_req_index_to_global = ConvertReqIndexToGlobal()
