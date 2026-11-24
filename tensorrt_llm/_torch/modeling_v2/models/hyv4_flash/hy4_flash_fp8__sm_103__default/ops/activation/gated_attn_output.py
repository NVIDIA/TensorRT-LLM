# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated attention epilogue before o_proj: ``attention_output *= sigmoid(gate)`` in place."""

from __future__ import annotations

import torch

# Registers torch.ops.trtllm.fused_sigmoid_mul_inplace; tensorrt_llm._torch.custom_ops does not.
import tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate  # noqa: F401
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_SPARSE = 32 * 256
_SPARSE_TP2 = 16 * 256
_SWA = 64 * 256


class GatedAttnOutput(OpWrapper):
    """`attention_output[t, i] *= sigmoid(gate[t, i])` in place, both `[T, H * D]`; returns `attention_output`."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sparse decode, one token: the attention epilogue before o_proj at 8192 wide",
            spec=dict(rows=1, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse verify and drafter: 6 rows (five drafts plus the anchor)",
            spec=dict(rows=6, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse decode at the widest captured batch: 64 rows",
            spec=dict(rows=64, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse verify at 64 requests: 6 x 64 = 384 rows",
            spec=dict(rows=384, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse chunked prefill: one 512-token chunk",
            spec=dict(rows=512, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse prefill at the envelope's largest row count: 5120 rows",
            spec=dict(rows=5120, width=_SPARSE, dtype=torch.bfloat16),
        ),
        Cell(
            why="sparse decode at attn_tp 2, one token: the attention epilogue before o_proj at 4096 wide",
            spec=dict(rows=1, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="the widest captured decode batch at attn_tp 2: 64 rows at 4096 wide",
            spec=dict(rows=64, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="a prefill chunk at attn_tp 2: 512 rows at 4096 wide",
            spec=dict(rows=512, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="the envelope's largest row count at attn_tp 2: 5120 rows at 4096 wide",
            spec=dict(rows=5120, width=_SPARSE_TP2, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA decode, one token: the same gate at 16384 wide",
            spec=dict(rows=1, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA verify: 6 rows at 16384 wide",
            spec=dict(rows=6, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA decode at the widest captured batch: 64 rows at 16384 wide",
            spec=dict(rows=64, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA verify at 64 requests: 384 rows at 16384 wide",
            spec=dict(rows=384, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA chunked prefill: one 512-token chunk at 16384 wide",
            spec=dict(rows=512, width=_SWA, dtype=torch.bfloat16),
        ),
        Cell(
            why="SWA prefill at the envelope's largest row count: 5120 rows at 16384 wide",
            spec=dict(rows=5120, width=_SWA, dtype=torch.bfloat16),
        ),
    )

    note = """
    No `is_valid`: the op's own bare asserts refuse a non-unit last stride and a mismatched gate shape.
    """

    def __call__(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.fused_sigmoid_mul_inplace(attention_output, gate)
        return attention_output

    def reference(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """What `attention_output` holds after the call; neither input is mutated."""
        num_tokens, hidden = attention_output.shape
        g = gate.reshape(num_tokens, hidden).float()
        return (attention_output.float() * torch.sigmoid(g)).to(attention_output.dtype)


gated_attn_output = GatedAttnOutput()
