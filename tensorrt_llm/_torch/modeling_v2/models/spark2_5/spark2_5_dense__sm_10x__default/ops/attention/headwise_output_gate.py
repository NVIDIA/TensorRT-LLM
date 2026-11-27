# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spark's headwise attention output gate: one sigmoid scalar per query head,
broadcast over the head's dims, in the activation dtype."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The published checkpoints' attention geometry: head_dim is 256 in both.
_GEOM_1_7B = dict(num_heads=8, head_dim=256, dtype=torch.bfloat16)
_GEOM_4B = dict(num_heads=16, head_dim=256, dtype=torch.bfloat16)


class HeadwiseOutputGate(OpWrapper):
    """`attention_output * sigmoid(gate)`, gate broadcast per head.

    `attention_output` is the packed `[T, num_heads * head_dim]` kernel
    output; `gate` is the `[T, num_heads]` raw g_proj logit. The sigmoid runs
    in fp32 and is cast back before the product, matching the HF reference.
    Returns a new tensor.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="1.7B decode: one token", spec=dict(_GEOM_1_7B, tokens=1)),
        Cell(why="1.7B odd token count", spec=dict(_GEOM_1_7B, tokens=17)),
        Cell(why="1.7B prefill-shaped", spec=dict(_GEOM_1_7B, tokens=2048)),
        Cell(why="4B decode: one token", spec=dict(_GEOM_4B, tokens=1)),
        Cell(why="4B prefill-shaped", spec=dict(_GEOM_4B, tokens=2048)),
    )

    note = """
    Applied to the attention output before the output projection. Native
    torch: one sigmoid, one broadcast multiply. Metadata consumed: none.
    """

    def __call__(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        weight = torch.sigmoid(gate.float()).to(attention_output.dtype).unsqueeze(-1)
        shape = attention_output.shape
        return (attention_output.view(*shape[:-1], num_heads, head_dim) * weight).view(shape)

    def reference(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """fp32 sigmoid, one rounding to the activation dtype, fp32 product
        rounded once -- the reference is the more accurate side."""
        weight = torch.sigmoid(gate.float())
        out = attention_output.float().view(*attention_output.shape[:-1], num_heads, head_dim)
        return (out * weight.unsqueeze(-1)).view(attention_output.shape).to(attention_output.dtype)

    def is_valid(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> None:
        # Broadcasting accepts a gate of the wrong head count when it happens
        # to divide the output width, and a packed-vs-[T,H,D] mix-up reshapes
        # silently; neither raises.
        assert attention_output.dim() == 2, (
            "attention_output must be packed [T, num_heads*head_dim]"
        )
        assert attention_output.shape[-1] == num_heads * head_dim, (
            f"output width {attention_output.shape[-1]} != {num_heads} * {head_dim}"
        )
        assert gate.shape == (attention_output.shape[0], num_heads), (
            f"gate must be [T, num_heads] = [{attention_output.shape[0]}, {num_heads}], "
            f"got {tuple(gate.shape)}"
        )
        assert attention_output.dtype == gate.dtype, (
            "attention_output and gate must share one dtype"
        )


headwise_output_gate = HeadwiseOutputGate()
