# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-head sigmoid output gate with a per-head bias, in the activation dtype."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Block 0's gate: 48 heads of 128, bf16. Block 3's gate has the same geometry.
_BLOCK0 = dict(num_heads=48, head_dim=128, dtype=torch.bfloat16)


class BeamOutputGate(OpWrapper):
    """`attention_output * sigmoid(gate + gate_bias)`, all `[T, H, D]` but `gate_bias` `[H]`.

    `gate_bias` is broadcast over tokens and the head dimension. The add, the sigmoid
    and the product are separate activation-dtype operations, each rounded to that
    dtype, which is what the source model computes; an fp32 chain rounded once is a
    different result. Returns a new tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block 0, 1 token", spec=dict(_BLOCK0, tokens=1)),
        Cell(why="block 0, 17 tokens", spec=dict(_BLOCK0, tokens=17)),
        Cell(
            why="block 0, 129 tokens: the longest context the block accepts",
            spec=dict(_BLOCK0, tokens=129),
        ),
    )

    note = """
    The gate is the second q-width projection of the attention input, not a slice of
    q, and it is applied to the attention core's output before the output projection.
    Omitting the bias, or using one value for every head, still returns a plausible
    gated output; only a head-varying bias makes either mistake visible, which is why
    the block's tests use one.
    """

    def __call__(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
    ) -> torch.Tensor:
        return attention_output * torch.sigmoid(gate + gate_bias.unsqueeze(-1))

    def reference(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Each step in fp32, rounded to the activation dtype before the next."""
        dtype = attention_output.dtype
        biased = (gate.float() + gate_bias.float().unsqueeze(-1)).to(dtype)
        weight = torch.sigmoid(biased.float()).to(dtype)
        return (attention_output.float() * weight.float()).to(dtype)

    def is_valid(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
    ) -> None:
        # Broadcasting accepts a bias of the wrong length-1 shape and a gate of fewer
        # dims; mixed dtypes promote and change every rounding. None of these raise.
        assert attention_output.dim() == 3 and gate.shape == attention_output.shape, (
            "attention_output and gate must both be [T, heads, head_dim]"
        )
        assert gate_bias.shape == (attention_output.shape[1],), (
            f"gate_bias must be [heads] = [{attention_output.shape[1]}], "
            f"got {tuple(gate_bias.shape)}"
        )
        assert attention_output.dtype == gate.dtype == gate_bias.dtype, (
            "attention_output, gate and gate_bias must share one dtype"
        )


beam_output_gate = BeamOutputGate()
