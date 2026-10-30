# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Position-dependent query scale of a full-attention layer without position embedding."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Block 5's full-attention layer: 48 q heads of 128, bf16.
_BLOCK5 = dict(num_heads=48, head_dim=128, dtype=torch.bfloat16)


class BeamNopeQueryScale(OpWrapper):
    """Scale `q` `[T, H, D]` per token: `q * exp(0.1 * log1p((position + 1) / 2048))`.

    The scale is computed in fp32 from the token's position, the product is the
    activation-dtype `q` promoted against that fp32 scale, and the result is rounded
    once to `q`'s dtype. It replaces the rotation of a full-attention layer that has no
    position embedding: it applies to the normalized `q` only, never to `k` or `v`,
    and it is not the attention's `1 / sqrt(D)` softmax scale. Returns a new tensor.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="block 5, 1 token: position 0 alone", spec=dict(_BLOCK5, tokens=1)),
        Cell(why="block 5, 17 tokens: positions 0..16", spec=dict(_BLOCK5, tokens=17)),
        Cell(
            why="block 5, 129 tokens: the longest context the block accepts",
            spec=dict(_BLOCK5, tokens=129),
        ),
    )

    note = """
    The scale grows from 1.00005 at position 0 to 1.0063 at position 128, so most
    bf16 query values are unchanged by it at these lengths: only values close to a
    rounding boundary move, and position 0 moves almost none. A test that has to see
    the scale therefore compares the rounded result exactly at nonzero positions.
    """

    def __call__(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))
        return (q * scale[:, None, None]).to(q.dtype)

    def reference(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Each step in fp32, the product of the widened `q` and the scale, one cast."""
        position = positions.to(torch.float32)
        ratio = torch.div(position + 1.0, 2048.0)
        scale = torch.exp(torch.mul(torch.log1p(ratio), 0.1))
        return (q.to(torch.float32) * scale.view(-1, 1, 1)).to(q.dtype)

    def is_valid(self, q: torch.Tensor, positions: torch.Tensor) -> None:
        # A [1] positions tensor broadcasts one scale over every token, and a float
        # tensor scales by fractional positions. Neither raises.
        assert q.dim() == 3, "q must be [T, heads, head_dim]"
        assert positions.shape == (q.shape[0],), (
            f"positions must be [T] with T = {q.shape[0]} tokens, got {tuple(positions.shape)}"
        )
        assert not positions.is_floating_point(), "positions must be integer token positions"


beam_nope_query_scale = BeamNopeQueryScale()
