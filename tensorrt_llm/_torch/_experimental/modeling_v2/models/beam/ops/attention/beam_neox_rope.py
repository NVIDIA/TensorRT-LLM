# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeoX half-rotation RoPE on per-head q and k, in fp32 with one rounding per tensor."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Block 0's sliding-attention layer: 48 q heads and 8 kv heads of 128, theta 10000, factor 1.
#: Block 3's sliding-attention layer has the same geometry and RoPE parameters.
_BLOCK0 = dict(
    num_heads=48, num_kv_heads=8, head_dim=128, theta=10000.0, factor=1.0, dtype=torch.bfloat16
)


def _cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 `[T, 1, head_dim]` cos and sin, both halves carrying the same angles."""
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (theta ** (exponent / head_dim)) / factor
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().unsqueeze(-2), emb.sin().unsqueeze(-2)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    first, second = x32.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (x32 * cos + rotated * sin).to(x.dtype)


class BeamNeoxRope(OpWrapper):
    """Rotate `q` `[T, Hq, D]` and `k` `[T, Hkv, D]` by their token's position.

    Frequencies are `1 / theta ** (2i / D) / factor` for `i < D / 2`; angles are
    `position * frequency` in fp32, and the two halves of the head dimension rotate
    together -- `x * cos + rotate_half(x) * sin`, NeoX order, not interleaved pairs.
    Each output is rounded once to its input's dtype. `q` and `k` arrive already
    normalized and rounded by their per-head RMSNorm; this op does not normalize, and
    `v` never passes through it. Returns two new tensors.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The three fresh-context lengths the dense block runs: a single token (position 0,
    # the identity rotation), a short and a long context.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="block 0, 1 token: position 0 is the identity rotation",
            spec=dict(_BLOCK0, tokens=1),
        ),
        Cell(why="block 0, 17 tokens: positions 0..16", spec=dict(_BLOCK0, tokens=17)),
        Cell(
            why="block 0, 129 tokens: the longest context the block accepts",
            spec=dict(_BLOCK0, tokens=129),
        ),
    )

    note = """
    Native torch, not a fused kernel: the source model rounds the per-head norm
    output to the activation dtype before promoting it again for the rotation, and
    no fused norm-plus-RoPE op in this tree has that intermediate rounding (the
    catalog's fused_qk_norm_rope keeps the normalized values in fp32 through the
    rotation). `q` and `k` share one cos/sin evaluation.

    `positions` is per token, not per sequence: a fresh context passes
    `arange(T)`. No scaling other than the linear `factor` is modelled.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = _cos_sin(positions, q.shape[-1], theta, factor)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rotation written per half: `(x1 c - x2 s, x2 c + x1 s)`, fp32, one cast."""
        half = q.shape[-1] // 2
        index = torch.arange(half, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (theta ** (2.0 * index / q.shape[-1])) / factor
        angle = positions.float()[:, None] * inv_freq[None, :]
        c, s = angle.cos()[:, None, :], angle.sin()[:, None, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.float()[..., :half], x.float()[..., half:]
            return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)

        return rotate(q), rotate(k)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
        factor: float,
    ) -> None:
        # A [1] or [B, 1] positions tensor broadcasts across every token and rotates
        # them all by one angle; a float tensor is accepted and rotates by fractional
        # positions. Neither raises.
        assert q.dim() == 3 and k.dim() == 3, "q and k must be [T, heads, head_dim]"
        assert positions.shape == (q.shape[0],) and k.shape[0] == q.shape[0], (
            f"positions must be [T] with T = {q.shape[0]} tokens; got {tuple(positions.shape)} "
            f"for k of {k.shape[0]} tokens"
        )
        assert not positions.is_floating_point(), "positions must be integer token positions"
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0, (
            "q and k must share an even head_dim"
        )


beam_neox_rope = BeamNeoxRope()
