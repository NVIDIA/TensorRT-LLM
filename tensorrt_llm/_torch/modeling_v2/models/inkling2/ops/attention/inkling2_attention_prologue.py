# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What Inkling 2's attention does to its projections before the attention core.

Per token and head, each step in fp32 and rounded to bf16 where the source model rounds:

    q = l2clip(q * q_scale, 22.627)            then on a global layer  q = q * tau
    k = l2clip(k, 22.627)                      (k after its short convolution)
    v = v / rms(v)                             (eps 1e-6, no weight; v after its convolution)
    r = l2clip(r * r_scale, 8)                 then on a global layer  r = r * tau

with `l2clip(x, c) = x * min(c / (|x| + 1e-6), 1)` over the head dimension, and the global
layers' log scaling `tau = 1 + alpha * ln(max((position + 1) / n_floor, 1))` per token. A
sliding layer has no tau.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_TP4 = dict(heads=12, kv_heads=2, head_dim=128, d_rel=16)


def log_scaling(positions: torch.Tensor, alpha: float, n_floor: float) -> torch.Tensor:
    """The global layers' per-token tau, fp32 `[T]`."""
    return 1.0 + alpha * torch.log(((positions.float() + 1.0) / n_floor).clamp(min=1.0))


def _clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


class Inkling2AttentionPrologue(OpWrapper):
    """`q` `[T, H, 128]`, `k`, `v` `[T, KV, 128]`, `r` `[T, H, 16]` bf16, `q_scale` `[H, 128]`,
    `r_scale` `[H, 16]`, `tau` `[T]` fp32 or None -> (q, k, v, r), new bf16 tensors."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="sliding layer at TP4, 33 tokens", spec=dict(_TP4, tokens=33, tau=False)),
        Cell(why="global layer at TP4, tokens past n_floor", spec=dict(_TP4, tokens=33, tau=True)),
    )

    note = """
    The clip thresholds are 22.627 for q and k and 8 for r; tau multiplies q and r after their
    clips, so a clipped q can exceed 22.627 on a global layer. tau is 1 below position 8191:
    only positions past n_floor show it.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        q_scale: torch.Tensor,
        r_scale: torch.Tensor,
        tau: Optional[torch.Tensor],
        qk_threshold: float,
        rel_threshold: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = q.dtype
        q = _clip(q.float() * q_scale.float(), qk_threshold).to(dtype)
        k = _clip(k.float(), qk_threshold).to(dtype)
        vf = v.float()
        v = (vf * torch.rsqrt(vf.square().mean(-1, keepdim=True) + eps)).to(dtype)
        r = _clip(r.float() * r_scale.float(), rel_threshold).to(dtype)
        if tau is not None:
            t = tau.view(-1, 1, 1)
            q = (q.float() * t).to(dtype)
            r = (r.float() * t).to(dtype)
        return q, k, v, r

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        q_scale: torch.Tensor,
        r_scale: torch.Tensor,
        tau: Optional[torch.Tensor],
        qk_threshold: float,
        rel_threshold: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same steps in fp64, rounded to bf16 at the same points."""
        dtype = q.dtype
        q = _clip(q.double() * q_scale.double(), qk_threshold).to(dtype)
        k = _clip(k.double(), qk_threshold).to(dtype)
        vd = v.double()
        v = (vd * torch.rsqrt(vd.square().mean(-1, keepdim=True) + eps)).to(dtype)
        r = _clip(r.double() * r_scale.double(), rel_threshold).to(dtype)
        if tau is not None:
            t = tau.double().view(-1, 1, 1)
            q = (q.double() * t).to(dtype)
            r = (r.double() * t).to(dtype)
        return q, k, v, r

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        r: torch.Tensor,
        q_scale: torch.Tensor,
        r_scale: torch.Tensor,
        tau: Optional[torch.Tensor],
        qk_threshold: float,
        rel_threshold: float,
        eps: float,
    ) -> None:
        assert q.dim() == k.dim() == v.dim() == r.dim() == 3, "q, k, v, r are [T, heads, dim]"
        assert q_scale.shape == q.shape[1:] and r_scale.shape == r.shape[1:], "per-head scales"
        assert tau is None or (tau.dtype == torch.float32 and tau.shape == (q.shape[0],)), (
            "tau is fp32 [T]"
        )


inkling2_attention_prologue = Inkling2AttentionPrologue()
