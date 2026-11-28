# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels.fused_rope import apply_minimax_h3_qk_norm_rope_bf16 as _launch_qk_norm_rope


def apply_minimax_h3_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply MiniMax-H3's partial split-half RoPE to ``[B, S, H, D]`` (eager path)."""

    rotary_dim = cos.shape[-1]
    if rotary_dim > hidden_states.shape[-1] or rotary_dim % 2:
        raise ValueError(
            f"Invalid rotary dimension {rotary_dim} for head dimension {hidden_states.shape[-1]}."
        )

    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


class FusedQKNormRoPE(OpWrapper):
    """H3 per-head RMS normalization and split-half rotation of the leading channels."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "partial rotation with an odd token count",
            dict(batch=1, seq=257, heads=8, dim=128, rotary_dim=96),
        ),
    )

    def __call__(self, qkv, weight_q, weight_k, cos, sin, eps, num_heads, head_dim):
        return _launch_qk_norm_rope(qkv, weight_q, weight_k, cos, sin, eps, num_heads, head_dim)

    def reference(self, qkv, weight_q, weight_k, cos, sin, eps, num_heads, head_dim):
        width = num_heads * head_dim
        q, k = qkv[..., : 2 * width].split(width, dim=-1)

        def transform(t, weight):
            t = t.unflatten(-1, (num_heads, head_dim)).float()
            t = t * torch.rsqrt(t.square().mean(dim=-1, keepdim=True) + eps) * weight.float()
            return apply_minimax_h3_rotary_emb(t, cos.float(), sin.float()).flatten(2).to(qkv.dtype)

        return transform(q, weight_q), transform(k, weight_k)


apply_minimax_h3_qk_norm_rope_bf16 = FusedQKNormRoPE()
