# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels.text_qk_norm_rope import apply_text_qk_norm_rope


class TextQkNormRope(OpWrapper):
    """In-place plain-gain Q/K RMSNorm and NeoX RoPE for one Qwen3-VL text prompt."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "Compiled TP2 text encoder",
            dict(
                tokens=73,
                fullgraph=True,
                num_heads_q=32,
                num_heads_kv=4,
                head_dim=128,
                rotary_dim=128,
                eps=1e-6,
                base=5e6,
            ),
        ),
        Cell(
            "Partial YaRN rotation on a replicated KV head",
            dict(
                tokens=19,
                num_heads_q=2,
                num_heads_kv=1,
                head_dim=64,
                rotary_dim=32,
                eps=1e-6,
                base=1e4,
                factor=4.0,
                low=2.0,
                high=10.0,
                attention_factor=1.2,
            ),
        ),
        Cell(
            "Partial YaRN rotation spanning twelve lane pairs",
            dict(
                tokens=17,
                num_heads_q=2,
                num_heads_kv=1,
                head_dim=128,
                rotary_dim=96,
                eps=1e-6,
                base=1e4,
                factor=3.0,
                low=15.0,
                high=27.0,
                attention_factor=1.1,
            ),
        ),
        Cell(
            "Single wide head",
            dict(
                tokens=1,
                num_heads_q=1,
                num_heads_kv=1,
                head_dim=256,
                rotary_dim=256,
                eps=1e-6,
                base=1e4,
            ),
        ),
    )

    def __call__(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        positions: torch.Tensor,
        *,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        base: float,
        factor: float = 1.0,
        low: float = 0.0,
        high: float = 0.0,
        attention_factor: float = 1.0,
    ) -> torch.Tensor:
        apply_text_qk_norm_rope(
            qkv,
            q_weight,
            k_weight,
            positions,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            base=base,
            factor=factor,
            low=low,
            high=high,
            attention_factor=attention_factor,
        )
        return qkv

    def reference(
        self,
        qkv,
        q_weight,
        k_weight,
        positions,
        *,
        num_heads_q,
        num_heads_kv,
        head_dim,
        rotary_dim,
        eps,
        base,
        factor=1.0,
        low=0.0,
        high=0.0,
        attention_factor=1.0,
    ):
        output = qkv.clone()
        x = output.view(qkv.shape[0], num_heads_q + 2 * num_heads_kv, head_dim)
        dims = torch.arange(rotary_dim, device=x.device)
        half = rotary_dim // 2
        pair = (dims % half).float()
        freq = base ** (-2 * pair / rotary_dim)
        if factor != 1.0:
            if abs(high - low) <= 1e-6:
                high += 0.001
            ramp = ((pair - low) / (high - low)).clamp(0, 1)
            freq = (freq / factor) * ramp + freq * (1 - ramp)
        angle = positions.float()[:, None, None] * freq
        cos, sin = angle.cos(), angle.sin()
        for start, count, gain in (
            (0, num_heads_q, q_weight),
            (num_heads_q, num_heads_kv, k_weight),
        ):
            head = x[:, start : start + count].float()
            head = head * (torch.rsqrt(head.square().mean(-1, keepdim=True) + eps) * gain.float())
            rotated = head[..., :rotary_dim] * cos + head.index_select(
                -1, dims ^ half
            ) * torch.where(dims < half, -sin, sin)
            x[:, start : start + count] = torch.cat(
                [rotated * attention_factor, head[..., rotary_dim:]], dim=-1
            ).to(qkv.dtype)
        return output


text_qk_norm_rope = TextQkNormRope()
