# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-sample FLUX 3 forward in plain PyTorch, following Diffusers ``Flux3Transformer2DModel``.

Uses no TRT-LLM module; dtypes follow Diffusers (model dtype, fp32 QK-norm and RoPE).
"""

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _timestep_proj(t: Tensor, channels: int) -> Tensor:
    """Diffusers ``Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`` of scaled ``t``."""
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    args = t[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _rope(ids: Tensor, axes: list[int], theta: float) -> tuple[Tensor, Tensor]:
    """Diffusers ``Flux3PosEmbed``: interleaved fp32 (cos, sin), each [S, head_dim]."""
    cos, sin = [], []
    for axis, dim in enumerate(axes):
        freqs = 1.0 / theta ** (
            torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim
        )
        angles = ids[:, axis].double()[:, None] * freqs[None, :]
        cos.append(angles.cos().repeat_interleave(2, dim=1).float())
        sin.append(angles.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _apply_rope(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    """Diffusers ``apply_rotary_emb(use_real=True, sequence_dim=1)`` on x [B, S, H, D]."""
    cos, sin = (freqs[None, :, None, :] for freqs in rope)
    real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imag, real], dim=-1).flatten(3)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class Flux3Reference:
    """Diffusers FLUX 3 math over a Diffusers-named state dict."""

    def __init__(self, weights: dict[str, Tensor], config: dict[str, Any]):
        self.weights = weights
        self.heads = config["num_attention_heads"]
        self.head_dim = config["attention_head_dim"]
        self.eps = config["eps"]
        self.axes = list(config["axes_dims_rope"])
        self.theta = config["rope_theta"]
        self.timestep_channels = config["timestep_channels"]
        self.num_layers = config["num_layers"]
        self.num_single_layers = config["num_single_layers"]

    def linear(self, name: str, x: Tensor) -> Tensor:
        return F.linear(x, self.weights[f"{name}.weight"])

    def layer_norm(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (x.shape[-1],), eps=self.eps)

    def embed(self, prefix: str, t: Tensor, dtype: torch.dtype) -> Tensor:
        temb = _timestep_proj(t, self.timestep_channels).to(dtype)
        return self.linear(f"{prefix}.linear_2", F.silu(self.linear(f"{prefix}.linear_1", temb)))

    def attention(self, prefix: str, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        q, k, v = (
            self.linear(f"{prefix}.to_{name}", x).unflatten(-1, (self.heads, -1))
            for name in ("q", "k", "v")
        )
        norm_q, norm_k = (self.weights[f"{prefix}.norm_{name}.weight"].float() for name in "qk")
        q = F.rms_norm(q.float(), (self.head_dim,), norm_q, self.eps)
        k = F.rms_norm(k.float(), (self.head_dim,), norm_k, self.eps)
        q, k = _apply_rope(q, rope).to(v.dtype), _apply_rope(k, rope).to(v.dtype)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.linear(f"{prefix}.to_out.0", out.transpose(1, 2).flatten(2))

    def feed_forward(self, prefix: str, x: Tensor) -> Tensor:
        gate, up = self.linear(f"{prefix}.linear_in", x).chunk(2, dim=-1)
        return self.linear(f"{prefix}.linear_out", F.silu(gate) * up)

    def mode_block(
        self, prefix: str, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor]
    ) -> Tensor:
        shift, scale, gate = mod.chunk(3, dim=-1)
        h = (1 + scale) * self.layer_norm(x) + shift
        return x + gate * (
            self.attention(f"{prefix}.attn", h, rope) + self.feed_forward(f"{prefix}.ff", h)
        )

    def single_block(
        self,
        prefix: str,
        content: Tensor,
        text: Tensor,
        content_mod: Tensor,
        text_mod: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        shift, scale, gate = content_mod.chunk(3, dim=-1)
        text_shift, text_scale, text_gate = text_mod.chunk(3, dim=-1)
        h = (1 + scale) * self.layer_norm(content) + shift
        text_h = (1 + text_scale) * self.layer_norm(text) + text_shift
        combined = torch.cat([text_h, h], dim=1)
        output = self.attention(f"{prefix}.attn", combined, rope) + self.feed_forward(
            f"{prefix}.ff", combined
        )
        text_out, content_out = output.split([text.shape[1], content.shape[1]], dim=1)
        return content + gate * content_out, text + text_gate * text_out

    def __call__(
        self,
        image: Tensor,
        image_ids: Tensor,
        text: Tensor,
        text_ids: Tensor,
        timestep: Tensor,
        image_cond: Optional[Tensor] = None,
        image_cond_ids: Optional[Tensor] = None,
        target_timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """Velocity [S, in_channels] of one sample's image tokens [S, in_channels]; t is [1]."""
        dtype = image.dtype
        image_temb = self.embed("timestep_embedder", timestep.to(dtype) * 1000, dtype)[:, None]
        zero_temb = self.embed("timestep_embedder", torch.zeros_like(timestep, dtype=dtype), dtype)
        zero_temb = zero_temb[:, None]
        if target_timestep is not None:
            target_temb = self.embed(
                "target_timestep_embedder", target_timestep.to(dtype) * 1000, dtype
            )[:, None]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_act, zero_act = F.silu(image_temb), F.silu(zero_temb)

        def modulation(stage: str, mode: str, act: Tensor) -> Tensor:
            return self.linear(f"{stage}_stream_modulations.{mode}.linear", act)

        text_rope = _rope(text_ids, self.axes, self.theta)
        image_rope = _rope(image_ids, self.axes, self.theta)
        text = self.linear("context_embedder", text[None])
        x = self.linear("x_embedders.image", image[None])
        for i in range(self.num_layers):
            text = self.mode_block(
                f"txt_mode_blocks.{i}", text, modulation("early", "txt", zero_act), text_rope
            )
            x = self.mode_block(
                f"mode_blocks.image.{i}", x, modulation("early", "image", image_act), image_rope
            )
        content_mod = modulation("single", "image", image_act).expand(-1, x.shape[1], -1)
        joint_rope = [text_rope, image_rope]
        if image_cond is not None:
            cond = self.linear("x_embedders.image_cond", image_cond[None])
            cond_rope = _rope(image_cond_ids, self.axes, self.theta)
            for i in range(self.num_layers):
                cond = self.mode_block(
                    f"mode_blocks.image_cond.{i}",
                    cond,
                    modulation("early", "image_cond", zero_act),
                    cond_rope,
                )
            cond_mod = modulation("single", "image_cond", zero_act).expand(-1, cond.shape[1], -1)
            x = torch.cat([x, cond], dim=1)
            content_mod = torch.cat([content_mod, cond_mod], dim=1)
            joint_rope.append(cond_rope)
        rope = tuple(torch.cat(parts, dim=0) for parts in zip(*joint_rope))
        text_mod = modulation("single", "txt", zero_act)
        for i in range(self.num_single_layers):
            x, text = self.single_block(
                f"single_transformer_blocks.{i}", x, text, content_mod, text_mod, rope
            )
        x = x[:, : image.shape[0]]
        scale, shift = self.linear("norm_out.linear", image_act).chunk(2, dim=-1)
        return self.linear("proj_out", self.layer_norm(x) * (1 + scale) + shift)[0]
