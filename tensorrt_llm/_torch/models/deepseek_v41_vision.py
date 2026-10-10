# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 image encoder and spatial downsampling projector."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ..configs.deepseek_v41 import DeepseekV41VisionConfig

__all__ = ["DeepseekV41VisionModel", "DeepseekV41Aligner"]


def _get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    height = torch.arange(n_h, device=device).view(n_h, 1).expand(n_h, n_w)
    width = torch.arange(n_w, device=device).view(1, n_w).expand(n_h, n_w)
    positions = torch.stack((height, width), dim=-1).reshape(-1, 2, 1).float()
    frequencies = (positions * inv_freq).flatten(1).unsqueeze(1)
    return frequencies.cos(), frequencies.sin()


def _apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)


class _DeepseekV41VisionRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> _DeepseekV41VisionRMSNorm:
        # Keep norm parameters FP32 when casting the enclosing encoder.
        def apply_fp32(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if converted.dtype != torch.float32:
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(apply_fp32, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * value).to(x.dtype)


class _DeepseekV41PatchEmbed(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class _DeepseekV41VisionAttention(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        num_patches = x.shape[0]
        query, key, value = (
            part.reshape(num_patches, self.num_heads, self.head_dim)
            for part in self.wqkv(x).chunk(3, dim=-1)
        )
        query = _apply_vision_rotary(query, cos, sin)
        key = _apply_vision_rotary(key, cos, sin)
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
        )
        return self.wo(attended.squeeze(0).transpose(0, 1).reshape(num_patches, -1))


class _DeepseekV41VisionMLP(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.norm1 = _DeepseekV41VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = _DeepseekV41VisionAttention(config)
        self.norm2 = _DeepseekV41VisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _DeepseekV41VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV41VisionModel(nn.Module):
    """Bidirectional image transformer with separate height/width rotary positions.

    Args:
        config: Vision tower dimensions and normalization configuration.
    """

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        if config.num_attention_heads <= 0 or config.hidden_size % config.num_attention_heads:
            raise ValueError("Vision hidden_size must be divisible by num_attention_heads")
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim <= 0 or head_dim % 4:
            raise ValueError(
                "Vision head dimension must be a positive multiple of four for 2D RoPE"
            )
        self.rope_dim = head_dim // 2
        self.rope_theta = config.rope_theta
        self.patch_embed = _DeepseekV41PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [_DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = _DeepseekV41VisionRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Encode one row-major patch grid into ``[n_h * n_w, hidden_size]``.

        Args:
            patches: Floating-point patches shaped ``[n_h * n_w, 3, P, P]``
                or ``[n_h * n_w, 3 * P * P]``, in the projection's dtype/device.
            n_h: Number of patch rows.
            n_w: Number of patch columns.
        """
        if n_h <= 0 or n_w <= 0 or patches.ndim < 2 or patches.shape[0] != n_h * n_w:
            raise ValueError("Vision patches must match a positive n_h by n_w grid")
        hidden_states = self.patch_embed(patches)
        cos, sin = _get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, patches.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return self.norm(hidden_states)


class DeepseekV41Aligner(nn.Module):
    """Pack spatial neighborhoods channel-first and project to text embeddings.

    Args:
        config: Vision width and spatial downsampling configuration.
        text_hidden_size: Output width required by the language model.
    """

    def __init__(self, config: DeepseekV41VisionConfig, text_hidden_size: int) -> None:
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        if self.downsample_ratio <= 0:
            raise ValueError("Vision downsample_ratio must be positive")
        self.w1 = nn.Linear(config.hidden_size * self.downsample_ratio**2, text_hidden_size)
        self.w2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Project ``[n_h * n_w, hidden_size]`` image features to text width.

        Right/bottom zero padding produces
        ``ceil(n_h / ratio) * ceil(n_w / ratio)`` output rows.
        """
        if n_h <= 0 or n_w <= 0 or x.ndim != 2 or x.shape[0] != n_h * n_w:
            raise ValueError("Aligner features must match a positive n_h by n_w grid")
        ratio = self.downsample_ratio
        grid = x.reshape(n_h, n_w, -1).permute(2, 0, 1)
        grid = F.pad(grid, (0, -n_w % ratio, 0, -n_h % ratio))
        neighborhoods = F.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(neighborhoods)))
