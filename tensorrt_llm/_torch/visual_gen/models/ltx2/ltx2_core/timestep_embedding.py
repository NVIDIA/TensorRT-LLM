# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math

import torch


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (DDPM-style)."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        make_linear=None,
    ):
        super().__init__()
        if make_linear is None:
            make_linear = torch.nn.Linear
        self.linear_1 = make_linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.cond_proj = (
            make_linear(cond_proj_dim, in_channels, bias=False)
            if cond_proj_dim is not None
            else None
        )
        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = make_linear(time_embed_dim, time_embed_dim_out, bias=sample_proj_bias)
        self.post_act = None

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """PixArt-Alpha combined timestep + size embeddings."""

    def __init__(self, embedding_dim: int, size_emb_dim: int, make_linear=None):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            make_linear=make_linear,
        )

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
