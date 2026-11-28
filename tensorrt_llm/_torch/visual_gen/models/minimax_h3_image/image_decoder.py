# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode-only image VAE for the H3 text-to-image pipeline (no temporal layers)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..minimax_h3.transformer_minimax_h3 import apply_minimax_h3_rotary_emb


class _DecoderAttention(nn.Module):
    def __init__(self, heads, head_dim, eps):
        super().__init__()
        dim = heads * head_dim
        self.heads = heads
        self.eps = eps
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x, rotary):
        def rotate(projection):
            t = projection(x).unflatten(-1, (self.heads, -1))
            norm = F.rms_norm(t.float(), (t.size(-1),), eps=self.eps).to(t.dtype)
            return apply_minimax_h3_rotary_emb(norm, *rotary).transpose(1, 2)

        v = self.to_v(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        attn = F.scaled_dot_product_attention(rotate(self.to_q), rotate(self.to_k), v)
        return self.to_out(attn.transpose(1, 2).flatten(2))


class _SwiGLU(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * inner)

    def forward(self, x):
        up, gate = self.proj(x).chunk(2, dim=-1)
        return up * F.silu(gate)


class _DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.decoder_num_attention_heads * cfg.decoder_attention_head_dim
        inner = dim * cfg.decoder_ffn_mult
        self.attn = _DecoderAttention(
            cfg.decoder_num_attention_heads, cfg.decoder_attention_head_dim, cfg.decoder_norm_eps
        )
        self.norm1 = nn.RMSNorm(dim, eps=cfg.decoder_norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=cfg.decoder_norm_eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.scale2 = nn.Parameter(torch.zeros(dim))
        self.ff = nn.ModuleDict(
            {"net": nn.Sequential(_SwiGLU(dim, inner), nn.Identity(), nn.Linear(inner, dim))}
        )

    def forward(self, x, rotary):
        dtype = self.attn.to_q.weight.dtype
        x = x + self.scale1 * self.attn(self.norm1(x.float()).to(dtype), rotary)
        return x + self.scale2 * self.ff["net"](self.norm2(x.float()).to(dtype))


class _TileDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.decoder_num_attention_heads * cfg.decoder_attention_head_dim
        self.patch = math.prod(cfg.spatial_downsample_factors)
        self.channels = cfg.out_channels
        self.proj_in = nn.Linear(cfg.latent_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.decoder_num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            _DecoderBlock(cfg) for _ in range(cfg.decoder_num_layers)
        )
        self.norm_out = nn.LayerNorm(dim, eps=cfg.decoder_norm_eps)
        self.proj_out = nn.Linear(dim, cfg.out_channels * self.patch**2)
        rotary_dim = int(cfg.decoder_rope_dim_ratio * cfg.decoder_attention_head_dim)
        if rotary_dim <= 0 or rotary_dim % 4:
            raise ValueError("Image-decoder rotary dimensions must split into two even axes")
        self.register_buffer(
            "inv_freq",
            1 / cfg.decoder_rope_theta ** torch.arange(0, 1, 4 / rotary_dim, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, z):
        b, _, h, w = z.shape
        x = self.proj_in(z.flatten(2).transpose(1, 2).to(self.proj_in.weight.dtype))
        suffix = self.register_tokens.expand(b, -1, -1).to(x.dtype)
        x = torch.cat((x, suffix, torch.zeros_like(x[:, :1])), dim=1)
        axes = [
            2 * ((torch.arange(n, device=z.device, dtype=torch.float32) + 0.5) / n) - 1
            for n in (h, w)
        ]
        positions = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
        positions = F.pad(positions, (0, 0, 0, suffix.size(1) + 1))
        phase = (2 * math.pi * positions[..., None] * self.inv_freq).flatten(1).repeat(1, 2)
        rotary = phase.cos(), phase.sin()
        for block in self.transformer_blocks:
            x = block(x, rotary)
        pixels = self.proj_out(self.norm_out(x.float()).to(self.proj_out.weight.dtype))[:, : h * w]
        p = self.patch
        return (
            pixels.reshape(b, h, w, self.channels, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(b, self.channels, h * p, w * p)
        )


def _tile_starts(length, tile=16, overlap=4):
    if length <= tile:
        return [0]
    intervals = math.ceil((length - tile) / (tile - overlap))
    stride, extra = divmod(length - tile, intervals)
    # Distribute surplus overlap to the earliest gaps, as in the release.
    return [i * stride + max(0, i - (intervals - extra)) for i in range(intervals + 1)]


def _blend(previous, tile, overlap, axis):
    n = tile.size(axis)
    weight = torch.arange(overlap, device=tile.device, dtype=tile.dtype) / overlap
    shape = [1] * tile.ndim
    shape[axis] = overlap
    weight = weight.reshape(shape)
    mixed = (
        previous.narrow(axis, previous.size(axis) - overlap, overlap) * (1 - weight)
        + tile.narrow(axis, 0, overlap) * weight
    )
    return torch.cat((mixed, tile.narrow(axis, overlap, n - overlap)), dim=axis)


class MiniMaxH3ImageDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.decoder = _TileDecoder(cfg)
        if self.decoder.patch != 16:
            raise ValueError("Expected the 16x spatial image VAE")
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)
        for name in ("latents_mean", "latents_std"):
            self.register_buffer(
                name, torch.tensor(getattr(cfg, name)).view(1, -1, 1, 1), persistent=False
            )

    def forward(self, latents):
        z = latents.float() * self.latents_std + self.latents_mean
        ys, xs = [_tile_starts(n) for n in z.shape[-2:]]
        previous = None
        rows = []
        for i, y in enumerate(ys):
            current = []
            pieces = []
            for j, x in enumerate(xs):
                tile = self.decoder(
                    self.post_quant_conv(
                        z[..., y : y + 16, x : x + 16].to(self.post_quant_conv.weight.dtype)
                    )
                )
                current.append(tile)
                if i:
                    tile = _blend(previous[j], tile, (16 + ys[i - 1] - y) * 16, -2)
                if j:
                    tile = _blend(current[j - 1], tile, (16 + xs[j - 1] - x) * 16, -1)
                ph = (ys[i + 1] - y) * 16 if i + 1 < len(ys) else tile.size(-2)
                pw = (xs[j + 1] - x) * 16 if j + 1 < len(xs) else tile.size(-1)
                pieces.append(tile[..., :ph, :pw])
            rows.append(torch.cat(pieces, dim=-1))
            previous = current
        return torch.cat(rows, dim=-2)
