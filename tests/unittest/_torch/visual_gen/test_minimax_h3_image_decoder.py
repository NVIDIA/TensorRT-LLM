# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.models.minimax_h3_image.image_decoder import (
    MiniMaxH3ImageDecoder,
    _DecoderBlock,
)


@torch.inference_mode()
def test_decoder_block_partial_rotary_and_gates():
    cfg = SimpleNamespace(
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=2,
        decoder_norm_eps=1e-5,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        block = _DecoderBlock(cfg).eval()
        # Nonzero scales are essential: the initializer gives the identity block.
        block.scale1.uniform_(0.25, 1)
        block.scale2.uniform_(0.25, 1)
        x = torch.randn(2, 7, 16)
        phase = torch.randn(7, 2).repeat(1, 2)

    cos, sin = phase.cos(), phase.sin()
    attn = block.attn

    def norm(x, weight=None):
        out = x * (x.square().mean(-1, keepdim=True) + cfg.decoder_norm_eps).rsqrt()
        return out if weight is None else out * weight

    def project(layer):
        return layer(norm(x, block.norm1.weight)).reshape(2, 7, 2, 8).transpose(1, 2)

    def rotate(t):
        left, right, tail = t.split([2, 2, 4], -1)
        c, s = cos[:, :2], sin[:, :2]
        return torch.cat([left * c - right * s, left * s + right * c, tail], dim=-1)

    q, k = (rotate(norm(project(layer))) for layer in (attn.to_q, attn.to_k))
    probs = ((q @ k.transpose(-1, -2)) / 8**0.5).softmax(-1)
    attended = (probs @ project(attn.to_v)).transpose(1, 2).reshape_as(x)
    residual = x + attn.to_out[0](attended) * block.scale1
    ff = block.ff["net"]
    up, gate = ff[0].proj(norm(residual, block.norm2.weight)).chunk(2, -1)
    expected = residual + ff[2](up * F.silu(gate)) * block.scale2
    torch.testing.assert_close(block(x, (cos, sin)), expected, atol=2e-6, rtol=2e-5)


class _TileContext(nn.Module):
    def forward(self, z):
        # Distinct tile contexts without a floating-point reduction whose order
        # depends on the strides of the input patch.
        return F.interpolate(z + z[..., :1, :1], scale_factor=16, mode="nearest")


@pytest.mark.parametrize(
    ("height", "width", "ys", "xs"),
    [
        (7, 13, [0], [0]),
        (28, 35, [0, 12], [0, 9, 19]),
    ],
)
@torch.inference_mode()
def test_decoder_pixel_space_tile_stitch(height, width, ys, xs):
    cfg = SimpleNamespace(
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=2,
        decoder_norm_eps=1e-5,
        spatial_downsample_factors=[2, 2, 2, 2],
        out_channels=1,
        latent_channels=1,
        decoder_num_register_tokens=1,
        decoder_num_layers=1,
        decoder_rope_dim_ratio=0.5,
        decoder_rope_theta=100.0,
        latents_mean=[0.75],
        latents_std=[1.5],
    )
    model = MiniMaxH3ImageDecoder(cfg)
    model.decoder = _TileContext()
    model.post_quant_conv.weight.fill_(1)
    model.post_quant_conv.bias.zero_()
    x = torch.randn(1, 1, height, width, generator=torch.Generator().manual_seed(31))
    z = x * 1.5 + 0.75
    tiles = [[model.decoder(z[..., y : y + 16, col : col + 16]) for col in xs] for y in ys]
    expected = torch.empty(1, 1, height * 16, width * 16)
    # The release blends against the unmodified neighbors, vertically first.
    for i, y in enumerate(ys):
        for j, col in enumerate(xs):
            tile = tiles[i][j].clone()
            if i:
                n = (ys[i - 1] + 16 - y) * 16
                alpha = torch.arange(n, dtype=tile.dtype).view(-1, 1) / n
                tile[..., :n, :] = (
                    tiles[i - 1][j][..., -n:, :] * (1 - alpha) + tile[..., :n, :] * alpha
                )
            if j:
                n = (xs[j - 1] + 16 - col) * 16
                alpha = torch.arange(n, dtype=tile.dtype) / n
                tile[..., :n] = tiles[i][j - 1][..., -n:] * (1 - alpha) + tile[..., :n] * alpha
            y2 = ys[i + 1] if i + 1 < len(ys) else height
            x2 = xs[j + 1] if j + 1 < len(xs) else width
            expected[..., y * 16 : y2 * 16, col * 16 : x2 * 16] = tile[
                ..., : (y2 - y) * 16, : (x2 - col) * 16
            ]
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
