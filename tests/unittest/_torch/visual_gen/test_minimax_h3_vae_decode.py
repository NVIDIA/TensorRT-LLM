# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the lossless MiniMax-H3 video-VAE decode preparation (batched tiles, fp16 Linear copies)."""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
AutoencoderKLMiniMaxH3 = pytest.importorskip(
    "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
).AutoencoderKLMiniMaxH3

from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (  # noqa: E402
    _batched_decode_clip,
    _prepare_vae_decoder,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 8, 16, 16, 16, 16),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=2,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )
    return vae.cuda().eval()


def _latent(vae: AutoencoderKLMiniMaxH3, frames: int, tiles: int) -> torch.Tensor:
    ratio = vae.spatial_compression_ratio
    side = vae.tile_sample_min_height * tiles // ratio
    torch.manual_seed(1)
    return torch.randn((1, vae.config.latent_channels, frames, side, side), device="cuda")


@requires_cuda
@pytest.mark.parametrize("tiles", [1, 2])
def test_batched_decode_clip_is_bitwise_identical_to_tile_loop(tiles):
    vae = _tiny_vae()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    z = _latent(vae, frames=2, tiles=tiles)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = vae._decode_clip(z)
        actual = _batched_decode_clip(vae, z)
    assert torch.equal(actual, expected)


@requires_cuda
def test_prepared_decoder_fp16_linear_copies_are_bitwise_identical_under_autocast():
    vae = _tiny_vae()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    z = _latent(vae, frames=2, tiles=2)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = vae._decode_clip(z)
    _prepare_vae_decoder(vae)
    assert all(
        m.weight.dtype == torch.float16
        for m in vae.decoder.modules()
        if isinstance(m, torch.nn.Linear)
    )
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        actual = vae._decode_clip(z)
    assert torch.equal(actual, expected)


@requires_cuda
def test_batched_decode_falls_back_when_tile_shapes_differ(monkeypatch):
    """The stock splitter always yields equal tiles; force unequal lengths to exercise the fallback loop."""
    vae = _tiny_vae()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    ratio = vae.spatial_compression_ratio

    def unequal_split(length, tile_size, min_overlap):
        # Two tiles: a full one and a shorter one, overlapping by one latent step.
        return [0, length - 48], [64, 48], [16]

    monkeypatch.setattr(vae, "_split_tiles", unequal_split)
    torch.manual_seed(2)
    z = torch.randn((1, vae.config.latent_channels, 2, 96 // ratio, 96 // ratio), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = vae._decode_clip(z)
        actual = _batched_decode_clip(vae, z)
    assert torch.equal(actual, expected)


@requires_cuda
def test_prepare_skips_batched_decode_for_subclasses_with_their_own_tiling():
    class Tiled(AutoencoderKLMiniMaxH3):
        def _decode_clip(self, z):  # pragma: no cover - identity marker only
            return super()._decode_clip(z)

    vae = _tiny_vae()
    vae.__class__ = Tiled
    _prepare_vae_decoder(vae)
    assert "_decode_clip" not in vae.__dict__
    assert all(
        m.weight.dtype == torch.float16
        for m in vae.decoder.modules()
        if isinstance(m, torch.nn.Linear)
    )
