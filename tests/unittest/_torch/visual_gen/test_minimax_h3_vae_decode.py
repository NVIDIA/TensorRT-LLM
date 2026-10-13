# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the lossless MiniMax-H3 video-VAE decode: batched tiles, memory-sized batches, fp16 Linear copies,
compiled decoder blocks."""

from types import SimpleNamespace

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
AutoencoderKLMiniMaxH3 = pytest.importorskip(
    "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
).AutoencoderKLMiniMaxH3

import tensorrt_llm._torch.visual_gen.models.minimax_h3.parallel_vae as parallel_vae  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.minimax_h3.parallel_vae import (  # noqa: E402
    TiledAutoencoderKLMiniMaxH3,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (  # noqa: E402
    MiniMaxH3Pipeline,
    _prepare_vae_decoder,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tiny_vae() -> TiledAutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    vae = TiledAutoencoderKLMiniMaxH3(
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
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    return vae.cuda().eval()


def _latent(vae: AutoencoderKLMiniMaxH3, frames: int, tiles: int, batch: int = 1) -> torch.Tensor:
    ratio = vae.spatial_compression_ratio
    side = vae.tile_sample_min_height * tiles // ratio
    torch.manual_seed(1)
    return torch.randn((batch, vae.config.latent_channels, frames, side, side), device="cuda")


def _reference_decode_clip(vae: AutoencoderKLMiniMaxH3, z: torch.Tensor) -> torch.Tensor:
    """The stock Diffusers tile loop on the same weights.

    The tests compare it bit for bit with batched decoder calls. Batching changes the GEMM and
    attention problem sizes, so a kernel-selection change on another GPU would surface here as
    a failure; that is a signal worth seeing, not a test to loosen quietly.
    """
    return AutoencoderKLMiniMaxH3._decode_clip(vae, z)


def _count_decoder_batches(monkeypatch, vae) -> list[int]:
    calls: list[int] = []
    forward = vae.decoder.forward

    def counting_forward(x):
        calls.append(x.shape[0])
        return forward(x)

    monkeypatch.setattr(vae.decoder, "forward", counting_forward)
    return calls


@requires_cuda
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(("tiles", "grid"), [(1, 1), (2, 9)])  # 2 tile widths + overlap -> 3 x 3
def test_single_rank_decode_batches_tiles_and_matches_the_tile_loop(
    monkeypatch, tiles, grid, batch
):
    vae = _tiny_vae()
    z = _latent(vae, frames=2, tiles=tiles, batch=batch)
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        calls.clear()
        actual = vae._decode_clip(z)
    assert calls == [grid * batch]  # uncalibrated: one call up to the upper bound
    assert torch.equal(actual, expected)


@requires_cuda
def test_calibration_sizes_the_batches_from_the_free_memory(monkeypatch):
    vae = _tiny_vae()
    _prepare_vae_decoder(vae)
    per_latent = vae._decode_bytes_per_latent
    assert per_latent is not None and per_latent > 0
    z = _latent(vae, frames=2, tiles=2)  # 3 x 3 tiles
    tile_elements = z[..., : z.shape[-2] // 2, : z.shape[-1] // 2].numel()
    per_tile = per_latent * tile_elements
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        for budget_tiles, batches in [(2.9, [2, 2, 2, 2, 1]), (0.4, [1] * 9), (100.0, [9])]:
            monkeypatch.setattr(
                parallel_vae,
                "available_device_bytes",
                lambda device, per_tile=per_tile, k=budget_tiles: int(
                    per_tile * k / parallel_vae.TILE_DECODE_MEMORY_FRACTION
                ),
            )
            calls.clear()
            actual = vae._decode_clip(z)
            assert calls == batches
            assert torch.equal(actual, expected)


@requires_cuda
def test_upper_bound_caps_the_batches_whatever_the_free_memory(monkeypatch):
    vae = _tiny_vae()
    _prepare_vae_decoder(vae)
    monkeypatch.setattr(parallel_vae, "available_device_bytes", lambda device: 2**60)
    vae.max_tiles_per_decoder_call = 4
    z = _latent(vae, frames=2, tiles=2)  # 3 x 3 tiles
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        calls.clear()
        actual = vae._decode_clip(z)
    assert calls == [4, 4, 1]
    assert torch.equal(actual, expected)


@requires_cuda
def test_prepared_decoder_fp16_linear_copies_are_bitwise_identical_under_autocast():
    vae = _tiny_vae()
    z = _latent(vae, frames=2, tiles=2)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
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
def test_unequal_tiles_fall_back_to_one_call_per_tile(monkeypatch):
    """The stock splitter always yields equal tiles; force unequal lengths to exercise the fallback."""
    vae = _tiny_vae()
    ratio = vae.spatial_compression_ratio

    def unequal_split(length, tile_size, min_overlap):
        # Two tiles: a full one and a shorter one, overlapping by one latent step.
        return [0, length - 48], [64, 48], [16]

    monkeypatch.setattr(vae, "_split_tiles", unequal_split)
    calls = _count_decoder_batches(monkeypatch, vae)
    torch.manual_seed(2)
    z = torch.randn((1, vae.config.latent_channels, 2, 96 // ratio, 96 // ratio), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        calls.clear()
        actual = vae._decode_clip(z)
    assert calls == [1, 1, 1, 1]
    assert torch.equal(actual, expected)


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_decode_batches_each_ranks_tiles_and_gathers_once(monkeypatch, rank):
    """A group of two: each rank decodes its every-second tile in one batch; one gather per clip."""
    vae = _tiny_vae()
    _prepare_vae_decoder(vae)
    monkeypatch.setattr(parallel_vae, "available_device_bytes", lambda device: 2**60)
    monkeypatch.setattr(parallel_vae.dist, "get_world_size", lambda g: 2)
    monkeypatch.setattr(parallel_vae.dist, "get_rank", lambda g: rank)
    monkeypatch.setattr(parallel_vae.dist, "all_reduce", lambda t, op, group: None)
    z = _latent(vae, frames=2, tiles=2)  # 3 x 3 tiles -> rank 0 owns 5, rank 1 owns 4
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        reference_tiles = [
            vae.decoder(vae.post_quant_conv(z[..., y : y + h, x : x + w]))
            for y, h, x, w in vae._tile_geometry(z)[0]
        ]
    gathers = []

    def fake_all_gather(gathered, local, group):
        gathers.append(local.shape[0])
        for offset, out in enumerate(gathered):
            # The peer's share, in the same (wave, tile) layout, zero padded past its last tile.
            for wave, slot in enumerate(out.split(1, dim=0)):
                index = wave * 2 + offset
                slot.copy_(reference_tiles[index] if index < 9 else torch.zeros_like(slot))

    monkeypatch.setattr(parallel_vae.dist, "all_gather", fake_all_gather)
    vae.tile_parallel_group = object()
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        out = vae._decode_clip(z)
    assert calls == [5 if rank == 0 else 4]
    assert gathers == [5]  # five waves in one gather; rank 1 sent one zero tile
    assert torch.equal(out, expected)


@requires_cuda
@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_decode_splits_batches_by_the_group_minimum(monkeypatch, rank):
    """The agreed count bounds every rank's batch so the gathered tensors have equal shapes."""
    vae = _tiny_vae()
    _prepare_vae_decoder(vae)
    monkeypatch.setattr(parallel_vae, "available_device_bytes", lambda device: 2**60)
    monkeypatch.setattr(parallel_vae.dist, "get_world_size", lambda g: 2)
    monkeypatch.setattr(parallel_vae.dist, "get_rank", lambda g: rank)

    def peer_has_memory_for_two(t, op, group):
        t.fill_(2)

    monkeypatch.setattr(parallel_vae.dist, "all_reduce", peer_has_memory_for_two)
    gathers = []

    def fake_all_gather(gathered, local, group):
        gathers.append(local.shape[0])
        for out in gathered:
            out.copy_(local)

    monkeypatch.setattr(parallel_vae.dist, "all_gather", fake_all_gather)
    vae.tile_parallel_group = object()
    z = _latent(vae, frames=2, tiles=2)  # rank 0 owns tiles 0, 2, 4, 6, 8; rank 1 owns 1, 3, 5, 7
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        vae._decode_clip(z)
    # Rank 1 has no tile for the fifth wave: its last batch is all padding, no decoder call.
    assert calls == ([2, 2, 1] if rank == 0 else [2, 2])
    assert gathers == [2, 2, 1]


@requires_cuda
def test_group_of_one_and_small_canvases_take_the_batched_path(monkeypatch):
    vae = _tiny_vae()
    monkeypatch.setattr(parallel_vae.dist, "get_world_size", lambda g: 1)
    vae.tile_parallel_group = object()
    z = _latent(vae, frames=2, tiles=2)
    calls = _count_decoder_batches(monkeypatch, vae)
    with torch.inference_mode(), torch.autocast("cuda", torch.float16):
        expected = _reference_decode_clip(vae, z)
        calls.clear()
        assert torch.equal(vae._decode_clip(z), expected)
        assert calls == [9]
        # Fewer tiles than ranks: a group of sixteen decodes the nine tiles locally.
        monkeypatch.setattr(parallel_vae.dist, "get_world_size", lambda g: 16)
        calls.clear()
        actual = vae._decode_clip(z)
        assert calls == [9]
    assert torch.equal(actual, expected)


def test_calibration_is_skipped_off_the_gpu():
    torch.manual_seed(0)
    vae = TiledAutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 8, 16, 16, 16, 16),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_ffn_mult=2,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )
    vae.calibrate_tile_decode_memory()
    assert vae._decode_bytes_per_latent is None


class _Wrapped(torch.nn.Module):
    def __init__(self, block, **kwargs):
        super().__init__()
        self.block = block
        self.kwargs = kwargs


@pytest.mark.parametrize("fullgraph", [True, False])
def test_torch_compile_compiles_every_decoder_block_with_the_configured_fullgraph(
    monkeypatch, fullgraph
):
    compiled = []

    def fake_compile(module, **kwargs):
        compiled.append(module)
        return _Wrapped(module, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    base_calls = []
    monkeypatch.setattr(BasePipeline, "torch_compile", lambda self: base_calls.append(self))
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_config = SimpleNamespace(
        torch_compile=SimpleNamespace(enable_fullgraph=fullgraph)
    )
    blocks = [torch.nn.Linear(2, 2) for _ in range(3)]
    pipeline.vae = SimpleNamespace(
        decoder=SimpleNamespace(transformer_blocks=torch.nn.ModuleList(blocks))
    )

    pipeline.torch_compile()

    assert base_calls == [pipeline]
    assert compiled == blocks
    new_blocks = list(pipeline.vae.decoder.transformer_blocks)
    assert [b.block for b in new_blocks] == blocks
    assert all(
        b.kwargs == {"mode": "default", "dynamic": None, "fullgraph": fullgraph} for b in new_blocks
    )


def test_torch_compile_leaves_a_skipped_vae_alone(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("compile called"))
    monkeypatch.setattr(BasePipeline, "torch_compile", lambda self: None)
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.pipeline_config = SimpleNamespace(torch_compile=SimpleNamespace(enable_fullgraph=True))
    pipeline.vae = None
    pipeline.torch_compile()
