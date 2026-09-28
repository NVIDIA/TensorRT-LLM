# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL coverage of H3 Ulysses padding and distributed spatial VAE tiles."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKLMiniMaxH3

from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.minimax_h3.tiled_vae import (
    MINIMAX_H3_VAE_DEFAULTS,
    TiledAutoencoderKLMiniMaxH3,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig


def _make_model_config() -> DiffusionModelConfig:
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=5,
            freq_dim=8,
            time_embed_hidden_dim=12,
            time_embed_dim=6,
            rope_freq_dim=1,
            rope_theta=10000.0,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        ),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
        attention_metadata_state=create_attention_metadata_state(),
    )


class _TileDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tile-dependent values expose wrong ordering and overlap blending.
        return (x + x.mean()).repeat_interleave(2, -2).repeat_interleave(2, -1)


def _make_vae() -> TiledAutoencoderKLMiniMaxH3:
    vae = TiledAutoencoderKLMiniMaxH3.__new__(TiledAutoencoderKLMiniMaxH3)
    torch.nn.Module.__init__(vae)
    vae.spatial_compression_ratio = 2
    vae.use_tiling = True
    vae.tile_sample_min_height = vae.tile_sample_min_width = 8
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 2
    vae.post_quant_conv = torch.nn.Conv3d(1, 1, 1)
    vae.decoder = _TileDecoder()
    vae.configure_tiling(MINIMAX_H3_VAE_DEFAULTS)
    return vae


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(42)
        config = _make_model_config()
        reference = MiniMaxH3Transformer3DModel(config).to(device).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.normal_(0, 0.02)
        config = _make_model_config()
        vgm = VisualGenMapping(world_size=2, rank=rank, ulysses_size=2, parallel_vae_size=2)
        config.visual_gen_mapping = vgm
        config.mapping = vgm.to_llm_mapping()
        candidate = MiniMaxH3Transformer3DModel(config).to(device).eval()
        candidate.load_state_dict(reference.state_dict())
        # Five packed rows and one text row both require padding for two ranks.
        inputs = {
            "hidden_states": torch.randn(1, 3, 2, device=device),
            "audio_hidden_states": torch.randn(1, 1, 3, device=device),
            "encoder_hidden_states": torch.randn(1, 1, 5, device=device),
            "timestep": torch.tensor([1.0], device=device),
            "conditioning_timesteps": torch.tensor([0.0, 0.75], device=device),
            "token_tags": torch.tensor([1, 0, 2, 0, 0], device=device),
            "timestep_indices": torch.tensor([0, 1, 1, 0, 1], device=device),
            "position_ids": torch.tensor(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1], [2, 1, 1]], device=device
            ),
            "video_indices": torch.tensor([1, 3, 4], device=device),
            "audio_indices": torch.tensor([2], device=device),
            "text_indices": torch.tensor([0], device=device),
        }
        with torch.inference_mode():
            expected = reference(**inputs)
            actual = candidate(**inputs)
        torch.testing.assert_close(actual.sample, expected.sample, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=2e-3, atol=2e-3)
        # Exercise the same VAE group and real gathers, including an idle tail rank
        # (3x3 tiles) and the fewer-tiles-than-ranks fallback.
        torch.manual_seed(43)
        vae = _make_vae().to(device).eval()
        for shape in ((2, 2), (10, 10), (7, 10)):
            z = torch.randn(1, 1, 3, *shape, device=device)
            with torch.inference_mode():
                expected_video = AutoencoderKLMiniMaxH3._decode_clip(vae, z)
                vae.tile_parallel_group = vgm.vae_group
                actual_video = vae._decode_clip(z)
            torch.testing.assert_close(actual_video, expected_video, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_h3_ulysses_and_parallel_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    from ._visual_gen_dist_utils import spawn_with_retry

    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    spawn_with_retry(lambda port: mp.spawn(_worker, args=(port,), nprocs=2, join=True))
