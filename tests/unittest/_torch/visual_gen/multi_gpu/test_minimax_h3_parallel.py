# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL coverage of H3 Ulysses padding and distributed spatial VAE tiles."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKLMiniMaxH3

from tensorrt_llm._torch.visual_gen.config import create_attention_metadata_state
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)

from ..test_minimax_h3_tiled_vae import _vae
from ..test_minimax_h3_transformer import _make_model_config, _model_inputs
from ._visual_gen_dist_utils import spawn_with_retry


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
        config = _make_model_config(
            num_layers=2, num_refiner_layers=1, hidden_size=32, attention_head_dim=16, ffn_dim=32
        )
        config.attention_metadata_state = create_attention_metadata_state()
        reference = MiniMaxH3Transformer3DModel(config).to(device).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.normal_(0, 0.02)
        config = _make_model_config(
            num_layers=2, num_refiner_layers=1, hidden_size=32, attention_head_dim=16, ffn_dim=32
        )
        vgm = VisualGenMapping(world_size=2, rank=rank, ulysses_size=2, parallel_vae_size=2)
        config.visual_gen_mapping = vgm
        config.mapping = vgm.to_llm_mapping()
        config.attention_metadata_state = create_attention_metadata_state()
        candidate = MiniMaxH3Transformer3DModel(config).to(device).eval()
        candidate.load_state_dict(reference.state_dict())
        inputs = _model_inputs(device)
        # Five packed rows and one text row both require padding for two ranks.
        inputs["hidden_states"] = torch.randn(1, 3, 2, device=device)
        inputs["token_tags"] = torch.tensor([1, 0, 2, 0, 0], device=device)
        inputs["timestep_indices"] = torch.tensor([0, 1, 1, 0, 1], device=device)
        inputs["position_ids"] = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1], [2, 1, 1]], device=device
        )
        inputs["video_indices"] = torch.tensor([1, 3, 4], device=device)
        with torch.inference_mode():
            expected = reference(**inputs)
            actual = candidate(**inputs)
        torch.testing.assert_close(actual.sample, expected.sample, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=2e-3, atol=2e-3)
        # Exercise the same VAE group and real gathers, including an idle tail rank
        # (3x3 tiles) and the fewer-tiles-than-ranks fallback.
        torch.manual_seed(43)
        vae = _vae().to(device).eval()
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
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    spawn_with_retry(lambda port: mp.spawn(_worker, args=(port,), nprocs=2, join=True))
