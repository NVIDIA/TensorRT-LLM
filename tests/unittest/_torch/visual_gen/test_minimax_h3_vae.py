# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic CPU tests for the MiniMax-H3 video and audio autoencoders."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.autoencoder_kl_minimax_h3 import (
    AutoencoderKLMiniMaxH3,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.autoencoder_kl_minimax_h3_audio import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3AudioCausalAttention,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.modeling_utils import _state_dict_files


def _tiny_video_vae() -> AutoencoderKLMiniMaxH3:
    model = AutoencoderKLMiniMaxH3(
        latent_channels=2,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(1, 2, 2),
        temporal_downsample_factors=(1, 2, 2),
        norm_num_groups=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        clip_length=17,
        token_drop=3,
        latents_mean=(0.0, 0.0),
        latents_std=(1.0, 1.0),
    )
    model.disable_tiling()
    return model.eval()


def _tiny_audio_vae() -> AutoencoderKLMiniMaxH3Audio:
    return AutoencoderKLMiniMaxH3Audio(
        encoder_dim=2,
        encoder_rates=(2,),
        latent_dim=4,
        latent_channels=2,
        num_attention_heads=2,
        decoder_dim=4,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=32000,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    ).eval()


def test_minimax_h3_video_vae_synthetic_round_trip_shapes() -> None:
    model = _tiny_video_vae()
    video = torch.randn(1, 3, 22, 8, 8)

    with torch.inference_mode():
        posterior = model.encode(video).latent_dist
        decoded = model.decode(posterior.mode()).sample

    assert posterior.mode().shape == (1, 2, 7, 2, 2)
    assert decoded.shape == video.shape
    assert model.temporal_compression_ratio == 4
    assert model.config.clip_length == 17
    assert model.config.token_drop == 3
    assert model.config.latent_channels == 2
    assert tuple(model.config.latents_mean) == (0.0, 0.0)


def test_minimax_h3_audio_vae_synthetic_round_trip_shapes() -> None:
    model = _tiny_audio_vae()
    waveform = torch.randn(1, 1, 8)

    with torch.inference_mode():
        posterior = model.encode(waveform).latent_dist
        decoded = model.decode(posterior.mode()).sample

    assert posterior.mode().shape == (1, 2, 4)
    assert decoded.shape == waveform.shape
    assert model.hop_length == 2
    assert model.config.latent_channels == 2
    assert model.config.sampling_rate == 32000


def test_minimax_h3_audio_projection_attention_is_causal() -> None:
    torch.manual_seed(0)
    attention = MiniMaxH3AudioCausalAttention(in_dim=8, out_dim=4, num_heads=2).eval()
    hidden_states = torch.randn(1, 5, 8)
    changed_future = hidden_states.clone()
    changed_future[:, -1] += 10.0

    with torch.inference_mode():
        actual = attention(hidden_states)
        perturbed = attention(changed_future)

    torch.testing.assert_close(actual[:, :-1], perturbed[:, :-1])


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_minimax_h3_sharded_checkpoint_paths_stay_inside_component(
    tmp_path: Path,
    filename: str,
) -> None:
    model_dir = tmp_path / "vae"
    model_dir.mkdir()
    index_path = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"weight": filename}}))

    with pytest.raises(ValueError, match="relative|escapes component directory"):
        _state_dict_files(model_dir)


def _state_dict_key_digest(model: torch.nn.Module) -> tuple[int, str]:
    keys = sorted(model.state_dict())
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return len(keys), digest


def test_minimax_h3_vae_state_dict_keys_match_converted_checkpoint_manifests() -> None:
    # Expected counts and digests come from the sorted keys in the official converted
    # video/audio VAE safetensors headers. Meta construction keeps this test CPU-memory cheap.
    with torch.device("meta"):
        video_vae = AutoencoderKLMiniMaxH3()
        audio_vae = AutoencoderKLMiniMaxH3Audio()

    assert _state_dict_key_digest(video_vae) == (
        703,
        "ee4f6076997b8b88be09a7d90f20c2f35af4e0783119e8d1a4a6c4173956c5a4",
    )
    assert _state_dict_key_digest(audio_vae) == (
        1087,
        "88311a6ddd74e9b42c33f85a41635c8ef5d20ef029983de12b4016cf98493f1c",
    )
