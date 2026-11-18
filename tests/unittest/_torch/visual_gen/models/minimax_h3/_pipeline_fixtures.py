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

"""Synthetic pipeline-level tests for MiniMax-H3."""

from types import SimpleNamespace

import torch
from diffusers import MiniMaxH3Scheduler
from PIL import Image

from tensorrt_llm._torch.visual_gen.models.minimax_h3.default.pipeline import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.fasth3.pipeline import FastH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from tensorrt_llm._torch.visual_gen.offloading import PipelineOffloader


class _FakeMiniMaxH3Transformer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(patch_size=(1, 2, 2))
        self.training = True
        self.static_context_calls = 0
        self.forward_calls = 0
        self.inference_modes: list[bool] = []
        self.attention_timesteps: list[torch.Tensor] = []

    def eval(self) -> "_FakeMiniMaxH3Transformer":
        self.training = False
        return self

    def prepare_static_context(
        self,
        prompt_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_context_calls += 1
        return prompt_embeds, position_ids

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        static_context: tuple[torch.Tensor, torch.Tensor],
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.attention_timesteps.append(kwargs["timestep"].clone())
        assert static_context is not None
        self.forward_calls += 1
        self.inference_modes.append(torch.is_inference_mode_enabled())
        # Non-zero: an all-zero velocity is what _check_denoise_step treats
        # as a corrupt step, so a fake must not emit one.
        return torch.ones_like(hidden_states), torch.ones_like(audio_hidden_states)


class _SyntheticInputs:
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        # BasePipeline.__init__ is bypassed here, so set the backing attribute
        # that BasePipeline.device reads.
        self.workflow = "fl2va"
        self._device = torch.device("cpu")
        self.pipeline_config = SimpleNamespace(
            visual_gen_mapping=None,
            primary_model_config=SimpleNamespace(
                attention=SimpleNamespace(backend="VANILLA", sparse_attention_config=None)
            ),
        )
        self.offloader = PipelineOffloader(self)
        self.transformer = _FakeMiniMaxH3Transformer()
        self.vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=2),
            spatial_compression_ratio=2,
        )
        self.audio_vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=3, sampling_rate=32000)
        )
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        self._is_warmup = True
        self._parallel_vae_enabled = False

    def _encode_prompt(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt, keyframes
        return torch.ones(1, 2, 4), torch.full((2,), MINIMAX_H3_TEXT_TAG)

    def _encode_keyframes(
        self,
        keyframes: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> None:
        del keyframes, latent_height, latent_width, generator
        return None

    def _decode_video(
        self,
        rows: torch.Tensor,
        num_condition_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        del rows, num_condition_rows, num_latent_frames, latent_height, latent_width
        return torch.zeros(1, 124, 32, 32, 3, dtype=torch.uint8)

    def _decode_audio(self, rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
        del rows
        return torch.zeros(1, 2, num_audio_latents * 800)


class SyntheticH3Pipeline(_SyntheticInputs, MiniMaxH3Pipeline):
    pass


class SyntheticFastH3Pipeline(_SyntheticInputs, FastH3Pipeline):
    pass
