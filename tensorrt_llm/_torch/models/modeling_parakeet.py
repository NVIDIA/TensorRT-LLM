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

from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import ParakeetEncoderConfig, ParakeetFeatureExtractor, PretrainedConfig

from ..modules.rms_norm import RMSNorm


class ParakeetExtractor(ParakeetFeatureExtractor):
    def __init__(self, config: PretrainedConfig) -> None:
        self.config = _ExtractorConfig(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
            subsampling_conv_stride=config.subsampling_conv_stride,
        )
        super().__init__(**self.config._asdict())

        self._clip_target_samples = int(round(self.config.clip_duration_s * self.sampling_rate))
        self._tail_min_samples = int(round(self.config.clip_min_duration_s * self.sampling_rate))

    def _clip_sizes(self, audio_len: int) -> list[int]:
        audio_len = max(audio_len, self._tail_min_samples)
        num_full_clips, remainder = divmod(audio_len, self._clip_target_samples)
        clip_sizes = [self._clip_target_samples] * num_full_clips
        if remainder > 0:
            clip_sizes.append(max(remainder, self._tail_min_samples))
        return clip_sizes

    def audio_token_count(self, audio_len: int) -> int:
        clip_sizes = self._clip_sizes(audio_len)
        num_frames = torch.tensor([cs // self.hop_length for cs in clip_sizes], dtype=torch.float)
        # NOTE: this is a massive hack in order not to duplicate the functionality here.
        n_tokens = HFParakeetEncoder._get_subsampling_output_length(self, num_frames)
        return max(1, int(n_tokens.sum().item()))

    def _split_audio_into_clips(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.ndim == 2:
            if audio.shape[1] == 0:
                raise ValueError(
                    f"Unsupported audio shape {audio.shape}: expected at least one channel"
                )
            audio = audio.mean(axis=1)
        elif audio.ndim != 1:
            raise ValueError(
                f"Unsupported audio shape {audio.shape}: "
                "expected 1-D (mono) or 2-D (samples x channels)"
            )
        audio_len = int(audio.shape[0])
        clip_sizes = self._clip_sizes(audio_len)
        target_len = sum(clip_sizes)
        if audio_len < target_len:
            audio = np.pad(audio, (0, target_len - audio_len))

        split_indices = np.cumsum(clip_sizes[:-1])
        return np.split(audio, split_indices)

    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs) -> torch.Tensor:
        audio_clips = list[np.ndarray]()
        audio_num_clips = list[int]()
        for audio in raw_speech:
            clips = self._split_audio_into_clips(audio)
            audio_clips.extend(clips)
            audio_num_clips.append(len(clips))

        outputs = super().__call__(audio_clips, *args, **kwargs)
        outputs["audio_num_clips"] = torch.tensor(audio_num_clips, dtype=torch.long)
        return outputs

    def audio_length(self, audio_tokens: int) -> int:
        return int(audio_tokens * self.config.subsampling_factor * self.hop_length)


# The sole purpose of this config object is so that we are able to make the call to
# `HFParakeetEncoder._get_subsampling_output_length`, which just reads a few of the below values.
class _ExtractorConfig(NamedTuple):
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    clip_duration_s: int = 30
    clip_min_duration_s: float = 0.1


def _make_parakeet_encoder_config(
    sound_config: PretrainedConfig,
) -> ParakeetEncoderConfig:
    """Build a `ParakeetEncoderConfig` from the HF `sound_config`.

    `HFParakeetEncoder` expects fields (`scale_input`, `attention_bias`, `max_position_embeddings`)
    that are not present in the raw HF composite config, so we map/default them here.
    """
    return ParakeetEncoderConfig(
        hidden_size=sound_config.hidden_size,
        num_hidden_layers=sound_config.num_hidden_layers,
        num_attention_heads=sound_config.num_attention_heads,
        intermediate_size=sound_config.intermediate_size,
        conv_kernel_size=sound_config.conv_kernel_size,
        convolution_bias=getattr(sound_config, "convolution_bias", False),
        subsampling_factor=sound_config.subsampling_factor,
        subsampling_conv_channels=sound_config.subsampling_conv_channels,
        subsampling_conv_kernel_size=sound_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=sound_config.subsampling_conv_stride,
        num_mel_bins=sound_config.num_mel_bins,
        # Fields required by HFParakeetEncoder but absent from the HF
        # composite config — use the defaults from ParakeetEncoderConfig.
        scale_input=getattr(sound_config, "scale_input", False),
        attention_bias=getattr(sound_config, "attention_bias", False),
        max_position_embeddings=getattr(sound_config, "max_position_embeddings", 5000),
    )


class ParakeetProjection(nn.Module):
    """MLP projection: RMSNorm → Linear → SquaredReLU → Linear."""

    def __init__(
        self,
        hidden_size: int,
        projection_hidden_size: int,
        llm_hidden_size: int,
        bias: bool = False,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype)
        self.linear1 = nn.Linear(hidden_size, projection_hidden_size, bias=bias, dtype=dtype)
        self.linear2 = nn.Linear(projection_hidden_size, llm_hidden_size, bias=bias, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = torch.square(torch.nn.functional.relu(hidden_states))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ProjectedParakeet(nn.Module):
    """Parakeet encoder + MLP projection into the LLM embedding space."""

    def __init__(
        self,
        sound_config: PretrainedConfig,
        llm_hidden_size: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        encoder_config = _make_parakeet_encoder_config(sound_config)
        self.encoder = HFParakeetEncoder(encoder_config).to(dtype=dtype)
        self.projection = ParakeetProjection(
            hidden_size=sound_config.hidden_size,
            projection_hidden_size=sound_config.projection_hidden_size,
            llm_hidden_size=llm_hidden_size,
            bias=getattr(sound_config, "projection_bias", False),
            dtype=dtype,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        return self.projection(hidden_states)

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        encoder_weights = {}
        projection_weights = {}
        for key, value in weights.items():
            if key.startswith("sound_encoder.encoder."):
                sub_key = key[len("sound_encoder.encoder.") :]
                # Skip feature extractor weights — handled by ParakeetExtractor.
                if sub_key.startswith("feature_extractor."):
                    continue
                encoder_weights[sub_key] = value
            elif key.startswith("sound_projection."):
                sub_key = key[len("sound_projection.") :]
                projection_weights[sub_key] = value

        # There was a bug in `transformers` prior to version 5.0, where the
        # `ParakeetEncoderConvolutionModule` would ignore the value of `config.convolution_bias`
        # and always use a bias.
        incompatible_keys = self.encoder.load_state_dict(encoder_weights, strict=False)
        non_conv_bias_keys = [
            key
            for key in incompatible_keys.missing_keys
            if not ("_conv" in key and key.endswith(".bias"))
        ]
        if len(non_conv_bias_keys) > 0:
            raise KeyError(
                f"The following keys were missing from the checkpoint: {non_conv_bias_keys}."
            )
        self.projection.load_state_dict(projection_weights, strict=True)
