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

from functools import cache
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import ParakeetEncoderConfig, PretrainedConfig
from transformers.audio_utils import mel_filter_bank

from ...logger import logger
from ..modules.rms_norm import RMSNorm

EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24


class ParakeetExtractor:
    def __init__(self, config: PretrainedConfig) -> None:
        # Keep `self.config` as the single source of truth because
        # `HFParakeetEncoder._get_subsampling_output_length` reads subsampling fields from it.
        self.config = _ExtractorConfig.from_hf_config(config)

        self._clip_target_samples = round(self.config.clip_duration_s * self.sampling_rate)
        self._tail_min_samples = round(self.config.clip_min_duration_s * self.sampling_rate)

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    @staticmethod
    @cache
    def _get_window(win_length: int, device: str) -> torch.Tensor:
        # Cache the Hann window used by STFT for each length/device pair.
        return torch.hann_window(win_length, periodic=False, device=device)

    @staticmethod
    @cache
    def _get_mel_filters(
        feature_size: int, sampling_rate: int, n_fft: int, device: str
    ) -> torch.Tensor:
        # Create overlapping triangular filters spaced on the Mel scale
        # to project FFT frequency bins into perceptually meaningful bands.
        filter_bank = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return torch.from_numpy(filter_bank.T).to(device=device, dtype=torch.float32)

    def _torch_extract_fbank_features(
        self, waveform: torch.Tensor, device: str | torch.device
    ) -> torch.Tensor:
        # Convert raw waveforms to complex spectra before applying Mel projection.
        device = str(torch.device(device))
        cfg = self.config
        window = self._get_window(cfg.win_length, device)
        stft = torch.stft(
            waveform,
            cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        mel_filters = self._get_mel_filters(cfg.feature_size, cfg.sampling_rate, cfg.n_fft, device)
        return self._apply_mel_filters(stft, mel_filters)

    @torch.compile(dynamic=True)
    def _apply_mel_filters(
        self, stft_output: torch.Tensor, mel_filters: torch.Tensor
    ) -> torch.Tensor:
        # Use power spectra before Mel projection to match the Parakeet feature extractor.
        magnitudes = stft_output.real.square() + stft_output.imag.square()
        mel_spec = mel_filters @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)
        return mel_spec.permute(0, 2, 1)

    @torch.compile(dynamic=True)
    def _apply_preemphasis(
        self, input_features: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        # Apply a first-order high-pass filter while keeping padded samples zeroed.
        preemphasis = self.config.preemphasis
        if preemphasis is None:
            return input_features
        timemask = torch.arange(input_features.shape[1], device=input_features.device).unsqueeze(
            0
        ) < audio_lengths.unsqueeze(1)
        input_features = torch.cat(
            [
                input_features[:, :1],
                input_features[:, 1:] - preemphasis * input_features[:, :-1],
            ],
            dim=1,
        )
        return input_features.masked_fill(~timemask, 0.0)

    @torch.compile(dynamic=True)
    def _normalize_mel_features(
        self, mel_features: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize each clip over valid frames only and return the frame mask.
        features_lengths = torch.floor_divide(
            audio_lengths + self.config.n_fft // 2 * 2 - self.config.n_fft,
            self.config.hop_length,
        )
        attention_mask = (
            torch.arange(mel_features.shape[1], device=mel_features.device)[None, :]
            < features_lengths[:, None]
        )
        mask = attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1)
        mel_features_masked = mel_features * mask
        mean = (mel_features_masked.sum(dim=1) / lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_features_masked - mean) ** 2 * mask).sum(dim=1) / (lengths - 1).unsqueeze(
            -1
        )
        std = torch.sqrt(variance).unsqueeze(1)
        return (mel_features - mean) / (std + EPSILON) * mask, attention_mask

    def _pad_raw_speech(
        self, raw_speech: list[torch.Tensor], max_len: int, device: str | torch.device
    ) -> torch.Tensor:
        # Pack variable-length clips into a dense batch for vectorized feature extraction.
        output = torch.full(
            (len(raw_speech), max_len),
            self.config.padding_value,
            device=device,
            dtype=torch.float32,
        )
        dsts = [output[i, : raw_speech[i].shape[0]] for i in range(len(raw_speech))]
        srcs = [speech.squeeze(-1) for speech in raw_speech]
        torch._foreach_copy_(dsts, srcs)
        return output

    def _clip_sizes(self, audio_len: int) -> list[int]:
        # Keep regular clips fixed-size and pad short tails to the minimum duration.
        audio_len = max(audio_len, self._tail_min_samples)
        num_full_clips, remainder = divmod(audio_len, self._clip_target_samples)
        clip_sizes = [self._clip_target_samples] * num_full_clips
        if remainder > 0:
            clip_sizes.append(max(remainder, self._tail_min_samples))
        return clip_sizes

    def audio_token_count(self, audio_len: int) -> int:
        # Reuse HF subsampling math so token estimates match the encoder output.
        clip_sizes = self._clip_sizes(audio_len)
        num_frames = torch.tensor(
            [cs // self.config.hop_length for cs in clip_sizes], dtype=torch.float
        )
        n_tokens = HFParakeetEncoder._get_subsampling_output_length(self, num_frames)
        return max(1, int(n_tokens.sum().item()))

    def _to_mono_tensor(
        self, audio: np.ndarray | torch.Tensor, device: str | torch.device
    ) -> torch.Tensor:
        # Convert NumPy or Torch audio to a float32 mono tensor on the target device.
        audio_tensor = torch.as_tensor(audio, device=device, dtype=torch.float32)
        if audio_tensor.ndim == 2:
            if audio_tensor.shape[1] == 0:
                raise ValueError(
                    f"Unsupported audio shape {audio_tensor.shape}: expected at least one channel"
                )
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__} "
                "We will take the mean of the channels to convert to mono."
            )
            audio_tensor = audio_tensor.mean(dim=-1)
        elif audio_tensor.ndim != 1:
            raise ValueError(
                f"Unsupported audio shape {audio_tensor.shape}: "
                "expected 1-D (mono) or 2-D (samples x channels)"
            )
        return audio_tensor

    def split_audio_into_clips(self, audio: torch.Tensor) -> list[torch.Tensor]:
        # Pad audio to whole clip boundaries, then slice it into model-sized chunks.
        if audio.ndim != 1:
            raise ValueError(f"Unsupported audio shape {audio.shape}: expected 1-D mono audio")
        audio_len = int(audio.shape[0])
        clip_sizes = self._clip_sizes(audio_len)
        target_len = sum(clip_sizes)
        if audio_len < target_len:
            audio = torch.nn.functional.pad(audio, (0, target_len - audio_len))

        clips: list[torch.Tensor] = []
        offset = 0
        for clip_size in clip_sizes:
            clips.append(audio[offset : offset + clip_size])
            offset += clip_size
        return clips

    def _split_audio_into_clips(self, audio: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        # Compatibility wrapper for callers that pass NumPy audio arrays.
        audio_tensor = self._to_mono_tensor(audio, "cpu")
        return self.split_audio_into_clips(audio_tensor)

    def __call__(
        self,
        raw_speech: list[np.ndarray | torch.Tensor],
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        if len(raw_speech) == 0:
            raise ValueError("raw_speech must contain at least one audio array.")

        raw_speech = [self._to_mono_tensor(audio, device) for audio in raw_speech]
        audio_clips: list[torch.Tensor] = []
        audio_num_clips: list[int] = []
        for audio in raw_speech:
            clips = self.split_audio_into_clips(audio)
            audio_clips.extend(clips)
            audio_num_clips.append(len(clips))

        audio_lengths = torch.tensor(
            [len(speech) for speech in audio_clips],
            dtype=torch.long,
            device=device,
        )
        max_length = max(len(speech) for speech in audio_clips)
        input_features = self._pad_raw_speech(audio_clips, max_length, device)
        if self.config.preemphasis is not None:
            input_features = self._apply_preemphasis(input_features, audio_lengths)
        input_features = self._torch_extract_fbank_features(input_features, device)
        input_features, attention_mask = self._normalize_mel_features(input_features, audio_lengths)

        return {
            "input_audio_features": input_features,
            "feature_attention_mask": attention_mask,
            "audio_num_clips": torch.tensor(audio_num_clips, dtype=torch.long, device=device),
        }

    def audio_length(self, audio_tokens: int) -> int:
        # Estimate raw sample count represented by a number of encoder tokens.
        return int(audio_tokens * self.config.subsampling_factor * self.config.hop_length)


# This config is the extractor's single source of truth. It also supplies the fields read by
# `HFParakeetEncoder._get_subsampling_output_length`.
class _ExtractorConfig(NamedTuple):
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    hop_length: int = 160
    win_length: int = 400
    preemphasis: float | None = 0.97
    n_fft: int = 512
    padding_value: float = 0.0
    clip_duration_s: int = 30
    clip_min_duration_s: float = 0.1

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "_ExtractorConfig":
        return cls(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
            subsampling_conv_stride=config.subsampling_conv_stride,
            hop_length=getattr(config, "hop_length", cls._field_defaults["hop_length"]),
            win_length=getattr(config, "win_length", cls._field_defaults["win_length"]),
            preemphasis=getattr(config, "preemphasis", cls._field_defaults["preemphasis"]),
            n_fft=getattr(config, "n_fft", cls._field_defaults["n_fft"]),
            padding_value=getattr(config, "padding_value", cls._field_defaults["padding_value"]),
        )


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
