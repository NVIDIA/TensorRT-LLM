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
import math
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import ParakeetEncoderConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.audio_utils import mel_filter_bank
from transformers.models.parakeet.modeling_parakeet import (
    ParakeetEncoderRelPositionalEncoding as HFParakeetEncoderRelPositionalEncoding,
)
from transformers.models.parakeet.modeling_parakeet import (
    ParakeetEncoderSubsamplingConv2D as HFParakeetEncoderSubsamplingConv2D,
)

from ...logger import logger
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear
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
        n_tokens = _subsampling_output_length(
            num_frames,
            self.config.subsampling_factor,
            self.config.subsampling_conv_kernel_size,
            self.config.subsampling_conv_stride,
        )
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


def _subsampling_output_length(
    input_lengths: torch.Tensor,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> torch.Tensor:
    num_layers = int(math.log2(subsampling_factor))
    add_pad = (subsampling_conv_kernel_size - 1) // 2 * 2 - subsampling_conv_kernel_size
    lengths = input_lengths
    for _ in range(num_layers):
        lengths = torch.div(lengths.to(torch.float) + add_pad, subsampling_conv_stride) + 1.0
        lengths = torch.floor(lengths)
    return lengths.to(torch.int)


# This config is the extractor's single source of truth. It also supplies the fields read by
# `_subsampling_output_length`.
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

    Several fields required by `ParakeetEncoderConfig` are absent from the raw
    HF composite config, so we map/default them here.
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
        self.linear1 = Linear(hidden_size, projection_hidden_size, bias=bias, dtype=dtype)
        self.linear2 = Linear(projection_hidden_size, llm_hidden_size, bias=bias, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = torch.square(torch.nn.functional.relu(hidden_states))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Conformer encoder
# ---------------------------------------------------------------------------


class ParakeetConformerAttention(Attention):
    """Conformer self-attention built on trtllm.Attention.

    qkv_proj and o_proj are inherited from trtllm.Attention (TP-sharded,
    quantization-ready, fused QKV kernel).  The Transformer-XL relative
    positional encoding (RPE) is computed here and passed directly to
    F.scaled_dot_product_attention as attn_mask.

    relative_k_proj, bias_u, and bias_v implement the content-position term;
    the content-content term uses the standard QK product inside SDPA.
    """

    def __init__(
        self,
        config: ParakeetEncoderConfig,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
    ):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=None,
            bias=config.attention_bias,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=config.attention_bias,
            config=ModelConfig(attn_backend="VANILLA"),
            q_scaling=1.0,
            head_dim=head_dim,
        )
        # Relative-position key projection: position_embeddings → [T_pos, H*d]
        self.relative_k_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=False,
            dtype=dtype,
        )
        # Transformer-XL global content bias added to q for the content-content term
        self.bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=dtype))
        # Transformer-XL global positional bias added to q for content-position term
        self.bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=dtype))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Shaw et al. relative shift. Appendix B of https://arxiv.org/abs/1901.02860."""
        B, H, T, T_pos = x.shape
        x = F.pad(x, (1, 0))
        x = x.view(B, H, -1, T)
        return x[:, :, 1:].view(B, H, T, T_pos)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:       [B, T, hidden_size]
            attention_mask:      Optional [B, 1, T, T] bool mask; True = attend.
            position_embeddings: [B, T_pos, hidden_size] relative position encodings.
        Returns:
            [B, T, hidden_size]
        """
        B, T, C = hidden_states.shape

        qkv = self.qkv_proj(hidden_states.reshape(-1, C))  # [B*T, (q+k+v)*d]
        q, k, v = self.split_qkv(qkv)  # each [B*T, H_rank*d]

        # Reshape to [B, H, T, d] for RPE computation
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ── Transformer-XL RPE: compute matrix_bd ────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)

        # Content-position term: (q + bias_v) @ rel_k^T, then _rel_shift
        q_v = q + self.bias_v.view(1, self.num_heads, 1, self.head_dim)
        rel_k = self.relative_k_proj(position_embeddings.reshape(-1, C)).view(
            B, -1, self.num_heads, self.head_dim
        )  # [B, T_pos, H, d]
        matrix_bd = q_v @ rel_k.permute(0, 2, 3, 1)  # [B, H, T, T_pos]
        matrix_bd = self._rel_shift(matrix_bd)[..., :T] * scale  # [B, H, T, T]

        # Fold in the padding mask (−∞ for masked/padded positions)
        if attention_mask is not None:
            matrix_bd = matrix_bd.masked_fill_(~attention_mask, float("-inf"))

        # Content-content term: add bias_u to q; SDPA computes (q+bias_u)@k^T
        q = q + self.bias_u.view(1, self.num_heads, 1, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=matrix_bd, scale=scale)
        attn_output = out.transpose(1, 2).reshape(B * T, self.q_size)
        return self.o_proj(attn_output).reshape(B, T, C)

    def load_weights(self, weights: Dict[str, torch.Tensor], prefix: str = "") -> None:
        """Load checkpoint weights.

        The checkpoint stores q/k/v as separate projections; qkv_proj is fused.
        """
        self.qkv_proj.load_weights(
            [
                {"weight": weights[f"{prefix}q_proj.weight"]},
                {"weight": weights[f"{prefix}k_proj.weight"]},
                {"weight": weights[f"{prefix}v_proj.weight"]},
            ]
        )
        self.o_proj.load_weights([{"weight": weights[f"{prefix}o_proj.weight"]}])
        self.relative_k_proj.load_weights([{"weight": weights[f"{prefix}relative_k_proj.weight"]}])
        self.bias_u.data.copy_(weights[f"{prefix}bias_u"])
        self.bias_v.data.copy_(weights[f"{prefix}bias_v"])


class ParakeetFeedForward(nn.Module):
    """Conformer FFN: Linear(hidden→intermediate) → activation → Linear(intermediate→hidden)."""

    def __init__(self, config: ParakeetEncoderConfig, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.linear1 = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=bool(config.attention_bias),
            dtype=dtype,
        )
        self.linear2 = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=bool(config.attention_bias),
            dtype=dtype,
        )
        self.activation = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(hidden_states)))


class ParakeetConvolutionModule(nn.Module):
    """Conformer depthwise convolution module.

    Checkpoint weights for pointwise_conv1/2 have shape [C_out, C_in, 1] and are
    squeezed to [C_out, C_in] during load_weights.
    """

    def __init__(self, config: ParakeetEncoderConfig, dtype: Optional[torch.dtype] = None):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        self.activation = ACT2FN[getattr(config, "hidden_act", "silu")]
        self.padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = Linear(
            channels, 2 * channels, bias=config.convolution_bias, dtype=dtype
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = Linear(channels, channels, bias=config.convolution_bias, dtype=dtype)
        if dtype is not None:
            self.depthwise_conv.to(dtype=dtype)
            self.norm.to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.pointwise_conv1(hidden_states)  # [B, T, 2C]
        hidden_states = nn.functional.glu(hidden_states, dim=-1)  # [B, T, C]

        # Depthwise conv operates on [B, C, T]
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                all_masked_rows = torch.all(~attention_mask, dim=2)
            else:
                all_masked_rows = torch.all(~(attention_mask == 0.0), dim=2)
            hidden_states = hidden_states.masked_fill(all_masked_rows, 0.0)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)  # [B, T, C]
        return self.pointwise_conv2(hidden_states)


class ParakeetConformerBlock(nn.Module):
    """Conformer block: FFN → self-attention → convolution → FFN → layer norm."""

    def __init__(
        self,
        config: ParakeetEncoderConfig,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.feed_forward1 = ParakeetFeedForward(config, dtype=dtype)
        self.self_attn = ParakeetConformerAttention(config, layer_idx, dtype=dtype)
        self.conv = ParakeetConvolutionModule(config, dtype=dtype)
        self.feed_forward2 = ParakeetFeedForward(config, dtype=dtype)
        self.norm_feed_forward1 = LayerNorm(hidden_size=config.hidden_size, eps=1e-5, dtype=dtype)
        self.norm_self_att = LayerNorm(hidden_size=config.hidden_size, eps=1e-5, dtype=dtype)
        self.norm_conv = LayerNorm(hidden_size=config.hidden_size, eps=1e-5, dtype=dtype)
        self.norm_feed_forward2 = LayerNorm(hidden_size=config.hidden_size, eps=1e-5, dtype=dtype)
        self.norm_out = LayerNorm(hidden_size=config.hidden_size, eps=1e-5, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 0.5 residual scaling on both FFN halves is part of the Conformer design.
        residual = hidden_states
        hidden_states = residual + 0.5 * self.feed_forward1(self.norm_feed_forward1(hidden_states))

        hidden_states = hidden_states + self.self_attn(
            self.norm_self_att(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        hidden_states = hidden_states + self.conv(
            self.norm_conv(hidden_states), attention_mask=attention_mask
        )

        hidden_states = hidden_states + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )

        return self.norm_out(hidden_states)


class ParakeetEncoder(nn.Module):
    """Conformer encoder for Parakeet audio."""

    def __init__(self, config: ParakeetEncoderConfig, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.config = config
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = HFParakeetEncoderSubsamplingConv2D(config)
        self.encode_positions = HFParakeetEncoderRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [
                ParakeetConformerBlock(config, layer_idx=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )

        if dtype is not None:
            self.subsampling.to(dtype=dtype)
            self.encode_positions.to(dtype=dtype)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return _subsampling_output_length(
            input_lengths,
            cfg.subsampling_factor,
            cfg.subsampling_conv_kernel_size,
            cfg.subsampling_conv_stride,
        )

    def _build_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Convert 2-D input mask to 4-D additive attention mask after subsampling."""
        output_lengths = self._get_subsampling_output_length(attention_mask.sum(-1))
        output_mask = torch.arange(seq_len, device=attention_mask.device) < output_lengths[:, None]
        mask_2d = output_mask.unsqueeze(1).expand(-1, seq_len, -1)
        mask_2d = mask_2d & mask_2d.transpose(1, 2)
        return mask_2d.unsqueeze(1)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        attn_mask_4d = None
        if attention_mask is not None:
            attn_mask_4d = self._build_attention_mask(attention_mask, hidden_states.shape[1])

        for block in self.layers:
            hidden_states = block(
                hidden_states,
                attention_mask=attn_mask_4d,
                position_embeddings=position_embeddings,
            )

        return hidden_states

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load weights from a flat dict with keys relative to the encoder root.

        ParakeetConformerAttention modules load their own weights (fused QKV from
        separate q/k/v checkpoint keys).  All other trtllm.Linear modules are loaded
        via their native API.  Remaining parameters are loaded via state_dict.
        """
        trtllm_param_keys: set[str] = set()
        # Track prefixes of attention modules whose sub-modules must be skipped.
        attn_prefixes: list[str] = []

        for mod_name, module in self.named_modules():
            # Skip sub-modules of already-handled attention blocks: named_modules()
            # recurses into ParakeetConformerAttention so qkv_proj/o_proj would
            # otherwise fall through to the generic Linear branch below.
            if any(mod_name.startswith(p) for p in attn_prefixes):
                continue

            if isinstance(module, ParakeetConformerAttention):
                # Attention owns its own weight loading (fused QKV + relative_k_proj
                # + bias_u/bias_v).  Prefix maps "layers.i.self_attn" → "layers.i.self_attn."
                prefix = f"{mod_name}." if mod_name else ""
                attn_prefixes.append(prefix)
                module.load_weights(weights, prefix=prefix)
                # Checkpoint keys (excluded from remaining dict):
                for suffix in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "relative_k_proj.weight",
                    "bias_u",
                    "bias_v",
                ):
                    trtllm_param_keys.add(f"{prefix}{suffix}")
                if module.qkv_proj.bias is not None:
                    for sfx in ("q_proj.bias", "k_proj.bias", "v_proj.bias"):
                        trtllm_param_keys.add(f"{prefix}{sfx}")
                if module.o_proj.bias is not None:
                    trtllm_param_keys.add(f"{prefix}o_proj.bias")
                # Model parameter names (allowed as missing in load_state_dict):
                # qkv_proj is fused in the model but split in the checkpoint.
                trtllm_param_keys.add(f"{prefix}qkv_proj.weight")
                if module.qkv_proj.bias is not None:
                    trtllm_param_keys.add(f"{prefix}qkv_proj.bias")
            elif isinstance(module, Linear):
                w = weights[f"{mod_name}.weight"]
                # Pointwise Conv1d weights are stored as [C_out, C_in, 1]; squeeze for Linear.
                entry: Dict[str, torch.Tensor] = {"weight": w.squeeze(-1) if w.dim() == 3 else w}
                bias_key = f"{mod_name}.bias"
                if bias_key in weights:
                    entry["bias"] = weights[bias_key]
                module.load_weights([entry])
                trtllm_param_keys.add(f"{mod_name}.weight")
                trtllm_param_keys.add(bias_key)

        remaining = {k: v for k, v in weights.items() if k not in trtllm_param_keys}
        incompatible = self.load_state_dict(remaining, strict=False)

        # Allowed gaps: trtllm weights (loaded above) and optional convolution bias keys.
        unexpected_missing = [
            k
            for k in incompatible.missing_keys
            if k not in trtllm_param_keys and not ("_conv" in k and k.endswith(".bias"))
        ]
        if unexpected_missing:
            raise KeyError(f"Missing encoder weights after loading: {unexpected_missing}")


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
        self.encoder = ParakeetEncoder(encoder_config, dtype=dtype)
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
        return self.projection(self.encoder(input_features, attention_mask))

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        encoder_weights: Dict[str, torch.Tensor] = {}
        projection_weights: Dict[str, torch.Tensor] = {}
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

        self.encoder.load_weights(encoder_weights)

        # projection.linear1/2 are trtllm.Linear; load them via their API.
        # projection.norm (RMSNorm) is loaded via state_dict.
        proj = self.projection
        proj.linear1.load_weights([{"weight": projection_weights["linear1.weight"]}])
        proj.linear2.load_weights([{"weight": projection_weights["linear2.weight"]}])
        norm_weights = {
            k: v
            for k, v in projection_weights.items()
            if not k.startswith("linear1.") and not k.startswith("linear2.")
        }
        proj.load_state_dict(norm_weights, strict=False)
