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
"""PyTorch-flow Whisper encoder-decoder (ASR) model for TensorRT-LLM.

Covers ``WhisperForConditionalGeneration`` (OpenAI Whisper).

Whisper is an audio encoder-decoder: a log-mel spectrogram is consumed by a
convolutional + Transformer audio encoder, and a text decoder attends to the
encoder output via cross-attention while generating a transcript
autoregressively.

The request carries the raw 30 s-padded waveform (not the mel): the input
processor only pads/validates audio on the host, and the encoder computes the
log-mel spectrogram on GPU inside the engine process (batched across the
encoder step; see :class:`WhisperLogMelFrontend`).

Key differences from BART:
    - Pre-norm (LayerNorm → sub-layer → residual add) instead of post-norm.
    - The encoder ingests a mel-feature tensor through a 2x ``Conv1d`` stem
      (the second conv has stride 2, halving the time axis), not token ids.
    - Encoder and decoder embeddings are NOT tied (the encoder has no vocab
      embedding); ``lm_head`` (``proj_out``) is tied to the decoder token
      embedding.
    - Learned absolute positional embeddings with NO index offset (BART uses 2).
    - No embedding scale.
    - ``k_proj`` has no bias (``q_proj``/``v_proj``/``out_proj`` do); the missing
      key bias is materialized as zeros at weight-load time.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, WhisperConfig

from ...inputs import (
    ExtraProcessedInputs,
    InputProcessor,
    MultimodalPlaceholderMetadata,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.cross_attention import CrossAttention
from ..modules.embedding import Embedding, LMHead
from ..modules.layer_norm import LayerNorm
from ..modules.linear import TensorParallelMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.mlp import MLP
from .modeling_utils import PostInitCaller, register_auto_model

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _packed_position_ids(
    position_ids: Optional[torch.IntTensor],
    hidden_states: torch.Tensor,
) -> Optional[torch.IntTensor]:
    if position_ids is None:
        return None

    position_ids = position_ids.reshape(-1)
    if position_ids.numel() != hidden_states.shape[0]:
        raise ValueError(
            "Whisper packed position_ids must match hidden_states tokens: "
            f"got {position_ids.numel()} positions for {hidden_states.shape[0]} tokens."
        )
    return position_ids


# ---------------------------------------------------------------------------
# Whisper Attention
# ---------------------------------------------------------------------------


class WhisperSelfAttention(Attention):
    """Whisper-style MHA with bias and no in-kernel positional encoding.

    Whisper adds learned positional embeddings to the input before the
    attention layer, so no RoPE or other in-kernel positional encoding is
    used. Query scaling by ``head_dim**-0.5`` is applied inside the standard
    attention path (``q_scaling=1.0``), which is numerically identical to
    Whisper's convention of pre-scaling the query and using ``scaling=1.0``.
    """

    def __init__(
        self,
        model_config: ModelConfig[WhisperConfig],
        num_heads: int,
        max_positions: int,
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.d_model,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=max_positions,
            bias=True,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def apply_rope(self, q, k, v, position_ids):
        """Whisper uses learned pos embeddings, not RoPE — pass through."""
        return q, k, v


class WhisperCrossAttention(CrossAttention):
    """Whisper-style cross-attention with bias (decoder attends to encoder)."""

    def __init__(
        self,
        model_config: ModelConfig[WhisperConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        num_heads = config.decoder_attention_heads
        super().__init__(
            hidden_size=config.d_model,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            encoder_hidden_size=config.d_model,
            bias=True,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


# ---------------------------------------------------------------------------
# Encoder layer (pre-norm)
# ---------------------------------------------------------------------------


class WhisperEncoderLayer(nn.Module):
    """Whisper encoder layer (pre-norm): LN → self-attn → add → LN → MLP → add."""

    def __init__(
        self,
        model_config: ModelConfig[WhisperConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model

        self.self_attn = WhisperSelfAttention(
            model_config,
            num_heads=config.encoder_attention_heads,
            max_positions=config.max_source_positions,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=config.encoder_ffn_dim,
            bias=True,
            activation=F.gelu,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.final_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention (pre-norm, bidirectional/FULL mask)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )
        hidden_states = residual + hidden_states

        # MLP (pre-norm)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer (pre-norm)
# ---------------------------------------------------------------------------


class WhisperDecoderLayer(nn.Module):
    """Whisper decoder layer (pre-norm): self-attn → cross-attn → MLP."""

    def __init__(
        self,
        model_config: ModelConfig[WhisperConfig],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        hidden_size = config.d_model

        self.self_attn = WhisperSelfAttention(
            model_config,
            num_heads=config.decoder_attention_heads,
            max_positions=config.max_target_positions,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

        self.cross_attn = WhisperCrossAttention(model_config, layer_idx=layer_idx)
        self.cross_attn_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=config.decoder_ffn_dim,
            bias=True,
            activation=F.gelu,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.final_layer_norm = LayerNorm(
            hidden_size=hidden_size,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Self-attention (pre-norm, causal)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )
        hidden_states = residual + hidden_states

        # Cross-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attn_metadata=attn_metadata,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
        )
        hidden_states = residual + hidden_states

        # MLP (pre-norm)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------


def _load_hf_feature_extractor(config: WhisperConfig):
    """The checkpoint's ``WhisperFeatureExtractor`` (``preprocessor_config.json``).

    STFT/mel parameters live there, not in the model config, so they are
    re-loaded from ``config._name_or_path``. Falls back to Whisper defaults
    with ``config.num_mel_bins`` filters when the checkpoint ships no
    preprocessor config or its ``feature_size`` contradicts the model config.
    """
    from transformers import WhisperFeatureExtractor

    extractor = None
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path:
        try:
            extractor = WhisperFeatureExtractor.from_pretrained(name_or_path)
        except (OSError, ValueError) as e:
            logger.warning(
                f"Could not load a Whisper feature-extractor config from "
                f"{name_or_path!r} ({e}); using Whisper default STFT/mel parameters."
            )
    if extractor is not None and int(extractor.feature_size) != int(config.num_mel_bins):
        logger.warning(
            f"preprocessor_config.json feature_size ({extractor.feature_size}) "
            f"contradicts config.num_mel_bins ({config.num_mel_bins}); using "
            f"Whisper default STFT/mel parameters with {config.num_mel_bins} mel bins."
        )
        extractor = None
    if extractor is None:
        extractor = WhisperFeatureExtractor(feature_size=config.num_mel_bins)
    return extractor


class WhisperLogMelFrontend(nn.Module):
    """GPU log-mel spectrogram front-end, numerics-identical to the HF
    ``WhisperFeatureExtractor`` torch path (``_torch_extract_fbank_features``).

    Consumes the raw zero-padded waveform batch shipped by
    :class:`WhisperInputProcessor` and produces ``[batch, num_mel_bins, frames]``
    in fp32 (STFT precision; the caller casts to the model dtype). Kept as a
    separate module so a future encoder CUDA-graph capture can choose to keep
    the STFT outside the graphed region.

    STFT/mel parameters and the filterbank come from the checkpoint's feature
    extractor; the class attributes below are the Whisper-family defaults.
    """

    N_FFT = 400
    HOP_LENGTH = 160
    SAMPLING_RATE = 16000

    def __init__(self, config: WhisperConfig):
        super().__init__()
        extractor = _load_hf_feature_extractor(config)
        self.n_fft = int(extractor.n_fft)
        self.hop_length = int(extractor.hop_length)
        # Pre-STFT Gaussian noise, applied where HF applies it; 0.0 (all
        # official checkpoints) disables it.
        self.dither = float(getattr(extractor, "dither", 0.0))
        self._mel_filters_np = extractor.mel_filters
        # Materialized lazily on the input device (NOT register_buffer: these
        # are derived constants that must stay fp32 and out of the state dict,
        # and lazy creation sidesteps meta-device module initialization).
        self._mel_filters: Optional[torch.Tensor] = None
        self._window: Optional[torch.Tensor] = None

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: [batch, n_samples] fp32, zero-padded to the fixed window.
        if self._mel_filters is None or self._mel_filters.device != waveforms.device:
            self._mel_filters = torch.from_numpy(self._mel_filters_np).to(
                waveforms.device, torch.float32
            )
            self._window = torch.hann_window(self.n_fft, device=waveforms.device)

        waveforms = waveforms.to(torch.float32)
        if self.dither != 0.0:
            # Out-of-place: the input tensor is the request's feature buffer.
            waveforms = waveforms + self.dither * torch.randn_like(waveforms)
        stft = torch.stft(
            waveforms,
            self.n_fft,
            self.hop_length,
            window=self._window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # Per-sample dynamic-range floor (batched samples must not share a max).
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        return (log_spec + 4.0) / 4.0


class WhisperEncoder(nn.Module):
    """Whisper audio encoder: log-mel front-end + 2x Conv1d stem + positions +
    self-attn layers."""

    def __init__(self, model_config: ModelConfig[WhisperConfig]):
        super().__init__()
        config = model_config.pretrained_config
        embed_dim = config.d_model

        self.log_mel = WhisperLogMelFrontend(config)

        # 2x Conv1d stem. conv2 has stride 2, halving the time axis
        # (3000 mel frames -> 1500 encoder positions).
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            embed_dim,
            kernel_size=3,
            padding=1,
            dtype=config.torch_dtype,
        )
        self.conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=config.torch_dtype,
        )

        # Learned absolute positional embedding for all source positions.
        self.embed_positions = Embedding(
            config.max_source_positions,
            embed_dim,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(model_config, layer_idx=i) for i in range(config.encoder_layers)]
        )
        self.layer_norm = LayerNorm(
            hidden_size=embed_dim,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # input_features: [batch, n_samples] raw 30 s-padded waveform (the
        # request contract), or a precomputed [batch, num_mel_bins, 3000] mel
        # (kept for direct-callers/validation harnesses).
        if input_features.dim() == 2:
            input_features = self.log_mel(input_features)
        input_features = input_features.to(self.conv1.weight.dtype)
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        # [batch, embed_dim, seq_len] -> [batch, seq_len, embed_dim]
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        hidden_states = inputs_embeds + self.embed_positions.weight
        # Pack to [num_tokens, embed_dim] for the attention backend.
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attn_metadata=attn_metadata)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):
    """Whisper text decoder: token + positional embedding + decoder layers."""

    def __init__(self, model_config: ModelConfig[WhisperConfig]):
        super().__init__()
        config = model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.d_model,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        # Whisper decoder positions have NO index offset (unlike BART's +2).
        self.embed_positions = Embedding(
            config.max_target_positions,
            config.d_model,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(model_config, layer_idx=i) for i in range(config.decoder_layers)]
        )
        self.layer_norm = LayerNorm(
            hidden_size=config.d_model,
            eps=1e-5,
            dtype=config.torch_dtype,
            has_bias=True,
        )

    def forward(
        self,
        input_ids: Optional[torch.IntTensor],
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        position_ids = _packed_position_ids(position_ids, hidden_states)
        if position_ids is not None:
            hidden_states = hidden_states + self.embed_positions(position_ids)

        for layer in self.layers:
            hidden_states = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
                cross_attn_metadata=cross_attn_metadata,
                skip_cross_kv_projection=skip_cross_kv_projection,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Input processor (audio -> padded waveform + forced decoder prompt)
# ---------------------------------------------------------------------------


class WhisperInputProcessor(InputProcessor):
    """Host-side Whisper preprocessing for the LLM API.

    Validates and zero-pads ``multi_modal_data["audio"]`` to the fixed 30 s
    window (the log-mel spectrogram itself is computed on GPU inside the
    engine, see :class:`WhisperLogMelFrontend`), and returns the forced
    decoder prompt as the request's token ids. An empty text prompt selects
    the checkpoint default
    (``<|startoftranscript|>[<|en|>][<|transcribe|>]<|notimestamps|>``); a
    non-empty text prompt is tokenized verbatim as the decoder prompt, which
    is how language/task are overridden, e.g.
    ``"<|startoftranscript|><|de|><|transcribe|><|notimestamps|>"``.
    Pre-tokenized ``prompt_token_ids`` are not consumed.

    The padded waveform rides ``multimodal_data["audio"]`` under
    ``encoder_input_features`` + ``encoder_output_len``, which
    ``executor_request_to_llm_request`` forwards into the request's native
    encoder fields.
    """

    # Marks this model as feature-driven for the encoder side: prompts
    # without audio cannot be served and are rejected at submission.
    requires_encoder_features = True

    # Whisper-family defaults (30 s at 16 kHz); the effective window comes
    # from the checkpoint's feature extractor in ``__init__``. Longer audio
    # is rejected instead of silently truncated (long-form chunking is a
    # separate feature).
    MAX_AUDIO_SECONDS = 30.0
    SAMPLING_RATE = 16000

    def __init__(self, model_path, config, tokenizer, trust_remote_code: bool = True, **kwargs):
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        # No decoder-side placeholder tokens exist to hash against; skip the
        # multimodal-hashing probe (the registry then always uses this
        # processor directly).
        self.multimodal_hashing_supported = False
        # WhisperProcessor = feature extractor (the reference log-mel
        # implementation) + tokenizer. Also consumed by accuracy evaluators
        # via ``llm.input_processor.processor``.
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Audio window/sampling rate from the checkpoint's feature extractor
        # (WhisperLogMelFrontend reads the same config engine-side).
        extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(extractor, "sampling_rate", None) or self.SAMPLING_RATE)
        self.n_samples = int(
            getattr(extractor, "n_samples", None) or self.MAX_AUDIO_SECONDS * self.sampling_rate
        )
        self.max_audio_seconds = self.n_samples / float(self.sampling_rate)
        # The mel frames halved by the conv stem must fill the encoder
        # position table exactly, or every downstream cross-KV size is wrong.
        hop_length = int(getattr(extractor, "hop_length", None) or WhisperLogMelFrontend.HOP_LENGTH)
        encoder_positions = self.n_samples // hop_length // 2
        if encoder_positions != int(self.config.max_source_positions):
            raise ValueError(
                f"Inconsistent Whisper checkpoint: the feature extractor's "
                f"{self.max_audio_seconds:.1f}s window at {self.sampling_rate} Hz "
                f"(hop {hop_length}) yields {encoder_positions} encoder "
                f"positions, but config.max_source_positions is "
                f"{self.config.max_source_positions}."
            )
        self._decoder_prompt = self._build_decoder_prompt()

    # The decoder prompt contains no multimodal placeholder tokens (the audio
    # feeds the encoder), so the embed-mask/cumsum machinery has nothing to
    # find — returning None from all three makes it skip cleanly.
    def get_vocab_size(self) -> Optional[int]:
        return None

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        return None

    def get_mm_special_token_ids(self) -> Optional[torch.Tensor]:
        return None

    def _build_decoder_prompt(self) -> List[int]:
        """``[decoder_start] + forced task tokens`` from the checkpoint."""
        start_id = getattr(self.config, "decoder_start_token_id", None)
        if start_id is None:
            raise ValueError(
                "Whisper requires config.decoder_start_token_id to build the decoder prompt."
            )
        try:
            # Multilingual checkpoints force <|lang|><|task|>. Defaults are
            # English transcription; user-facing overrides are a follow-up.
            forced = self.processor.get_decoder_prompt_ids(
                language="en", task="transcribe", no_timestamps=True
            )
        except ValueError:
            # English-only checkpoints (*.en) have no language/task tokens.
            forced = self.processor.get_decoder_prompt_ids(no_timestamps=True)
        return [int(start_id)] + [int(tok) for _, tok in sorted(forced)]

    def _resolve_decoder_prompt(self, prompt_text: Optional[str]) -> List[int]:
        """Checkpoint-default forced prompt, or the user's decoder prompt.

        A non-empty text prompt is tokenized verbatim (special tokens
        resolve to their ids) and must start with ``<|startoftranscript|>``;
        this is the language/task override mechanism.
        """
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            return list(self._decoder_prompt)

        decoder_prompt = self.processor.tokenizer.encode(prompt_text, add_special_tokens=False)
        start_id = int(self.config.decoder_start_token_id)
        if not decoder_prompt or decoder_prompt[0] != start_id:
            raise ValueError(
                "A Whisper text prompt overrides the decoder prompt and must "
                f"start with <|startoftranscript|> (token {start_id}), e.g. "
                "'<|startoftranscript|><|de|><|transcribe|><|notimestamps|>'; "
                f"got {prompt_text[:80]!r}."
            )
        max_prompt = int(self.config.max_target_positions) - 1
        if len(decoder_prompt) > max_prompt:
            raise ValueError(
                f"Whisper decoder prompt has {len(decoder_prompt)} tokens; at "
                f"most {max_prompt} fit the decoder position table."
            )
        return [int(token) for token in decoder_prompt]

    def _load_waveform(self, item: Any) -> np.ndarray:
        if isinstance(item, str):
            from ...inputs.utils import load_audio

            waveform, sample_rate = load_audio(item)
        elif isinstance(item, dict) and "array" in item:
            # HF datasets audio format: {"array": ..., "sampling_rate": ...}
            waveform = item["array"]
            sample_rate = item.get("sampling_rate", self.sampling_rate)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            waveform, sample_rate = item
        else:
            raise TypeError(
                "Unsupported audio item for Whisper: expected a file "
                "path/URL, an (array, sample_rate) tuple, or a dict with "
                f"'array'/'sampling_rate'; got {type(item).__name__}."
            )

        if int(sample_rate) != self.sampling_rate:
            raise ValueError(
                f"Whisper expects {self.sampling_rate} Hz audio; got "
                f"{sample_rate} Hz. Resample on the client side."
            )

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            # soundfile returns [frames, channels]; downmix to mono.
            waveform = waveform.mean(axis=1)
        if waveform.shape[0] > self.n_samples:
            duration = waveform.shape[0] / float(self.sampling_rate)
            raise ValueError(
                f"Audio is {duration:.2f}s long, but Whisper supports at "
                f"most {self.max_audio_seconds:.1f}s per request; chunk the "
                "input on the client side."
            )
        return waveform

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        mm_data = inputs.get("multi_modal_data") or {}
        audio_items = mm_data.get("audio")
        if audio_items is None:
            raise ValueError(
                "Whisper requires multi_modal_data['audio'] (the text prompt "
                "carries no encoder input)."
            )
        if not isinstance(audio_items, list):
            audio_items = [audio_items]
        if len(audio_items) != 1:
            raise ValueError(
                f"Whisper supports exactly one audio clip per request; got {len(audio_items)}."
            )

        decoder_prompt = self._resolve_decoder_prompt(inputs.get("prompt"))

        # The engine's max_seq_len covers the 1500-position encoder pass, so it
        # can't protect the smaller decoder position table (max_target_positions
        # rows). Cap generation to the table here instead; re-capping a shared
        # SamplingParams is idempotent.
        decoder_budget = int(self.config.max_target_positions) - len(decoder_prompt)
        if sampling_params.max_tokens is None or sampling_params.max_tokens > decoder_budget:
            if sampling_params.max_tokens is not None:
                logger.warning(
                    f"Capping max_tokens from {sampling_params.max_tokens} to "
                    f"{decoder_budget}: Whisper's decoder position table has "
                    f"{self.config.max_target_positions} rows and the prompt "
                    f"uses {len(decoder_prompt)}."
                )
            sampling_params.max_tokens = decoder_budget

        waveform = self._load_waveform(audio_items[0])
        # Zero-pad to the fixed window, as the HF extractor does. Shipped as
        # fp32 [1, n_samples]: the STFT needs fp32, and the leading dim of 1
        # keeps the C++ request's encoder-input-length bookkeeping unchanged.
        n_samples = self.n_samples
        padded = np.zeros((1, n_samples), dtype=np.float32)
        padded[0, : waveform.shape[0]] = waveform
        input_features = torch.from_numpy(padded)

        extra = {
            "multimodal_data": {
                "audio": {
                    # Purpose-specific key (not the generic HF "input_features"):
                    # its presence routes the request through the enc-dec
                    # encoder step.
                    "encoder_input_features": input_features.contiguous(),
                    # Post-conv position count (mel frames // 2): the cross-KV
                    # capacity every downstream consumer sizes against.
                    "encoder_output_len": int(self.config.max_source_positions),
                }
            }
        }
        return decoder_prompt, extra


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class WhisperModel(nn.Module):
    """Whisper encoder-decoder body (no lm_head)."""

    def __init__(self, model_config: ModelConfig[WhisperConfig]):
        super().__init__()
        self.model_config = model_config

        self.encoder = WhisperEncoder(model_config)
        self.decoder = WhisperDecoder(model_config)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        encoder_attn_metadata: Optional[AttentionMetadata] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Run the audio encoder unless its output is already supplied.
        if encoder_hidden_states is None and input_features is not None:
            assert encoder_attn_metadata is not None
            encoder_hidden_states = self.encoder(
                input_features=input_features,
                attn_metadata=encoder_attn_metadata,
            )

        decoder_output = self.decoder(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
            inputs_embeds=inputs_embeds,
        )
        return decoder_output


@register_input_processor(
    WhisperInputProcessor,
    model_type="whisper",
    # Whisper has no decoder-side audio placeholder tokens (the audio goes to
    # the encoder); the metadata only affects chat-template serving.
    placeholder_metadata=MultimodalPlaceholderMetadata(placeholder_map={"audio": ""}),
)
@register_auto_model("WhisperForConditionalGeneration")
class WhisperForConditionalGeneration(nn.Module, metaclass=PostInitCaller):
    """Whisper encoder-decoder model with LM head."""

    def __init__(self, model_config: ModelConfig[WhisperConfig]):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.model = WhisperModel(model_config)

        self.lm_head = LMHead(
            config.vocab_size,
            config.d_model,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        )

        # Whisper ties the LM head (``proj_out``) to the decoder token embedding.
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.decoder.embed_tokens.weight

        self.logits_processor = LogitsProcessor()

    def __post_init__(self):
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def __pp_init__(self):
        pass

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attn_metadata: Optional[AttentionMetadata] = None,
        cross_attn_metadata: Optional[AttentionMetadata] = None,
        skip_cross_kv_projection: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            input_features=input_features,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
            encoder_attn_metadata=encoder_attn_metadata,
            cross_attn_metadata=cross_attn_metadata,
            skip_cross_kv_projection=skip_cross_kv_projection,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor.forward(
            hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def infer_max_seq_len(self) -> int:
        # One engine-level max_seq_len must cover the encoder pass, which packs
        # max_source_positions (1500) — more than the decoder's
        # max_target_positions (448). Decoder generation length is capped to the
        # position table separately, in WhisperInputProcessor.
        config = self.model_config.pretrained_config
        return max(
            getattr(config, "max_target_positions", 448),
            getattr(config, "max_source_positions", 1500),
        )

    def load_weights(self, weights: Dict, **kwargs):
        config = self.model_config.pretrained_config
        tllm_weights = _convert_hf_whisper_weights(
            weights, config, dtype=self.model_config.torch_dtype
        )

        consumed = set()
        for name, module in self.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            w = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=w)
            else:
                for n, p in module.named_parameters(recurse=False):
                    if n in w[0]:
                        p.data.copy_(w[0][n][:])
            consumed.add(name)

        # A converter key that matches no module would silently leave that
        # module at its random init (usually non-NaN) — fail loudly instead.
        unconsumed = sorted(set(tllm_weights) - consumed)
        if unconsumed:
            raise ValueError(
                "Converted Whisper weights match no module in the model tree "
                f"(converter/module-name mismatch): {unconsumed}"
            )


def _convert_hf_whisper_weights(
    hf_weights: Dict[str, torch.Tensor],
    config: WhisperConfig,
    dtype: Optional[torch.dtype] = None,
) -> Dict:
    """Map HuggingFace Whisper state_dict keys to TRT-LLM module-tree keys.

    HF Whisper weight layout (prefix ``model.``; ``proj_out`` is the LM head,
    tied to ``model.decoder.embed_tokens``):
        model.encoder.conv1.{weight,bias}
        model.encoder.conv2.{weight,bias}
        model.encoder.embed_positions.weight
        model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias?}
        model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
        model.encoder.layers.{i}.fc1.{weight,bias}, fc2.{weight,bias}
        model.encoder.layers.{i}.final_layer_norm.{weight,bias}
        model.encoder.layer_norm.{weight,bias}
        model.decoder.embed_tokens.weight
        model.decoder.embed_positions.weight
        model.decoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias?}
        model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
        model.decoder.layers.{i}.encoder_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias?}
        model.decoder.layers.{i}.encoder_attn_layer_norm.{weight,bias}
        model.decoder.layers.{i}.fc1.{weight,bias}, fc2.{weight,bias}
        model.decoder.layers.{i}.final_layer_norm.{weight,bias}
        model.decoder.layer_norm.{weight,bias}
        proj_out.weight

    Whisper's ``k_proj`` has no bias while ``q``/``v``/``out`` do; because the
    TRT-LLM fused QKV / cross-attn projections carry a bias when ``bias=True``,
    a zero bias is materialized for the key projection (numerically identical).
    """
    out: Dict[str, list] = {}
    enc_layers = config.encoder_layers
    dec_layers = config.decoder_layers

    def _get(key: str) -> torch.Tensor:
        # Cast lazily per-tensor: an eager whole-dict rewrite would hold a
        # second full-checkpoint copy in host memory at peak.
        if key in hf_weights:
            w = hf_weights[key]
            return w.to(dtype) if dtype is not None else w
        raise KeyError(f"Missing expected HF weight: {key}")

    def _maybe(key: str):
        w = hf_weights.get(key, None)
        if w is not None and dtype is not None:
            w = w.to(dtype)
        return w

    def _wb(prefix: str) -> dict:
        d = {"weight": _get(f"{prefix}.weight")}
        b = _maybe(f"{prefix}.bias")
        if b is not None:
            d["bias"] = b
        return d

    def _attn_qkv(hpfx: str) -> list:
        """Fused QKV weight-dicts; synthesize a zero bias for the (bias-less)
        key. MHA with num_heads * head_dim == d_model, so the bias length is
        d_model for encoder and decoder alike."""
        q = _wb(f"{hpfx}.q_proj")
        k = _wb(f"{hpfx}.k_proj")
        v = _wb(f"{hpfx}.v_proj")
        if "bias" not in k and ("bias" in q or "bias" in v):
            k = dict(k)
            k["bias"] = torch.zeros(
                config.d_model,
                dtype=q["weight"].dtype,
            )
        return [q, k, v]

    def _cross_kv(hpfx: str, proj: str) -> list:
        """Separate cross-attn projection; synthesize a zero bias for the key."""
        d = _wb(f"{hpfx}.{proj}")
        if proj == "k_proj" and "bias" not in d:
            d = dict(d)
            d["bias"] = torch.zeros(
                config.d_model,
                dtype=d["weight"].dtype,
            )
        return [d]

    # LM head (proj_out); tied to decoder embed_tokens but load explicitly too.
    if "proj_out.weight" in hf_weights:
        out["lm_head"] = [{"weight": _get("proj_out.weight")}]

    # ------------------------------------------------------------------ Encoder
    out["model.encoder.conv1"] = [_wb("model.encoder.conv1")]
    out["model.encoder.conv2"] = [_wb("model.encoder.conv2")]
    out["model.encoder.embed_positions"] = [
        {"weight": _get("model.encoder.embed_positions.weight")}
    ]
    out["model.encoder.layer_norm"] = [_wb("model.encoder.layer_norm")]

    for i in range(enc_layers):
        hpfx = f"model.encoder.layers.{i}"
        tgt = f"model.encoder.layers.{i}"

        out[f"{tgt}.self_attn.qkv_proj"] = _attn_qkv(f"{hpfx}.self_attn")
        out[f"{tgt}.self_attn.o_proj"] = [_wb(f"{hpfx}.self_attn.out_proj")]
        out[f"{tgt}.self_attn_layer_norm"] = [_wb(f"{hpfx}.self_attn_layer_norm")]

        out[f"{tgt}.mlp.up_proj"] = [_wb(f"{hpfx}.fc1")]
        out[f"{tgt}.mlp.down_proj"] = [_wb(f"{hpfx}.fc2")]
        out[f"{tgt}.final_layer_norm"] = [_wb(f"{hpfx}.final_layer_norm")]

    # ------------------------------------------------------------------ Decoder
    out["model.decoder.embed_tokens"] = [{"weight": _get("model.decoder.embed_tokens.weight")}]
    out["model.decoder.embed_positions"] = [
        {"weight": _get("model.decoder.embed_positions.weight")}
    ]
    out["model.decoder.layer_norm"] = [_wb("model.decoder.layer_norm")]

    for i in range(dec_layers):
        hpfx = f"model.decoder.layers.{i}"
        tgt = f"model.decoder.layers.{i}"

        out[f"{tgt}.self_attn.qkv_proj"] = _attn_qkv(f"{hpfx}.self_attn")
        out[f"{tgt}.self_attn.o_proj"] = [_wb(f"{hpfx}.self_attn.out_proj")]
        out[f"{tgt}.self_attn_layer_norm"] = [_wb(f"{hpfx}.self_attn_layer_norm")]

        # Cross-attention (separate q/k/v/o projections).
        out[f"{tgt}.cross_attn.q_proj"] = [_wb(f"{hpfx}.encoder_attn.q_proj")]
        out[f"{tgt}.cross_attn.k_proj"] = _cross_kv(f"{hpfx}.encoder_attn", "k_proj")
        out[f"{tgt}.cross_attn.v_proj"] = [_wb(f"{hpfx}.encoder_attn.v_proj")]
        out[f"{tgt}.cross_attn.o_proj"] = [_wb(f"{hpfx}.encoder_attn.out_proj")]
        out[f"{tgt}.cross_attn_layer_norm"] = [_wb(f"{hpfx}.encoder_attn_layer_norm")]

        out[f"{tgt}.mlp.up_proj"] = [_wb(f"{hpfx}.fc1")]
        out[f"{tgt}.mlp.down_proj"] = [_wb(f"{hpfx}.fc2")]
        out[f"{tgt}.final_layer_norm"] = [_wb(f"{hpfx}.final_layer_norm")]

    return out
