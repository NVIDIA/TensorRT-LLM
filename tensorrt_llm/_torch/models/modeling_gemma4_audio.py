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
"""Native TRT-LLM Gemma4 audio tower (Conformer).

Replaces ``AutoModel.from_config(config.audio_config)`` in
``modeling_gemma4mm.py``. Greenfield: first native Conformer port in the
fork. The audio attention is **chunked local attention with relative-position
bias + softcap tanh + per-dim query scale** — fundamentally non-standard
shape, so it stays pure PyTorch instead of inheriting TRT-LLM ``Attention``
(unlike vision, which routes through the standard backend dispatch).
``Linear`` comes from TRT-LLM for layout parity with the vision tower.
RMSNorm reuses HF's ``Gemma4RMSNorm`` directly (plain ``w * x / rms`` — not
the Gemma3 ``(1 + w) * x / rms`` LLM variant) instead of the TRT-LLM
``RMSNorm`` module — the TRT-LLM module dispatches to ``flashinfer_rmsnorm``
whose CUTE kernel rejects non-contiguous row strides, and
``Gemma4AudioLightConv1d`` feeds it exactly such tensors via
``depthwise_conv1d(x.transpose(1,2)).transpose(1,2)``. Audio is prefill-only;
the unfused path's overhead is negligible. The subsample conv stack uses
plain ``nn.LayerNorm`` (HF uses real LN there, not RMSNorm).

Architecture references (HF transformers 5.5.3 ``modeling_gemma4.py``):
- ``Gemma4ClippableLinear``                     @ ``:128``
- ``Gemma4RMSNorm``                             @ ``:157``
- ``Gemma4AudioRelPositionalEncoding``          @ ``:178``
- ``Gemma4AudioAttention``                      @ ``:209``
- ``Gemma4AudioSubSampleConvProjectionLayer``   @ ``:317``
- ``Gemma4AudioSubSampleConvProjection``        @ ``:345``
- ``Gemma4AudioFeedForward``                    @ ``:375``
- ``Gemma4AudioCausalConv1d``                   @ ``:411``
- ``Gemma4AudioLightConv1d``                    @ ``:444``
- ``Gemma4AudioLayer``                          @ ``:485``
- ``Gemma4AudioModel``                          @ ``:1800``
"""

import math
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm as _HFGemma4RMSNorm

from ..model_config import ModelConfig
from ..modules.linear import Linear
from .modeling_utils import _load_weights_impl

# DEBUG_PROBE_BEGIN: gated by env var, fires only when DEBUG_AUDIO_TOWER=1.
_DEBUG_AUDIO = os.environ.get("DEBUG_AUDIO_TOWER", "0") == "1"


def _audio_stats(name: str, t: Optional[torch.Tensor]) -> None:
    if not _DEBUG_AUDIO or t is None:
        return
    f = t.detach().float()
    finite_frac = torch.isfinite(f).float().mean().item()
    print(
        f"  [audio] {name:32s} shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={f.mean().item():+.4g} std={f.std().item():.4g} "
        f"absmax={f.abs().max().item():.4g} finite={finite_frac:.4f}",
        flush=True,
    )


# DEBUG_PROBE_END


@dataclass
class _AudioOutput:
    """Stand-in for ``transformers.Gemma4AudioModelOutput`` (fields used by caller)."""

    last_hidden_state: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# RMSNorm: thin kwarg/dtype adapter over HF's ``Gemma4RMSNorm`` (see module
# docstring for why this isn't the TRT-LLM ``RMSNorm`` module).
# ---------------------------------------------------------------------------


class _Gemma4AudioRMSNorm(_HFGemma4RMSNorm):
    """Kwarg + dtype adapter around HF ``Gemma4RMSNorm`` (``w * x / rms(x)``)."""

    def __init__(self, *, hidden_size: int, eps: float, dtype: Optional[torch.dtype] = None):
        super().__init__(hidden_size, eps=eps)
        if dtype is not None:
            with torch.no_grad():
                self.weight.data = self.weight.data.to(dtype)


# ---------------------------------------------------------------------------
# Clippable linear wrapper (same shape as Gemma4VisionClippableLinear)
# ---------------------------------------------------------------------------


class Gemma4AudioClippableLinear(nn.Module):
    """TRT-LLM ``Linear`` + optional input/output clamping.

    Mirrors HF ``Gemma4ClippableLinear``: four scalar buffers initialised to
    sentinel -inf / +inf so an unclipped checkpoint is a no-op; populated by
    the checkpoint when ``use_clipped_linears=True``.
    """

    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        bias: bool = False,
        mapping=None,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = bool(getattr(config, "use_clipped_linears", False))
        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
        )
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


# ---------------------------------------------------------------------------
# Relative positional encoding (fixed sinusoidal, concat[sin, cos] layout)
# ---------------------------------------------------------------------------


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative position embeddings used by ``Gemma4AudioAttention``.

    HF parity: ``forward`` returns ``[1, context_size, hidden_size]`` with the
    *fixed* ``position_ids = arange(12, -1, -1)`` (per the reference impl —
    note this hard-codes 13 positions; the value is preserved verbatim).
    """

    inv_timescales: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales) * -log_timescale_increment
        )
        self.register_buffer(
            "inv_timescales", inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False
        )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Verbatim from HF: hard-coded 13-step range (see :201).
        position_ids = torch.arange(12, -1, -1, device=hidden_states.device)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


# ---------------------------------------------------------------------------
# Chunked local attention with relative-position bias
# ---------------------------------------------------------------------------


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias and softcap tanh.

    Stays pure PyTorch — the shape (5D blocked, ``[B, 1, num_blocks,
    chunk_size, context_size]``) and the rel-shift trick don't map to TRT-LLM
    flashinfer / trtllm-gen kernels. Audio sequences are short (≤750 tokens
    for the 30s window), so the perf cost vs vision is minimal.
    """

    def __init__(self, config, layer_idx: int, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        # See HF :220-221 — these scale factors are baked into q/k pre-attn.
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_proj = Gemma4AudioClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim, dtype, mapping=mapping
        )
        self.k_proj = Gemma4AudioClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim, dtype, mapping=mapping
        )
        self.v_proj = Gemma4AudioClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim, dtype, mapping=mapping
        )
        self.post = Gemma4AudioClippableLinear(
            config, config.hidden_size, config.hidden_size, dtype, mapping=mapping
        )

        # Relative-position key projection is *not* clipped in HF.
        self.relative_k_proj = Linear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            mapping=mapping,
        )
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.register_buffer(
            "softcap", torch.tensor(self.attention_logits_soft_cap), persistent=False
        )

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Split ``(B, T, H, D)`` into ``(B, num_blocks, chunk_size, H, D)``."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(
            batch_size, num_blocks, self.chunk_size, num_heads, head_dim
        ).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract overlapping context windows of ``context_size`` per block."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = F.pad(
            hidden_states,
            (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer-XL appendix-B relative position shift."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        # HF runs q/k/v in float32 to keep the softcap + softmax stable.
        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        # AC: content × content. BD: content × relative positions (rel-shifted).
        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_blocks * self.chunk_size, -1
        )
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Subsample conv stack (mel → encoder embeddings, stride 4 total)
# ---------------------------------------------------------------------------


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    """Conv2d 3x3 stride 2 + LayerNorm (real LN, not RMSNorm) + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, norm_eps: float, dtype: torch.dtype):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(
            out_channels, eps=norm_eps, elementwise_affine=True, bias=False, dtype=dtype
        )
        self.act = nn.ReLU()

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        # LayerNorm on the channel dim — permute, norm, permute back.
        hidden_states = self.act(
            self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )
        if mask is not None:
            mask = mask[:, ::2]
        return hidden_states, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    """Two-stage stride-2 subsample (4× downsample) + linear projection to hidden_size."""

    def __init__(self, config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
            dtype=dtype,
        )
        # HF :358 — both Conv2d strides are 2, so input mel dim shrinks by 4.
        # Output flatten dim = (mel_dim // 4) * out_channels[1] — but HF
        # parameterises off subsampling_conv_channels[0] // 4. We preserve
        # that verbatim to avoid any silent off-by-one vs the checkpoint.
        proj_input_dim = (
            config.subsampling_conv_channels[0] // 4
        ) * config.subsampling_conv_channels[1]
        self.input_proj_linear = Linear(
            in_features=proj_input_dim,
            out_features=config.hidden_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = input_features.unsqueeze(1)  # (B, 1, T, mel)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        batch_size, _, seq_len, _ = hidden_states.shape
        # (B, C, T', mel') → (B, T', mel'*C) for the linear projection.
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        )
        return self.input_proj_linear(hidden_states), mask


# ---------------------------------------------------------------------------
# Macaron feedforward + light conv1d branch
# ---------------------------------------------------------------------------


class Gemma4AudioFeedForward(nn.Module):
    """Macaron-style FFN with pre/post norm and residual_weight scaling."""

    def __init__(self, config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = config
        self.ffw_layer_1 = Gemma4AudioClippableLinear(
            config, config.hidden_size, config.hidden_size * 4, dtype, mapping=mapping
        )
        self.ffw_layer_2 = Gemma4AudioClippableLinear(
            config, config.hidden_size * 4, config.hidden_size, dtype, mapping=mapping
        )
        self.pre_layer_norm = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_layer_norm = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.ffw_layer_1.linear.weight.dtype).max,
        )
        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = hidden_states * self.post_layer_scale
        hidden_states = hidden_states + residual
        return hidden_states


class Gemma4AudioCausalConv1d(nn.Conv1d):
    """Left-padded Conv1d so the receptive field stays causal."""

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class Gemma4AudioLightConv1d(nn.Module):
    """Depthwise causal conv branch with GLU activation and pre/conv norms."""

    def __init__(self, config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = config
        self.linear_start = Gemma4AudioClippableLinear(
            config, config.hidden_size, config.hidden_size * 2, dtype, mapping=mapping
        )
        self.linear_end = Gemma4AudioClippableLinear(
            config, config.hidden_size, config.hidden_size, dtype, mapping=mapping
        )
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
            dtype=dtype,
        )
        self.pre_layer_norm = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.conv_norm = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = F.glu(hidden_states, dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.linear_start.linear.weight.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


# ---------------------------------------------------------------------------
# Conformer encoder layer + top-level model
# ---------------------------------------------------------------------------


class Gemma4AudioLayer(nn.Module):
    """Conformer block: FF1 → attn → FF2 → light-conv → out_norm."""

    def __init__(self, config, layer_idx: int, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = config
        self.feed_forward1 = Gemma4AudioFeedForward(config, dtype=dtype, mapping=mapping)
        self.feed_forward2 = Gemma4AudioFeedForward(config, dtype=dtype, mapping=mapping)
        self.self_attn = Gemma4AudioAttention(
            config, layer_idx=layer_idx, dtype=dtype, mapping=mapping
        )
        self.lconv1d = Gemma4AudioLightConv1d(config, dtype=dtype, mapping=mapping)
        self.norm_pre_attn = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.norm_post_attn = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.norm_out = _Gemma4AudioRMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.norm_pre_attn.weight.dtype).max,
        )

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class Gemma4AudioModel(nn.Module):
    """Gemma4 audio tower (USM-Conformer encoder).

    Input contract (preserved from HF ``Gemma4AudioModel.forward``):
        input_features:  (B, mel_T, mel_bins) — mel-spectrogram frames
        attention_mask:  (B, mel_T) bool — True for valid frames

    Returns ``_AudioOutput`` with:
        last_hidden_state: (B, T_audio, output_proj_dims)
        attention_mask:    (B, T_audio) bool — post-subsample frame validity
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        dtype = getattr(self.config, "torch_dtype", None) or torch.bfloat16
        mapping = getattr(model_config, "mapping", None)

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(
            self.config, dtype=dtype, mapping=mapping
        )
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(self.config)
        self.layers = nn.ModuleList(
            [
                Gemma4AudioLayer(self.config, layer_idx=i, dtype=dtype, mapping=mapping)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.output_proj = Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.output_proj_dims,
            bias=True,
            dtype=dtype,
            mapping=mapping,
        )

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, 1, T, T]`` causal/window mask → ``[B, 1, num_blocks, chunk_size, context_size]``."""
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)

        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)
        return mask_5d.gather(-1, kv_indices)

    def _build_chunked_local_mask(
        self, output_mask: torch.Tensor, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build the blocked 5D attention mask from the post-subsample frame mask.

        HF builds the dense 4D mask via ``create_bidirectional_mask`` +
        ``sliding_window_mask_function((ctx_left-1, ctx_right))``. We replicate
        that semantics directly: True where both row and column frames are
        valid AND ``j`` is within ``[i - max_past, i + max_future]``.
        """
        # Valid (i, j) cells: both frames valid.
        valid = output_mask.bool()  # (B, T)
        valid_4d = valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(-1)

        # Sliding-window: |i - j| within (-max_past, +max_future).
        idx = torch.arange(seq_len, device=device)
        rel = idx.unsqueeze(0) - idx.unsqueeze(1)  # (T, T): j - i
        window = (rel >= -self.max_past_horizon) & (rel <= self.max_future_horizon)
        return valid_4d & window.unsqueeze(0).unsqueeze(0)

    @property
    def max_past_horizon(self) -> int:
        return self.config.attention_context_left - 1

    @property
    def max_future_horizon(self) -> int:
        return self.config.attention_context_right

    @torch.inference_mode()
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> _AudioOutput:
        # DEBUG_PROBE_BEGIN
        if _DEBUG_AUDIO:
            print("[audio] === Gemma4AudioModel.forward ===", flush=True)
            _audio_stats("input_features (mel)", input_features)
            _audio_stats(
                "attention_mask", attention_mask.float() if attention_mask is not None else None
            )
        # DEBUG_PROBE_END
        hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
        # DEBUG_PROBE_BEGIN
        _audio_stats("after subsample_conv_proj", hidden_states)
        _audio_stats(
            "output_mask (subsample)", output_mask.float() if output_mask is not None else None
        )
        # DEBUG_PROBE_END
        position_embeddings = self.rel_pos_enc(hidden_states)
        # DEBUG_PROBE_BEGIN
        _audio_stats("rel_pos_enc", position_embeddings)
        # DEBUG_PROBE_END

        # output_mask: (B, T_audio) bool. Frames where mask is False are pad.
        if output_mask is None:
            output_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        attn_mask_4d = self._build_chunked_local_mask(
            output_mask, seq_len=hidden_states.shape[1], device=hidden_states.device
        )
        attn_mask_5d = self._convert_4d_mask_to_blocked_5d(attn_mask_4d)
        # DEBUG_PROBE_BEGIN
        _audio_stats("attn_mask_5d", attn_mask_5d.float())
        # DEBUG_PROBE_END

        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attn_mask_5d,
                position_embeddings=position_embeddings,
            )
            # DEBUG_PROBE_BEGIN: log layers 0 / mid / last
            if _DEBUG_AUDIO and (i == 0 or i == n_layers // 2 or i == n_layers - 1):
                _audio_stats(f"after layer[{i}]", hidden_states)
            # DEBUG_PROBE_END

        hidden_states = self.output_proj(hidden_states)
        # DEBUG_PROBE_BEGIN
        _audio_stats("after output_proj (final)", hidden_states)
        # DEBUG_PROBE_END
        return _AudioOutput(last_hidden_state=hidden_states, attention_mask=output_mask)

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load HF ``Gemma4AudioModel`` weights.

        Layout differences vs HF:
          - Our ``Gemma4AudioClippableLinear.linear`` is TRT-LLM ``Linear``;
            HF's ``Gemma4ClippableLinear.linear`` is ``nn.Linear``. Both store
            weight at ``.linear.weight`` so the keys match without remapping.
          - Top-level ``output_proj`` is a TRT-LLM ``Linear`` (HF uses
            ``nn.Linear``); both expose ``.weight`` / ``.bias``.
          - ``Gemma4AudioCausalConv1d`` subclasses ``nn.Conv1d`` directly, no
            wrapper. Conv2d in the subsample stack is also plain ``nn.Conv2d``.
          - Audio attention does *not* fuse q/k/v (HF keeps them separate too).

        Clipping buffers (``input_min`` etc. on ``Gemma4AudioClippableLinear``)
        live as named buffers and aren't iterated by ``_load_weights_impl``'s
        parameter walk, so we copy them explicitly below.
        """
        _load_weights_impl(self, weights)
        for name, buf in self.named_buffers():
            if name in weights:
                buf.data.copy_(weights[name].to(buf.dtype).to(buf.device))
        # DEBUG_PROBE_BEGIN: verify checkpoint actually landed
        if _DEBUG_AUDIO:
            print("[audio] === load_weights summary ===", flush=True)
            print(f"  checkpoint keys provided: {len(weights)}", flush=True)
            loaded_param_keys = {n for n, _ in self.named_parameters()}
            covered_params = sum(1 for k in weights if k in loaded_param_keys)
            print(
                f"  params covered by ckpt:   {covered_params}/{len(loaded_param_keys)}", flush=True
            )
            # Spot-check a few _Gemma4AudioRMSNorm weights — if they're all
            # zeros, the (1+w) convention reduces to identity scaling and
            # explains "BLEU=1.9, model ignores audio". Sample 3 RMSNorms.
            picks = []
            for name, mod in self.named_modules():
                if isinstance(mod, _Gemma4AudioRMSNorm):
                    picks.append((name, mod.weight))
                    if len(picks) >= 3:
                        break
            for name, w in picks:
                f = w.detach().float()
                print(
                    f"  rmsnorm[{name}].weight stats: "
                    f"mean={f.mean().item():+.4g} std={f.std().item():.4g} "
                    f"absmax={f.abs().max().item():.4g} zero_frac={(f == 0).float().mean().item():.4f}",
                    flush=True,
                )
            # Also spot-check one clippable linear's input_min/output_max buffers.
            for name, mod in self.named_modules():
                if isinstance(mod, Gemma4AudioClippableLinear):
                    for bname in ("input_min", "input_max", "output_min", "output_max"):
                        b = getattr(mod, bname, None)
                        if b is not None:
                            print(f"  clip[{name}].{bname}: {b.item():+.4g}", flush=True)
                    break
        # DEBUG_PROBE_END
