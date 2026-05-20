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
"""Native TRT-LLM Gemma4 vision tower.

Replaces the HF ``AutoModel.from_config(config.vision_config)`` call in
``modeling_gemma4mm.py`` so the vision encoder uses TRT-LLM building blocks
(``Attention``, ``Linear``, ``RMSNorm``) instead of importing
``transformers.Gemma4VisionModel``.

Attention inherits from TRT-LLM ``Attention`` and runs through the standard
backend dispatch (default ``FLASHINFER`` via ``Gemma4ForCausalLM`` model
defaults). Vision is context-only — ``kv_cache_manager=None`` and a tower-local
``attn_metadata`` are built per forward following the SigLip pattern.

Architecture references (HF transformers 5.5.3 ``modeling_gemma4.py``):
- ``Gemma4ClippableLinear`` @ ``:128``
- ``Gemma4RMSNorm`` @ ``:157``
- ``Gemma4VisionPatchEmbedder`` @ ``:539``
- ``Gemma4VisionPooler`` @ ``:573``
- ``Gemma4VisionMLP`` @ ``:632``
- ``Gemma4VisionRotaryEmbedding`` @ ``:648``
- ``apply_multidimensional_rope`` @ ``:802``
- ``Gemma4VisionAttention`` @ ``:859``
- ``Gemma4VisionEncoderLayer`` @ ``:928``
- ``Gemma4VisionEncoder`` @ ``:972``
- ``Gemma4VisionModel`` @ ``:1885``
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm as _HFGemma4RMSNorm

from tensorrt_llm._utils import maybe_pin_memory

from ..attention_backend.interface import AttentionMetadata, PredefinedAttentionMask
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.linear import Linear
from .modeling_utils import _load_weights_impl


# Use HF's ``Gemma4RMSNorm`` directly instead of TRT-LLM's ``RMSNorm`` module.
# TRT-LLM ``RMSNorm`` dispatches to ``flashinfer_rmsnorm`` whose CUTE kernel
# rejects non-fp32 / non-contiguous strides and 1152-wide bf16 inputs from
# SigLip. HF's class is pure PyTorch and handles ``with_scale=False`` (the
# weightless ``v_norm`` case) via a flag.
class Gemma4VisionRMSNorm(_HFGemma4RMSNorm):
    """Kwarg + dtype adapter around HF ``Gemma4RMSNorm``."""

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        has_weights: bool = True,
    ):
        super().__init__(hidden_size, eps=eps, with_scale=has_weights)
        if dtype is not None and has_weights:
            with torch.no_grad():
                self.weight.data = self.weight.data.to(dtype)


@dataclass
class VisionOutput:
    """Stand-in for ``transformers.BaseModelOutputWithPast`` (only field used)."""

    last_hidden_state: torch.Tensor


# ---------------------------------------------------------------------------
# RoPE helpers (ported from transformers/models/gemma4/modeling_gemma4.py)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """2D RoPE: split channels into ``ndim`` parts, rotate each with its own freqs."""
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))
    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, "
            f"got {num_rotated_channels_per_dim} (num_input_channels="
            f"{num_input_channels}, ndim={ndim})"
        )
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _apply_rotary_pos_emb(
            x=x_parts[k], cos=cos_parts[k], sin=sin_parts[k], unsqueeze_dim=unsqueeze_dim
        )
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Computes (cos, sin) for 2D RoPE; ``inv_freq`` is per-spatial-dim."""

    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        base = config.rope_parameters["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        # Per-spatial-dim partitioning: HF allocates head_dim//2 channels per dim.
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(dtype=torch.float)
                / spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, N, ndim=2). Compute cos/sin independently per spatial dim
        # and concatenate so apply_multidimensional_rope can split them back.
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        all_cos, all_sin = [], []
        for i in range(position_ids.shape[-1]):
            dim_position_ids = position_ids[:, :, i]
            dim_position_ids_expanded = dim_position_ids[:, None, :].float()
            with torch.autocast(device_type=x.device.type, enabled=False):
                freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(
                    1, 2
                )
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Gemma4VisionClippableLinear(nn.Module):
    """TRT-LLM ``Linear`` + optional per-projection input/output clamping.

    HF parity: ``Gemma4ClippableLinear`` registers four scalar buffers (sentinel
    -inf / +inf so that an unloaded checkpoint behaves as a no-op). The buffers
    are populated by the checkpoint when ``use_clipped_linears=True``.
    """

    def __init__(
        self,
        config,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        mapping=None,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = bool(getattr(config, "use_clipped_linears", False))
        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
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


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.gate_proj = Gemma4VisionClippableLinear(
            config, config.hidden_size, config.intermediate_size, dtype, mapping
        )
        self.up_proj = Gemma4VisionClippableLinear(
            config, config.hidden_size, config.intermediate_size, dtype, mapping
        )
        self.down_proj = Gemma4VisionClippableLinear(
            config, config.intermediate_size, config.hidden_size, dtype, mapping
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4VisionAttention(Attention):
    """Bidirectional attention with q/k/v RMSNorms and 2D RoPE.

    Inherits ``Attention`` to participate in the TRT-LLM backend dispatch.
    Vision sub_config is forced to ``attn_backend="TRTLLM"`` in
    ``modeling_gemma4mm.get_sub_model_config`` (same pattern Qwen2.5-VL uses).
    Vision is context-only and runs through ``forward_impl`` with
    ``PredefinedAttentionMask.FULL`` and a tower-local ``attn_metadata``
    (``kv_cache_manager=None``).

    The ``Attention`` base class uses a fused ``qkv_proj`` and a vanilla
    ``o_proj``. To preserve HF Gemma4's ``ClippableLinear`` semantics we
    register clamp buffers directly on this module and apply them around the
    projections in ``forward``:
      - ``qkv_input_{min,max}``: shared input clamp (HF stores 3 copies, one
        per q/k/v; ``load_weights`` asserts they match and collapses to one).
      - ``{q,k,v}_output_{min,max}``: per-section output clamp applied after
        ``split_qkv``.
      - ``o_input_{min,max}`` / ``o_output_{min,max}``: o_proj clamps.

    HF Gemma4VisionAttention sets ``self.scaling = 1.0`` (no head-dim
    rescaling). TRT-LLM uses ``qk_scale = 1 / (sqrt(head_dim) * q_scaling)``,
    so we pass ``q_scaling = 1 / sqrt(head_dim)`` to neutralize the sqrt.
    """

    # trtllm-gen FMHA on sm100a ships cubins for these head_dim sizes only.
    # Variants whose HF ``head_dim`` is not in this set are padded up to the
    # next supported size; the kernel sees zero-padded q/k/v while RMSNorm,
    # RoPE, and o_proj math run on the unpadded HF channels (see ``forward``
    # for the slice / re-pad dance and ``Gemma4VisionModel.load_weights`` for
    # weight zero-padding). Gemma4 26B/31B vision: ``head_dim=72`` → 80.
    _FMHA_SUPPORTED_HEAD_DIMS = (64, 80, 128, 256, 512)

    def __init__(
        self, model_config: ModelConfig, vision_config, layer_idx: int, dtype: torch.dtype
    ):
        hf_head_dim = getattr(
            vision_config,
            "head_dim",
            vision_config.hidden_size // vision_config.num_attention_heads,
        )
        if hf_head_dim in self._FMHA_SUPPORTED_HEAD_DIMS:
            padded_head_dim = hf_head_dim
        else:
            padded_head_dim = next(d for d in self._FMHA_SUPPORTED_HEAD_DIMS if d >= hf_head_dim)
        # TRT-LLM ``Attention`` applies ``qk_scale = 1 / (sqrt(self.head_dim) *
        # q_scaling)``; HF ``Gemma4VisionAttention`` sets ``scaling = 1.0`` (no
        # ``1/sqrt(d)`` rescale). Use the padded ``head_dim`` here so the scale
        # exactly cancels — the padded q/k channels are zero, so the dot product
        # is numerically the same regardless of ``sqrt(padded)`` vs
        # ``sqrt(hf_head_dim)``.
        q_scaling = 1.0 / math.sqrt(padded_head_dim)
        max_pe = getattr(vision_config, "max_position_embeddings", None) or (
            getattr(vision_config, "position_embedding_size", 0) ** 2 or 4096
        )
        super().__init__(
            hidden_size=vision_config.hidden_size,
            num_attention_heads=vision_config.num_attention_heads,
            num_key_value_heads=vision_config.num_key_value_heads,
            max_position_embeddings=max_pe,
            bias=False,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
            head_dim=padded_head_dim,
        )
        self.vision_config = vision_config
        # ``self.head_dim`` is now ``padded_head_dim`` (the kernel-facing value).
        # ``self.hf_head_dim`` is the original HF head_dim; norm + RoPE see
        # this dim only. ``self.head_dim_pad`` is the zero-padding width.
        self.hf_head_dim = hf_head_dim
        self.head_dim_pad = padded_head_dim - hf_head_dim

        # HF Gemma4RMSNorm is the plain variant (``x_normed * weight``, no
        # ``(1+w)``). v_norm has ``with_scale=False`` (no learnable weight).
        # Norms operate on ``hf_head_dim``; padded channels are sliced off
        # before norm and re-padded with zeros after RoPE (see ``forward``).
        self.q_norm = Gemma4VisionRMSNorm(
            hidden_size=self.hf_head_dim, eps=vision_config.rms_norm_eps, dtype=dtype
        )
        self.k_norm = Gemma4VisionRMSNorm(
            hidden_size=self.hf_head_dim, eps=vision_config.rms_norm_eps, dtype=dtype
        )
        self.v_norm = Gemma4VisionRMSNorm(
            hidden_size=self.hf_head_dim,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
            has_weights=False,
        )

        self.use_clipped_linears = bool(getattr(vision_config, "use_clipped_linears", False))
        if self.use_clipped_linears:
            neg_inf = torch.tensor(-float("inf"))
            pos_inf = torch.tensor(float("inf"))
            for name in (
                "qkv_input_min",
                "q_output_min",
                "k_output_min",
                "v_output_min",
                "o_input_min",
                "o_output_min",
            ):
                self.register_buffer(name, neg_inf.clone())
            for name in (
                "qkv_input_max",
                "q_output_max",
                "k_output_max",
                "v_output_max",
                "o_input_max",
                "o_output_max",
            ):
                self.register_buffer(name, pos_inf.clone())

    def _head_norm(self, x: torch.Tensor, norm: Gemma4VisionRMSNorm) -> torch.Tensor:
        # HF ``Gemma4RMSNorm`` is pure PyTorch and normalises over the last
        # dim, so we can pass the (..., head_dim) tensor through directly.
        return norm(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Flat (num_tokens, hidden) layout — encoder flattens (B, N, H) before
        # dispatching to attention.
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.qkv_input_min, self.qkv_input_max)

        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)

        if self.use_clipped_linears:
            q = torch.clamp(q, self.q_output_min, self.q_output_max)
            k = torch.clamp(k, self.k_output_min, self.k_output_max)
            v = torch.clamp(v, self.v_output_min, self.v_output_max)

        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_key_value_heads, self.head_dim)
        v = v.view(num_tokens, self.num_key_value_heads, self.head_dim)

        # When ``head_dim`` was padded to satisfy the FMHA cubin set, the
        # qkv_proj last ``head_dim_pad`` channels are produced by zero-padded
        # weight rows (see ``Gemma4VisionModel.load_weights``) — slice them
        # off before RMSNorm so the mean-of-squares averages over
        # ``hf_head_dim`` (HF semantics), not the padded width.
        if self.head_dim_pad > 0:
            q_real = q[..., : self.hf_head_dim]
            k_real = k[..., : self.hf_head_dim]
            v_real = v[..., : self.hf_head_dim]
        else:
            q_real, k_real, v_real = q, k, v

        q_real = self._head_norm(q_real, self.q_norm)
        k_real = self._head_norm(k_real, self.k_norm)
        v_real = self._head_norm(v_real, self.v_norm)

        cos, sin = position_embeddings
        q_real = _apply_multidimensional_rope(q_real, cos, sin, position_ids, unsqueeze_dim=1)
        k_real = _apply_multidimensional_rope(k_real, cos, sin, position_ids, unsqueeze_dim=1)

        # Re-pad with zeros for the FMHA kernel. Zeros in q_pad/k_pad don't
        # contribute to QK^T; zeros in v_pad produce zero output channels — so
        # ``attn_output[..., hf_head_dim:]`` is identically 0 and o_proj's
        # last padded columns (also zero) don't see real signal either way.
        if self.head_dim_pad > 0:
            pad_shape = q_real.shape[:-1] + (self.head_dim_pad,)
            q = torch.cat([q_real, q_real.new_zeros(pad_shape)], dim=-1)
            k = torch.cat([k_real, k_real.new_zeros(pad_shape)], dim=-1)
            v = torch.cat([v_real, v_real.new_zeros(pad_shape)], dim=-1)
        else:
            q, k, v = q_real, k_real, v_real

        q = q.reshape(num_tokens, self.num_heads * self.head_dim)
        k = k.reshape(num_tokens, self.num_key_value_heads * self.head_dim)
        v = v.reshape(num_tokens, self.num_key_value_heads * self.head_dim)

        # Defensive dtype cast: HF ``Gemma4RMSNorm.forward`` casts internals to
        # fp32; RoPE / clamp paths can also promote. The FMHA dispatcher
        # rejects non-bf16/half input ("Fall back to unfused MHA because of
        # unsupported data type") and falls into the O(B × T²) unfused path.
        # Casting q/k/v back to the projection weight dtype (matches model
        # dtype, typically bf16; fp32 in unit tests) keeps the kernel
        # dispatch on the FMHA path in production and is a no-op when the
        # tower was constructed in fp32 anyway.
        target_dtype = self.qkv_proj.weight.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.forward_impl(
            q=q,
            k=k,
            v=v,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )

        if self.use_clipped_linears:
            attn_output = torch.clamp(attn_output, self.o_input_min, self.o_input_max)
        attn_output = self.o_proj(attn_output, layer_idx=self.layer_idx)
        if self.use_clipped_linears:
            attn_output = torch.clamp(attn_output, self.o_output_min, self.o_output_max)
        return attn_output


class Gemma4VisionEncoderLayer(nn.Module):
    """Sandwich norm pattern: input → attn → post_attn → +res; pre_ffn → mlp → post_ffn → +res."""

    def __init__(
        self,
        model_config: ModelConfig,
        vision_config,
        layer_idx: int,
        dtype: torch.dtype,
        mapping=None,
    ):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            model_config=model_config,
            vision_config=vision_config,
            layer_idx=layer_idx,
            dtype=dtype,
        )
        self.mlp = Gemma4VisionMLP(vision_config, dtype, mapping)
        self.input_layernorm = Gemma4VisionRMSNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = Gemma4VisionRMSNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
        )
        self.pre_feedforward_layernorm = Gemma4VisionRMSNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
        )
        self.post_feedforward_layernorm = Gemma4VisionRMSNorm(
            hidden_size=vision_config.hidden_size,
            eps=vision_config.rms_norm_eps,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma4VisionEncoder(nn.Module):
    """Encoder runs in flat ``(sum_valid_tokens, hidden_size)`` layout.

    Padding (``pixel_position_ids == -1``) is dropped at the encoder entry so
    every token participates in the same FULL attention block via cu_seqlens.
    Mapping back to ``(B, N)`` for the pooler is the caller's job.
    """

    def __init__(self, model_config: ModelConfig, vision_config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = vision_config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(vision_config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    model_config=model_config,
                    vision_config=vision_config,
                    layer_idx=i,
                    dtype=dtype,
                    mapping=mapping,
                )
                for i in range(vision_config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> VisionOutput:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
        return VisionOutput(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# Patch embedder + pooler (small, kept native)
# ---------------------------------------------------------------------------


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config, dtype: torch.dtype, mapping=None):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = Linear(
            in_features=3 * self.patch_size**2,
            out_features=config.hidden_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
        )
        # (2, P, H) — one table per spatial axis, H = hidden_size.
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, config.hidden_size, dtype=dtype)
        )

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Gemma4 scales pixel inputs in model code (no normalization in the processor).
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


class Gemma4VisionPooler(nn.Module):
    """Optional 2D spatial pooling by patch positions."""

    def __init__(self, config):
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times "
                f"{length=} must be {input_seq_len}."
            )
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than "
                f"there are patches ({hidden_states.shape[1]})."
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


# ---------------------------------------------------------------------------
# Top-level vision tower
# ---------------------------------------------------------------------------


class Gemma4VisionModel(nn.Module):
    """Gemma4 vision tower.

    Input contract (preserved from HF ``Gemma4VisionModel.forward``):
        pixel_values: (B, num_patches, 3 * patch_size**2)
        pixel_position_ids: (B, num_patches, 2) -- (x, y) per patch; -1 = padding
        output_length: post-pool token count (== num_patches // pooling_k**2)

    Returns ``VisionOutput`` with ``last_hidden_state`` of shape
    ``(N_valid_tokens, hidden_size)`` (flat across the batch after pooler_mask strip).
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        dtype = getattr(self.config, "torch_dtype", None) or torch.bfloat16
        mapping = getattr(model_config, "mapping", None)

        self.patch_embedder = Gemma4VisionPatchEmbedder(self.config, dtype=dtype, mapping=mapping)
        self.encoder = Gemma4VisionEncoder(
            model_config=model_config,
            vision_config=self.config,
            dtype=dtype,
            mapping=mapping,
        )
        self.pooler = Gemma4VisionPooler(self.config)
        self.standardize = bool(getattr(self.config, "standardize", False))
        if self.standardize:
            self.register_buffer("std_bias", torch.zeros(self.config.hidden_size, dtype=dtype))
            self.register_buffer("std_scale", torch.ones(self.config.hidden_size, dtype=dtype))

        # SigLip-style context-only metadata: kv_cache_manager=None, no decode
        # phase. Re-prepared each forward with the actual per-image seq lens.
        # Vision tower is called once per LLM step across all images in the
        # batch (``modeling_gemma4mm._get_image_features``), so the batch
        # axis here is the cross-request image count.
        #
        # Sizing rationale: vision attention dispatches to the trtllm-gen
        # FMHA kernel (see ``Gemma4VisionAttention``: dtype cast on q/k/v
        # before ``forward_impl``; head_dim padded to an FMHA-supported
        # cubin size in ``__init__`` + ``load_weights``). FMHA workspace is
        # O(max_num_tokens × hidden_size) — linear, no batch² term — so we
        # can safely pull the LLM-side ``max_num_requests`` (typically
        # 8192) without OOM. ``max_num_tokens`` stays at a vision-scale
        # 8192 (covers worst-case patch counts) — pulling the LLM-side
        # value of 16384 would still bloat workspace 4× on no reason.
        self.metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self.attn_metadata = self.metadata_cls(
            max_num_requests=getattr(model_config, "max_num_requests", 8192) or 8192,
            max_num_tokens=8192,
            kv_cache_manager=None,
        )

    def _prepare_attn_metadata(self, seq_lens_cpu: torch.Tensor):
        batch_size = int(seq_lens_cpu.numel())
        prompt_lens = seq_lens_cpu.tolist()
        self.attn_metadata.num_contexts = batch_size
        self.attn_metadata.request_ids = list(range(1, batch_size + 1))
        self.attn_metadata.prompt_lens = prompt_lens
        self.attn_metadata.seq_lens = maybe_pin_memory(seq_lens_cpu.to(dtype=torch.int, copy=True))
        self.attn_metadata.max_seq_len = max(prompt_lens) if prompt_lens else 0
        self.attn_metadata.prepare()
        return self.attn_metadata

    @torch.inference_mode()
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> VisionOutput:
        if output_length is None:
            k = self.config.pooling_kernel_size
            output_length = pixel_values.shape[-2] // (k * k)

        # (B, N) bool: True = padding.
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        valid = ~padding_positions
        B, N = padding_positions.shape

        # Patch embed produces (B, N, H) including padded slots zeroed.
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        # Position embeddings (cos/sin) are (B, N, head_dim) per spatial dim.
        position_embeddings_btn = self.encoder.rotary_emb(inputs_embeds, pixel_position_ids)

        # Flatten to (sum_valid, H) and drop padded tokens — encoder runs
        # FULL attention with per-image cu_seqlens.
        flat_embeds = inputs_embeds[valid]
        cos_btn, sin_btn = position_embeddings_btn
        flat_cos = cos_btn[valid]
        flat_sin = sin_btn[valid]
        flat_position_ids = pixel_position_ids[valid]
        seq_lens_cpu = valid.sum(dim=-1).to(torch.int).cpu()

        attn_metadata = self._prepare_attn_metadata(seq_lens_cpu)

        enc_out = self.encoder(
            inputs_embeds=flat_embeds,
            attn_metadata=attn_metadata,
            position_embeddings=(flat_cos, flat_sin),
            position_ids=flat_position_ids,
        )

        # Scatter back to (B, N, H) for the pooler (padded slots → 0).
        last = enc_out.last_hidden_state.new_zeros(B, N, inputs_embeds.shape[-1])
        last[valid] = enc_out.last_hidden_state

        hidden_states, pooler_mask = self.pooler(
            hidden_states=last,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        # Strip padding tokens — flat (N_valid, H).
        hidden_states = hidden_states[pooler_mask]
        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return VisionOutput(last_hidden_state=hidden_states)

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        # HF Gemma4ClippableLinear has nested ``.linear`` weights. We collapse
        # them as follows:
        #   - MLP/o_proj/patch_embedder: ``Gemma4VisionClippableLinear`` /
        #     plain TRT-LLM ``Linear`` — peel the ``.linear`` indirection
        #     where it exists (for MLP/o_proj wrappers), and route to fused
        #     ``qkv_proj`` for attention via ``_load_weights_impl``'s
        #     ``params_map`` fusion.
        #   - Attention input clamps: HF has 3 (q/k/v), we collapse to one
        #     ``qkv_input_{min,max}`` after asserting they match.
        #   - Attention output clamps: q/k/v stay separate; o_proj input/output
        #     clamps move from ``self_attn.o_proj.{input,output}_{min,max}``
        #     into the attention module as ``o_{input,output}_{min,max}``.
        #   - If the model's HF ``head_dim`` is not an FMHA-supported cubin
        #     size (e.g. 72 for 26B/31B), zero-pad q/k/v/o_proj head dim to
        #     the next supported size (e.g. 80) so the trtllm-gen FMHA
        #     kernel can dispatch. Norm/RoPE math sees only the unpadded
        #     channels (see ``Gemma4VisionAttention.forward``).
        remapped = self._remap_clamp_buffers(dict(weights))
        remapped = self._pad_attention_head_dim(remapped)
        # Strip the HF ``.linear`` indirection inside our ClippableLinear
        # wrappers (MLP and patch_embedder.input_proj are also wrapped/standalone).
        # The attention's qkv_proj absorbs HF q_proj/k_proj/v_proj via
        # ``params_map`` inside ``_load_weights_impl``.
        pattern_mapping = {
            # MLP wrappers: ``mlp.gate_proj.linear.weight`` → ``mlp.gate_proj.linear.weight``
            # (no change needed — wrapper keeps ``.linear`` attr).
            # Attention QKV: HF q_proj.linear.weight → q_proj.weight (so
            # ``_load_weights_impl`` fusion sees q_proj/k_proj/v_proj).
            r"(.*?self_attn\.)q_proj\.linear\.(weight|bias)$": r"\1q_proj.\2",
            r"(.*?self_attn\.)k_proj\.linear\.(weight|bias)$": r"\1k_proj.\2",
            r"(.*?self_attn\.)v_proj\.linear\.(weight|bias)$": r"\1v_proj.\2",
            # Attention o_proj: peel ``.linear`` (base class has plain Linear).
            r"(.*?self_attn\.)o_proj\.linear\.(weight|bias)$": r"\1o_proj.\2",
            # patch_embedder.input_proj is a plain TRT-LLM Linear now.
            r"(.*?patch_embedder\.)input_proj\.(weight|bias)$": r"\1input_proj.\2",
        }
        _load_weights_impl(self, remapped, params_map=pattern_mapping)

        # ``_load_weights_impl`` only walks ``named_parameters``; it skips
        # buffers. Copy clamp scalars and optional standardize buffers here.
        for name, buf in self.named_buffers():
            if name in remapped:
                buf.data.copy_(remapped[name].to(buf.dtype).to(buf.device))

    def _remap_clamp_buffers(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Rename HF clamp buffer keys onto our consolidated layout.

        HF: ``encoder.layers.{i}.self_attn.{q,k,v}_proj.{input,output}_{min,max}``
            ``encoder.layers.{i}.self_attn.o_proj.{input,output}_{min,max}``
        Ours: ``encoder.layers.{i}.self_attn.qkv_input_{min,max}`` (collapsed,
              must match across q/k/v)
              ``encoder.layers.{i}.self_attn.{q,k,v}_output_{min,max}``
              ``encoder.layers.{i}.self_attn.o_{input,output}_{min,max}``
        """
        import re

        remapped = dict(weights)
        # Find all attention prefixes by scanning for q_proj.input_min
        prefix_re = re.compile(r"^(.*self_attn)\.q_proj\.input_min$")
        prefixes = sorted({m.group(1) for k in weights for m in [prefix_re.match(k)] if m})
        for prefix in prefixes:
            # Collapse q/k/v input clamps → one qkv_input_*, asserting equality.
            for kind in ("min", "max"):
                q_key = f"{prefix}.q_proj.input_{kind}"
                k_key = f"{prefix}.k_proj.input_{kind}"
                v_key = f"{prefix}.v_proj.input_{kind}"
                if q_key in weights and k_key in weights and v_key in weights:
                    q_val = weights[q_key].clone()
                    k_val = weights[k_key].clone()
                    v_val = weights[v_key].clone()
                    if not (torch.equal(q_val, k_val) and torch.equal(k_val, v_val)):
                        raise ValueError(
                            f"Gemma4 vision clamp fusion: q/k/v input clamp "
                            f"{kind} mismatch at {prefix}; cannot collapse to "
                            f"fused qkv_input_{kind}."
                        )
                    remapped[f"{prefix}.qkv_input_{kind}"] = q_val
                    remapped.pop(q_key, None)
                    remapped.pop(k_key, None)
                    remapped.pop(v_key, None)
                # Per-projection output clamps: keep the values, rename the keys.
                for sec in ("q", "k", "v"):
                    src = f"{prefix}.{sec}_proj.output_{kind}"
                    if src in weights:
                        remapped[f"{prefix}.{sec}_output_{kind}"] = weights[src].clone()
                        remapped.pop(src, None)
                # o_proj clamps: move into the attention module.
                for io in ("input", "output"):
                    src = f"{prefix}.o_proj.{io}_{kind}"
                    if src in weights:
                        remapped[f"{prefix}.o_{io}_{kind}"] = weights[src].clone()
                        remapped.pop(src, None)
        return remapped

    def _pad_attention_head_dim(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Zero-pad q/k/v_proj and o_proj head dim from HF to FMHA-supported.

        No-op when ``head_dim`` was already in the FMHA cubin set (E2B/E4B
        vision: head_dim=64). For 26B/31B vision (head_dim=72), pads to 80
        so the trtllm-gen FMHA dispatcher finds a matching kernel and
        keeps workspace linear in seq_len.

        Math invariant: zeros in the appended channels mean (a) qkv_proj
        outputs zero on those channels (zero-rowed weight × any input = 0),
        (b) softmax(QK^T) is the same since pad·pad=0 doesn't contribute,
        (c) ``attn_output[..., hf:padded]`` is identically zero (V's pad
        channels are zero), and (d) o_proj's zero-padded last columns × the
        zero attn_output channels = 0. Net: byte-equivalent to the unpadded
        HF math, plus 11% extra compute on the kernel side (80/72).
        """
        import re

        first_attn = self.encoder.layers[0].self_attn
        if getattr(first_attn, "head_dim_pad", 0) == 0:
            return weights

        hf_hd = first_attn.hf_head_dim
        padded_hd = first_attn.head_dim
        nh = first_attn.num_heads
        nkv = first_attn.num_key_value_heads
        pad_w = padded_hd - hf_hd

        # HF keys at this point have already had ``.linear.`` stripped if the
        # rename pattern landed first, but the rename happens INSIDE
        # ``_load_weights_impl``. So we see the raw HF keys here:
        #   self_attn.{q,k,v}_proj.linear.{weight,bias}
        #   self_attn.o_proj.linear.{weight,bias}
        qkv_re = re.compile(r"(.*self_attn\.)([qkv])_proj\.linear\.(weight|bias)$")
        o_re = re.compile(r"(.*self_attn\.)o_proj\.linear\.(weight|bias)$")
        out = dict(weights)
        for k, v in list(weights.items()):
            m = qkv_re.match(k)
            if m is not None and m.group(3) == "weight":
                # HF (sec=q): (nh × hf_hd, hidden_size)
                # HF (sec=k,v): (nkv × hf_hd, hidden_size)
                heads = nh if m.group(2) == "q" else nkv
                w = v
                assert w.shape[0] == heads * hf_hd, (
                    f"Unexpected qkv shape {tuple(w.shape)} for {k} "
                    f"(expected first dim = {heads}*{hf_hd})"
                )
                w = w.view(heads, hf_hd, -1)
                zeros = w.new_zeros(heads, pad_w, w.shape[-1])
                out[k] = torch.cat([w, zeros], dim=1).reshape(heads * padded_hd, -1)
                continue
            if m is not None and m.group(3) == "bias":
                heads = nh if m.group(2) == "q" else nkv
                b = v
                assert b.shape[0] == heads * hf_hd
                b = b.view(heads, hf_hd)
                zeros = b.new_zeros(heads, pad_w)
                out[k] = torch.cat([b, zeros], dim=1).reshape(heads * padded_hd)
                continue
            m_o = o_re.match(k)
            if m_o is not None and m_o.group(2) == "weight":
                # HF o_proj.weight: (hidden_size, nh × hf_hd) — pad the input dim.
                w = v
                assert w.shape[1] == nh * hf_hd, (
                    f"Unexpected o_proj shape {tuple(w.shape)} for {k} "
                    f"(expected second dim = {nh}*{hf_hd})"
                )
                w = w.view(-1, nh, hf_hd)
                zeros = w.new_zeros(w.shape[0], nh, pad_w)
                out[k] = torch.cat([w, zeros], dim=2).reshape(-1, nh * padded_hd)
                continue
            # o_proj.bias is shape (hidden_size,) — output dim, not head dim.
            # No padding needed.
        return out
