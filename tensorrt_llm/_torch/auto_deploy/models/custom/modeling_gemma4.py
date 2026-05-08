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

"""Slimmed down Gemma 4 text implementation for AutoDeploy export.

This implementation follows the HuggingFace Gemma 4 text stack closely while
keeping only the prefill path needed by AutoDeploy.  The outer
``Gemma4ForConditionalGeneration`` wrapper preserves the HF checkpoint layout
(``model.language_model.*``). The text model is exported, while the outer
wrapper remains eager so it can run Gemma4's multimodal vision merge path
before delegating to the exported language model.

Key architectural features of Gemma 4 vs standard transformers:
- K=V attention on full-attention layers (v_proj is absent; k_proj output is
  reused as value)
- Different head dimensions for full vs sliding attention (global_head_dim vs
  head_dim)
- Proportional RoPE with partial_rotary_factor on full-attention layers
- Dense MLP running in parallel with Mixture-of-Experts (MoE) in every layer
- Per-layer scalar multiplier
- Final logit softcapping
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from torch.export import Dim
from torch.fx import GraphModule
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerFast
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, cached_file

try:
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.registry import (
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        MultimodalPlaceholderMetadata,
    )
except ModuleNotFoundError:
    ContentFormat = None
    MULTIMODAL_PLACEHOLDER_REGISTRY = None
    MultimodalPlaceholderMetadata = None

from ..._compat import ActivationType
from ...custom_ops.semantic_mask_registry import SemanticMaskLoweringSpec, SemanticMaskRegistry
from ..factory import ModelFactoryRegistry
from ..hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
    TextModelExportInfo,
)

# ---------------------------------------------------------------------------
# Multimodal semantic mask custom ops — prefill-only and cached variants.
# Defined here so the ops are registered when the module is imported.
# ---------------------------------------------------------------------------


def _blob_ids_from_spans(
    seq_len: int,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build contiguous blob ids from multimodal span positions and lengths."""
    blob_ids = torch.zeros(seq_len, dtype=torch.int64, device=device)
    next_blob_id = 1
    for position, length in zip(positions.tolist(), lengths.tolist(), strict=True):
        if length <= 0:
            continue
        start = max(int(position), 0)
        end = min(start + int(length), seq_len)
        if start >= end:
            continue
        blob_ids[start:end] = next_blob_id
        next_blob_id += 1
    return blob_ids


def _causal_or_bidirectional_mask(blob_ids: torch.Tensor) -> torch.Tensor:
    """Build a bool allow-mask from blob ids."""
    seq_len = blob_ids.shape[0]
    causal = torch.tril(torch.ones(seq_len, seq_len, device=blob_ids.device, dtype=torch.bool))
    q_blob = blob_ids.unsqueeze(1)
    kv_blob = blob_ids.unsqueeze(0)
    bidirectional = (q_blob == kv_blob) & (q_blob != 0)
    return causal | bidirectional


@torch.library.custom_op("auto_deploy::gemma4_multimodal_mask", mutates_args=())
def gemma4_multimodal_mask(
    input_ids: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a dense source-graph attention mask from multimodal spans."""
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    batch_size, seq_len = input_ids.shape[:2]
    masks = []
    for batch_idx in range(batch_size):
        span_start = int(mm_item_cu_seqlen[batch_idx].item())
        span_end = int(mm_item_cu_seqlen[batch_idx + 1].item())
        blob_ids = _blob_ids_from_spans(
            seq_len,
            mm_token_positions[span_start:span_end],
            mm_token_lengths[span_start:span_end],
            input_ids.device,
        )
        masks.append(_causal_or_bidirectional_mask(blob_ids))
    return torch.stack(masks, dim=0).unsqueeze(1)


@gemma4_multimodal_mask.register_fake
def _gemma4_multimodal_mask_fake(
    input_ids: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del mm_token_positions, mm_token_lengths, mm_item_cu_seqlen
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    batch_size, seq_len = input_ids.shape[:2]
    return torch.empty(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=input_ids.device)


@torch.library.custom_op("auto_deploy::gemma4_prepare_multimodal_mask", mutates_args=())
def gemma4_prepare_multimodal_mask(
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    input_pos: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build backend-consumable prefill masks from multimodal spans and batch geometry."""
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    num_prefill = int(batch_info_host[0].item())
    if num_prefill == 0 or mm_token_positions.numel() == 0:
        return torch.empty(0, 1, 0, 0, dtype=torch.bool, device=input_pos.device)

    device = input_pos.device
    masks = []
    max_q = 0
    max_kv = 0
    for batch_idx in range(num_prefill):
        q_start = int(input_pos[batch_idx].item())
        q_len = int(cu_seqlen[batch_idx + 1].item()) - int(cu_seqlen[batch_idx].item())
        kv_len = q_start + q_len

        span_start = int(mm_item_cu_seqlen[batch_idx].item())
        span_end = int(mm_item_cu_seqlen[batch_idx + 1].item())
        blob_ids = _blob_ids_from_spans(
            kv_len,
            mm_token_positions[span_start:span_end],
            mm_token_lengths[span_start:span_end],
            device,
        )

        q_blob = blob_ids[q_start : q_start + q_len].unsqueeze(1)
        kv_blob = blob_ids.unsqueeze(0)
        bidirectional = (q_blob == kv_blob) & (q_blob != 0)

        q_pos = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(1)
        kv_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        causal = kv_pos <= q_pos

        mask = (causal | bidirectional).unsqueeze(0)
        masks.append(mask)
        max_q = max(max_q, q_len)
        max_kv = max(max_kv, kv_len)

    padded = []
    for mask in masks:
        _, q_len, kv_len = mask.shape
        padded.append(F.pad(mask, (0, max_kv - kv_len, 0, max_q - q_len), value=False))
    return torch.stack(padded, dim=0)


@gemma4_prepare_multimodal_mask.register_fake
def _gemma4_prepare_multimodal_mask_fake(
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    input_pos: torch.Tensor,
    mm_token_positions: torch.Tensor,
    mm_token_lengths: torch.Tensor,
    mm_item_cu_seqlen: torch.Tensor,
    mm_item_types: Optional[torch.Tensor] = None,
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
    mm_special_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del mm_token_positions, mm_token_lengths, mm_item_cu_seqlen
    del mm_item_types, mm_special_offsets_cu_seqlen, mm_special_offsets
    num_prefill = int(batch_info_host[0].item())
    q_lens = torch.diff(cu_seqlen[: num_prefill + 1])
    max_q = int(q_lens.max().item()) if num_prefill > 0 else 0
    max_kv = int((input_pos[:num_prefill] + q_lens).max().item()) if num_prefill > 0 else 0
    return torch.empty(num_prefill, 1, max_q, max_kv, dtype=torch.bool, device=input_pos.device)


# ---------------------------------------------------------------------------
# Self-register semantic mask lowering specs.  The registry lives in
# ``custom_ops/`` which is imported before ``models/``, so this runs at
# model import time with no circular-dependency issues.
# ---------------------------------------------------------------------------
_GEMMA4_MASK_SPEC = SemanticMaskLoweringSpec(
    prepare_op=torch.ops.auto_deploy.gemma4_prepare_multimodal_mask.default,
    num_outputs=1,
)
SemanticMaskRegistry.register(
    torch.ops.auto_deploy.gemma4_multimodal_mask, "torch", _GEMMA4_MASK_SPEC
)
SemanticMaskRegistry.register(
    torch.ops.auto_deploy.gemma4_multimodal_mask, "triton_paged", _GEMMA4_MASK_SPEC
)


# ---------------------------------------------------------------------------
# Bundled config classes — enables loading on transformers <5.3 where
# Gemma4 is not natively registered.
# ---------------------------------------------------------------------------


class Gemma4TextConfig(PretrainedConfig):
    """Minimal Gemma4 text config for AutoDeploy."""

    model_type = "gemma4_text"

    def __init__(
        self,
        vocab_size: int = 262_144,
        hidden_size: int = 2816,
        intermediate_size: int = 2112,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        global_head_dim: int = 512,
        num_global_key_value_heads: int = 2,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 131_072,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attention_k_eq_v: bool = True,
        sliding_window: int = 1024,
        layer_types: Optional[list] = None,
        rope_parameters: Optional[dict] = None,
        final_logit_softcapping: Optional[float] = 30.0,
        hidden_size_per_layer_input: int = 0,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
        use_bidirectional_attention: Optional[str] = "vision",
        enable_moe_block: bool = True,
        num_experts: Optional[int] = 128,
        top_k_experts: Optional[int] = 8,
        expert_intermediate_size: Optional[int] = 704,
        stream_and_decode_in_f32: bool = True,
        vocab_size_per_layer_input: int = 262_144,
        routed_layer_pattern: Optional[list] = None,
        pad_token_id: Optional[int] = 0,
        eos_token_id=1,
        bos_token_id: Optional[int] = 2,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.sliding_window = sliding_window
        self.layer_types = layer_types or (["sliding_attention"] * num_hidden_layers)
        self.rope_parameters = rope_parameters or {
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        }
        self.final_logit_softcapping = final_logit_softcapping
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        self.use_bidirectional_attention = use_bidirectional_attention
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.stream_and_decode_in_f32 = stream_and_decode_in_f32
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.routed_layer_pattern = routed_layer_pattern
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Gemma4VisionConfig(PretrainedConfig):
    """Gemma4 vision config."""

    model_type = "gemma4_vision"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        head_dim: int = 64,
        hidden_activation: str = "gelu_pytorch_tanh",
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131_072,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_parameters: Optional[dict] = None,
        pooling_kernel_size: int = 3,
        patch_size: int = 16,
        position_embedding_size: int = 10 * 1024,
        standardize: bool = False,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters or {"rope_type": "default", "rope_theta": 100.0}
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.standardize = standardize
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Gemma4Config(PretrainedConfig):
    """Top-level Gemma4 multimodal config."""

    model_type = "gemma4"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        initializer_range: float = 0.02,
        boi_token_id: int = 255_999,
        eoi_token_id: int = 258_882,
        image_token_id: int = 258_880,
        video_token_id: int = 258_884,
        audio_token_id: int = 258_881,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.initializer_range = initializer_range
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        if text_config is None:
            self.text_config = Gemma4TextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = Gemma4VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.audio_config = audio_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


AutoConfig.register("gemma4", Gemma4Config, exist_ok=True)
AutoConfig.register("gemma4_text", Gemma4TextConfig, exist_ok=True)
AutoConfig.register("gemma4_vision", Gemma4VisionConfig, exist_ok=True)

# ---------------------------------------------------------------------------
# RoPE cache builder
# ---------------------------------------------------------------------------


def _build_rope_cache(
    config: Gemma4TextConfig,
    layer_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin RoPE cache for the given layer type."""
    rope_params = config.rope_parameters[layer_type]
    rope_type = rope_params.get("rope_type", "default")
    base = rope_params["rope_theta"]
    factor = rope_params.get("factor", 1.0)
    attention_scaling = 1.0

    if rope_type == "default":
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    elif rope_type == "proportional":
        # Proportional RoPE: only partial_rotary_factor of head dims are rotated,
        # remaining dims get zero inv_freq → cos=1, sin=0 (no rotation).
        head_dim = config.global_head_dim
        rope_proportion = rope_params.get("partial_rotary_factor", 1.0)
        rope_angles = int(rope_proportion * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)),
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated
        inv_freq = inv_freq / factor
    else:
        # Fallback to HF ROPE_INIT_FUNCTIONS for other types (e.g. yarn, longrope)
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(config, device=None, layer_type=layer_type)

    positions = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------


class Gemma4RMSNorm(nn.Module):
    """RMSNorm matching HF Gemma4 (transformers >= 5.5).

    The checkpoint stores effective weights directly — no ``+1.0`` offset.
    Uses the ``torch_rmsnorm`` canonical op for AD transform compatibility.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(dtype=self.weight.dtype)


class Gemma4ClippableLinear(nn.Module):
    """Wrapper matching the upstream Gemma4 ``*.linear.weight`` checkpoint layout."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


class Gemma4RotaryEmbedding(nn.Module):
    """Pre-computed RoPE cache for a single layer type (global or local)."""

    def __init__(self, config: Gemma4TextConfig, layer_type: str):
        super().__init__()
        (
            cos,
            sin,
        ) = _build_rope_cache(config, layer_type)
        self.register_buffer("_ad_cos_cached", cos, persistent=False)
        self.register_buffer("_ad_sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
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
    ndim = position_ids.shape[-1]
    num_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_channels // (2 * ndim))
    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            f"Invalid Gemma4 vision RoPE configuration: num_channels={num_channels}, ndim={ndim}"
        )

    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    outputs = [
        _apply_rotary_pos_emb(
            x=x_parts[idx],
            cos=cos_parts[idx],
            sin=sin_parts[idx],
            unsqueeze_dim=unsqueeze_dim,
        )
        for idx in range(ndim)
    ]
    return torch.cat(outputs, dim=-1)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        position_embeddings = (
            self.position_embedding_table[0][clamped_positions[..., 0]]
            + self.position_embedding_table[1][clamped_positions[..., 1]]
        )
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + position_embeddings


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        kernel_size = int((input_seq_len // length) ** 0.5)
        if kernel_size**2 * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: incompatible kernel size"
            )

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_indices = torch.div(clamped_positions, kernel_size, rounding_mode="floor")
        kernel_indices = kernel_indices[..., 0] + (max_x // kernel_size) * kernel_indices[..., 1]
        weights = F.one_hot(kernel_indices.long(), length).float() / (kernel_size**2)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if output_length is None:
            output_length = hidden_states.shape[1]
        if output_length > hidden_states.shape[1]:
            raise ValueError("Gemma4 vision pooler cannot increase the number of tokens")

        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states *= self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = Gemma4ClippableLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = Gemma4ClippableLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        rope_theta = config.rope_parameters["rope_theta"]
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        all_cos = []
        all_sin = []
        for dim_idx in range(2):
            dim_position_ids = position_ids[:, None, :, dim_idx].float().to(hidden_states.device)
            freqs = (inv_freq_expanded.to(hidden_states.device) @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        sin = torch.cat(all_sin, dim=-1).to(dtype=hidden_states.dtype, device=hidden_states.device)
        return cos, sin


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        del layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = Gemma4ClippableLinear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = Gemma4ClippableLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = Gemma4ClippableLinear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = Gemma4ClippableLinear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = _apply_multidimensional_rope(
            query_states, cos, sin, torch.zeros_like(cos[..., :2])
        )
        query_states = query_states.transpose(1, 2)

        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        key_states = _apply_multidimensional_rope(
            key_states, cos, sin, torch.zeros_like(cos[..., :2])
        )
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))
        value_states = value_states.transpose(1, 2)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=1.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output), None


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(config=config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
    ) -> ModelOutput:
        # Full bidirectional attention over valid patches only.
        valid = attention_mask.to(torch.bool)
        attention_mask_4d = valid[:, None, :, None] & valid[:, None, None, :]
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
        return ModelOutput(last_hidden_state=hidden_states)


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        if self.config.standardize:
            self.register_buffer("std_bias", torch.empty(self.config.hidden_size))
            self.register_buffer("std_scale", torch.empty(self.config.hidden_size))

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_position_ids: torch.LongTensor,
    ) -> ModelOutput:
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return ModelOutput(last_hidden_state=hidden_states)


def _canonicalize_optional_int_tensor(
    tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0, dtype=torch.int32, device=device)
    return tensor.to(device=device, dtype=torch.int32)


def _canonicalize_mm_span_tensors(
    *,
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    mm_item_cu_seqlen: Optional[torch.Tensor],
    mm_token_positions: Optional[torch.Tensor],
    mm_token_lengths: Optional[torch.Tensor],
    mm_item_types: Optional[torch.Tensor],
    mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
    mm_special_offsets: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if input_ids is not None:
        batch_size = input_ids.shape[0]
        device = input_ids.device
    elif inputs_embeds is not None:
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
    else:
        raise ValueError("Either input_ids or inputs_embeds must be provided")

    if mm_item_cu_seqlen is None:
        mm_item_cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    else:
        mm_item_cu_seqlen = mm_item_cu_seqlen.to(device=device, dtype=torch.int32)

    if mm_special_offsets_cu_seqlen is None:
        mm_special_offsets_cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    else:
        mm_special_offsets_cu_seqlen = mm_special_offsets_cu_seqlen.to(
            device=device, dtype=torch.int32
        )

    return {
        "mm_item_cu_seqlen": mm_item_cu_seqlen,
        "mm_token_positions": _canonicalize_optional_int_tensor(mm_token_positions, device=device),
        "mm_token_lengths": _canonicalize_optional_int_tensor(mm_token_lengths, device=device),
        "mm_item_types": _canonicalize_optional_int_tensor(mm_item_types, device=device),
        "mm_special_offsets_cu_seqlen": mm_special_offsets_cu_seqlen,
        "mm_special_offsets": _canonicalize_optional_int_tensor(mm_special_offsets, device=device),
    }


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE Router + Experts
# ---------------------------------------------------------------------------


class Gemma4Router(nn.Module):
    """Gemma4-style MoE router: RMSNorm(no-scale) -> per-dim scale -> linear -> softmax -> topk."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.register_buffer("root_size", torch.tensor(config.hidden_size**-0.5), persistent=False)
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # RMSNorm without learnable scale
        normed = hidden_states.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = normed.type_as(hidden_states)
        # Apply scalar and per-dim scaling
        normed = normed * self.root_size.to(hidden_states.dtype)
        normed = normed * self.scale.to(hidden_states.dtype)
        # Route
        expert_scores = self.proj(normed)
        probs = F.softmax(expert_scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_index


class Gemma4Expert(nn.Module):
    """Single MoE expert: gated MLP (gate_proj, up_proj, down_proj)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class Gemma4MoEBlock(nn.Module):
    """Mixture-of-Experts block with fused checkpoint weight conversion.

    Checkpoint stores fused parameters:
      - gate_up_proj: [num_experts, 2*intermediate, hidden]
      - down_proj: [num_experts, hidden, intermediate]
      - per_expert_scale: [num_experts]

    We unfuse these into per-expert nn.Linear modules at load time so that
    torch_moe can consume them as weight lists.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.expert_intermediate_size
        self.experts = nn.ModuleList(
            [
                Gemma4Expert(config.hidden_size, config.expert_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_moe(
            hidden_states,
            top_k_index,
            top_k_weights,
            w1_weight=[e.gate_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[e.up_proj.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Gelu),
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Full-attention layers may use different head dim and K=V
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding
        if not self.is_sliding and config.global_head_dim:
            self.head_dim = config.global_head_dim
        else:
            self.head_dim = config.head_dim

        num_kv_heads = (
            config.num_global_key_value_heads if self.use_k_eq_v else config.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = (
            None
            if self.use_k_eq_v
            else nn.Linear(
                config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias
            )
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # v_norm has no learnable scale
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.v_proj is not None:
            v = self.v_proj(hidden_states).view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim
            )
        else:
            v = k  # K=V: reuse key as value

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        cos, sin = position_embeddings
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attention_mask,
            0.0,  # dropout_p
            True,  # is_causal
            1.0,  # scale (QK norms handle scaling)
            None,  # sinks
            self.sliding_window,
            None,  # logit_cap
            "bsnd",
            self.layer_idx,
        )
        return self.o_proj(attn_output.reshape(batch_size, seq_len, -1))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.expert_intermediate_size = config.expert_intermediate_size
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4Router(config)
            self.moe = Gemma4MoEBlock(config)
            self._register_load_state_dict_pre_hook(self._unfuse_moe_weights)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def _unfuse_moe_weights(self, state_dict, prefix, *_args, **_kwargs):
        """Convert layer-level fused Gemma4 MoE checkpoint weights to per-expert weights."""
        candidates = [
            (
                prefix + "experts.gate_up_proj",
                prefix + "experts.down_proj",
                prefix + "router.per_expert_scale",
            ),
            (
                prefix + "moe.gate_up_proj",
                prefix + "moe.down_proj",
                prefix + "moe.per_expert_scale",
            ),
        ]

        gate_up_key = down_key = scale_key = None
        for gate_up_candidate, down_candidate, scale_candidate in candidates:
            if (
                gate_up_candidate in state_dict
                and down_candidate in state_dict
                and scale_candidate in state_dict
            ):
                gate_up_key = gate_up_candidate
                down_key = down_candidate
                scale_key = scale_candidate
                break

        if gate_up_key is None or down_key is None or scale_key is None:
            return

        gate_up = state_dict.pop(gate_up_key)  # [E, 2*I, H]
        down = state_dict.pop(down_key)  # [E, H, I]
        scale = state_dict.pop(scale_key)  # [E]

        inter = self.expert_intermediate_size
        for e in range(self.num_experts):
            state_dict[f"{prefix}moe.experts.{e}.gate_proj.weight"] = gate_up[e, :inter, :]
            state_dict[f"{prefix}moe.experts.{e}.up_proj.weight"] = gate_up[e, inter:, :]
            state_dict[f"{prefix}moe.experts.{e}.down_proj.weight"] = down[e] * scale[e]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward (dense MLP ± MoE)
        residual = hidden_states

        if self.enable_moe_block:
            # Dense MLP path
            hs_dense = self.pre_feedforward_layernorm(hidden_states)
            hs_dense = self.mlp(hs_dense)
            hs_dense = self.post_feedforward_layernorm_1(hs_dense)

            # MoE path
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            top_k_weights, top_k_index = self.router(hs_flat)
            hs_moe = self.pre_feedforward_layernorm_2(hs_flat)
            hs_moe = self.moe(hs_moe, top_k_index, top_k_weights)
            hs_moe = hs_moe.reshape(hidden_states.shape)
            hs_moe = self.post_feedforward_layernorm_2(hs_moe)

            hidden_states = hs_dense + hs_moe
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


@dataclass
class Gemma4TextOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Gemma4CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class Gemma4ConditionalOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Gemma4TextPreTrainedModel(PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4TextDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma4TextModel(Gemma4TextPreTrainedModel):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Separate RoPE caches for global (full) and local (sliding) attention
        self.rotary_emb_global = Gemma4RotaryEmbedding(config, "full_attention")
        self.rotary_emb_local = Gemma4RotaryEmbedding(config, "sliding_attention")

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4TextOutput:
        del kwargs
        assert position_ids is not None, "position_ids must be provided"

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        pos_emb_global = self.rotary_emb_global(inputs_embeds, position_ids)
        pos_emb_local = self.rotary_emb_local(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if decoder_layer.attention_type == "sliding_attention":
                pos_emb = pos_emb_local
            else:
                pos_emb = pos_emb_global
            hidden_states = decoder_layer(hidden_states, pos_emb, attention_mask)

        hidden_states = self.norm(hidden_states)
        return Gemma4TextOutput(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# CausalLM wrapper (text config)
# ---------------------------------------------------------------------------


class Gemma4ForCausalLM(Gemma4TextPreTrainedModel, GenerationMixin):
    config_class = Gemma4TextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma4TextConfig, **kwargs):
        del kwargs
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Keep the text backbone modules directly on the CausalLM wrapper so
        # checkpoint keys match the HF layout: `language_model.layers.*`
        # instead of `language_model.model.layers.*`.
        self.rotary_emb_global = Gemma4RotaryEmbedding(config, "full_attention")
        self.rotary_emb_local = Gemma4RotaryEmbedding(config, "sliding_attention")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.register_load_state_dict_post_hook(self._retie_lm_head_weight)
        self.post_init()
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _retie_lm_head_weight(module, incompatible_keys):
        del incompatible_keys
        if not hasattr(module, "config") or not hasattr(module, "lm_head"):
            return
        if getattr(module.config, "tie_word_embeddings", True):
            module.lm_head.weight = module.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mm_item_cu_seqlen: Optional[torch.Tensor] = None,
        mm_token_positions: Optional[torch.Tensor] = None,
        mm_token_lengths: Optional[torch.Tensor] = None,
        mm_item_types: Optional[torch.Tensor] = None,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
        mm_special_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        del kwargs
        assert position_ids is not None, "position_ids must be provided"

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None
        mm_tensors = _canonicalize_mm_span_tensors(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mm_item_cu_seqlen=mm_item_cu_seqlen,
            mm_token_positions=mm_token_positions,
            mm_token_lengths=mm_token_lengths,
            mm_item_types=mm_item_types,
            mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
            mm_special_offsets=mm_special_offsets,
        )
        mask_input_ids = input_ids
        if mask_input_ids is None:
            mask_input_ids = torch.zeros(
                inputs_embeds.shape[:2], dtype=torch.int64, device=inputs_embeds.device
            )
        attention_mask = torch.ops.auto_deploy.gemma4_multimodal_mask.default(
            mask_input_ids,
            mm_tensors["mm_token_positions"],
            mm_tensors["mm_token_lengths"],
            mm_tensors["mm_item_cu_seqlen"],
            mm_tensors["mm_item_types"],
            mm_tensors["mm_special_offsets_cu_seqlen"],
            mm_tensors["mm_special_offsets"],
        )
        pos_emb_global = self.rotary_emb_global(inputs_embeds, position_ids)
        pos_emb_local = self.rotary_emb_local(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if decoder_layer.attention_type == "sliding_attention":
                pos_emb = pos_emb_local
            else:
                pos_emb = pos_emb_global
            hidden_states = decoder_layer(hidden_states, pos_emb, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return Gemma4CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Multimodal embedder stub (for weight loading)
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """Projects multimodal hidden states into language-model space."""

    def __init__(self, vision_config: Gemma4VisionConfig, text_config: Gemma4TextConfig):
        super().__init__()
        self.eps = vision_config.rms_norm_eps
        self.embedding_projection = nn.Linear(
            vision_config.hidden_size, text_config.hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            vision_config.hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(hidden_states)


# ---------------------------------------------------------------------------
# ConditionalGeneration wrapper (multimodal config, text-only forward)
# ---------------------------------------------------------------------------


class Gemma4PreTrainedModel(PreTrainedModel):
    config_class = Gemma4Config
    base_model_prefix = "model"
    _no_split_modules = ["Gemma4TextDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma4Model(Gemma4PreTrainedModel):
    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.language_model = Gemma4ForCausalLM(config.text_config)
        self.vision_tower = (
            Gemma4VisionModel(config.vision_config) if config.vision_config is not None else None
        )
        self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        self._register_load_state_dict_pre_hook(self._remap_and_drop_weights)
        self.post_init()

    @staticmethod
    def _remap_and_drop_weights(state_dict, prefix, *_args, **_kwargs):
        unsupported_prefixes = (
            prefix + "audio_tower.",
            prefix + "embed_audio.",
        )
        for key in list(state_dict):
            if key.startswith(unsupported_prefixes):
                state_dict.pop(key)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: Optional[torch.LongTensor] = None,
    ) -> ModelOutput:
        if self.vision_tower is None:
            raise ValueError("Gemma4 vision_tower is not initialized")
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        return ModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=self.embed_vision(inputs_embeds=last_hidden_state),
        )

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.BoolTensor:
        if input_ids is not None:
            return input_ids == self.config.image_token_id
        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        image_embedding = self.get_input_embeddings()(
            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        return (inputs_embeds == image_embedding).all(-1)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_decoder(self):
        return self.language_model

    def _split_image_features_by_item(
        self,
        image_features: torch.Tensor,
        image_position_ids: Optional[torch.LongTensor],
    ) -> List[torch.Tensor]:
        if image_position_ids is None:
            raise ValueError("image_position_ids is required to split Gemma4 image features")

        pooling_kernel_size = self.config.vision_config.pooling_kernel_size
        image_feature_slices: List[torch.Tensor] = []
        feature_offset = 0

        for item_position_ids in image_position_ids:
            valid_positions = item_position_ids[(item_position_ids != -1).all(dim=-1)]
            if valid_positions.numel() == 0:
                num_feature_tokens = 0
            else:
                max_x = int(valid_positions[:, 0].max().item()) + 1
                max_y = int(valid_positions[:, 1].max().item()) + 1
                pooled_width = (max_x + pooling_kernel_size - 1) // pooling_kernel_size
                pooled_height = (max_y + pooling_kernel_size - 1) // pooling_kernel_size
                num_feature_tokens = pooled_width * pooled_height

            image_feature_slices.append(
                image_features[feature_offset : feature_offset + num_feature_tokens]
            )
            feature_offset += num_feature_tokens

        if feature_offset != image_features.shape[0]:
            raise ValueError(
                "Gemma4 image feature splitting mismatch: "
                f"consumed={feature_offset}, actual={image_features.shape[0]}"
            )

        return image_feature_slices

    def _select_request_chunk_image_features(
        self,
        req_input_pos: int,
        req_seq_len: int,
        req_mm_positions: Sequence[int],
        req_mm_lengths: Sequence[int],
        req_special_offsets: Sequence[int],
        req_image_feature_slices: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        chunk_end = req_input_pos + req_seq_len
        mm_cumulative_offset = 0
        chunks: List[torch.Tensor] = []
        hidden_size = self.config.text_config.hidden_size
        special_offsets_set = set(int(x) for x in req_special_offsets)

        for item_embeds, mm_start, mm_len in zip(
            req_image_feature_slices, req_mm_positions, req_mm_lengths
        ):
            item_mm_offset = mm_cumulative_offset
            item_mm_len = int(mm_len)
            item_abs_start = int(mm_start)
            item_abs_end = item_abs_start + item_mm_len
            overlap_start = max(req_input_pos, item_abs_start)
            overlap_end = min(chunk_end, item_abs_end)

            local_to_feature_idx: List[Optional[int]] = []
            feature_idx = 0
            for rel in range(item_mm_len):
                if item_mm_offset + rel in special_offsets_set:
                    local_to_feature_idx.append(None)
                else:
                    local_to_feature_idx.append(feature_idx)
                    feature_idx += 1

            if feature_idx != item_embeds.shape[0]:
                raise ValueError(
                    "Gemma4 multimodal embedding length mismatch for image item: "
                    f"expected={feature_idx}, actual={item_embeds.shape[0]}, "
                    f"mm_len={item_mm_len}, item_start={item_abs_start}, "
                    f"special_offsets={sorted(special_offsets_set)}"
                )

            if overlap_start < overlap_end:
                selected_indices = [
                    local_to_feature_idx[rel]
                    for rel in range(overlap_start - item_abs_start, overlap_end - item_abs_start)
                    if local_to_feature_idx[rel] is not None
                ]
                if selected_indices:
                    chunks.append(item_embeds[selected_indices])

            mm_cumulative_offset += item_mm_len

        if chunks:
            return torch.cat(chunks, dim=0)

        device = req_image_feature_slices[0].device
        dtype = req_image_feature_slices[0].dtype
        return torch.empty(0, hidden_size, device=device, dtype=dtype)

    def _build_chunked_image_features(
        self,
        image_features: torch.Tensor,
        image_position_ids: Optional[torch.LongTensor],
        batch_info_host: torch.Tensor,
        cu_seqlen: torch.Tensor,
        input_pos: torch.Tensor,
        mm_item_cu_seqlen: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor],
        mm_special_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_prefill_seqs = int(batch_info_host[0].item())
        seq_len = cu_seqlen[1:] - cu_seqlen[:-1]
        image_feature_slices = self._split_image_features_by_item(
            image_features, image_position_ids
        )
        img_idx = 0
        chunks: List[torch.Tensor] = []

        for req_idx in range(num_prefill_seqs):
            item_start = int(mm_item_cu_seqlen[req_idx].item())
            item_end = int(mm_item_cu_seqlen[req_idx + 1].item())
            req_mm_positions = mm_token_positions[item_start:item_end].tolist()
            req_mm_lengths = mm_token_lengths[item_start:item_end].tolist()
            req_num_images = item_end - item_start
            req_image_feature_slices = image_feature_slices[img_idx : img_idx + req_num_images]
            img_idx += req_num_images

            req_special_offsets: List[int] = []
            if mm_special_offsets_cu_seqlen is not None and mm_special_offsets is not None:
                special_start = int(mm_special_offsets_cu_seqlen[req_idx].item())
                special_end = int(mm_special_offsets_cu_seqlen[req_idx + 1].item())
                req_special_offsets = mm_special_offsets[special_start:special_end].tolist()

            req_chunk_features = self._select_request_chunk_image_features(
                req_input_pos=int(input_pos[req_idx].item()),
                req_seq_len=int(seq_len[req_idx].item()),
                req_mm_positions=req_mm_positions,
                req_mm_lengths=req_mm_lengths,
                req_special_offsets=req_special_offsets,
                req_image_feature_slices=req_image_feature_slices,
            )
            chunks.append(req_chunk_features)

        if chunks:
            return torch.cat(chunks, dim=0)

        return image_features[:0]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        batch_info_host: Optional[torch.Tensor] = None,
        cu_seqlen: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        mm_item_cu_seqlen: Optional[torch.Tensor] = None,
        mm_token_positions: Optional[torch.Tensor] = None,
        mm_token_lengths: Optional[torch.Tensor] = None,
        mm_item_types: Optional[torch.Tensor] = None,
        mm_special_offsets_cu_seqlen: Optional[torch.Tensor] = None,
        mm_special_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided"
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            image_mask = self.get_placeholder_mask(input_ids=input_ids)
            llm_input_ids = input_ids.clone()
            llm_input_ids = torch.where(
                image_mask,
                torch.full_like(llm_input_ids, self.config.text_config.pad_token_id),
                llm_input_ids,
            )
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        else:
            image_mask = self.get_placeholder_mask(inputs_embeds=inputs_embeds)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
            ).pooler_output
            has_chunk_mm_layout = (
                batch_info_host is not None
                and cu_seqlen is not None
                and input_pos is not None
                and mm_item_cu_seqlen is not None
                and mm_token_positions is not None
                and mm_token_lengths is not None
                and mm_item_cu_seqlen.numel() > 0
                and int(mm_item_cu_seqlen[-1].item()) > 0
                and mm_token_positions.numel() > 0
                and mm_token_lengths.numel() > 0
            )
            if has_chunk_mm_layout:
                image_features = self._build_chunked_image_features(
                    image_features=image_features,
                    image_position_ids=image_position_ids,
                    batch_info_host=batch_info_host,
                    cu_seqlen=cu_seqlen,
                    input_pos=input_pos,
                    mm_item_cu_seqlen=mm_item_cu_seqlen,
                    mm_token_positions=mm_token_positions,
                    mm_token_lengths=mm_token_lengths,
                    mm_special_offsets_cu_seqlen=mm_special_offsets_cu_seqlen,
                    mm_special_offsets=mm_special_offsets,
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            # _dbg.debug(
            #     "[Gemma4Model EMBED INJECTION] image_mask sum=%d | image_mask shape=%s | "
            #     "image_features shape=%s | image_features mean=%.6f std=%.6f | "
            #     "inputs_embeds mean=%.6f std=%.6f (before scatter) | "
            #     "pixel_values shape=%s mean=%.6f",
            #     image_mask.sum().item(),
            #     tuple(image_mask.shape),
            #     tuple(image_features.shape),
            #     image_features.mean().item(),
            #     image_features.std().item(),
            #     inputs_embeds.mean().item(),
            #     inputs_embeds.std().item(),
            #     tuple(pixel_values.shape),
            #     pixel_values.mean().item(),
            # )
            if inputs_embeds[expanded_image_mask].numel() != image_features.numel():
                placeholder_token_count = int(image_mask.sum().item())
                feature_token_count = int(image_features.shape[0])
                raise ValueError(
                    "Image features and image placeholder tokens do not match: "
                    f"placeholder_tokens={placeholder_token_count}, "
                    f"feature_tokens={feature_token_count}, "
                    f"pixel_values_shape={tuple(pixel_values.shape)}, "
                    f"image_position_ids_shape="
                    f"{tuple(image_position_ids.shape) if image_position_ids is not None else None}, "
                    f"input_pos={input_pos.tolist() if input_pos is not None else None}, "
                    f"cu_seqlen={cu_seqlen.tolist() if cu_seqlen is not None else None}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_features)
            # _dbg.debug(
            #     "[Gemma4Model EMBED INJECTION] inputs_embeds mean=%.6f std=%.6f (after scatter)",
            #     inputs_embeds.mean().item(),
            #     inputs_embeds.std().item(),
            # )

        language_model_kwargs = dict(kwargs)
        if batch_info_host is not None:
            language_model_kwargs["batch_info_host"] = batch_info_host
        if cu_seqlen is not None:
            language_model_kwargs["cu_seqlen"] = cu_seqlen
        if input_pos is not None:
            language_model_kwargs["input_pos"] = input_pos
        if mm_item_cu_seqlen is not None:
            language_model_kwargs["mm_item_cu_seqlen"] = mm_item_cu_seqlen
        if mm_token_positions is not None:
            language_model_kwargs["mm_token_positions"] = mm_token_positions
        if mm_token_lengths is not None:
            language_model_kwargs["mm_token_lengths"] = mm_token_lengths
        if mm_item_types is not None:
            language_model_kwargs["mm_item_types"] = mm_item_types
        if mm_special_offsets_cu_seqlen is not None:
            language_model_kwargs["mm_special_offsets_cu_seqlen"] = mm_special_offsets_cu_seqlen
        if mm_special_offsets is not None:
            language_model_kwargs["mm_special_offsets"] = mm_special_offsets

        return Gemma4ForConditionalGeneration._call_language_model(
            self.language_model,
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **language_model_kwargs,
        )


class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    config_class = Gemma4Config
    _tied_weights_keys = {
        "model.language_model.lm_head.weight": "model.language_model.embed_tokens.weight"
    }

    def __init__(self, config: Gemma4Config, **kwargs):
        del kwargs
        super().__init__(config)
        self.model = Gemma4Model(config)
        self._register_load_state_dict_pre_hook(self._remap_lm_head_weight)
        self.post_init()

    @staticmethod
    def _remap_lm_head_weight(state_dict, prefix, *_args, **_kwargs):
        """Remap lm_head into language_model so the export info exports it."""
        old_key = prefix + "lm_head.weight"
        new_key = prefix + "model.language_model.lm_head.weight"
        if old_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.lm_head

    def set_output_embeddings(self, value):
        self.model.language_model.lm_head = value

    def get_decoder(self):
        return self.model.get_decoder()

    @staticmethod
    def _call_language_model(
        language_model: nn.Module,
        input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor],
        **kwargs,
    ):
        """Call eager modules and exported FX graphs using their expected input structure."""
        if not isinstance(language_model, GraphModule):
            model_kwargs = dict(kwargs)
            model_kwargs["position_ids"] = position_ids
            if inputs_embeds is not None:
                model_kwargs["inputs_embeds"] = inputs_embeds
            else:
                model_kwargs["input_ids"] = input_ids
            return language_model(**model_kwargs)

        available_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }
        placeholder_names = [
            node.target for node in language_model.graph.nodes if node.op == "placeholder"
        ]
        in_spec = getattr(language_model, "_in_spec", None)
        if in_spec is not None and in_spec.type is tuple and in_spec.num_children == 2:
            pos_spec = in_spec.child(0)
            kw_spec = in_spec.child(1)
            num_positional = pos_spec.num_children if pos_spec.type is tuple else 0
            positional_names = placeholder_names[:num_positional]
            keyword_names = list(kw_spec.context) if kw_spec.type is dict else []

            positional_args = [available_args.get(name) for name in positional_names]
            keyword_args = {name: available_args.get(name) for name in keyword_names}
            return language_model(*positional_args, **keyword_args)

        positional_args = [available_args.get(name) for name in placeholder_names]
        return language_model(*positional_args)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4ConditionalOutput:
        kwargs.pop("token_type_ids", None)
        model_kwargs = dict(kwargs)
        if "cu_seqlen" not in model_kwargs and "cu_seqlen_host" in model_kwargs:
            model_kwargs["cu_seqlen"] = model_kwargs.pop("cu_seqlen_host")
        if "seq_len_with_cache" not in model_kwargs and "seq_len_with_cache_host" in model_kwargs:
            model_kwargs["seq_len_with_cache"] = model_kwargs.pop("seq_len_with_cache_host")
        model_kwargs.pop("mm_chunk_flat_start", None)
        model_kwargs.pop("mm_chunk_count", None)
        mm_tensors = _canonicalize_mm_span_tensors(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            mm_item_cu_seqlen=model_kwargs.pop("mm_item_cu_seqlen", None),
            mm_token_positions=model_kwargs.pop("mm_token_positions", None),
            mm_token_lengths=model_kwargs.pop("mm_token_lengths", None),
            mm_item_types=model_kwargs.pop("mm_item_types", None),
            mm_special_offsets_cu_seqlen=model_kwargs.pop("mm_special_offsets_cu_seqlen", None),
            mm_special_offsets=model_kwargs.pop("mm_special_offsets", None),
        )
        model_kwargs.update(mm_tensors)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )
        return Gemma4ConditionalOutput(logits=outputs.logits)


# ---------------------------------------------------------------------------
# Wrapper tokenizer for Gemma 4
#
# The upstream HF checkpoint ships ``extra_special_tokens`` as a *list* in
# tokenizer_config.json, which is incompatible with transformers <5.3.
# This thin wrapper loads the tokenizer assets directly, bypassing the
# problematic codepath.
# ---------------------------------------------------------------------------

_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_PROCESSOR_CONFIG_FILE = "processor_config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_FILE = "tokenizer.json"
_SUPPORTED_GEMMA4_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> Tuple[int, int]:
    """Resize within the Gemma4 patch budget while preserving aspect ratio."""
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = (target_px / total_px) ** 0.5
    ideal_height = factor * height
    ideal_width = factor * width
    side_multiple = pooling_kernel_size * patch_size

    target_height = int(ideal_height // side_multiple) * side_multiple
    target_width = int(ideal_width // side_multiple) * side_multiple

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. "
            f"Resized height should be divisible by `pooling_kernel_size * patch_size`={side_multiple}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if target_height == 0:
        target_height = side_multiple
        target_width = min((width // height) * side_multiple, max_side_length)
    elif target_width == 0:
        target_width = side_multiple
        target_height = min((height // width) * side_multiple, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"exceeds {max_patches} patches with patch_size {patch_size}."
        )

    return target_height, target_width


class ADGemma4Tokenizer(PreTrainedTokenizerFast):
    """Wrapper that loads the upstream Gemma 4 tokenizer on current transformers."""

    vocab_files_names = {"tokenizer_file": _TOKENIZER_FILE}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *inputs,
        **kwargs,
    ) -> "ADGemma4Tokenizer":
        del inputs
        for k in ("_from_auto", "_commit_hash", "trust_remote_code"):
            kwargs.pop(k, None)

        config_path = cached_file(pretrained_model_name_or_path, _TOKENIZER_CONFIG_FILE, **kwargs)
        assert config_path is not None
        config = json.loads(Path(config_path).read_text())

        tokenizer_file = cached_file(pretrained_model_name_or_path, _TOKENIZER_FILE, **kwargs)
        assert tokenizer_file is not None

        # ``extra_special_tokens`` is a list in the upstream config; map it to
        # the standard ``additional_special_tokens`` field.
        extra = config.get("extra_special_tokens", [])
        if isinstance(extra, list):
            additional = extra
        else:
            additional = list(extra.keys()) if isinstance(extra, dict) else []

        tokenizer = cls(
            tokenizer_object=Tokenizer.from_file(tokenizer_file),
            name_or_path=str(pretrained_model_name_or_path),
            bos_token=config.get("bos_token"),
            eos_token=config.get("eos_token"),
            unk_token=config.get("unk_token"),
            pad_token=config.get("pad_token"),
            additional_special_tokens=additional,
            clean_up_tokenization_spaces=config.get("clean_up_tokenization_spaces", False),
            model_max_length=config.get("model_max_length"),
            padding_side=config.get("padding_side", "left"),
            truncation_side=config.get("truncation_side", "left"),
        )

        tokenizer.image_token = config.get("image_token", "<|image|>")
        tokenizer.boi_token = config.get("boi_token", "<|image>")
        tokenizer.eoi_token = config.get("eoi_token", "<image|>")
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
        tokenizer.boi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.boi_token)
        tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eoi_token)

        template_path = cached_file(
            pretrained_model_name_or_path,
            _CHAT_TEMPLATE_FILE,
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        if template_path is not None:
            tokenizer.chat_template = Path(template_path).read_text()

        return tokenizer


class ADGemma4ImageProcessor:
    """Minimal Gemma4 image processor compatible with the local transformers version."""

    def __init__(
        self,
        *,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = False,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        resample: int = Image.BICUBIC,
    ) -> None:
        if max_soft_tokens not in _SUPPORTED_GEMMA4_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {_SUPPORTED_GEMMA4_SOFT_TOKENS}, got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.0, 0.0, 0.0]
        self.image_std = image_std or [1.0, 1.0, 1.0]
        self.resample = resample

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "ADGemma4ImageProcessor":
        for key in ("_from_auto", "_commit_hash", "trust_remote_code"):
            kwargs.pop(key, None)

        config_path = cached_file(pretrained_model_name_or_path, _PROCESSOR_CONFIG_FILE, **kwargs)
        assert config_path is not None
        processor_config = json.loads(Path(config_path).read_text())
        image_config = processor_config.get("image_processor", {})
        allowed_keys = {
            "patch_size",
            "max_soft_tokens",
            "pooling_kernel_size",
            "do_convert_rgb",
            "do_resize",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "resample",
        }
        filtered_config = {key: value for key, value in image_config.items() if key in allowed_keys}
        return cls(**filtered_config)

    @staticmethod
    def fetch_images(images):
        return images

    @staticmethod
    def _to_tensor(image, do_convert_rgb: bool) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            if do_convert_rgb:
                image = image.convert("RGB")
            array = np.array(image, copy=True)
            tensor = torch.from_numpy(array)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            return tensor.permute(2, 0, 1).contiguous().to(torch.float32)

        if torch.is_tensor(image):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Expected a 3D image tensor, got shape {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                return tensor.to(torch.float32)
            if tensor.shape[-1] in (1, 3):
                return tensor.permute(2, 0, 1).contiguous().to(torch.float32)
            raise ValueError(f"Unsupported tensor image shape {tuple(tensor.shape)}")

        array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"Unsupported image with shape {array.shape}")
        tensor = torch.from_numpy(array)
        if tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous().to(torch.float32)

    @staticmethod
    def _convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
        channels, image_height, image_width = image.shape
        num_patches_height = image_height // patch_size
        num_patches_width = image_width // patch_size
        patched = image.reshape(
            channels,
            num_patches_height,
            patch_size,
            num_patches_width,
            patch_size,
        )
        patched = patched.permute(1, 3, 2, 4, 0)
        return patched.reshape(num_patches_height * num_patches_width, -1)

    @staticmethod
    def _pad_along_first_dim(
        image: torch.Tensor, positions: torch.Tensor, target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_length = image.shape[0]
        padding_length = target_length - current_length
        if padding_length <= 0:
            return image, positions
        image_padding = torch.zeros(
            (padding_length, image.shape[1]), dtype=image.dtype, device=image.device
        )
        pos_padding = torch.full(
            (padding_length, 2), -1, dtype=positions.dtype, device=positions.device
        )
        return torch.cat([image, image_padding], dim=0), torch.cat([positions, pos_padding], dim=0)

    def _aspect_ratio_preserving_resize(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=self.patch_size,
            max_patches=max_patches,
            pooling_kernel_size=self.pooling_kernel_size,
        )
        if target_height == height and target_width == width:
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def __call__(
        self,
        images,
        *,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        return_tensors: Optional[str] = None,
        **_kwargs,
    ) -> dict[str, Any]:
        del return_tensors
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb
        do_resize = self.do_resize if do_resize is None else do_resize
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std

        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []
        mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)
        target_patches = self.max_soft_tokens * self.pooling_kernel_size**2

        for image in images:
            tensor = self._to_tensor(image, do_convert_rgb=do_convert_rgb)
            if do_resize:
                tensor = self._aspect_ratio_preserving_resize(tensor)
            if do_rescale:
                tensor = tensor * rescale_factor
            if do_normalize:
                tensor = (tensor - mean) / std

            patches = self._convert_image_to_patches(tensor, self.patch_size)
            num_soft_tokens_per_image.append(patches.shape[0] // self.pooling_kernel_size**2)

            patch_height = tensor.shape[-2] // self.patch_size
            patch_width = tensor.shape[-1] // self.patch_size
            grid_y, grid_x = torch.meshgrid(
                torch.arange(patch_height, dtype=torch.int64),
                torch.arange(patch_width, dtype=torch.int64),
                indexing="ij",
            )
            positions = torch.stack([grid_x, grid_y], dim=-1).reshape(patches.shape[0], 2)

            pixel_values.append(patches)
            position_ids.append(positions)
        pixel_values_padded = []
        position_ids_padded = []
        for patches, positions in zip(pixel_values, position_ids):
            padded_patches, padded_positions = self._pad_along_first_dim(
                patches, positions, target_patches
            )
            pixel_values_padded.append(padded_patches)
            position_ids_padded.append(padded_positions)

        return {
            "pixel_values": torch.stack(pixel_values_padded, dim=0),
            "image_position_ids": torch.stack(position_ids_padded, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }


class ADGemma4Processor:
    """Minimal Gemma4 multimodal processor for image-text requests."""

    def __init__(
        self,
        *,
        tokenizer: ADGemma4Tokenizer,
        image_processor: ADGemma4ImageProcessor,
        image_seq_length: int = 280,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.image_token_id = tokenizer.image_token_id
        self.boi_token_id = tokenizer.boi_token_id
        self.eoi_token_id = tokenizer.eoi_token_id
        self.chat_template = getattr(tokenizer, "chat_template", None)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "ADGemma4Processor":
        tokenizer = ADGemma4Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor = ADGemma4ImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        config_path = cached_file(pretrained_model_name_or_path, _PROCESSOR_CONFIG_FILE, **kwargs)
        assert config_path is not None
        processor_config = json.loads(Path(config_path).read_text())
        return cls(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_length=processor_config.get(
                "image_seq_length", image_processor.max_soft_tokens
            ),
        )

    @staticmethod
    def _ensure_text_list(text) -> List[str]:
        if text is None:
            return []
        if isinstance(text, str):
            return [text]
        return list(text)

    @staticmethod
    def _normalize_batched_images(images) -> List[List[Any]]:
        if images is None:
            return []
        if not isinstance(images, list):
            return [[images]]
        if not images:
            return []
        if isinstance(images[0], list):
            return [list(batch) for batch in images]
        return [list(images)]

    def _expand_image_placeholders(
        self, text: List[str], batched_images: List[List[Any]], image_inputs: dict[str, Any]
    ) -> List[str]:
        num_soft_tokens = image_inputs.pop("num_soft_tokens_per_image")
        if not text:
            text = [" ".join([self.image_token] * len(images)) for images in batched_images]
        if len(text) != len(batched_images):
            raise ValueError(
                f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
            )

        replacements = [
            f"{self.boi_token}{self.image_token * num_tokens}{self.eoi_token}"
            for num_tokens in num_soft_tokens
        ]
        replacements_iter = iter(replacements)
        pattern = re.escape(self.image_token)
        return [re.sub(pattern, lambda _match: next(replacements_iter), prompt) for prompt in text]

    def _build_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_type_ids = torch.zeros_like(input_ids)
        next_blob_id = 1
        for batch_idx in range(input_ids.shape[0]):
            in_blob = False
            current_blob_id = 0
            for token_idx, token in enumerate(input_ids[batch_idx].tolist()):
                if token == self.boi_token_id:
                    in_blob = True
                    current_blob_id = next_blob_id
                    next_blob_id += 1
                    token_type_ids[batch_idx, token_idx] = current_blob_id
                elif token == self.eoi_token_id and in_blob:
                    token_type_ids[batch_idx, token_idx] = current_blob_id
                    in_blob = False
                    current_blob_id = 0
                elif in_blob:
                    token_type_ids[batch_idx, token_idx] = current_blob_id
        return token_type_ids

    @staticmethod
    def _render_messages(messages) -> Tuple[List[str], List[Any]]:
        """Extract text + images from a single conversation.

        Returns ``(rendered_text, images)`` where ``rendered_text`` has an
        ``<|image|>`` placeholder for each image.
        """
        parts: List[str] = []
        images: List[Any] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image":
                    parts.append("<|image|>")
                    images.append(item.get("image"))
        return " ".join(part for part in parts if part), images

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: Optional[str] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ):
        is_batched = messages and isinstance(messages[0], list)
        batched_messages = messages if is_batched else [messages]

        rendered_texts: List[str] = []
        batched_images: List[List[Any]] = []
        has_chat_template = bool(self.chat_template)

        for conversation in batched_messages:
            if has_chat_template:
                # Use the Jinja chat template for proper turn formatting.
                # Strip image content items so the template only sees text;
                # we insert image placeholders ourselves afterwards.
                text_only_conv = []
                conv_images: List[Any] = []
                for message in conversation:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text_only_conv.append(message)
                        continue
                    text_parts: List[str] = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            text_parts.append(self.image_token)
                            conv_images.append(item.get("image"))
                    text_only_conv.append({**message, "content": " ".join(text_parts)})
                rendered_texts.append(
                    self.tokenizer.apply_chat_template(
                        text_only_conv,
                        chat_template=self.chat_template,
                        add_generation_prompt=add_generation_prompt,
                        tokenize=False,
                    )
                )
                batched_images.append(conv_images)
            else:
                # No chat template — render messages directly.
                text, conv_images = self._render_messages(conversation)
                rendered_texts.append(text)
                batched_images.append(conv_images)

        if not tokenize:
            return rendered_texts if is_batched else rendered_texts[0]

        result = self(
            text=rendered_texts,
            images=batched_images,
            return_dict=True,
            return_tensors=return_tensors,
            **kwargs,
        )
        if return_dict:
            return result
        return result["input_ids"]

    def __call__(
        self,
        *,
        images=None,
        text=None,
        return_dict: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        del return_dict
        batched_images = self._normalize_batched_images(images)
        flat_images = [image for batch in batched_images for image in batch]
        text_list = self._ensure_text_list(text)

        image_inputs = {}
        if flat_images:
            image_inputs = self.image_processor(
                flat_images,
                return_tensors=return_tensors,
                **kwargs,
            )
            text_list = self._expand_image_placeholders(text_list, batched_images, image_inputs)

        tokenizer_kwargs = dict(kwargs)
        tokenizer_kwargs.pop("do_rescale", None)
        tokenizer_kwargs.pop("do_convert_rgb", None)
        tokenizer_kwargs.pop("rescale_factor", None)
        tokenizer_kwargs.pop("do_resize", None)
        tokenizer_kwargs.pop("do_normalize", None)
        tokenizer_kwargs["return_tensors"] = return_tensors
        tokenizer_kwargs["return_attention_mask"] = return_attention_mask
        text_inputs = self.tokenizer(text=text_list, **tokenizer_kwargs)
        text_inputs["token_type_ids"] = self._build_token_type_ids(text_inputs["input_ids"])
        return {**text_inputs, **image_inputs}


class Gemma4ADInputProcessor:
    """Input processor that ensures ``multimodal_input`` is set.

    For multimodal requests, ``multimodal_input`` is computed with image token
    positions and lengths so the AD executor can stage span metadata
    (``mm_token_positions``, ``mm_token_lengths``, ``mm_item_cu_seqlen``) for
    the exported text model to emit the Gemma4 semantic attention mask op.
    """

    def __init__(self, base, image_token_id: int, boi_token_id: int, eoi_token_id: int):
        self.base = base
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        # Bypass the generic multimodal hashing probe. This processor already
        # constructs ``multimodal_input`` directly from Gemma4 image spans.
        self.multimodal_hashing_supported = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_num_tokens_per_image(self, *, image: Image.Image, **kwargs) -> int:
        processor = getattr(self, "processor", None)
        if processor is None:
            raise AttributeError(
                "Gemma4ADInputProcessor requires a processor for image token sizing."
            )

        image_inputs = processor.image_processor([image], **kwargs)
        num_soft_tokens = int(image_inputs["num_soft_tokens_per_image"][0].item())
        return num_soft_tokens + 2  # include BOI + EOI

    def get_vocab_size(self) -> Optional[int]:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "vocab_size"):
            return int(tokenizer.vocab_size)
        wrapped_tokenizer = getattr(tokenizer, "tokenizer", None)
        if wrapped_tokenizer is not None and hasattr(wrapped_tokenizer, "vocab_size"):
            return int(wrapped_tokenizer.vocab_size)
        processor = getattr(self, "processor", None)
        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None and hasattr(processor_tokenizer, "vocab_size"):
            return int(processor_tokenizer.vocab_size)
        return None

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([self.image_token_id], dtype=torch.int32)

    def get_mm_special_token_ids(self) -> torch.Tensor:
        return torch.tensor(sorted({self.boi_token_id, self.eoi_token_id}), dtype=torch.int32)

    def _find_image_spans(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Find start positions and lengths of each image blob (boi…eoi) span."""
        positions: List[int] = []
        lengths: List[int] = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.boi_token_id:
                start = i
                # Scan to the matching eoi token
                j = i + 1
                while j < len(token_ids) and token_ids[j] != self.eoi_token_id:
                    j += 1
                end = j + 1 if j < len(token_ids) else j  # include eoi
                positions.append(start)
                lengths.append(end - start)
                i = end
            else:
                i += 1
        return positions, lengths

    def __call__(self, inputs, sampling_params):
        token_ids, extra = self.base(inputs, sampling_params)
        if extra is None:
            extra = {}

        # Remove token_type_ids if the base processor added it — mask is now
        # built from span metadata in the eager wrapper.
        mm_data = extra.get("multimodal_data")
        if mm_data is not None:
            mm_data.pop("token_type_ids", None)

        # Compute multimodal_input so the executor knows where image spans are.
        if "multimodal_input" not in extra:
            positions, lengths = self._find_image_spans(token_ids)
            if positions:
                from tensorrt_llm.inputs.multimodal import MultimodalInput

                # Dummy hashes — KV-cache reuse for images is not yet supported.
                dummy_hashes = [[0] * 8 for _ in positions]
                extra["multimodal_input"] = MultimodalInput.from_components(
                    mm_hashes=dummy_hashes,
                    mm_positions=positions,
                    mm_lengths=lengths,
                )
                multimodal_data = extra.get("multimodal_data", {})
                special_offsets: List[int] = []
                mm_offset = 0
                for length in lengths:
                    special_offsets.extend([mm_offset, mm_offset + length - 1])
                    mm_offset += length
                multimodal_data["layout_metadata"] = {
                    "special_token_offsets": torch.tensor(special_offsets, dtype=torch.int32),
                    "item_types": torch.zeros(len(positions), dtype=torch.int32),
                }
                extra["multimodal_data"] = multimodal_data

        return token_ids, extra


class Gemma4TextExportInfo(TextModelExportInfo):
    """Export info for Gemma4 text graphs with multimodal span tensor inputs."""

    def _init_dynamic_shape_lookup(self):
        dynamic_shapes = super()._init_dynamic_shape_lookup()
        dynamic_shapes["mm_item_cu_seqlen"] = {0: Dim.DYNAMIC}
        dynamic_shapes["mm_item_types"] = {0: Dim.DYNAMIC}
        dynamic_shapes["mm_token_positions"] = {0: Dim.DYNAMIC}
        dynamic_shapes["mm_token_lengths"] = {0: Dim.DYNAMIC}
        dynamic_shapes["mm_special_offsets_cu_seqlen"] = {0: Dim.DYNAMIC}
        dynamic_shapes["mm_special_offsets"] = {0: Dim.DYNAMIC}
        return dynamic_shapes


@ModelFactoryRegistry.register("Gemma4ForConditionalGeneration")
class Gemma4ForConditionalGenerationFactory(AutoModelForImageTextToTextFactory):
    """Factory for Gemma 4 VLM with custom attention mask support."""

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=torch.int32,
        )
        return {
            "input_ids": input_ids,
            "mm_item_cu_seqlen": torch.tensor([0, 1, 2], dtype=torch.int32),
            "mm_item_types": torch.tensor([0, 0], dtype=torch.int32),
            "mm_token_positions": torch.tensor([1, 1], dtype=torch.int32),
            "mm_token_lengths": torch.tensor([2, 2], dtype=torch.int32),
            "mm_special_offsets_cu_seqlen": torch.tensor([0, 2, 4], dtype=torch.int32),
            "mm_special_offsets": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        }

    def get_export_infos(self, model: nn.Module):
        return [Gemma4TextExportInfo.from_autoinferred(model)]

    def init_tokenizer(self) -> Optional[Any]:
        if self.tokenizer is None:
            return None
        return ADGemma4Tokenizer.from_pretrained(self.tokenizer)

    def init_processor(self) -> Optional[Any]:
        """Return the local Gemma4 multimodal processor."""
        if self.tokenizer is None:
            return None
        return ADGemma4Processor.from_pretrained(self.tokenizer)

    def init_input_processor(self, base):
        processor = self.init_processor()
        image_token_id = getattr(processor, "image_token_id", 258_880)
        boi_token_id = getattr(processor, "boi_token_id", 255_999)
        eoi_token_id = getattr(processor, "eoi_token_id", 258_882)
        return Gemma4ADInputProcessor(
            base,
            image_token_id=image_token_id,
            boi_token_id=boi_token_id,
            eoi_token_id=eoi_token_id,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("Gemma4TextConfig", Gemma4ForCausalLM)
Gemma4ForConditionalGenerationFactory.register_custom_model_cls(
    "Gemma4Config", Gemma4ForConditionalGeneration
)
if MULTIMODAL_PLACEHOLDER_REGISTRY is not None:
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "gemma4",
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<|image|>"},
            content_format=ContentFormat.STRING,
        ),
    )
