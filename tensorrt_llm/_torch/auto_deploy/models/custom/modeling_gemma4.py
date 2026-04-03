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
(``model.language_model.*``) and drops unsupported vision/audio tower weights
at load time.  The forward path supports text-only export.

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
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from torch import nn
from torch.fx import GraphModule
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerFast
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, cached_file

from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry, SubModuleExportInfo
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
    TextModelExportInfo,
)
from tensorrt_llm._torch.utils import ActivationType

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
    """Minimal Gemma4 vision config stub."""

    model_type = "gemma4_vision"

    def __init__(self, hidden_size: int = 1152, rms_norm_eps: float = 1e-6, **kwargs):
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
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
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.initializer_range = initializer_range
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
            None,  # attn_mask
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
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
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
            hidden_states = decoder_layer(hidden_states, pos_emb)

        hidden_states = self.norm(hidden_states)
        return Gemma4TextOutput(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# CausalLM wrapper (text config)
# ---------------------------------------------------------------------------


class Gemma4ForCausalLM(Gemma4TextPreTrainedModel, GenerationMixin):
    config_class = Gemma4TextConfig
    _tied_weights_keys = ["lm_head.weight"]

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
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        del kwargs
        assert position_ids is not None, "position_ids must be provided"

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None
        pos_emb_global = self.rotary_emb_global(inputs_embeds, position_ids)
        pos_emb_local = self.rotary_emb_local(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if decoder_layer.attention_type == "sliding_attention":
                pos_emb = pos_emb_local
            else:
                pos_emb = pos_emb_global
            hidden_states = decoder_layer(hidden_states, pos_emb)

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
    """Minimal stub to accept embed_vision checkpoint weights."""

    def __init__(self, vision_config: Gemma4VisionConfig, text_config: Gemma4TextConfig):
        super().__init__()
        self.embedding_projection = nn.Linear(
            vision_config.hidden_size, text_config.hidden_size, bias=False
        )


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
        self.vision_tower = nn.Module()  # stub
        self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        self._register_load_state_dict_pre_hook(self._remap_and_drop_weights)
        self.post_init()

    @staticmethod
    def _remap_and_drop_weights(state_dict, prefix, *_args, **_kwargs):
        unsupported_prefixes = (
            prefix + "vision_tower.",
            prefix + "audio_tower.",
            prefix + "embed_audio.",
        )
        for key in list(state_dict):
            if key.startswith(unsupported_prefixes):
                state_dict.pop(key)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_decoder(self):
        return self.language_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        del kwargs
        assert position_ids is not None, "position_ids must be provided"
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    config_class = Gemma4Config
    _tied_weights_keys = ["model.language_model.lm_head.weight"]

    def __init__(self, config: Gemma4Config, **kwargs):
        del kwargs
        super().__init__(config)
        self.model = Gemma4Model(config)
        self._register_load_state_dict_pre_hook(self._remap_lm_head_weight)
        self.post_init()

    @staticmethod
    def _remap_lm_head_weight(state_dict, prefix, *_args, **_kwargs):
        """Remap lm_head into language_model so TextModelExportInfo exports it."""
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

    @staticmethod
    def _build_attention_mask(token_type_ids: torch.Tensor) -> torch.Tensor:
        """Build a bool attention mask from ``token_type_ids``.

        Returns a ``[batch, 1, seq, seq]`` bool mask that is causal for text
        tokens and bidirectional within contiguous media blobs.
        """
        batch_size, seq_len = token_type_ids.shape
        device = token_type_ids.device

        # Identify non-text tokens and detect blob boundaries
        non_text = token_type_ids.ne(0)
        prev = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=token_type_ids.dtype, device=device),
                token_type_ids[:, :-1],
            ],
            dim=1,
        )
        blob_starts = non_text & token_type_ids.ne(prev)
        blob_ids = torch.cumsum(blob_starts.to(torch.int64), dim=1)
        token_blob_ids = torch.where(non_text, blob_ids, torch.zeros_like(blob_ids))

        # Bidirectional within same media blob
        blob_q = token_blob_ids.unsqueeze(2)  # [B, S, 1]
        blob_k = token_blob_ids.unsqueeze(1)  # [B, 1, S]
        bidirectional_media = (blob_q == blob_k) & (blob_q != 0)

        # Standard causal mask
        positions = torch.arange(seq_len, device=device)
        causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)  # [S, S]
        causal_mask = causal_mask.unsqueeze(0)  # [1, S, S]

        # Combine: causal OR bidirectional-within-blob
        combined = causal_mask | bidirectional_media  # [B, S, S]
        return combined.unsqueeze(1)  # [B, 1, S, S]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4ConditionalOutput:
        # Compute custom attention mask from token_type_ids outside the graph.
        # Pass None during warmup / text-only / decode so the attention backend
        # uses its fast causal kernel instead of the per-sequence fallback.
        token_type_ids = kwargs.pop("token_type_ids", None)
        if token_type_ids is not None and token_type_ids.any():
            kwargs["custom_attn_mask"] = self._build_attention_mask(token_type_ids)
        else:
            kwargs["custom_attn_mask"] = None

        outputs = self._call_language_model(
            self.model.language_model,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
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

        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []
        mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)

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
            patches, positions = self._pad_along_first_dim(patches, positions, max_patches)

            pixel_values.append(patches)
            position_ids.append(positions)

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
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

    def _render_messages(self, messages) -> Tuple[List[str], List[List[Any]]]:
        batched_messages = messages if messages and isinstance(messages[0], list) else [messages]
        rendered_prompts: List[str] = []
        batched_images: List[List[Any]] = []

        for conversation in batched_messages:
            parts: List[str] = []
            conversation_images: List[Any] = []
            for message in conversation:
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                    continue
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(item.get("text", ""))
                    elif item_type == "image":
                        parts.append(self.image_token)
                        conversation_images.append(item.get("image"))
            rendered_prompts.append(" ".join(part for part in parts if part))
            batched_images.append(conversation_images)

        return rendered_prompts, batched_images

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
        del add_generation_prompt
        text, batched_images = self._render_messages(messages)
        if not tokenize:
            if messages and isinstance(messages[0], list):
                return text
            return text[0]

        result = self(
            text=text,
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
    """Input processor that ensures ``token_type_ids`` is always present.

    Every request (text-only or multimodal) gets a ``token_type_ids`` tensor in
    its multimodal data.  For text-only requests this is all-zeros (standard
    causal attention).  For multimodal requests the local Gemma4 processor
    provides it with per-blob IDs.  This guarantees the batched
    ``token_type_ids`` tensor in ``extra_args`` always covers the full
    flattened batch, including mixed image + text-only batches.
    """

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, inputs, sampling_params):
        token_ids, extra = self.base(inputs, sampling_params)
        if extra is None:
            extra = {}
        mm_data = extra.setdefault("multimodal_data", {})
        if "token_type_ids" not in mm_data:
            mm_data["token_type_ids"] = torch.zeros(1, len(token_ids), dtype=torch.int64)
        return token_ids, extra


class Gemma4TextModelExportInfo(TextModelExportInfo):
    """Export config for the Gemma4 text submodule.

    Extends the base ``TextModelExportInfo`` with ``token_type_ids`` as a
    dynamically-shaped input so that it is exported with symbolic batch/seq
    dimensions rather than concrete static shapes.
    """

    def _init_dynamic_shape_lookup(self):
        lookup = super()._init_dynamic_shape_lookup()
        # Reuse the same dynamic dim objects from input_ids so export can verify
        # that batch and seq dimensions are semantically identical across inputs.
        batch_dim = lookup["input_ids"][0]
        seq_dim = lookup["input_ids"][1]
        lookup["token_type_ids"] = {0: batch_dim, 1: seq_dim}
        return lookup


@ModelFactoryRegistry.register("Gemma4ForConditionalGeneration")
class Gemma4ForConditionalGenerationFactory(AutoModelForImageTextToTextFactory):
    """Factory for Gemma 4 VLM with custom attention mask support."""

    def get_export_infos(self, model) -> List[SubModuleExportInfo]:
        """Return export info with token_type_ids as a dynamic input."""
        base_info = TextModelExportInfo.from_autoinferred(model)
        return [Gemma4TextModelExportInfo(base_info.submodule_name)]

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
        return Gemma4ADInputProcessor(base)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("Gemma4TextConfig", Gemma4ForCausalLM)
Gemma4ForConditionalGenerationFactory.register_custom_model_cls(
    "Gemma4Config", Gemma4ForConditionalGeneration
)
