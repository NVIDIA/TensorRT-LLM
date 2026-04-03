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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
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
        self.model = Gemma4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Gemma4ConditionalOutput:
        # Ensure token_type_ids is always present for the custom attention mask.
        # Text-only / decode / warmup steps won't have it in named_args.
        if "token_type_ids" not in kwargs:
            ref = input_ids if input_ids is not None else inputs_embeds
            if ref is not None:
                kwargs["token_type_ids"] = torch.zeros(
                    ref.shape[0], ref.shape[1], dtype=torch.int64, device=ref.device
                )

        outputs = self.model.language_model(
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
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_FILE = "tokenizer.json"


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

        template_path = cached_file(
            pretrained_model_name_or_path,
            _CHAT_TEMPLATE_FILE,
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        if template_path is not None:
            tokenizer.chat_template = Path(template_path).read_text()

        return tokenizer


class Gemma4ADInputProcessor:
    """Input processor that ensures ``token_type_ids`` is always present.

    Every request (text-only or multimodal) gets a ``token_type_ids`` tensor in
    its multimodal data.  For text-only requests this is all-zeros (standard
    causal attention).  For multimodal requests the HF processor already
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
        """Return the tokenizer as the processor for ADInputProcessor."""
        if self.tokenizer is None:
            return None
        return ADGemma4Tokenizer.from_pretrained(self.tokenizer)

    def init_input_processor(self, base):
        return Gemma4ADInputProcessor(base)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("Gemma4TextConfig", Gemma4ForCausalLM)
Gemma4ForConditionalGenerationFactory.register_custom_model_cls(
    "Gemma4Config", Gemma4ForConditionalGeneration
)
