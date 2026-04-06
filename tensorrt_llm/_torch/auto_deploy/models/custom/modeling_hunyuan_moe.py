# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prefill-only HunYuan A13B MoE model implementation for auto_deploy export.

Source:
https://huggingface.co/tencent/Hunyuan-A13B-Instruct

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* Simplified MoE routing: softmax top-k (no capacity buffers / dispatch-combine)

The HunYuan A13B model is an MoE transformer with:
* Grouped-Query Attention (GQA) with QK normalization (RMSNorm on Q/K) applied after RoPE
* Dynamic NTK-Alpha RoPE scaling
* Mixed MLP-MoE: each layer has a shared dense MLP + 64 routed experts with top-8 routing
* SiLU-gated MLP for all experts
* Checkpoint keys: model.layers.N.mlp.gate.wg, mlp.shared_mlp, mlp.experts.N
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType

# ---------------------------------------------------------------------------
# Helper: extract per-layer value from list-or-scalar config field
# ---------------------------------------------------------------------------


def _get_layer_val(val, layer_idx: int, default=None):
    """Return val[layer_idx] if val is a list, else val, else default."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return val[layer_idx]
    return val


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class HunYuanMoERMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary Embedding (Dynamic NTK-Alpha)
# ---------------------------------------------------------------------------


class HunYuanMoERotaryEmbedding(nn.Module):
    """Dynamic NTK-Alpha Rotary Position Embedding.

    When rope_scaling = {"type": "dynamic", "alpha": alpha_value, ...}:
        base = rope_theta * alpha^(dim / (dim - 2))

    Returns full cached cos/sin tables (not sliced). Uses _ad_ prefix for
    buffer names (required by AutoDeploy lift_to_meta).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Apply Dynamic NTK-Alpha scaling if configured
        if (
            rope_scaling is not None
            and rope_scaling.get("type") == "dynamic"
            and rope_scaling.get("alpha")
        ):
            alpha = rope_scaling["alpha"]
            base = base * alpha ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class HunYuanMoEMLP(nn.Module):
    """SiLU-gated MLP used for both shared expert and routed experts."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE Gate
# ---------------------------------------------------------------------------


class HunYuanMoEGate(nn.Module):
    """Softmax top-k MoE gate.

    Checkpoint key: model.layers.N.mlp.gate.wg.weight
    The gate weight is float32 in the checkpoint (matching HF implementation).
    """

    def __init__(self, hidden_size: int, num_experts: int, topk: int):
        super().__init__()
        self.topk = topk
        self.num_experts = num_experts
        # float32 weight to match HF checkpoint dtype
        self.wg = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (topk_indices, topk_weights), shapes [T, K] each."""
        T, D = hidden_states.shape
        # Cast both input and weight to float32 for gate computation
        logits = F.linear(hidden_states.float(), self.wg.weight.float())
        gates = F.softmax(logits, dim=-1)  # [T, num_experts]
        topk_weights, topk_indices = gates.topk(self.topk, dim=-1)  # [T, K]
        # Normalize routing weights across selected experts
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_indices, topk_weights.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------


class HunYuanMoEMoE(nn.Module):
    """Mixed MLP-MoE: shared dense expert + top-k routed experts.

    Checkpoint structure:
        mlp.gate.wg.weight                         — gate linear
        mlp.shared_mlp.{gate,up,down}_proj.weight  — shared expert
        mlp.experts.N.{gate,up,down}_proj.weight   — N routed experts
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        topk: int,
        hidden_act: str,
        use_mixed_mlp_moe: bool,
    ):
        super().__init__()
        self.topk = topk
        self.use_mixed_mlp_moe = use_mixed_mlp_moe

        # MoE gate
        self.gate = HunYuanMoEGate(hidden_size, num_experts, topk)

        # Shared (dense) expert — always present when use_mixed_mlp_moe=True
        if use_mixed_mlp_moe:
            self.shared_mlp = HunYuanMoEMLP(hidden_size, shared_intermediate_size, hidden_act)

        # Routed experts (per-expert ModuleList for checkpoint compatibility)
        self.experts = nn.ModuleList(
            [
                HunYuanMoEMLP(hidden_size, moe_intermediate_size, hidden_act)
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.shape

        # Compute shared expert output
        if self.use_mixed_mlp_moe:
            shared_output = self.shared_mlp(hidden_states)

        # Routing
        hidden_flat = hidden_states.view(-1, hidden_size)
        topk_indices, topk_weights = self.gate(hidden_flat)  # [T, K] each

        # Routed experts via torch_moe custom op
        routed_output = torch.ops.auto_deploy.torch_moe(
            hidden_flat,
            topk_indices,
            topk_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )
        routed_output = routed_output.view(bsz, seq_len, hidden_size)

        if self.use_mixed_mlp_moe:
            return shared_output + routed_output
        return routed_output


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class HunYuanMoEAttention(nn.Module):
    """GQA attention with QK normalization for HunYuan A13B.

    Applies RMSNorm to Q and K *after* RoPE (matching HF implementation order).
    Uses torch_attention and torch_rope_with_explicit_cos_sin AD ops.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # QK normalization applied per-head after RoPE
        if config.use_qk_norm:
            self.query_layernorm = HunYuanMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunYuanMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.use_qk_norm = config.use_qk_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V → [B, S, N, D] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Slice cos/sin by position_ids
        cos = position_embeddings[0]  # [max_seq_len, head_dim]
        sin = position_embeddings[1]  # [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE in BSND layout (unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # QK normalization after RoPE (matches HF order)
        if self.use_qk_norm:
            q = torch.ops.auto_deploy.torch_rmsnorm(
                q, self.query_layernorm.weight, self.query_layernorm.variance_epsilon
            )
            k = torch.ops.auto_deploy.torch_rmsnorm(
                k, self.key_layernorm.weight, self.key_layernorm.variance_epsilon
            )

        # Attention in BSND layout
        attn_output = torch.ops.auto_deploy.torch_attention(
            q, k, v, is_causal=True, scale=self.scaling, layout="bsnd"
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class HunYuanMoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HunYuanMoEAttention(config, layer_idx=layer_idx)

        # Per-layer MoE parameters
        moe_intermediate_size = _get_layer_val(
            config.moe_intermediate_size, layer_idx, default=config.intermediate_size
        )
        num_shared_expert = _get_layer_val(config.num_shared_expert, layer_idx, default=1)
        topk = _get_layer_val(config.moe_topk, layer_idx, default=1)
        shared_intermediate_size = config.intermediate_size * num_shared_expert

        self.mlp = HunYuanMoEMoE(
            hidden_size=config.hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            num_experts=config.num_experts,
            topk=topk,
            hidden_act=config.hidden_act,
            use_mixed_mlp_moe=config.use_mixed_mlp_moe,
        )
        self.input_layernorm = HunYuanMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanMoERMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------


@dataclass
class HunYuanMoEOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class HunYuanMoECausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class HunYuanMoEPreTrainedModel(PreTrainedModel):
    config_class = None  # set dynamically below when used with real HunYuanConfig
    base_model_prefix = "model"
    _no_split_modules = ["HunYuanMoEDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HunYuanMoEModel(HunYuanMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                HunYuanMoEDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = HunYuanMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = HunYuanMoERotaryEmbedding(
            dim=config.attention_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> HunYuanMoEOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return HunYuanMoEOutput(last_hidden_state=hidden_states)


class HunYuanMoEForCausalLM(HunYuanMoEPreTrainedModel, GenerationMixin):
    """HunYuan A13B MoE model with language modeling head.

    Registered as the custom AD model for checkpoints with config class 'HunYuanConfig'
    (the dynamically loaded custom config from tencent/Hunyuan-A13B-Instruct).
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = HunYuanMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> HunYuanMoECausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()
        return HunYuanMoECausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Register with AutoModelForCausalLMFactory
# ---------------------------------------------------------------------------

# Register with the AD factory under the REAL checkpoint config class name
# (HunYuanConfig — dynamically loaded via trust_remote_code from the HF repo).
AutoModelForCausalLMFactory.register_custom_model_cls("HunYuanConfig", HunYuanMoEForCausalLM)
