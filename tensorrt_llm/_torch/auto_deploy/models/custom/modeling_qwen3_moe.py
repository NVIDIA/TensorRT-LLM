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

"""Slimmed down PyTorch Qwen3 MoE model implementation for auto_deploy export.

Source:
https://huggingface.co/Qwen/Qwen3-30B-A3B

This implementation differs from the original HuggingFace version in the following ways:
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* Removed sliding window attention (not needed for prefill-only)
* MoE routing uses torch_moe canonical op instead of data-dependent sparse dispatch

The Qwen3 MoE model uses Grouped Query Attention (GQA) with per-head Q/K normalization
(RMSNorm on head_dim), same as the dense Qwen3 model. The MLP layers are replaced with
Sparse MoE blocks using softmax top-k routing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.utils import ModelOutput

from ..hf import AutoModelForCausalLMFactory


class Qwen3MoeRMSNorm(nn.Module):
    """RMS Normalization for Qwen3 MoE using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Qwen3MoeRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Qwen3 MoE.

    Simplified version that precomputes and caches cos/sin values.
    Returns pre-sliced values indexed by position_ids.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache with AD-specific naming
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Use _ad_ prefix for AutoDeploy compatibility with lift_to_meta
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Slice cos/sin by position_ids here (once) instead of in every attention layer
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


class Qwen3MoeMLP(nn.Module):
    """MLP layer for Qwen3 MoE (SwiGLU activation)."""

    def __init__(self, config: Qwen3MoeConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block for Qwen3 MoE.

    Uses softmax top-k routing with optional probability normalization.
    Expert computation is handled by the torch_moe canonical op.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self._register_load_state_dict_pre_hook(self._load_experts_from_fused_checkpoint)

    def _owned_expert_ids(self) -> list[int]:
        """Derive expert ids from params/buffers that currently exist on this module.

        Stays shard-agnostic: on an unsharded module the state dict contains all
        experts; after AD sharding it contains only the surviving local experts.
        """
        expert_ids = set()
        for key in self.state_dict().keys():
            if not key.startswith("experts."):
                continue
            parts = key.split(".", 3)
            if len(parts) < 4:
                continue
            try:
                expert_ids.add(int(parts[1]))
            except ValueError:
                continue
        return sorted(expert_ids)

    def _load_experts_from_fused_checkpoint(self, state_dict, prefix, *args):
        """Convert HF Qwen3MoeExperts fused tensors into per-expert split form.

        HF stores fused 3D tensors:
            experts.gate_up_proj  -> [num_experts, 2 * intermediate, hidden]
            experts.down_proj     -> [num_experts, hidden, intermediate]
        AutoDeploy's Qwen3MoeMLP expects per-expert ``Linear`` weights:
            experts.{i}.gate_proj.weight -> [intermediate, hidden]
            experts.{i}.up_proj.weight   -> [intermediate, hidden]
            experts.{i}.down_proj.weight -> [hidden, intermediate]
        """
        gate_up_key = prefix + "experts.gate_up_proj"
        down_key = prefix + "experts.down_proj"
        local_expert_ids = self._owned_expert_ids()

        if gate_up_key in state_dict:
            fused = state_dict.pop(gate_up_key)
            intermediate_dim = fused.shape[1] // 2
            gate_weights = fused[:, :intermediate_dim, :]
            up_weights = fused[:, intermediate_dim:, :]
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.gate_proj.weight"] = gate_weights[idx]
                state_dict[f"{prefix}experts.{idx}.up_proj.weight"] = up_weights[idx]

        if down_key in state_dict:
            fused = state_dict.pop(down_key)
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.down_proj.weight"] = fused[idx]

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router logits: (batch * sequence_length, num_experts)
        router_logits = self.gate(hidden_states_flat)

        # Softmax routing with top-k selection
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # Use torch_moe canonical op for expert computation
        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen3MoeAttention(nn.Module):
    """Grouped Query Attention for Qwen3 MoE with per-head Q/K normalization.

    Qwen3 MoE applies RMSNorm to query and key states after projection and reshaping,
    but before RoPE application. This per-head normalization on head_dim is a key
    architectural feature shared with the dense Qwen3 model.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim ** (-0.5)

        # Q/K/V/O projections
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

        # Per-head Q/K normalization (Qwen3-specific)
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to [B, S, N, head_dim] (BSND layout)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Apply per-head Q/K normalization (Qwen3-specific, on head_dim dimension)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Get pre-sliced cos/sin from position_embeddings (already indexed by position_ids)
        cos, sin = position_embeddings  # [B, S, head_dim]

        # Apply RoPE using custom op (BSND layout, unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Attention using custom op with GQA support (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,  # [B, S, N, head_dim]
            k,  # [B, S, N_kv, head_dim]
            v,  # [B, S, N_kv, head_dim]
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape [B, S, N, head_dim] -> [B, S, N * head_dim] and project
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3MoeDecoderLayer(nn.Module):
    """Transformer decoder layer for Qwen3 MoE.

    Supports both dense MLP and Sparse MoE layers based on layer index
    and config settings (decoder_sparse_step, mlp_only_layers).
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx=layer_idx)

        # Determine MLP type: MoE or dense based on layer index
        is_moe_layer = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # Unpack tuple from MoE layer (hidden_states, router_logits)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Qwen3MoeModelOutput(ModelOutput):
    """Output for Qwen3MoeModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3MoeCausalLMOutput(ModelOutput):
    """Output for Qwen3MoeForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class Qwen3MoePreTrainedModel(PreTrainedModel):
    """Base class for Qwen3 MoE models."""

    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    supports_gradient_checkpointing = False

    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        """No-op override.

        ``transformers >= 5.x``'s ``PreTrainedModel.__init__`` calls this method
        which dispatches to ``_grouped_mm_can_dispatch`` and raises
        ``ValueError`` for any class that hasn't opted into the new MoE
        ``experts_implementation`` contract. AutoDeploy uses its own
        ``torch_moe`` canonical op for routing, so no dispatch decision is
        needed here.
        """
        return None

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


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    """Qwen3 MoE transformer decoder model."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_parameters["rope_theta"],
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
    ) -> Qwen3MoeModelOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype (e.g., bfloat16) for FP8 models where embedding
        # output may be FP8 but downstream ops (RMSNorm, attention) require FP16/BF16
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute position embeddings once (sliced by position_ids in RoPE)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return Qwen3MoeModelOutput(last_hidden_state=hidden_states)


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    """Qwen3 MoE model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
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
    ) -> Qwen3MoeCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return Qwen3MoeCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3MoeConfig", Qwen3MoeForCausalLM)
