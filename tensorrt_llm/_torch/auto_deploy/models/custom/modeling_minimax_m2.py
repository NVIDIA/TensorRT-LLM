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

"""Slimmed down PyTorch MiniMax-M2 model implementation for auto_deploy export.

Source:
https://huggingface.co/MiniMaxAI/MiniMax-M2

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class (model requires trust_remote_code; not in transformers yet)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The MiniMax-M2 model uses:
- GQA with per-layer QK normalization and partial RoPE (rotary_dim < head_dim)
- MoE with sigmoid routing and e_score_correction_bias (noaux_tc style)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class MiniMaxM2Config(PretrainedConfig):
    """Configuration class for MiniMax-M2 model.

    Bundled with the custom model implementation because transformers does not
    natively register this model type (it uses trust_remote_code).
    """

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096 * 32,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1e6,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts

        # MiniMax-M2 specific fields (from config.json kwargs)
        self.use_qk_norm = kwargs.pop("use_qk_norm", False)
        self.rotary_dim = kwargs.pop("rotary_dim", self.head_dim)
        # Consume unused config.json fields to prevent them leaking into super().__init__
        kwargs.pop("partial_rotary_factor", None)
        kwargs.pop("sliding_window", None)
        kwargs.pop("attention_dropout", None)
        kwargs.pop("output_router_logits", None)
        kwargs.pop("router_aux_loss_coef", None)
        kwargs.pop("router_jitter_noise", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config so AutoConfig.from_pretrained can find it
try:
    AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("minimax_m2", MiniMaxM2Config)
    except ValueError:
        pass


class MiniMaxM2RMSNorm(nn.Module):
    """RMS Normalization using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class MiniMaxM2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for MiniMax-M2 with partial RoPE support.

    Precomputes and caches cos/sin values. The dim parameter is the rotary_dim
    (which may be less than head_dim for partial RoPE).

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
        # Return full cached cos/sin (not sliced) for export compatibility
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class MiniMaxM2MLP(nn.Module):
    """Expert MLP (SwiGLU activation) for MiniMax-M2.

    Uses w1/w2/w3 naming to match checkpoint keys:
    - w1: gate projection
    - w2: down projection
    - w3: up projection
    """

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


class MiniMaxM2SparseMoeBlock(nn.Module):
    """MoE block with sigmoid routing and e_score_correction_bias.

    Routing follows the noaux_tc pattern:
    1. Sigmoid activation on router logits
    2. Add bias for expert selection (biased scores)
    3. Top-k on biased scores
    4. Gather original (unbiased) sigmoid weights
    5. Normalize weights to sum to 1
    """

    def __init__(self, config: MiniMaxM2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MiniMaxM2MLP(config) for _ in range(self.num_experts)])
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router
        router_logits = self.gate(hidden_states_flat)

        # Sigmoid routing with bias correction (noaux_tc pattern)
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)

        # Gather original (unbiased) sigmoid weights and normalize
        top_k_weights = routing_weights.gather(1, selected_experts)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states_flat.dtype)

        # MoE computation using canonical op
        output = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            top_k_weights,
            w1_weight=[expert.w1.weight for expert in self.experts],
            w2_weight=[expert.w2.weight for expert in self.experts],
            w3_weight=[expert.w3.weight for expert in self.experts],
        )

        return output.view(batch_size, sequence_length, hidden_dim)


class MiniMaxM2Attention(nn.Module):
    """GQA with per-layer QK normalization and partial RoPE.

    MiniMax-M2 applies QK norm on the full projected Q/K vectors (per-layer,
    not per-head), then reshapes to heads, then applies partial RoPE.
    """

    def __init__(self, config: MiniMaxM2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        ) or (config.hidden_size // config.num_attention_heads)
        self.rotary_dim = getattr(config, "rotary_dim", self.head_dim) or self.head_dim
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Per-layer QK normalization
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = MiniMaxM2RMSNorm(
                self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Per-layer QK norm (before reshaping to heads)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Reshape to [B, S, N, head_dim] (BSND layout)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Get cos/sin from position_embeddings (full cached tables)
        cos_full, sin_full = position_embeddings  # [max_seq_len, rotary_dim]
        cos = cos_full[position_ids]  # [B, S, rotary_dim]
        sin = sin_full[position_ids]  # [B, S, rotary_dim]

        # Handle partial RoPE: split, apply, concatenate
        if self.rotary_dim < self.head_dim:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

            q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
                q_rot,
                k_rot,
                cos,
                sin,
                2,  # unsqueeze_dim=2 for BSND layout
            )

            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # Attention using canonical op with GQA support (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
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


class MiniMaxM2DecoderLayer(nn.Module):
    """Transformer decoder layer for MiniMax-M2."""

    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MiniMaxM2Attention(config, layer_idx=layer_idx)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class MiniMaxM2Output(ModelOutput):
    """Output for MiniMaxM2Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class MiniMaxM2CausalLMOutput(ModelOutput):
    """Output for MiniMaxM2ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class MiniMaxM2PreTrainedModel(PreTrainedModel):
    """Base class for MiniMax-M2 models."""

    config_class = MiniMaxM2Config
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
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


class MiniMaxM2Model(MiniMaxM2PreTrainedModel):
    """MiniMax-M2 transformer decoder model."""

    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                MiniMaxM2DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding using rotary_dim (partial RoPE)
        rotary_dim = getattr(config, "rotary_dim", None)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        if rotary_dim is None:
            rotary_dim = head_dim
        self.rotary_emb = MiniMaxM2RotaryEmbedding(
            rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
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
    ) -> MiniMaxM2Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype for FP8 models
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute position embeddings once (full cached tables)
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return MiniMaxM2Output(last_hidden_state=hidden_states)


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel, GenerationMixin):
    """MiniMax-M2 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = MiniMaxM2Model(config)
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
    ) -> MiniMaxM2CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return MiniMaxM2CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("MiniMaxM2Config", MiniMaxM2ForCausalLM)
