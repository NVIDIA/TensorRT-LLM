# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Slimmed down PyTorch GLM MoE DSA model implementation for auto_deploy export.

Source:
https://huggingface.co/zai-org/GLM-5

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class (model_type "glm_moe_dsa" not yet in transformers)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_mla custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* Skipped Dynamic Sparse Attention (DSA) indexer — not needed for prefill
* Skipped Multi-Token Prediction (MTP) layers — not needed for prefill

The GLM MoE DSA model uses Multi-head Latent Attention (MLA), similar to DeepSeek V3,
combined with a Mixture of Experts (MoE) architecture with noaux_tc routing.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.custom import mla_rope_utils
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class GlmMoeDsaConfig(PretrainedConfig):
    """Configuration class for GLM MoE DSA model.

    This config class is bundled because model_type "glm_moe_dsa" is not yet
    registered in the installed transformers version.
    """

    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        rms_norm_eps: float = 1e-5,
        # MLA parameters
        q_lora_rank: int = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        # MoE parameters
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 2048,
        moe_layer_freq: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 3,
        # RoPE parameters
        rope_theta: float = 1000000.0,
        rope_parameters: Optional[dict] = None,
        rope_interleave: bool = True,
        # Other parameters
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 154820,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        # RoPE — extract rope_theta from nested rope_parameters if present
        self.rope_parameters = rope_parameters
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            self.rope_theta = rope_parameters["rope_theta"]
        else:
            self.rope_theta = rope_theta
        self.rope_interleave = rope_interleave
        # Other
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config with transformers' AutoConfig
try:
    AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig)
    except ValueError:
        pass

# Register GlmMoeDsaConfig in TOKENIZER_MAPPING so AutoTokenizer.from_pretrained
# can look up the tokenizer class. All GLM variants use PreTrainedTokenizerFast.
TOKENIZER_MAPPING.register(GlmMoeDsaConfig, (None, PreTrainedTokenizerFast), exist_ok=True)


# GLM-5-FP8 specifies tokenizer_class="TokenizersBackend" in its tokenizer_config.json.
# AutoTokenizer.from_pretrained resolves the class name via tokenizer_class_from_name(),
# which scans TOKENIZER_MAPPING._extra_content for a tokenizer whose __name__ matches.
# We define a thin alias so that name lookup succeeds and falls back to PreTrainedTokenizerFast.
class TokenizersBackend(PreTrainedTokenizerFast):
    """Alias for PreTrainedTokenizerFast to satisfy the TokenizersBackend tokenizer_class name
    declared in zai-org/GLM-5-FP8's tokenizer_config.json."""

    pass


TOKENIZER_MAPPING.register(GlmMoeDsaConfig, (None, TokenizersBackend), exist_ok=True)


class GlmMoeDsaRMSNorm(nn.Module):
    """RMS Normalization using the canonical AutoDeploy torch_rmsnorm op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class GlmMoeDsaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Precomputes and caches cos/sin values. Returns full cached values for export.
    Uses _ad_ prefix for buffer names for AutoDeploy lift_to_meta compatibility.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.attention_scaling = attention_scaling

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * self.attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class GlmMoeDsaMLP(nn.Module):
    """MLP layer (SwiGLU activation)."""

    def __init__(
        self, config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GlmMoeDsaMoEGate(nn.Module):
    """MoE Gating with noaux_tc top-k selection.

    Uses fused TensorRT-LLM custom ops for efficient routing:
    - dsv3_router_gemm_op: Fused router GEMM for non-float32 weights
    - noaux_tc_op: Fused sigmoid + bias + group top-k + normalize + scale
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)

        if not self.norm_topk_prob:
            raise ValueError(
                "GlmMoeDsaMoEGate requires norm_topk_prob=True when using fused ops. "
                "The noaux_tc_op kernel always normalizes routing weights."
            )

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router GEMM - use fused op when weights are not float32
        if self.weight.dtype == torch.float32:
            router_logits = F.linear(hidden_states_flat.float(), self.weight)
        else:
            router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_states_flat, self.weight.t(), bias=None, out_dtype=torch.float32
            )

        # Fused routing: sigmoid + bias + group top-k + normalize + scale
        topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        return topk_indices, topk_weights


class GlmMoeDsaMoE(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                GlmMoeDsaMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        self.gate = GlmMoeDsaMoEGate(config)

        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = GlmMoeDsaMLP(config, intermediate_size=intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        selected_experts, routing_weights = self.gate(hidden_states)

        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            selected_experts,
            routing_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )

        final_hidden_states = final_hidden_states.view(*orig_shape)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        return final_hidden_states


class GlmMoeDsaAttention(nn.Module):
    """Multi-head Latent Attention (MLA).

    Uses compressed KV representation with latent projections.
    The DSA indexer is skipped — not needed for prefill-only AD export.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Q projection (with LoRA)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        # KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Softmax scale
        self.softmax_scale = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Q projection
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        # Shape: [B, S, N, qk_head_dim] (BSND layout)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection - keep compressed form
        kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # k_pe: [B, S, 1, qk_rope_head_dim] (BSND layout, shared across heads)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        # Get cos/sin from position_embeddings (full cached from shared rotary embedding)
        cos = position_embeddings[0]  # [max_seq_len, head_dim]
        sin = position_embeddings[1]  # [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE using custom op (weights pre-permuted to NeoX format at load time)
        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_pe,
            k_pe,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Call MLA with compressed KV
        attn_output = torch.ops.auto_deploy.torch_mla(
            q_nope,  # [B, S, N, qk_nope_head_dim]
            q_pe_rotated,  # [B, S, N, qk_rope_head_dim]
            compressed_kv,  # [B, S, kv_lora_rank]
            kpe,  # [B, S, 1, qk_rope_head_dim]
            self.kv_b_proj.weight,  # [N*(qk_nope+v), kv_lora_rank]
            True,  # is_causal
            self.softmax_scale,
            "bsnd",  # layout
        )

        # Output: [B, S, N, v_head_dim] -> [B, S, N * v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class GlmMoeDsaDecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = GlmMoeDsaAttention(config, layer_idx=layer_idx)

        # Dense layers for first first_k_dense_replace layers, MoE for the rest
        # (subject to moe_layer_freq — all layers after dense use MoE when freq=1)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = GlmMoeDsaMoE(config)
        else:
            self.mlp = GlmMoeDsaMLP(config)

        self.input_layernorm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmMoeDsaRMSNorm(
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


@dataclass
class GlmMoeDsaOutput(ModelOutput):
    """Output for GlmMoeDsaModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GlmMoeDsaCausalLMOutput(ModelOutput):
    """Output for GlmMoeDsaForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class GlmMoeDsaPreTrainedModel(PreTrainedModel):
    """Base class for GLM MoE DSA models."""

    config_class = GlmMoeDsaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
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


class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    """GLM MoE DSA transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = GlmMoeDsaRotaryEmbedding(
            config.qk_rope_head_dim,
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
    ) -> GlmMoeDsaOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert position_ids is not None, "position_ids is required"

        # Compute position embeddings once from shared rotary embedding
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return GlmMoeDsaOutput(last_hidden_state=hidden_states)


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel, GenerationMixin):
    """GLM MoE DSA model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Pre-permute RoPE weight rows from interleaved to NeoX format at load time
        self._register_load_state_dict_pre_hook(
            partial(
                mla_rope_utils._rope_deinterleave_load_hook,
                qk_rope_head_dim=config.qk_rope_head_dim,
                qk_nope_head_dim=config.qk_nope_head_dim,
                num_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                num_layers=config.num_hidden_layers,
            )
        )

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
    ) -> GlmMoeDsaCausalLMOutput:
        assert position_ids is not None, "position_ids is required"

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return GlmMoeDsaCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("GlmMoeDsaConfig", GlmMoeDsaForCausalLM)
