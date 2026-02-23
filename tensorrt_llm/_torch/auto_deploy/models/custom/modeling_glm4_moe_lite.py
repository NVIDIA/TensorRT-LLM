# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Slimmed down PyTorch GLM4 MoE Lite model implementation for auto_deploy export.

Source:
https://huggingface.co/zai-org/GLM-4.7-Flash

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class to work with transformers v4.57 (model requires v5.0)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_mla custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

The GLM4 MoE Lite model uses Multi-head Latent Attention (MLA), similar to DeepSeek V3.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class Glm4MoeLiteConfig(PretrainedConfig):
    """Configuration class for GLM4 MoE Lite model.

    This config class is bundled with the custom model implementation to enable
    loading on transformers v4.57 (the model requires v5.0 where the config is
    natively registered).
    """

    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        rms_norm_eps: float = 1e-5,
        # MLA parameters
        q_lora_rank: int = 768,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        # MoE parameters
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        moe_intermediate_size: int = 1536,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.8,
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 1,
        # RoPE parameters
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
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
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        # RoPE
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Other
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config with transformers' AutoConfig so it can be loaded from HF hub
# Use exist_ok=True to handle cases where transformers already has this model type registered
# (e.g., transformers v5.0+). In those cases, AutoConfig will use the built-in config,
# but AutoModelForCausalLMFactory will still use our custom model implementation.
try:
    AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig, exist_ok=True)
except TypeError:
    # Older transformers versions don't support exist_ok parameter
    try:
        AutoConfig.register("glm4_moe_lite", Glm4MoeLiteConfig)
    except ValueError:
        # Already registered by transformers, that's fine
        pass


class Glm4MoeLiteRMSNorm(nn.Module):
    """RMS Normalization for GLM4 MoE Lite.

    Uses standard torch operations so AD fusion passes can replace with
    the appropriate backend (flashinfer/triton) based on config.
    """

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


class Glm4MoeLiteRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for GLM4 MoE Lite.

    Simplified version that precomputes and caches cos/sin values.
    Returns full cached values (not sliced by seq_len) to enable export.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
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

        # Build cos/sin cache with AD-specific naming
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Use _ad_ prefix for AutoDeploy compatibility with lift_to_meta
        self.register_buffer("_ad_cos_cached", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * self.attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full cached cos/sin (not sliced) for export compatibility
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Glm4MoeLiteYarnRotaryEmbedding(Glm4MoeLiteRotaryEmbedding):
    """YaRN-extended rotary embedding for GLM4 MoE Lite."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        attention_scaling: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, attention_scaling)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        # Use _ad_ prefix for AutoDeploy compatibility with lift_to_meta
        # Note: attention_scaling is already incorporated in _mscale for YaRN
        self.register_buffer("_ad_cos_cached", (emb.cos() * _mscale), persistent=False)
        self.register_buffer("_ad_sin_cached", (emb.sin() * _mscale), persistent=False)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)


class Glm4MoeLiteMLP(nn.Module):
    """MLP layer for GLM4 MoE Lite (SwiGLU activation)."""

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


class Glm4MoeLiteMoEGate(nn.Module):
    """MoE Gating for GLM4 MoE Lite with top-k selection.

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

        # noaux_tc_op always normalizes, so norm_topk_prob must be True
        if not self.norm_topk_prob:
            raise ValueError(
                "Glm4MoeLiteMoEGate requires norm_topk_prob=True when using fused ops. "
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
        """Initialize gate weights using kaiming uniform."""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (selected_experts, routing_weights).

        Uses fused TensorRT-LLM ops for efficient routing:
        1. dsv3_router_gemm_op: Router GEMM (when weights are not float32)
        2. noaux_tc_op: Fused sigmoid + bias + group top-k + normalize + scale
        """
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
        # noaux_tc_op internally applies:
        # 1. Sigmoid to router_logits
        # 2. Adds e_score_correction_bias
        # 3. Group-wise top-2 scoring and top group selection
        # 4. Top-k expert selection from selected groups
        # 5. Gathers weights from sigmoid scores
        # 6. Normalizes and scales by routed_scaling_factor
        topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        return topk_indices, topk_weights


class Glm4MoeLiteMoE(nn.Module):
    """Mixture of Experts layer for GLM4 MoE Lite."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Routed experts - use ModuleList with individual expert modules
        # This creates state_dict keys like: experts.0.gate_proj.weight
        # which matches the checkpoint structure
        self.experts = nn.ModuleList(
            [
                Glm4MoeLiteMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        # Gate
        self.gate = Glm4MoeLiteMoEGate(config)

        # Shared experts
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeLiteMLP(config, intermediate_size=intermediate_size)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        selected_experts, routing_weights = self.gate(hidden_states)

        # Use torch_moe custom op for routed experts
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

        # Add shared experts output if present
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)

        return final_hidden_states


class Glm4MoeLiteAttention(nn.Module):
    """Multi-head Latent Attention (MLA) for GLM4 MoE Lite.

    Uses compressed KV representation with latent projections.
    Receives position embeddings from the model level (shared rotary embedding).
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

        # Q projection (with optional LoRA)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = Glm4MoeLiteRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        # KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = Glm4MoeLiteRMSNorm(self.kv_lora_rank)
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
        # Apply mscale adjustment if using YaRN scaling with factor
        if (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        ):
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = Glm4MoeLiteYarnRotaryEmbedding._yarn_get_mscale(
                    scaling_factor, mscale_all_dim
                )
                self.softmax_scale = self.softmax_scale * mscale * mscale

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

        # Apply layernorm to compressed_kv
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        # k_pe: [B, S, 1, qk_rope_head_dim] (BSND layout, shared across heads)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        # Get cos/sin from position_embeddings (full cached from shared rotary embedding)
        cos, sin = position_embeddings  # Full table: [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE using custom op
        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
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


class Glm4MoeLiteDecoderLayer(nn.Module):
    """Transformer decoder layer for GLM4 MoE Lite."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Glm4MoeLiteAttention(config, layer_idx=layer_idx)

        # MLP or MoE
        # Layer 0 to first_k_dense_replace-1 use dense MLP, rest use MoE
        use_moe = config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
        if use_moe:
            self.mlp = Glm4MoeLiteMoE(config)
        else:
            self.mlp = Glm4MoeLiteMLP(config)

        # Layer norms
        self.input_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLiteRMSNorm(
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

        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Glm4MoeLiteOutput(ModelOutput):
    """Output for Glm4MoeLiteModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Glm4MoeLiteCausalLMOutput(ModelOutput):
    """Output for Glm4MoeLiteForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class Glm4MoeLitePreTrainedModel(PreTrainedModel):
    """Base class for GLM4 MoE Lite models."""

    config_class = Glm4MoeLiteConfig
    base_model_prefix = "model"
    _no_split_modules = ["Glm4MoeLiteDecoderLayer"]
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


class Glm4MoeLiteModel(Glm4MoeLitePreTrainedModel):
    """GLM4 MoE Lite transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Glm4MoeLiteDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level (not per-layer)
        # This creates a single set of cos/sin buffers for all layers
        self.rotary_emb = self._init_rope(config)

        self.post_init()

    def _init_rope(self, config):
        """Initialize shared rotary embedding for all layers."""
        qk_rope_head_dim = config.qk_rope_head_dim

        # Compute attention_scaling for RoPE (same logic as in attention)
        attention_scaling = 1.0
        if (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        ):
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = Glm4MoeLiteYarnRotaryEmbedding._yarn_get_mscale(
                    scaling_factor, mscale_all_dim
                )
                attention_scaling = mscale

        # Check if rope_scaling is None, empty, or missing required "factor" key
        use_yarn = (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        )

        if not use_yarn:
            return Glm4MoeLiteRotaryEmbedding(
                qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                attention_scaling=attention_scaling,
            )
        else:
            scaling_factor = config.rope_scaling["factor"]
            kwargs = {
                key: config.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in config.rope_scaling
            }
            return Glm4MoeLiteYarnRotaryEmbedding(
                qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=config.rope_theta,
                attention_scaling=attention_scaling,
                **kwargs,
            )

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
    ) -> Glm4MoeLiteOutput:
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

        # Compute position embeddings once from shared rotary embedding
        # This returns full cached cos/sin tables
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return Glm4MoeLiteOutput(last_hidden_state=hidden_states)


class Glm4MoeLiteForCausalLM(Glm4MoeLitePreTrainedModel, GenerationMixin):
    """GLM4 MoE Lite model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = Glm4MoeLiteModel(config)
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
    ) -> Glm4MoeLiteCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return Glm4MoeLiteCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("Glm4MoeLiteConfig", Glm4MoeLiteForCausalLM)
