# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Slimmed down PyTorch Kimi-K2.5 model implementation for auto_deploy export.

Source:
https://huggingface.co/moonshotai/Kimi-K2.5

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config classes (KimiK2Config, KimiK25Config) for transformers compatibility
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility (torch_mla, torch_moe,
  torch_rope_with_qk_interleaving)
* Vanilla PyTorch MoE routing (sigmoid + bias + group top-k); AD transforms
  can replace with fused kernels at deployment time
* Removed flash attention variants
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
* Vision tower kept in eager mode; only text path is exported

The Kimi-K2.5 text model is a DeepSeek-V3-style architecture with:
* Multi-head Latent Attention (MLA)
* Mixture of Experts (MoE) with noaux_tc routing (sigmoid + bias + top-k)
* YaRN rotary position embeddings
* SwiGLU activation in FFN layers
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType

# =============================================================================
# Configuration
# =============================================================================


class KimiK2Config(PretrainedConfig):
    """Configuration class for Kimi-K2 text model (DeepSeek-V3 variant).

    This config class is bundled with the custom model implementation to enable
    loading on transformers versions that don't natively have Kimi-K2 registered.
    """

    model_type = "kimi_k2"

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-5,
        # MLA parameters
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # MoE parameters
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 2.827,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        # RoPE parameters
        rope_theta: float = 50000.0,
        rope_scaling: Optional[dict] = None,
        # Other parameters
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 163584,
        eos_token_id: int = 163585,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
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
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        # RoPE
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # Other
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class KimiK25Config(PretrainedConfig):
    """Configuration class for Kimi-K2.5 (vision-language model wrapper).

    The text model config is stored as text_config and uses DeepSeek-V3 architecture.
    Vision config is stored but not used for AD export (vision stays in eager mode).
    """

    model_type = "kimi_k25"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 163839,
        use_unified_vision_chunk: bool = True,
        video_placeholder: str = "<|kimi_k25_video_placeholder|>",
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = KimiK2Config(**text_config)
        if text_config is None:
            text_config = KimiK2Config()
        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder

        super().__init__(pad_token_id=pad_token_id, **kwargs)


# Register configs with transformers' AutoConfig
try:
    AutoConfig.register("kimi_k2", KimiK2Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("kimi_k2", KimiK2Config)
    except ValueError:
        pass

try:
    AutoConfig.register("kimi_k25", KimiK25Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("kimi_k25", KimiK25Config)
    except ValueError:
        pass


# =============================================================================
# Model Components
# =============================================================================


class KimiK2RMSNorm(nn.Module):
    """RMS Normalization for Kimi-K2."""

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


class KimiK2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Kimi-K2.

    Returns full cached cos/sin (not sliced by seq_len) to enable export.
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


class KimiK2YarnRotaryEmbedding(KimiK2RotaryEmbedding):
    """YaRN-extended rotary embedding for Kimi-K2."""

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
        self.register_buffer("_ad_cos_cached", (emb.cos() * _mscale), persistent=False)
        self.register_buffer("_ad_sin_cached", (emb.sin() * _mscale), persistent=False)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048,
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self,
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float,
        max_position_embeddings: int,
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


# =============================================================================
# MLP / MoE
# =============================================================================


class KimiK2MLP(nn.Module):
    """MLP layer (SwiGLU activation)."""

    def __init__(
        self,
        config,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
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


class KimiK2MoEGate(nn.Module):
    """MoE Gating with noaux_tc routing (sigmoid + bias + group top-k).

    Vanilla PyTorch implementation of DeepSeek-V3-style routing:
    sigmoid scoring → bias-adjusted group selection → top-k → normalize → scale.
    AutoDeploy transforms can replace this with fused kernels at deployment time.
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

    @torch.no_grad()
    def _get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Select top-k expert indices using group-based routing with bias."""
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.to(device=scores.device).unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router GEMM in float32
        router_logits = F.linear(
            hidden_states_flat.to(torch.float32), self.weight.to(torch.float32)
        )

        # Sigmoid scoring
        scores = router_logits.sigmoid()

        # Group-based top-k selection (uses bias-adjusted scores for selection)
        topk_indices = self._get_topk_indices(scores)

        # Gather original scores (not bias-adjusted) for the selected experts
        topk_weights = scores.gather(1, topk_indices)

        # Normalize
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        # Scale
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


class KimiK2MoE(nn.Module):
    """Mixture of Experts layer for Kimi-K2."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                KimiK2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        self.gate = KimiK2MoEGate(config)

        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = KimiK2MLP(config, intermediate_size=intermediate_size)
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


# =============================================================================
# Attention
# =============================================================================


class KimiK2Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for Kimi-K2.

    Uses compressed KV representation with latent projections, identical
    to the DeepSeek-V3 MLA mechanism.
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
            self.q_a_layernorm = KimiK2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        # KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = KimiK2RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Softmax scale with mscale adjustment
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        if (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        ):
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = KimiK2YarnRotaryEmbedding._yarn_get_mscale(scaling_factor, mscale_all_dim)
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

        # KV projection
        kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        # Get cos/sin from position_embeddings
        cos = position_embeddings[0]  # Full table: [max_seq_len, head_dim]
        sin = position_embeddings[1]  # Full table: [max_seq_len, head_dim]
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Apply RoPE on native interleaved q/k channels.
        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
            q_pe,
            k_pe,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # MLA with compressed KV
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


# =============================================================================
# Decoder Layer
# =============================================================================


class KimiK2DecoderLayer(nn.Module):
    """Transformer decoder layer for Kimi-K2."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = KimiK2Attention(config, layer_idx=layer_idx)

        # Layer 0 to first_k_dense_replace-1 use dense MLP, rest use MoE
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = KimiK2MoE(config)
        else:
            self.mlp = KimiK2MLP(config)

        self.input_layernorm = KimiK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


# =============================================================================
# Model Outputs
# =============================================================================


@dataclass
class KimiK2ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class KimiK2CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class KimiK25ConditionalOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# =============================================================================
# Full Models
# =============================================================================


class KimiK2PreTrainedModel(PreTrainedModel):
    """Base class for Kimi-K2 models."""

    config_class = KimiK2Config
    base_model_prefix = "model"
    _no_split_modules = ["KimiK2DecoderLayer"]
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


class KimiK2Model(KimiK2PreTrainedModel):
    """Kimi-K2 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [KimiK2DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Shared rotary embedding at model level
        self.rotary_emb = self._init_rope(config)

        self.post_init()

    def _init_rope(self, config):
        qk_rope_head_dim = config.qk_rope_head_dim

        attention_scaling = 1.0
        if (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        ):
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = KimiK2YarnRotaryEmbedding._yarn_get_mscale(scaling_factor, mscale_all_dim)
                attention_scaling = mscale

        use_yarn = (
            config.rope_scaling is not None
            and isinstance(config.rope_scaling, dict)
            and "factor" in config.rope_scaling
        )

        if not use_yarn:
            return KimiK2RotaryEmbedding(
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
            return KimiK2YarnRotaryEmbedding(
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
    ) -> KimiK2ModelOutput:
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
        position_embeddings = self.rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return KimiK2ModelOutput(last_hidden_state=hidden_states)


class KimiK2ForCausalLM(KimiK2PreTrainedModel, GenerationMixin):
    """Kimi-K2 model with language modeling head.

    Weight layout matches DeepseekV3ForCausalLM from the HF checkpoint:
      model.embed_tokens, model.layers.*, model.norm, lm_head
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = KimiK2Model(config)
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
    ) -> KimiK2CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return KimiK2CausalLMOutput(logits=logits)


# =============================================================================
# VLM Wrapper (Conditional Generation)
# =============================================================================


class KimiK25PreTrainedModel(PreTrainedModel):
    """Base class for Kimi-K2.5 VLM models."""

    config_class = KimiK25Config
    base_model_prefix = "language_model"
    _no_split_modules = ["KimiK2DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = getattr(self.config.text_config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KimiK25ForConditionalGeneration(KimiK25PreTrainedModel):
    """Kimi-K2.5 conditional generation model (text-only path for AD export).

    Weight layout matches HF checkpoint:
      language_model.model.embed_tokens, language_model.model.layers.*,
      language_model.model.norm, language_model.lm_head
    Vision tower weights are ignored during export.
    """

    def __init__(self, config: KimiK25Config, **kwargs):
        super().__init__(config)
        self.language_model = KimiK2ForCausalLM(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> KimiK25ConditionalOutput:
        outputs = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return KimiK25ConditionalOutput(logits=outputs.logits)


# =============================================================================
# Registration
# =============================================================================

# Register text model for direct text-only usage
AutoModelForCausalLMFactory.register_custom_model_cls("KimiK2Config", KimiK2ForCausalLM)

# Register VLM wrapper for full KimiK25 config (used by HF's auto_map)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "KimiK25Config", KimiK25ForConditionalGeneration
)
