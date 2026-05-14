# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slimmed-down PyTorch GPT-OSS model for AutoDeploy export (prefill only).

Source:
    https://huggingface.co/openai/gpt-oss-20b
    https://huggingface.co/openai/gpt-oss-120b

Both 20b and 120b share the same architecture (only num_hidden_layers and
num_local_experts differ), so this file covers both variants.

Key architecture features:
* GQA: 64 Q heads / 8 KV heads, head_dim=64, hidden_size=2880
* Attention sinks: per-head learnable scalar concatenated into softmax denominator
* Alternating sliding/full attention by layer (sliding_window=128)
* YaRN-scaled RoPE (factor=32, original_max=4096), Llama-style half-rotary
* MoE: 32 experts (20b) / 128 experts (120b), top-4 routing
* Stacked MoE weights with biases on both gate_up and down projections
* Custom GLU activation: ``(up + 1) * gate * sigmoid(gate * 1.702)`` with
  ``gate.clamp(max=7)`` and ``up.clamp(-7, 7)``
* MXFP4 quantized MoE weights handled by the AD ``quantize_mxfp4_moe`` transform

Differences from the HF reference (modeling_gpt_oss.py):
* Stripped KV cache, training paths, dropout, mask construction, deprecated kwargs
* Uses AD canonical ops:
    - ``torch_rmsnorm``                 (normalization)
    - ``torch_attention``               (with ``sinks=`` and ``sliding_window=``)
    - ``torch_rope_with_explicit_cos_sin``
    - ``torch_moe_router``              (linear + topk + softmax + scatter)
    - ``torch_moe_dense_mlp``           (dense bmm-based GPT-OSS expert math)
* No ``repeat_kv`` (``torch_attention`` handles GQA natively)
* RoPE cos/sin is computed once per forward and pre-sliced by ``position_ids``
* The HF config class ``GptOssConfig`` is reused directly from ``transformers``
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ..hf import AutoModelForCausalLMFactory

# GPT-OSS hard-codes these in the HF reference (see modeling_gpt_oss.GptOssExperts).
# ``alpha`` controls the SwiGLU sigmoid scaling, ``limit`` clamps gate/up before the GLU.
_GPTOSS_GLU_ALPHA = 1.702
_GPTOSS_GLU_LIMIT_FALLBACK = 7.0


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GptOssModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GptOssCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# YaRN helpers (faithful copy of transformers._compute_yarn_parameters)
# ---------------------------------------------------------------------------


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rot: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_pos: int, truncate: bool
) -> Tuple[float, float]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_pos)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_pos)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_factor(min_v: float, max_v: float, dim: int) -> torch.Tensor:
    if min_v == max_v:
        max_v = max_v + 0.001
    factor = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(factor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# RMSNorm (using AD canonical op)
# ---------------------------------------------------------------------------


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# Rotary Embedding (YaRN, pre-cached, sliced once per forward)
# ---------------------------------------------------------------------------


class GptOssRotaryEmbedding(nn.Module):
    """YaRN-scaled rotary embedding for GPT-OSS.

    The HF reference applies RoPE via ``torch.chunk(x, 2, dim=-1)`` with cos/sin
    of length ``head_dim/2``. This is mathematically identical to the standard
    Llama RoPE (``rotate_half`` + ``cos = sin = cat(freqs, freqs)``), so we cache
    a duplicated ``[max_pos, head_dim]`` table and feed it to the AD canonical
    ``torch_rope_with_explicit_cos_sin`` op.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()

        attention_scaling = 1.0
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        else:
            rope_type = "default"

        if rope_type == "yarn":
            factor = float(rope_scaling["factor"])
            beta_fast = float(rope_scaling.get("beta_fast", 32.0))
            beta_slow = float(rope_scaling.get("beta_slow", 1.0))
            mscale = rope_scaling.get("mscale", None)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", None)
            attention_factor = rope_scaling.get("attention_factor", None)
            original_max = int(
                rope_scaling.get("original_max_position_embeddings") or max_position_embeddings
            )
            truncate = bool(rope_scaling.get("truncate", True))

            if attention_factor is None:
                if mscale and mscale_all_dim:
                    attention_scaling = float(
                        _yarn_get_mscale(factor, float(mscale))
                        / _yarn_get_mscale(factor, float(mscale_all_dim))
                    )
                else:
                    attention_scaling = _yarn_get_mscale(factor)
            else:
                attention_scaling = float(attention_factor)

            pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            inv_freq_extra = 1.0 / pos_freqs
            inv_freq_inter = 1.0 / (factor * pos_freqs)

            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, head_dim, rope_theta, original_max, truncate
            )
            extra_factor = 1.0 - _yarn_linear_ramp_factor(low, high, head_dim // 2)
            inv_freq = inv_freq_inter * (1.0 - extra_factor) + inv_freq_extra * extra_factor
        else:
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin


# ---------------------------------------------------------------------------
# Router (replaces HF GptOssTopKRouter; eliminates the gptoss_topk_router patch)
# ---------------------------------------------------------------------------


class GptOssTopKRouter(nn.Module):
    """Top-K router: linear projection + topk + softmax + scatter.

    Produces ``router_scores`` of shape ``[B*S, num_experts]`` with non-zero
    entries only at the top-k expert positions, summing to 1 along dim=-1.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_local_experts)
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_moe_router(
            hidden_states, self.weight, self.bias, self.top_k
        )


# ---------------------------------------------------------------------------
# Experts (stacked weights with biases; uses torch_moe_dense_mlp)
# ---------------------------------------------------------------------------


class GptOssExperts(nn.Module):
    """GPT-OSS dense experts module.

    Holds the four stacked parameters that match the HF safetensors layout:
        gate_up_proj      : [E, H, 2I]  (gate and up interleaved on the last dim)
        gate_up_proj_bias : [E, 2I]
        down_proj         : [E, I, H]
        down_proj_bias    : [E, H]

    The forward delegates to ``torch_moe_dense_mlp``, which encodes GPT-OSS's
    custom GLU: ``(up + 1) * gate * sigmoid(alpha * gate)`` with clamps on
    gate (max=limit) and up (-limit, limit).

    The MXFP4 quantization path replaces this op (and the upstream router op)
    with ``triton_mxfp4_moe`` in the AD ``quantize_mxfp4_moe`` graph transform;
    the ``_blocks`` / ``_scales`` parameters are registered there at transform
    time so we do not declare them here.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = int(config.num_local_experts)
        self.hidden_size = int(config.hidden_size)
        self.expert_dim = int(config.intermediate_size)
        self.alpha = _GPTOSS_GLU_ALPHA
        # The HF safetensors / config carry ``swiglu_limit``; fall back to 7.0
        # for synthetic configs that omit it.
        self.limit = float(getattr(config, "swiglu_limit", _GPTOSS_GLU_LIMIT_FALLBACK))

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_moe_dense_mlp(
            hidden_states,
            routing_weights,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            self.alpha,
            self.limit,
        )


class GptOssMLP(nn.Module):
    """Router + experts. Drop-in replacement for HF ``GptOssMLP`` in prefill."""

    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        routing_weights = self.router(hidden_states)  # [B*S, E]
        out = self.experts(hidden_states, routing_weights)
        return out.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Attention (GQA + sinks + per-layer sliding window)
# ---------------------------------------------------------------------------


class GptOssAttention(nn.Module):
    """GPT-OSS attention with learnable per-head sinks and optional sliding window."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_bias = bool(getattr(config, "attention_bias", True))

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=self.attention_bias
        )

        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        # Per-layer sliding window: only enabled on layers tagged "sliding_attention".
        layer_types = getattr(config, "layer_types", None)
        is_sliding = layer_types is not None and layer_types[layer_idx] == "sliding_attention"
        sliding_window = getattr(config, "sliding_window", None)
        self.sliding_window = int(sliding_window) if (is_sliding and sliding_window) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to [B, S, N, head_dim] (BSND layout).
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings  # [B, S, head_dim]
        # Apply RoPE with unsqueeze_dim=2 for BSND layout.
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # ``torch_attention`` handles GQA natively; sinks/sliding_window are
        # per-call kwargs. Causal mask is applied internally for prefill.
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scaling,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
            layout="bsnd",
        )
        # [B, S, N, D] -> [B, S, N*D]
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Model + CausalLM
# ---------------------------------------------------------------------------


class GptOssPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["GptOssDecoderLayer"]
    supports_gradient_checkpointing = False


class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.rotary_emb = GptOssRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssModelOutput:
        assert position_ids is not None, "position_ids is required"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.norm(hidden_states)
        return GptOssModelOutput(last_hidden_state=hidden_states)


class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssCausalLMOutput:
        assert position_ids is not None, "position_ids is required"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return GptOssCausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("GptOssConfig", GptOssForCausalLM)
