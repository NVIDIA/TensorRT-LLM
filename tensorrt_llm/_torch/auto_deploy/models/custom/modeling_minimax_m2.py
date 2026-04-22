# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slimmed-down PyTorch MiniMax-M2 model for AutoDeploy export (prefill only).

Source:
https://huggingface.co/MiniMaxAI/MiniMax-M2.7

Key architecture features:
* GQA: 48 Q heads / 8 KV heads, head_dim=128
* Partial RoPE: rotary_dim=64 (half of head_dim), NeoX format
* Per-layer QK norm: RMSNorm on flat projection before head reshape
* 256 routed experts, top-8 selection, sigmoid routing with e_score_correction_bias
* No shared experts
* FP8 quantized checkpoint (handled by AD transforms, not model code)

Differences from the HF reference (modeling_minimax_m2.py):
* Stripped KV cache, training paths, dropout, flash attention variants
* Uses AD canonical ops: torch_rmsnorm, torch_attention, torch_moe
* No repeat_kv (torch_attention handles GQA natively)
* Position embeddings computed once and passed to all layers
* MTP modules removed (not used in prefill export)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MiniMaxM2ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class MiniMaxM2CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# RoPE helpers (partial rotation, NeoX format)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE to query and key in bsnd layout.

    Only the first ``rotary_dim`` dimensions are rotated; the rest pass through.
    ``unsqueeze_dim=2`` broadcasts over the head dimension (N) in bsnd.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# ---------------------------------------------------------------------------
# Rotary Embedding (pre-cached with _ad_ prefix)
# ---------------------------------------------------------------------------


class MiniMaxM2RotaryEmbedding(nn.Module):
    """Pre-computed RoPE cache for partial rotation (rotary_dim < head_dim)."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin


# ---------------------------------------------------------------------------
# RMSNorm (using AD canonical op)
# ---------------------------------------------------------------------------


class MiniMaxM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# MLP (expert body — uses w1/w2/w3 naming to match checkpoint)
# ---------------------------------------------------------------------------


class MiniMaxM2MLP(nn.Module):
    """SwiGLU expert MLP. Naming matches HF checkpoint: w1=gate, w3=up, w2=down."""

    def __init__(self, hidden_size: int, ffn_dim: int, hidden_act: str):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Sparse MoE Block
# ---------------------------------------------------------------------------


class MiniMaxM2SparseMoeBlock(nn.Module):
    """MoE with sigmoid routing + e_score_correction_bias (DeepSeek-V3 style)."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MiniMaxM2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
                for _ in range(config.num_local_experts)
            ]
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Sigmoid routing with bias correction and top-k selection
        router_logits = self.gate(hidden_flat)
        routing_weights = torch.sigmoid(router_logits.float())
        scores = routing_weights + self.e_score_correction_bias
        _, top_k_indices = torch.topk(scores, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_indices)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_flat.dtype)

        # Dispatch via AD canonical MoE op
        output = torch.ops.auto_deploy.torch_moe(
            hidden_flat,
            top_k_indices,
            top_k_weights,
            w1_weight=[e.w1.weight for e in self.experts],
            w2_weight=[e.w2.weight for e in self.experts],
            w3_weight=[e.w3.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )

        return output.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MiniMaxM2Attention(nn.Module):
    """GQA attention with per-layer QK norm and partial RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Per-layer QK norm: applied to flat projection BEFORE head reshape
        self.q_norm = MiniMaxM2RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM2RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Per-layer QK norm on flat projection
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to bsnd layout
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Partial RoPE in bsnd layout
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # GQA attention via AD canonical op
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxM2Attention(config, layer_idx)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MiniMaxM2PreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM2DecoderLayer"]
    supports_gradient_checkpointing = False


class MiniMaxM2Model(MiniMaxM2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MiniMaxM2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniMaxM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embedding — partial rotation (rotary_dim < head_dim)
        self.rotary_emb = MiniMaxM2RotaryEmbedding(
            dim=config.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MiniMaxM2ModelOutput:
        assert position_ids is not None, "position_ids is required"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Compute position embeddings once (pre-sliced by position_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.norm(hidden_states)
        return MiniMaxM2ModelOutput(last_hidden_state=hidden_states)


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniMaxM2Model(config)
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
    ) -> MiniMaxM2CausalLMOutput:
        assert position_ids is not None, "position_ids is required"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return MiniMaxM2CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("MiniMaxM2Config", MiniMaxM2ForCausalLM)
