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

"""Tests for HunYuan A13B MoE custom model implementation.

Hierarchical test levels:
1. Block equivalence — RMSNorm, Attention, MLP expert, MoE layer individually
2. Layer equivalence — Full decoder layer
3. Full model equivalence — End-to-end logits comparison
4. Export test — torch_export_to_gm with dynamic shapes

HF reference classes are copied inline from the HF repo source
(modeling_hunyuan.py) to keep tests self-contained without requiring
trust_remote_code or a specific transformers version.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_hunyuan_moe import (
    HunYuanMoEAttention,
    HunYuanMoEDecoderLayer,
    HunYuanMoEForCausalLM,
    HunYuanMoEMLP,
    HunYuanMoEMoE,
    HunYuanMoERMSNorm,
    HunYuanMoERotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# ===========================================================================
# Inline HF reference classes (from modeling_hunyuan.py)
# ===========================================================================


class _HFRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _HFRotaryEmbedding(nn.Module):
    """NTK-Alpha dynamic RoPE (HF reference)."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).float()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached or self.inv_freq.dtype != torch.float32:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class _HFDynamicNTKAlphaRotaryEmbedding(_HFRotaryEmbedding):
    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_alpha=1.0
    ):
        self.scaling_alpha = scaling_alpha
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        base = self.base * self.scaling_alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def _hf_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (_hf_rotate_half(k) * sin)
    return q_embed, k_embed


def _hf_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _HFAttention(nn.Module):
    """Minimal HF eager attention reference (GQA + QK norm after RoPE)."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_qk_norm = config.use_qk_norm
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.use_qk_norm:
            self.query_layernorm = _HFRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = _HFRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Init RoPE
        if config.rope_scaling is not None and config.rope_scaling.get("alpha"):
            self.rotary_emb = _HFDynamicNTKAlphaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                scaling_alpha=config.rope_scaling["alpha"],
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = _HFRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = _hf_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        key_states = _hf_repeat_kv(key_states, self.num_key_value_groups)
        value_states = _hf_repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class _HFMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _top1gating_simple(logits):
    """Simplified top-1 gating for reference (matches HF logic for dispatch/combine)."""
    gates = F.softmax(logits.float(), dim=1)
    indices = torch.argmax(gates, dim=1)  # [T]
    expert_weights = gates[torch.arange(len(indices)), indices]  # [T]
    return indices, expert_weights


class _HFMoE(nn.Module):
    """HF-faithful MoE reference using simple per-token dispatch (no capacity buffers)."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        topk_val = config.moe_topk
        self.topk = topk_val[0] if isinstance(topk_val, list) else topk_val
        num_shared = config.num_shared_expert
        num_shared = num_shared[0] if isinstance(num_shared, list) else num_shared
        moe_inter = config.moe_intermediate_size
        moe_inter = moe_inter[0] if isinstance(moe_inter, list) else moe_inter
        shared_inter = config.intermediate_size * num_shared

        # Gate linear (float32 weights, matching HF)
        self.gate_wg = nn.Linear(
            config.hidden_size, config.num_experts, bias=False, dtype=torch.float32
        )
        # Shared expert
        if getattr(config, "use_mixed_mlp_moe", True):
            self.shared_mlp = _HFMLP(config, intermediate_size=shared_inter)
        else:
            self.shared_mlp = None
        # Routed experts
        self.experts = nn.ModuleList(
            [_HFMLP(config, intermediate_size=moe_inter) for _ in range(config.num_experts)]
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        if self.shared_mlp is not None:
            shared_out = self.shared_mlp(hidden_states)

        reshaped = hidden_states.reshape(-1, hidden_size)
        logits = F.linear(reshaped.float(), self.gate_wg.weight.float())
        gates = F.softmax(logits, dim=-1)  # [T, E]

        topk_weights, topk_indices = gates.topk(self.topk, dim=-1)  # [T, K]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # normalize
        topk_weights = topk_weights.to(hidden_states.dtype)

        T, K = topk_indices.shape
        output = torch.zeros_like(reshaped)
        for k in range(K):
            expert_idx = topk_indices[:, k]  # [T]
            weight = topk_weights[:, k]  # [T]
            for e_id in range(self.num_experts):
                mask = expert_idx == e_id
                if not mask.any():
                    continue
                tokens = reshaped[mask]
                out = self.experts[e_id](tokens)
                output[mask] += weight[mask].unsqueeze(-1) * out

        output = output.reshape(bsz, seq_len, hidden_size)
        if self.shared_mlp is not None:
            output = output + shared_out
        return output


class _HFDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.self_attn = _HFAttention(config, layer_idx=layer_idx)
        self.mlp = _HFMoE(config)
        self.input_layernorm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _HFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_HFDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, position_ids=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            )

        # Build causal mask
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, -1, -1)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class _HFForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = _HFModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, position_ids=None):
        hidden = self.model(input_ids, position_ids=position_ids)
        return self.lm_head(hidden).float()


# ===========================================================================
# Small configs for testing
# ===========================================================================

# Small number of experts to keep tests fast
_NUM_EXPERTS = 4
_TOPK = 2


def _create_small_custom_config() -> PretrainedConfig:
    return PretrainedConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_qk_norm=True,
        num_experts=_NUM_EXPERTS,
        num_shared_expert=1,
        moe_topk=_TOPK,
        use_mixed_mlp_moe=True,
        tie_word_embeddings=True,
        initializer_range=0.02,
        pad_token_id=0,
    )


def _hf_config_from_custom(cfg: PretrainedConfig):
    """Return a namespace-like config usable by HF reference classes."""

    class _Cfg:
        pass

    c = _Cfg()
    for k, v in vars(cfg).items():
        setattr(c, k, v)
    c._attn_implementation = "eager"
    # Flatten per-layer lists into scalar for small config (all same value)
    return c


# ===========================================================================
# Weight-transfer helpers
# ===========================================================================


def _transfer_mlp_weights(hf_mlp, custom_mlp):
    custom_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
    custom_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
    custom_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)


def _transfer_attention_weights(hf_attn, custom_attn):
    custom_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
    custom_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
    custom_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
    custom_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)
    if custom_attn.use_qk_norm:
        custom_attn.query_layernorm.weight.data.copy_(hf_attn.query_layernorm.weight.data)
        custom_attn.key_layernorm.weight.data.copy_(hf_attn.key_layernorm.weight.data)


def _transfer_moe_weights(hf_moe, custom_moe):
    # Gate
    custom_moe.gate.wg.weight.data.copy_(hf_moe.gate_wg.weight.data)
    # Shared MLP
    if hf_moe.shared_mlp is not None and custom_moe.use_mixed_mlp_moe:
        _transfer_mlp_weights(hf_moe.shared_mlp, custom_moe.shared_mlp)
    # Experts
    for i, (hf_exp, custom_exp) in enumerate(zip(hf_moe.experts, custom_moe.experts)):
        _transfer_mlp_weights(hf_exp, custom_exp)


def _transfer_decoder_layer_weights(hf_layer, custom_layer):
    _transfer_attention_weights(hf_layer.self_attn, custom_layer.self_attn)
    _transfer_moe_weights(hf_layer.mlp, custom_layer.mlp)
    custom_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    custom_layer.post_attention_layernorm.weight.data.copy_(
        hf_layer.post_attention_layernorm.weight.data
    )


def _transfer_full_model_weights(hf_model, custom_model):
    custom_model.model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
    custom_model.model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    for i in range(len(hf_model.model.layers)):
        _transfer_decoder_layer_weights(hf_model.model.layers[i], custom_model.model.layers[i])
    if custom_model.config.tie_word_embeddings:
        # Both models should use the same weights for embed_tokens and lm_head.
        # Custom model already has them tied; align HF reference too.
        hf_model.lm_head.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
    else:
        custom_model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)


# ===========================================================================
# Level 1: Block equivalence
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_rmsnorm_equivalence(B, S, dtype):
    """RMSNorm produces identical output to HF reference."""
    hidden_size = 64
    eps = 1e-5
    hf_norm = _HFRMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    custom_norm = HunYuanMoERMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    custom_norm.weight.data.copy_(hf_norm.weight.data)

    x = torch.randn(B, S, hidden_size, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mlp_equivalence(B, S, dtype):
    """Expert MLP produces identical output to HF MLP reference."""
    cfg = _create_small_custom_config()
    hf_cfg = _hf_config_from_custom(cfg)
    hf_mlp = _HFMLP(hf_cfg, intermediate_size=cfg.moe_intermediate_size).to(dtype=dtype)
    custom_mlp = HunYuanMoEMLP(cfg.hidden_size, cfg.moe_intermediate_size).to(dtype=dtype)
    _transfer_mlp_weights(hf_mlp, custom_mlp)

    x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_attention_equivalence(B, S, dtype):
    """Attention block approximately matches HF eager reference."""
    cfg = _create_small_custom_config()
    hf_cfg = _hf_config_from_custom(cfg)

    hf_attn = _HFAttention(hf_cfg, layer_idx=0).to(dtype=dtype)
    custom_attn = HunYuanMoEAttention(cfg, layer_idx=0).to(dtype=dtype)
    _transfer_attention_weights(hf_attn, custom_attn)

    x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # HF forward with causal mask
    causal_mask = torch.full((S, S), float("-inf"), dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None].expand(B, 1, -1, -1)
    hf_out, _ = hf_attn(x, attention_mask=causal_mask, position_ids=position_ids)

    # Custom forward
    custom_rope = HunYuanMoERotaryEmbedding(
        dim=cfg.attention_head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        base=cfg.rope_theta,
        rope_scaling=cfg.rope_scaling,
    ).to(dtype=dtype)
    pos_emb = custom_rope(x)
    custom_out = custom_attn(x, position_ids, pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_moe_layer_equivalence(B, S, dtype):
    """MoE layer approximately matches HF reference (shared + routed experts)."""
    cfg = _create_small_custom_config()
    hf_cfg = _hf_config_from_custom(cfg)

    hf_moe = _HFMoE(hf_cfg).to(dtype=dtype)
    custom_moe = HunYuanMoEMoE(
        hidden_size=cfg.hidden_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        shared_intermediate_size=cfg.intermediate_size,
        num_experts=cfg.num_experts,
        topk=cfg.moe_topk,
        hidden_act=cfg.hidden_act,
        use_mixed_mlp_moe=cfg.use_mixed_mlp_moe,
    ).to(dtype=dtype)
    _transfer_moe_weights(hf_moe, custom_moe)

    x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


# ===========================================================================
# Level 2: Layer equivalence
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decoder_layer_equivalence(B, S, dtype):
    """Full decoder layer approximately matches HF reference."""
    cfg = _create_small_custom_config()
    hf_cfg = _hf_config_from_custom(cfg)

    hf_layer = _HFDecoderLayer(hf_cfg, layer_idx=0).to(dtype=dtype)
    custom_layer = HunYuanMoEDecoderLayer(cfg, layer_idx=0).to(dtype=dtype)
    _transfer_decoder_layer_weights(hf_layer, custom_layer)

    x = torch.randn(B, S, cfg.hidden_size, dtype=dtype)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    causal_mask = torch.full((S, S), float("-inf"), dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None].expand(B, 1, -1, -1)
    hf_out = hf_layer(x, attention_mask=causal_mask, position_ids=position_ids)

    custom_rope = HunYuanMoERotaryEmbedding(
        dim=cfg.attention_head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        base=cfg.rope_theta,
        rope_scaling=cfg.rope_scaling,
    ).to(dtype=dtype)
    pos_emb = custom_rope(x)
    custom_out = custom_layer(x, position_ids, pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="DecoderLayer: ")


# ===========================================================================
# Level 3: Full model equivalence
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_full_model_equivalence(B, S, dtype):
    """Full model logits approximately match HF reference."""
    cfg = _create_small_custom_config()
    hf_cfg = _hf_config_from_custom(cfg)

    hf_model = _HFForCausalLM(hf_cfg).to(dtype=dtype)
    hf_model.eval()

    custom_model = HunYuanMoEForCausalLM(cfg).to(dtype=dtype)
    custom_model.eval()

    _transfer_full_model_weights(hf_model, custom_model)

    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits, hf_logits, rmse_ratio_tol=0.05, msg="FullModel: ")


# ===========================================================================
# Level 4: Export test
# ===========================================================================


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@torch.no_grad()
def test_model_can_be_exported(device):
    """Model can be exported with torch_export_to_gm and produces finite output."""
    dtype = torch.bfloat16
    cfg = _create_small_custom_config()

    model = HunYuanMoEForCausalLM(cfg)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    batch_dim = Dim.DYNAMIC
    seq_dim = Dim.DYNAMIC
    dynamic_shapes = {
        "input_ids": {0: batch_dim, 1: seq_dim},
        "position_ids": {0: batch_dim, 1: seq_dim},
    }

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "Exported logits should be finite"

    assert_rmse_close(logits, eager_out.logits, rmse_ratio_tol=0.05, msg="Export: ")

    # Verify dynamic shapes work with a different size
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, cfg.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)
    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, cfg.vocab_size)
    assert_rmse_close(logits2, eager_out2.logits, rmse_ratio_tol=0.05, msg="Export dynamic: ")


# ===========================================================================
# Registration and structure tests
# ===========================================================================


def test_config_registration():
    """Factory knows the model class under the real HF config name."""
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    assert "HunYuanConfig" in AutoModelForCausalLMFactory._custom_model_mapping


def test_tied_weights():
    """tie_word_embeddings: lm_head.weight is the same tensor as embed_tokens.weight."""
    cfg = _create_small_custom_config()
    model = HunYuanMoEForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_moe_structure():
    """Every decoder layer has the expected MoE submodule structure."""
    cfg = _create_small_custom_config()
    model = HunYuanMoEForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        moe = layer.mlp
        assert hasattr(moe, "gate"), f"Layer {i}: missing mlp.gate"
        assert hasattr(moe.gate, "wg"), f"Layer {i}: missing mlp.gate.wg"
        assert hasattr(moe, "experts"), f"Layer {i}: missing mlp.experts"
        assert len(moe.experts) == cfg.num_experts, f"Layer {i}: wrong expert count"
        if cfg.use_mixed_mlp_moe:
            assert hasattr(moe, "shared_mlp"), f"Layer {i}: missing mlp.shared_mlp"


def test_qk_norm_structure():
    """QK normalization layers exist in each attention module."""
    cfg = _create_small_custom_config()
    model = HunYuanMoEForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert hasattr(attn, "query_layernorm"), f"Layer {i}: missing query_layernorm"
        assert hasattr(attn, "key_layernorm"), f"Layer {i}: missing key_layernorm"


def test_state_dict_keys_match_checkpoint():
    """State dict keys match the HF checkpoint naming convention."""
    cfg = _create_small_custom_config()
    model = HunYuanMoEForCausalLM(cfg)
    sd = model.state_dict()

    expected = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.query_layernorm.weight",
        "model.layers.0.self_attn.key_layernorm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate.wg.weight",
        "model.layers.0.mlp.shared_mlp.gate_proj.weight",
        "model.layers.0.mlp.shared_mlp.up_proj.weight",
        "model.layers.0.mlp.shared_mlp.down_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    for key in expected:
        assert key in sd, f"Expected key '{key}' not in state_dict"
