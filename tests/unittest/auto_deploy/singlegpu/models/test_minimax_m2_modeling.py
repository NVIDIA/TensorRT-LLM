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

"""Tests for MiniMax-M2 custom model implementation.

This module tests the custom MiniMax-M2 model implementation which uses
auto_deploy custom ops for export compatibility. MiniMax-M2 uses:
- GQA with per-layer QK normalization and partial RoPE
- MoE with sigmoid routing and e_score_correction_bias (noaux_tc style)

Since the HF MiniMax-M2 model requires trust_remote_code (not in transformers
natively), we include minimal HF reference classes directly in this test file.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2Config,
    MiniMaxM2DecoderLayer,
    MiniMaxM2ForCausalLM,
    MiniMaxM2MLP,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2SparseMoeBlock,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> MiniMaxM2Config:
    """Create a small MiniMax-M2 config for testing.

    Models partial RoPE (rotary_dim < head_dim), QK norm, and MoE with
    sigmoid routing.
    """
    return MiniMaxM2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_experts_per_tok=2,
        num_local_experts=4,
        use_qk_norm=True,
        rotary_dim=8,  # Partial RoPE: 8 out of 16 head_dim
    )


# =========================================================================
# HF Reference Classes (minimal copies for equivalence testing)
# =========================================================================


class _HFRMSNorm(nn.Module):
    """HF-style RMSNorm reference."""

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


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _HFAttention(nn.Module):
    """Minimal HF MiniMax-M2 Attention reference for testing."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = _HFRMSNorm(self.head_dim * self.num_heads, eps=config.rms_norm_eps)
            self.k_norm = _HFRMSNorm(self.head_dim * self.num_kv_heads, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class _HFMLP(nn.Module):
    """Minimal HF MiniMax-M2 MLP reference."""

    def __init__(self, config):
        super().__init__()
        from transformers.activations import ACT2FN

        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class _HFSparseMoeBlock(nn.Module):
    """Minimal HF MiniMax-M2 MoE reference with sigmoid routing."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([_HFMLP(config) for _ in range(self.num_experts)])
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, selected_experts)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states_flat.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
            current_hidden = (
                self.experts[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden.to(hidden_states_flat.dtype))

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


class _HFDecoderLayer(nn.Module):
    """Minimal HF MiniMax-M2 decoder layer reference."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.self_attn = _HFAttention(config)
        self.block_sparse_moe = _HFSparseMoeBlock(config)
        self.input_layernorm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class _HFForCausalLM(nn.Module):
    """Minimal HF MiniMax-M2 full model reference."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_HFDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, cos, sin):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        return logits


def _compute_hf_rope(config, x, position_ids):
    """Compute HF-style rotary embeddings for reference model."""
    rotary_dim = getattr(config, "rotary_dim", None)
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    if rotary_dim is None:
        rotary_dim = head_dim

    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    inv_freq_expanded = inv_freq_expanded.to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)
    return cos, sin


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_minimax_m2_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    # Create HF MLP
    hf_mlp = _HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = MiniMaxM2MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_minimax_m2_attention_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    # Create HF attention reference
    hf_attn = _HFAttention(config)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention with same weights
    custom_attn = MiniMaxM2Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute RoPE embeddings
    hf_cos, hf_sin = _compute_hf_rope(config, x, position_ids)

    rotary_dim = getattr(config, "rotary_dim", None) or config.head_dim
    custom_rotary = MiniMaxM2RotaryEmbedding(
        rotary_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos_full, custom_sin_full = custom_rotary(x)

    # Run both
    hf_out = hf_attn(x, hf_cos, hf_sin)
    custom_out = custom_attn(x, position_ids, (custom_cos_full, custom_sin_full))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_minimax_m2_moe_equivalence(B, S, dtype):
    """Test MoE block produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    # Create HF MoE
    hf_moe = _HFSparseMoeBlock(config)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    # Create custom MoE and load same weights
    custom_moe = MiniMaxM2SparseMoeBlock(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_minimax_m2_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    # Create HF decoder layer
    hf_layer = _HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = MiniMaxM2DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF RoPE
    hf_cos, hf_sin = _compute_hf_rope(config, x, position_ids)

    # Custom RoPE (full tables)
    rotary_dim = getattr(config, "rotary_dim", None) or config.head_dim
    custom_rotary = MiniMaxM2RotaryEmbedding(
        rotary_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos_full, custom_sin_full = custom_rotary(x)

    hf_out = hf_layer(x, hf_cos, hf_sin)
    custom_out = custom_layer(x, position_ids, (custom_cos_full, custom_sin_full))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_minimax_m2_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF reference."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = _create_small_config()

    # Create HF model
    hf_model = _HFForCausalLM(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model
    custom_model = MiniMaxM2ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    # Build state dict mapping from HF reference to custom model
    hf_sd = hf_model.state_dict()
    custom_sd = {}
    for key, val in hf_sd.items():
        custom_sd[f"model.{key}" if not key.startswith("lm_head") else key] = val
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF RoPE (need dummy tensor for dtype)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    hf_cos, hf_sin = _compute_hf_rope(config, dummy, position_ids)

    hf_logits = hf_model(input_ids, hf_cos, hf_sin)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(), hf_logits.float(), rmse_ratio_tol=0.05, msg="Full model: "
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_minimax_m2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = MiniMaxM2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test with different input shape (dynamic shapes)
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert_rmse_close(
        logits2.float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_minimax_m2_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "minimax_m2"
    assert hasattr(config, "use_qk_norm")
    assert hasattr(config, "rotary_dim")
    assert config.use_qk_norm is True
    assert config.rotary_dim == 8


def test_minimax_m2_partial_rope():
    """Test that attention correctly handles partial RoPE."""
    config = _create_small_config()
    attn = MiniMaxM2Attention(config, layer_idx=0)
    assert attn.rotary_dim == 8
    assert attn.head_dim == 16
    assert attn.rotary_dim < attn.head_dim


def test_minimax_m2_gqa_structure():
    """Test that attention uses GQA."""
    config = _create_small_config()
    model = MiniMaxM2ForCausalLM(config)
    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert hasattr(attn, "q_norm")
    assert hasattr(attn, "k_norm")


def test_minimax_m2_moe_structure():
    """Test MoE expert structure matches checkpoint format."""
    config = _create_small_config()
    moe = MiniMaxM2SparseMoeBlock(config)

    assert isinstance(moe.experts, nn.ModuleList)
    assert len(moe.experts) == config.num_local_experts

    state_dict = moe.state_dict()
    expected_keys = [
        "gate.weight",
        "experts.0.w1.weight",
        "experts.0.w2.weight",
        "experts.0.w3.weight",
        "e_score_correction_bias",
    ]
    for key in expected_keys:
        assert key in state_dict, f"Missing key '{key}' in state_dict"


def test_minimax_m2_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = MiniMaxM2ForCausalLM(config)
    state_dict = model.state_dict()

    expected_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.0.block_sparse_moe.e_score_correction_bias",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for key in expected_keys:
        assert key in state_dict, f"Missing key '{key}'"
