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

"""Tests for EXAONE custom model implementation.

This module tests the custom EXAONE model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility. EXAONE uses GQA with SwiGLU MLP, RMSNorm,
and llama3-style RoPE.

Since ExaoneForCausalLM is not in the installed transformers, this file
includes self-contained HF reference implementations for equivalence testing.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_exaone import (
    ExaoneAttention,
    ExaoneAttentionBlock,
    ExaoneConfig,
    ExaoneDecoderLayer,
    ExaoneForCausalLM,
    ExaoneMLP,
    ExaoneRMSNorm,
    ExaoneRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> ExaoneConfig:
    """Create a small EXAONE config for testing."""
    return ExaoneConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        activation_function="silu",
        max_position_embeddings=512,
        layer_norm_epsilon=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference implementations (self-contained, since EXAONE is not in
# the installed transformers)
# =========================================================================


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE. cos/sin: [B, S, head_dim]. Q/K: [B, N, S, head_dim] (BNSD)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _HFExaoneRMSNorm(nn.Module):
    """Reference RMSNorm matching HF ExaoneRMSNorm."""

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


class _HFExaoneMLP(nn.Module):
    """Reference MLP matching HF ExaoneMLP (c_fc_0/c_fc_1/c_proj naming)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc_0 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_fc_1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc_0(x)) * self.c_fc_1(x))


class _HFExaoneAttention(nn.Module):
    """Reference attention matching HF ExaoneAttention (eager mode)."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.shape

        # BNSD layout
        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # cos/sin already pre-sliced by position_ids
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.out_proj(attn_output)


class _HFExaoneAttentionBlock(nn.Module):
    """Wrapper matching HF ExaoneAttentionBlock (attn.attention.* keys)."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.attention = _HFExaoneAttention(config, layer_idx)

    def forward(self, hidden_states, position_embeddings):
        return self.attention(hidden_states, position_embeddings)


class _HFExaoneDecoderLayer(nn.Module):
    """Reference decoder layer matching HF ExaoneDecoderLayer."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.ln_1 = _HFExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = _HFExaoneAttentionBlock(config, layer_idx)
        self.ln_2 = _HFExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _HFExaoneMLP(config)

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class _HFExaoneCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class _HFExaoneModel(nn.Module):
    """Reference EXAONE model (no PreTrainedModel to keep simple)."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.h = nn.ModuleList(
            [_HFExaoneDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = _HFExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, position_embeddings):
        hidden_states = self.wte(input_ids)
        for layer in self.h:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class _HFExaoneForCausalLM(nn.Module):
    """Reference EXAONE CausalLM. State dict keys match our custom model."""

    def __init__(self, config):
        super().__init__()
        self.transformer = _HFExaoneModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_embeddings):
        hidden_states = self.transformer(input_ids, position_embeddings)
        logits = self.lm_head(hidden_states).float()
        return _HFExaoneCausalLMOutput(logits=logits)


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_exaone_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm layer produces numerically equivalent output to HF implementation."""
    device = "cuda"
    config = _create_small_config()

    hf_norm = _HFExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = ExaoneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    custom_norm.to(device=device, dtype=dtype)
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_exaone_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    device = "cuda"
    config = _create_small_config()

    hf_mlp = _HFExaoneMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = ExaoneMLP(config)
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
def test_exaone_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = _HFExaoneAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = ExaoneAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute pre-sliced position embeddings
    rotary = ExaoneRotaryEmbedding(config)
    rotary.to(device=device, dtype=dtype)
    position_embeddings = rotary(x, position_ids)

    hf_out = hf_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
    )
    custom_out = custom_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_exaone_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    device = "cuda"
    config = _create_small_config()

    hf_layer = _HFExaoneDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = ExaoneDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    rotary = ExaoneRotaryEmbedding(config)
    rotary.to(device=device, dtype=dtype)
    position_embeddings = rotary(x, position_ids)

    hf_out = hf_layer(
        hidden_states=x,
        position_embeddings=position_embeddings,
    )
    custom_out = custom_layer(
        hidden_states=x,
        position_embeddings=position_embeddings,
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_exaone_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF reference."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()

    # Build custom model and extract state_dict
    custom_model = ExaoneForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.eval()

    # Build HF reference model with same weights
    hf_model = _HFExaoneForCausalLM(config)
    hf_model.to(device=device, dtype=dtype)
    result = hf_model.load_state_dict(custom_model.state_dict(), strict=False)
    assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
    assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"
    hf_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Custom model takes input_ids + position_ids
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    # HF reference needs pre-sliced position embeddings explicitly
    rotary = ExaoneRotaryEmbedding(config)
    rotary.to(device=device, dtype=dtype)
    dummy_x = custom_model.transformer.wte(input_ids)
    position_embeddings = rotary(dummy_x, position_ids)
    hf_out = hf_model(
        input_ids=input_ids,
        position_embeddings=position_embeddings,
    )

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_exaone_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = ExaoneForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
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
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    assert_rmse_close(
        logits.float(),
        eager_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager: ",
    )

    # Test with different input shape to verify dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    expected_shape = (B2, S2, config.vocab_size)
    assert logits2.shape == expected_shape, (
        f"Dynamic shape test failed: expected {expected_shape}, got {logits2.shape}"
    )
    assert_rmse_close(
        logits2.float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_exaone_config_attribute_map():
    """Test that config attribute_map correctly aliases EXAONE-specific names."""
    config = _create_small_config()
    # num_hidden_layers -> num_layers
    assert config.num_hidden_layers == 3
    assert config.num_layers == 3
    # hidden_act -> activation_function
    assert config.hidden_act == "silu"
    assert config.activation_function == "silu"
    # rms_norm_eps -> layer_norm_epsilon
    assert config.rms_norm_eps == 1e-6
    assert config.layer_norm_epsilon == 1e-6


def test_exaone_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = ExaoneForCausalLM(config)

    attn = model.transformer.h[0].attn.attention
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_exaone_state_dict_keys():
    """Test that state_dict keys match expected EXAONE checkpoint format."""
    config = _create_small_config()
    model = ExaoneForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "transformer.wte.weight",
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.attn.attention.q_proj.weight",
        "transformer.h.0.attn.attention.k_proj.weight",
        "transformer.h.0.attn.attention.v_proj.weight",
        "transformer.h.0.attn.attention.out_proj.weight",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.mlp.c_fc_0.weight",
        "transformer.h.0.mlp.c_fc_1.weight",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )


def test_exaone_attention_block_wrapper():
    """Test that ExaoneAttentionBlock correctly wraps ExaoneAttention."""
    config = _create_small_config()
    block = ExaoneAttentionBlock(config, layer_idx=0)

    assert hasattr(block, "attention"), "ExaoneAttentionBlock should have 'attention' attribute"
    assert isinstance(block.attention, ExaoneAttention), (
        "block.attention should be an ExaoneAttention instance"
    )

    sd_keys = list(block.state_dict().keys())
    for key in sd_keys:
        assert key.startswith("attention."), f"Expected key to start with 'attention.', got '{key}'"
