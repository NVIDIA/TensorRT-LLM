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

"""Tests for InternLM3 custom model implementation.

This module tests the custom InternLM3 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility. InternLM3 uses GQA with SwiGLU MLP, RMSNorm,
and dynamic NTK-scaled RoPE.

HF reference classes are defined inline for equivalence testing because the
HF modeling_internlm3.py (from the HF checkpoint) cannot be imported on the
installed transformers version — it requires ``LossKwargs`` from
``transformers.utils`` which is only available in transformers >=4.48.
The config class *can* be loaded via AutoConfig with trust_remote_code.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_internlm3 import (
    InternLM3Attention,
    InternLM3DecoderLayer,
    InternLM3ForCausalLM,
    InternLM3MLP,
    InternLM3RMSNorm,
    InternLM3RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))

# Load InternLM3Config from the HF checkpoint cache (trust_remote_code).
# The config class is not in the installed transformers but is bundled with the HF repo.
try:
    _hf_config = AutoConfig.from_pretrained(
        "internlm/internlm3-8b-instruct", trust_remote_code=True
    )
    InternLM3Config = type(_hf_config)
except Exception:
    InternLM3Config = None


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config():
    """Create a small InternLM3 config for testing."""
    if InternLM3Config is None:
        pytest.skip("InternLM3Config not available (internlm/internlm3-8b-instruct not cached)")
    return InternLM3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,  # No dynamic scaling for tests (simplifies comparison)
        qkv_bias=False,
        attention_dropout=0.0,
        bias=False,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference classes (copied from internlm3 HF repo for equivalence tests)
# InternLM3 is not in installed transformers, so we inline minimal references.
# =========================================================================


class _HFInternLM3RMSNorm(nn.Module):
    """HF reference RMSNorm."""

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


def _hf_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (_hf_rotate_half(k) * sin)
    return q_embed, k_embed


def _hf_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _HFInternLM3RotaryEmbedding(nn.Module):
    """HF reference RotaryEmbedding using ROPE_INIT_FUNCTIONS."""

    def __init__(self, config):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"

        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _HFInternLM3MLP(nn.Module):
    """HF reference MLP."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _HFInternLM3Attention(nn.Module):
    """HF reference Attention (eager implementation)."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _hf_apply_rotary_pos_emb(q, k, cos, sin)

        k = _hf_repeat_kv(k, self.num_key_value_groups)
        v = _hf_repeat_kv(v, self.num_key_value_groups)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=hidden_states.device), diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _HFInternLM3DecoderLayer(nn.Module):
    """HF reference DecoderLayer."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.self_attn = _HFInternLM3Attention(config, layer_idx=layer_idx)
        self.mlp = _HFInternLM3MLP(config)
        self.input_layernorm = _HFInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _HFInternLM3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class _HFInternLM3ForCausalLM(nn.Module):
    """HF reference ForCausalLM (minimal, for full-model equivalence test)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_HFInternLM3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _HFInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = _HFInternLM3RotaryEmbedding(config)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, (cos, sin))

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def _convert_hf_ref_state_dict_to_custom(hf_state_dict):
    """Convert HF reference model state dict keys to custom model format.

    The HF reference uses flat structure (embed_tokens, layers, norm, lm_head)
    while the custom model wraps them in a `model` submodule.
    """
    new_sd = {}
    for k, v in hf_state_dict.items():
        if k.startswith("lm_head."):
            new_sd[k] = v
        else:
            new_sd[f"model.{k}"] = v
    return new_sd


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm layer produces numerically equivalent output to HF."""
    device = "cuda"
    config = _create_small_config()

    hf_norm = _HFInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = InternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
def test_internlm3_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF."""
    device = "cuda"
    config = _create_small_config()

    hf_mlp = _HFInternLM3MLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = InternLM3MLP(config)
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
def test_internlm3_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = _HFInternLM3Attention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = InternLM3Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = _HFInternLM3RotaryEmbedding(config)
    hf_rotary.to(device=device, dtype=dtype)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Run HF attention
    hf_out = hf_attn(hidden_states=x, position_embeddings=(hf_cos, hf_sin))

    # Run custom attention
    custom_out = custom_attn(hidden_states=x, position_embeddings=(custom_cos, custom_sin))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF."""
    device = "cuda"
    config = _create_small_config()

    hf_layer = _HFInternLM3DecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = InternLM3DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = _HFInternLM3RotaryEmbedding(config)
    hf_rotary.to(device=device, dtype=dtype)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Run HF decoder layer
    hf_out = hf_layer(hidden_states=x, position_embeddings=(hf_cos, hf_sin))

    # Run custom decoder layer
    custom_out = custom_layer(hidden_states=x, position_embeddings=(custom_cos, custom_sin))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_internlm3_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()

    hf_model = _HFInternLM3ForCausalLM(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = InternLM3ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(_convert_hf_ref_state_dict_to_custom(hf_model.state_dict()))
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_internlm3_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = InternLM3ForCausalLM(config)
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


def test_internlm3_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "internlm3"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "head_dim")
    assert hasattr(config, "bias")
    assert hasattr(config, "qkv_bias")


def test_internlm3_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = InternLM3ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_internlm3_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = InternLM3ForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )
