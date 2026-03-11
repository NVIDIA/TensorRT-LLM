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

"""Tests for DeciLM (Nemotron-NAS) custom model implementation.

Tests the custom AD model against an inline HF reference implementation.
The HF config is loaded via trust_remote_code=True at runtime; for unit tests
we create a small config using the same class loaded from the HF checkpoint.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import AutoConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_decilm import (
    DeciLMAttention,
    DeciLMDecoderLayer,
    DeciLMForCausalLM,
    DeciLMMLP,
    DeciLMRotaryEmbedding,
    _ffn_mult_to_intermediate_size,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Small test config (mimics HF DeciLMConfig structure loaded via trust_remote_code)
# =============================================================================

# Block configs for a small test model:
# Layer 0: attention (GQA, n_heads_in_group=2) + FFN (ffn_mult=2.625)
# Layer 1: FFN-only (attention no_op, ffn_mult=1.3125)
# Layer 2: attention (GQA, n_heads_in_group=2) + FFN (ffn_mult=5.25)
_SMALL_BLOCK_CONFIGS = [
    {
        "attention": {"n_heads_in_group": 2, "no_op": False},
        "ffn": {"ffn_mult": 2.625, "no_op": False},
    },
    {
        "attention": {"no_op": True},
        "ffn": {"ffn_mult": 1.3125, "no_op": False},
    },
    {
        "attention": {"n_heads_in_group": 2, "no_op": False},
        "ffn": {"ffn_mult": 5.25, "no_op": False},
    },
]


def _create_small_config():
    """Create a small DeciLM config for testing.

    Uses AutoConfig with trust_remote_code=True to load the HF DeciLMConfig class,
    then creates a small instance for testing.
    """
    try:
        hf_config = AutoConfig.from_pretrained(
            "nvidia/Llama-3_3-Nemotron-Super-49B-v1", trust_remote_code=True
        )
        ConfigCls = type(hf_config)
    except Exception:
        pytest.skip("Cannot load DeciLMConfig from HF (network or cache unavailable)")

    return ConfigCls(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 64,
        },
        block_configs=_SMALL_BLOCK_CONFIGS,
    )


# =============================================================================
# HF Reference Classes (standalone, for equivalence testing)
# =============================================================================


class _HFDeciLMRMSNorm(nn.Module):
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


class _HFDeciLMRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type", "default")
            if rope_type == "llama3":
                factor = config.rope_scaling["factor"]
                low_freq_factor = config.rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = config.rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = config.rope_scaling.get("original_max_position_embeddings", 8192)
                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                wavelen = 2 * math.pi / inv_freq
                inv_freq_scaled = inv_freq / factor
                smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)
                smoothed = (1 - smooth_factor) * inv_freq_scaled + smooth_factor * inv_freq
                inv_freq = torch.where(wavelen < high_freq_wavelen, inv_freq, smoothed)
                inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq_scaled, inv_freq)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


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


def _hf_repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _HFDeciLMAttention(nn.Module):
    def __init__(self, config, n_heads_in_group, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = n_heads_in_group
        self.num_key_value_heads = self.num_heads // self.num_key_value_groups

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = _HFDeciLMRotaryEmbedding(config)

    def forward(self, hidden_states, position_ids):
        bsz, q_len, _ = hidden_states.size()
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = _hf_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _hf_repeat_kv(key_states, self.num_key_value_groups)
        value_states = _hf_repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        causal_mask = torch.triu(
            torch.full(
                (q_len, q_len),
                float("-inf"),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _HFDeciLMMLP(nn.Module):
    def __init__(self, config, ffn_mult):
        super().__init__()
        intermediate_size = _ffn_mult_to_intermediate_size(ffn_mult, config.hidden_size)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _HFDeciLMDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        block_config = config.block_configs[layer_idx]
        self.has_attention = not block_config.attention.no_op
        self.has_ffn = not block_config.ffn.no_op

        if self.has_attention:
            self.input_layernorm = _HFDeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = _HFDeciLMAttention(
                config, block_config.attention.n_heads_in_group, layer_idx
            )
        if self.has_ffn:
            self.post_attention_layernorm = _HFDeciLMRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = _HFDeciLMMLP(config, block_config.ffn.ffn_mult)

    def forward(self, hidden_states, position_ids):
        if self.has_attention:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, position_ids)
            hidden_states = residual + hidden_states
        if self.has_ffn:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states


class _HFDeciLMForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_HFDeciLMDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = _HFDeciLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states).float()


# =============================================================================
# Unit Tests: Structure
# =============================================================================


def test_decilm_layer_structure():
    """Test that layers have correct structure based on block_configs."""
    config = _create_small_config()
    model = DeciLMForCausalLM(config)

    layer0 = model.model.layers[0]
    assert layer0.has_attention is True
    assert hasattr(layer0, "self_attn")
    assert hasattr(layer0, "input_layernorm")

    layer1 = model.model.layers[1]
    assert layer1.has_attention is False
    assert not hasattr(layer1, "self_attn")

    layer2 = model.model.layers[2]
    assert layer2.has_attention is True


def test_decilm_ffn_mult_to_intermediate_size():
    assert _ffn_mult_to_intermediate_size(5.25, 8192) == 28672
    assert _ffn_mult_to_intermediate_size(2.625, 8192) == 14336
    assert _ffn_mult_to_intermediate_size(1.0, 8192) == 5632
    assert _ffn_mult_to_intermediate_size(0.5, 8192) == 2816


def test_decilm_weight_keys_match_checkpoint():
    config = _create_small_config()
    model = DeciLMForCausalLM(config)
    keys = set(model.state_dict().keys())

    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.layers.1.self_attn.q_proj.weight" not in keys
    assert "model.layers.1.input_layernorm.weight" not in keys
    assert "model.layers.1.post_attention_layernorm.weight" in keys
    assert "model.layers.1.mlp.gate_proj.weight" in keys


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decilm_mlp_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    ffn_mult = 2.625
    intermediate_size = _ffn_mult_to_intermediate_size(ffn_mult, config.hidden_size)

    hf_mlp = _HFDeciLMMLP(config, ffn_mult).to(device=device, dtype=dtype).eval()
    custom_mlp = DeciLMMLP(config, intermediate_size).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decilm_attention_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    n_heads_in_group = 2

    hf_attn = _HFDeciLMAttention(config, n_heads_in_group, 0).to(device=device, dtype=dtype).eval()
    custom_attn = DeciLMAttention(config, n_heads_in_group, 0).to(device=device, dtype=dtype)
    custom_sd = {k: v for k, v in hf_attn.state_dict().items() if not k.startswith("rotary_emb.")}
    custom_attn.load_state_dict(custom_sd)
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_attn(x, position_ids)

    rotary_emb = DeciLMRotaryEmbedding(config).to(device=device, dtype=dtype)
    position_embeddings = rotary_emb(x, position_ids)
    custom_out = custom_attn(x, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decilm_decoder_layer_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    layer_idx = 0

    hf_layer = _HFDeciLMDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
    custom_layer = DeciLMDecoderLayer(config, layer_idx).to(device=device, dtype=dtype)
    custom_sd = {k: v for k, v in hf_layer.state_dict().items() if "rotary_emb." not in k}
    custom_layer.load_state_dict(custom_sd)
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_layer(x, position_ids)

    rotary_emb = DeciLMRotaryEmbedding(config).to(device=device, dtype=dtype)
    position_embeddings = rotary_emb(x, position_ids)
    custom_out = custom_layer(x, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decilm_ffn_only_layer_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    layer_idx = 1  # FFN-only

    hf_layer = _HFDeciLMDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
    custom_layer = DeciLMDecoderLayer(config, layer_idx).to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_layer(x, position_ids)

    rotary_emb = DeciLMRotaryEmbedding(config).to(device=device, dtype=dtype)
    position_embeddings = rotary_emb(x, position_ids)
    custom_out = custom_layer(x, position_embeddings)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decilm_full_model_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()

    hf_model = _HFDeciLMForCausalLM(config).to(device=device, dtype=dtype).eval()
    custom_model = DeciLMForCausalLM(config).to(device=device, dtype=dtype)

    hf_sd = hf_model.state_dict()
    # Map HF reference keys to custom model keys (model. prefix for custom)
    custom_sd = {}
    for k, v in hf_sd.items():
        if "rotary_emb." in k:
            continue
        if k.startswith("embed_tokens.") or k.startswith("layers.") or k.startswith("norm."):
            custom_sd[f"model.{k}"] = v
        else:
            custom_sd[k] = v

    custom_expected = {k for k in custom_model.state_dict().keys() if "rotary_emb." not in k}
    assert set(custom_sd.keys()) == custom_expected, (
        f"Key mismatch.\n"
        f"  Extra: {set(custom_sd.keys()) - custom_expected}\n"
        f"  Missing: {custom_expected - set(custom_sd.keys())}"
    )
    custom_model.load_state_dict(custom_sd, strict=False)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids, position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits, hf_logits, rmse_ratio_tol=0.05, msg="Full model: ")


# =============================================================================
# Export Test
# =============================================================================


def test_decilm_model_can_be_exported():
    """Test export with torch_export_to_gm, dynamic shapes, and equivalence."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = DeciLMForCausalLM(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    with torch.inference_mode():
        ref_out = model(input_ids=input_ids, position_ids=position_ids)

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

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert_rmse_close(logits, ref_out.logits, rmse_ratio_tol=0.05, msg="Export: ")

    # Test different shape
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        ref_out2 = model(input_ids=input_ids2, position_ids=position_ids2)
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()
    assert_rmse_close(logits2, ref_out2.logits, rmse_ratio_tol=0.05, msg="Export dynamic: ")
