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

"""Tests for OpenELM custom model implementation.

Compares the custom AD model against an inline HF reference implementation.
The HF config is loaded via trust_remote_code=True; for unit tests we create
a small config using the same class.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_openelm import (
    OpenELMAttention,
    OpenELMDecoderLayer,
    OpenELMFeedForwardNetwork,
    OpenELMForCausalLM,
    OpenELMRotaryEmbedding,
    _make_divisible,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Small test config
# =============================================================================


class _TestOpenELMConfig(PretrainedConfig):
    """Standalone test config matching OpenELM's attribute interface."""

    model_type = "openelm"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()


def _create_small_config():
    """Create a small OpenELM config for testing (no network access needed)."""
    return _TestOpenELMConfig(
        vocab_size=1000,
        max_context_length=128,
        num_transformer_layers=3,
        model_dim=64,
        head_dim=16,
        num_gqa_groups=2,
        num_query_heads=[2, 3, 4],
        num_kv_heads=[1, 1, 2],
        ffn_multipliers=[0.5, 2.0, 4.0],
        ffn_with_glu=True,
        ffn_dim_divisor=16,
        activation_fn_name="swish",
        normalize_qk_projections=True,
        share_input_output_layers=True,
        rope_freq_constant=10000,
        rope_max_length=256,
        initializer_range=0.02,
    )


# =============================================================================
# HF Reference Classes (standalone)
# =============================================================================


class _HFRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        output = (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return output * self.weight


class _HFRotaryEmbedding(nn.Module):
    def __init__(self, model_dim, max_seq_length, freq_constant=10000):
        super().__init__()
        inv_freq = 1.0 / (
            freq_constant ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        pos_index = torch.arange(max_seq_length, dtype=torch.float32)
        pos_index_theta = torch.einsum("i,j->ij", pos_index, inv_freq)
        emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k):
        # q, k: [B, N, S, D] (bnsd layout)
        key_len = k.shape[2]
        query_len = q.shape[2]
        cos = self.cos_cached[:, :, key_len - query_len : key_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, key_len - query_len : key_len, :].to(q.dtype)

        def _rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_out = (q.float() * cos) + (_rotate_half(q.float()) * sin)
        k_out = (k.float() * cos) + (_rotate_half(k.float()) * sin)
        return q_out.type_as(q), k_out.type_as(k)


class _HFAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_query_heads[layer_idx]
        self.num_k_heads = config.num_kv_heads[layer_idx]
        self.num_v_heads = config.num_kv_heads[layer_idx]
        self.num_groups = self.num_q_heads // self.num_k_heads

        self.qkv_proj = nn.Linear(
            config.model_dim,
            (self.num_q_heads + self.num_k_heads + self.num_v_heads) * self.head_dim,
            bias=False,
        )
        self.q_norm = _HFRMSNorm(config.head_dim) if config.normalize_qk_projections else None
        self.k_norm = _HFRMSNorm(config.head_dim) if config.normalize_qk_projections else None
        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.model_dim, bias=False)
        self.pos_embedding = _HFRotaryEmbedding(
            config.head_dim, config.rope_max_length, config.rope_freq_constant
        )

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            bsz, seq_len, self.num_q_heads + self.num_k_heads + self.num_v_heads, self.head_dim
        )
        qkv = qkv.transpose(1, 2)  # [B, N_total, S, D]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            values = values.repeat_interleave(self.num_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, seq_len, self.num_q_heads * self.head_dim)
        )
        return self.out_proj(attn_output)


class _HFFFN(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            _make_divisible(ffn_multiplier * config.model_dim, divisor=config.ffn_dim_divisor)
        )
        self.ffn_with_glu = config.ffn_with_glu
        if config.ffn_with_glu:
            self.proj_1 = nn.Linear(config.model_dim, 2 * intermediate_dim, bias=False)
        else:
            self.proj_1 = nn.Linear(config.model_dim, intermediate_dim, bias=False)
        self.proj_2 = nn.Linear(intermediate_dim, config.model_dim, bias=False)
        self.act = F.silu

    def forward(self, x):
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            return self.proj_2(self.act(y_1) * y_2)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class _HFDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = _HFAttention(config, layer_idx)
        self.ffn = _HFFFN(config, layer_idx)
        self.attn_norm = _HFRMSNorm(config.model_dim)
        self.ffn_norm = _HFRMSNorm(config.model_dim)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _HFOpenELMForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList(
            [_HFDecoderLayer(config, idx) for idx in range(config.num_transformer_layers)]
        )
        self.norm = _HFRMSNorm(config.model_dim)
        self.share_input_output_layers = config.share_input_output_layers
        if not self.share_input_output_layers:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.token_embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        if self.share_input_output_layers:
            return F.linear(hidden_states, self.token_embeddings.weight).float()
        return self.lm_head(hidden_states).float()


# =============================================================================
# Structure Tests
# =============================================================================


@torch.no_grad()
def test_openelm_layer_structure_and_forward():
    config = _create_small_config()
    model = OpenELMForCausalLM(config).eval()

    assert len(model.transformer.layers) == 3
    assert model.lm_head is None  # shared embeddings
    assert hasattr(model.transformer, "token_embeddings")

    # Check per-layer head counts vary
    heads = [layer.attn.num_q_heads for layer in model.transformer.layers]
    assert len(set(heads)) > 1 or len(heads) == 1

    # Verify forward produces valid output
    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    out = model(input_ids=input_ids, position_ids=position_ids)
    assert out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(out.logits).all()


@torch.no_grad()
def test_openelm_weight_keys_match_checkpoint():
    config = _create_small_config()
    model = OpenELMForCausalLM(config).eval()
    keys = set(model.state_dict().keys())

    # Check fused QKV proj
    assert "transformer.layers.0.attn.qkv_proj.weight" in keys
    assert "transformer.layers.0.attn.q_norm.weight" in keys
    assert "transformer.layers.0.attn.k_norm.weight" in keys
    assert "transformer.layers.0.ffn.proj_1.weight" in keys
    assert "transformer.layers.0.ffn.proj_2.weight" in keys
    assert "lm_head.weight" not in keys  # shared embeddings
    assert "transformer.token_embeddings.weight" in keys

    # Verify forward produces valid output with these weights
    B, S = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    out = model(input_ids=input_ids, position_ids=position_ids)
    assert out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(out.logits).all()


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_openelm_ffn_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()

    hf_ffn = _HFFFN(config, 0).to(device=device, dtype=dtype).eval()
    custom_ffn = OpenELMFeedForwardNetwork(config, 0).to(device=device, dtype=dtype)
    custom_ffn.load_state_dict(hf_ffn.state_dict())
    custom_ffn.eval()

    x = torch.randn(B, S, config.model_dim, device=device, dtype=dtype)
    torch.testing.assert_close(custom_ffn(x), hf_ffn(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_openelm_attention_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    layer_idx = 0

    hf_attn = _HFAttention(config, layer_idx).to(device=device, dtype=dtype).eval()
    custom_attn = OpenELMAttention(config, layer_idx).to(device=device, dtype=dtype)
    # Transfer weights (exclude pos_embedding buffers from HF)
    custom_sd = {
        k: v for k, v in hf_attn.state_dict().items() if not k.startswith("pos_embedding.")
    }
    custom_attn.load_state_dict(custom_sd)
    custom_attn.eval()

    x = torch.randn(B, S, config.model_dim, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_attn(x)
    rotary_emb = OpenELMRotaryEmbedding(
        config.head_dim, config.rope_max_length, config.rope_freq_constant
    ).to(device=device, dtype=dtype)
    position_embeddings = rotary_emb(x, position_ids)
    custom_out = custom_attn(x, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_openelm_decoder_layer_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()
    layer_idx = 0

    hf_layer = _HFDecoderLayer(config, layer_idx).to(device=device, dtype=dtype).eval()
    custom_layer = OpenELMDecoderLayer(config, layer_idx).to(device=device, dtype=dtype)
    custom_sd = {k: v for k, v in hf_layer.state_dict().items() if "pos_embedding." not in k}
    custom_layer.load_state_dict(custom_sd)
    custom_layer.eval()

    x = torch.randn(B, S, config.model_dim, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_layer(x)
    rotary_emb = OpenELMRotaryEmbedding(
        config.head_dim, config.rope_max_length, config.rope_freq_constant
    ).to(device=device, dtype=dtype)
    position_embeddings = rotary_emb(x, position_ids)
    custom_out = custom_layer(x, position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_openelm_full_model_equivalence(B, S, dtype):
    device = "cpu"
    config = _create_small_config()

    hf_model = _HFOpenELMForCausalLM(config).to(device=device, dtype=dtype).eval()
    custom_model = OpenELMForCausalLM(config).to(device=device, dtype=dtype)

    # Transfer weights: HF reference has pos_embedding buffers per attention layer
    hf_sd = hf_model.state_dict()
    custom_sd = {}
    for k, v in hf_sd.items():
        if "pos_embedding." in k:
            continue
        custom_sd[f"transformer.{k}"] = v

    custom_expected = {k for k in custom_model.state_dict().keys() if "rotary_emb." not in k}
    assert set(custom_sd.keys()) == custom_expected, (
        f"Key mismatch.\n  Extra: {set(custom_sd.keys()) - custom_expected}\n"
        f"  Missing: {custom_expected - set(custom_sd.keys())}"
    )
    custom_model.load_state_dict(custom_sd, strict=False)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits, hf_logits, rmse_ratio_tol=0.05, msg="Full model: ")


# =============================================================================
# Export Test
# =============================================================================


def test_openelm_model_can_be_exported():
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = OpenELMForCausalLM(config).to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    with torch.inference_mode():
        ref_out = model(input_ids=input_ids, position_ids=position_ids)

    dynamic_shapes = {
        "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
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
