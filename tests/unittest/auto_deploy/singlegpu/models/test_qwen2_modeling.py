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

"""Tests for Qwen2 custom model implementation.

This module tests the custom Qwen2 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_rmsnorm) for export compatibility. Qwen2 uses GQA with SwiGLU MLP
and RMSNorm, similar to Llama but with bias=True on Q/K/V projections.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Qwen2Config:
    """Create a small Qwen2 config for testing."""
    return Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 heads, 2 KV heads
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF Qwen2ForCausalLM class."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as HFQwen2ForCausalLM

        return HFQwen2ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Qwen2Attention class."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as HFQwen2Attention

        return HFQwen2Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Qwen2MLP class."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP

        return HFQwen2MLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Qwen2DecoderLayer class."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2DecoderLayer as HFQwen2DecoderLayer,
        )

        return HFQwen2DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Qwen2RotaryEmbedding class."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2RotaryEmbedding as HFQwen2RotaryEmbedding,
        )

        return HFQwen2RotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen2_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm layer produces numerically equivalent output to HF implementation."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as HFQwen2RMSNorm
    except ImportError:
        pytest.skip("transformers doesn't have Qwen2RMSNorm")

    device = "cuda"
    config = _create_small_config()

    hf_norm = HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
def test_qwen2_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Qwen2MLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Qwen2MLP(config)
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
def test_qwen2_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Qwen2Attention or Qwen2RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Qwen2Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = Qwen2RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Build causal mask so HF attention uses causal masking (matching our is_causal=True)
    causal_mask = (
        torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, 1, S, S)
    )

    # Run HF attention
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Run custom attention
    custom_out = custom_attn(
        hidden_states=x,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen2_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Qwen2DecoderLayer or Qwen2RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Qwen2DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = Qwen2RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Build causal mask so HF attention uses causal masking (matching our is_causal=True)
    causal_mask = (
        torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, 1, S, S)
    )

    # Run HF decoder layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Run custom decoder layer
    custom_out = custom_layer(
        hidden_states=x,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_qwen2_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Qwen2ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = Qwen2ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

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
def test_qwen2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Qwen2ForCausalLM(config)
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


def test_qwen2_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "qwen2"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")


def test_qwen2_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = Qwen2ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_qwen2_attention_has_qkv_bias():
    """Test that Q/K/V projections have bias=True and O projection has bias=False."""
    config = _create_small_config()
    model = Qwen2ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is not None, "Q projection should have bias"
    assert attn.k_proj.bias is not None, "K projection should have bias"
    assert attn.v_proj.bias is not None, "V projection should have bias"
    assert attn.o_proj.bias is None, "O projection should NOT have bias"


def test_qwen2_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = Qwen2ForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
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

    # Verify O projection does NOT have bias
    assert "model.layers.0.self_attn.o_proj.bias" not in state_dict, (
        "O projection should NOT have bias in state_dict"
    )
