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

"""Tests for Gemma custom model implementation.

This module tests the custom Gemma model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_rmsnorm) for export compatibility. Gemma uses MHA with gelu_pytorch_tanh
MLP, RMSNorm with (1+weight) scaling, RoPE, and embedding normalization.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.gemma.configuration_gemma import GemmaConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma import (
    GemmaADAttention,
    GemmaADDecoderLayer,
    GemmaADForCausalLM,
    GemmaADMLP,
    GemmaADRMSNorm,
    GemmaADRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> GemmaConfig:
    """Create a small Gemma config for testing.

    Gemma has head_dim=256 in the real model, but we use 16 for tests.
    Key Gemma specifics: attention_bias=False, hidden_act=gelu_pytorch_tanh,
    tie_word_embeddings=True, and the (1+weight) RMSNorm variant.
    """
    return GemmaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,  # MHA: same as num_attention_heads
        head_dim=16,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF GemmaForCausalLM class."""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM as HFGemmaForCausalLM

        return HFGemmaForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF GemmaAttention class."""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaAttention as HFGemmaAttention

        return HFGemmaAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF GemmaMLP class."""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaMLP as HFGemmaMLP

        return HFGemmaMLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF GemmaDecoderLayer class."""
    try:
        from transformers.models.gemma.modeling_gemma import (
            GemmaDecoderLayer as HFGemmaDecoderLayer,
        )

        return HFGemmaDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF GemmaRotaryEmbedding class."""
    try:
        from transformers.models.gemma.modeling_gemma import (
            GemmaRotaryEmbedding as HFGemmaRotaryEmbedding,
        )

        return HFGemmaRotaryEmbedding
    except ImportError:
        return None


def _get_hf_rmsnorm_class():
    """Get the HF GemmaRMSNorm class."""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm

        return HFGemmaRMSNorm
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


def _convert_hf_norm_weights(hf_state_dict):
    """Convert HF Gemma norm weights to AD format by adding 1.0.

    HF Gemma stores norm weights as bias around zero and applies (1+weight).
    Our custom model absorbs the +1.0 into the weight directly.
    """
    converted = {}
    for key, value in hf_state_dict.items():
        if key.endswith("layernorm.weight") or key.endswith("norm.weight"):
            converted[key] = value + 1.0
        else:
            converted[key] = value
    return converted


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_gemma_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm with (1+weight) scaling produces equivalent output to HF."""
    HFRMSNorm = _get_hf_rmsnorm_class()
    if HFRMSNorm is None:
        pytest.skip("transformers doesn't have GemmaRMSNorm")

    device = "cuda"
    config = _create_small_config()

    hf_norm = HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = GemmaADRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    custom_norm.to(device=device, dtype=dtype)
    # HF stores weight as bias (0-centered); custom stores weight + 1.0
    sd = hf_norm.state_dict()
    sd["weight"] = sd["weight"] + 1.0
    custom_norm.load_state_dict(sd)
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_gemma_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have GemmaMLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = GemmaADMLP(config)
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
def test_gemma_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have GemmaAttention or GemmaRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = GemmaADAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = GemmaADRotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Run HF attention
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=None,
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
def test_gemma_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have GemmaDecoderLayer or GemmaRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = GemmaADDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(_convert_hf_norm_weights(hf_layer.state_dict()))
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = GemmaADRotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Run HF decoder layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=None,
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
def test_gemma_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have GemmaForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = GemmaADForCausalLM(config)
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
def test_gemma_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = GemmaADForCausalLM(config)
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


def test_gemma_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "gemma"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "head_dim")


def test_gemma_mha_structure():
    """Test that attention uses MHA (same Q and KV head count)."""
    config = _create_small_config()
    model = GemmaADForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 4, f"Expected 4 KV heads (MHA), got {attn.num_kv_heads}"


def test_gemma_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = GemmaADForCausalLM(config)
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


def test_gemma_embedding_normalization():
    """Test that Gemma applies sqrt(hidden_size) normalization to embeddings."""
    config = _create_small_config()
    model = GemmaADForCausalLM(config)

    # Check normalizer value
    expected_normalizer = config.hidden_size**0.5
    assert model.model.normalizer == expected_normalizer, (
        f"Expected normalizer {expected_normalizer}, got {model.model.normalizer}"
    )


def test_gemma_tied_weights():
    """Test that Gemma properly supports tied embed/lm_head weights."""
    config = _create_small_config()
    assert config.tie_word_embeddings is True
    model = GemmaADForCausalLM(config)
    assert "lm_head.weight" in model._tied_weights_keys
