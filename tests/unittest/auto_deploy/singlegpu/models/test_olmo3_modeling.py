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

"""Tests for OLMo-3 custom model implementation.

This module tests the custom OLMo-3 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_rmsnorm) for export compatibility.

OLMo-3 key features:
* Post-norm residual pattern (RMSNorm after attention/MLP, not before)
* QK normalization on full projection (not per-head)
* Two separate RoPE embeddings: "default" for sliding_attention, "yarn" for full_attention
* Mixed attention types per layer via config.layer_types
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3RotaryEmbedding,
    Olmo3YarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


def _create_causal_mask(B: int, S: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create the additive causal mask expected by HF eager attention."""
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Olmo3Config:
    """Create a small OLMo-3 config for testing.

    Uses a minimal layer_types pattern with both sliding and full attention
    to cover both code paths. Includes GQA (4 Q heads, 2 KV heads).
    """
    return Olmo3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 heads, 2 KV heads
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.2,
        },
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF Olmo3ForCausalLM class."""
    try:
        from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM as HFOlmo3ForCausalLM

        return HFOlmo3ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Olmo3Attention class."""
    try:
        from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention as HFOlmo3Attention

        return HFOlmo3Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Olmo3MLP class."""
    try:
        from transformers.models.olmo3.modeling_olmo3 import Olmo3MLP as HFOlmo3MLP

        return HFOlmo3MLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Olmo3DecoderLayer class."""
    try:
        from transformers.models.olmo3.modeling_olmo3 import (
            Olmo3DecoderLayer as HFOlmo3DecoderLayer,
        )

        return HFOlmo3DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Olmo3RotaryEmbedding class."""
    try:
        from transformers.models.olmo3.modeling_olmo3 import (
            Olmo3RotaryEmbedding as HFOlmo3RotaryEmbedding,
        )

        return HFOlmo3RotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_olmo3_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Olmo3MLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Olmo3MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # MLP uses identical math, so use tight tolerance
    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 3])  # sliding (idx=0) and full (idx=3)
@torch.no_grad()
def test_olmo3_attention_equivalence(B, S, dtype, layer_idx):
    """Test Attention layer produces numerically equivalent output to HF implementation.

    Tests both sliding_attention (layer_idx=0) and full_attention (layer_idx=3) types.
    """
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Olmo3Attention or Olmo3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    attention_type = config.layer_types[layer_idx]

    # Create HF attention
    hf_attn = HFAttention(config, layer_idx=layer_idx)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights
    custom_attn = Olmo3Attention(config, layer_idx=layer_idx)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings (creates the right type based on attention_type)
    if attention_type == "sliding_attention":
        hf_rotary = HFRotary(config=config, device=device, rope_type="default")
    else:
        hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings
    if attention_type == "sliding_attention":
        custom_rotary = Olmo3RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    else:
        rope_scaling = config.rope_scaling
        custom_rotary = Olmo3YarnRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=rope_scaling.get(
                "original_max_position_embeddings", 64
            ),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
            attention_factor=rope_scaling.get("attention_factor", 1.0),
        )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # HF eager attention is non-causal when attention_mask=None, but the
    # AutoDeploy custom attention models prefill-only causal attention.
    causal_mask = _create_causal_mask(B, S, device, dtype)

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

    assert_rmse_close(
        custom_out, hf_out, rmse_ratio_tol=0.10, msg=f"Attention ({attention_type}): "
    )


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 3])  # sliding and full
@torch.no_grad()
def test_olmo3_decoder_layer_equivalence(B, S, dtype, layer_idx):
    """Test decoder layer produces numerically equivalent output to HF implementation.

    Tests both sliding_attention and full_attention decoder layers.
    """
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Olmo3DecoderLayer or Olmo3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    attention_type = config.layer_types[layer_idx]

    # Create HF decoder layer
    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = Olmo3DecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    if attention_type == "sliding_attention":
        hf_rotary = HFRotary(config=config, device=device, rope_type="default")
    else:
        hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings
    if attention_type == "sliding_attention":
        custom_rotary = Olmo3RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    else:
        rope_scaling = config.rope_scaling
        custom_rotary = Olmo3YarnRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=rope_scaling.get(
                "original_max_position_embeddings", 64
            ),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
            attention_factor=rope_scaling.get("attention_factor", 1.0),
        )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Match the custom decoder layer's causal prefill behavior.
    causal_mask = _create_causal_mask(B, S, device, dtype)

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

    assert_rmse_close(
        custom_out, hf_out, rmse_ratio_tol=0.05, msg=f"Decoder layer ({attention_type}): "
    )


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_olmo3_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Olmo3ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = Olmo3ForCausalLM(config)
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
def test_olmo3_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output to eager model
    3. Dynamic shapes work correctly
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Olmo3ForCausalLM(config)
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
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test with different input shape to verify dynamic shapes work
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


def test_olmo3_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "olmo3"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "layer_types")


def test_olmo3_post_norm_structure():
    """Test that OLMo-3 uses post-norm (not pre-norm)."""
    config = _create_small_config()
    model = Olmo3ForCausalLM(config)

    layer = model.model.layers[0]
    # OLMo-3 has post_attention_layernorm and post_feedforward_layernorm
    assert hasattr(layer, "post_attention_layernorm"), "Should have post_attention_layernorm"
    assert hasattr(layer, "post_feedforward_layernorm"), "Should have post_feedforward_layernorm"
    # OLMo-3 does NOT have input_layernorm (pre-norm)
    assert not hasattr(layer, "input_layernorm"), (
        "Should NOT have input_layernorm (post-norm model)"
    )


def test_olmo3_qk_norm_structure():
    """Test that attention uses QK normalization on full projection."""
    config = _create_small_config()
    model = Olmo3ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert hasattr(attn, "q_norm"), "Attention should have q_norm"
    assert hasattr(attn, "k_norm"), "Attention should have k_norm"

    # Q norm weight size should be num_heads * head_dim (full projection)
    head_dim = config.hidden_size // config.num_attention_heads
    assert attn.q_norm.weight.shape[0] == config.num_attention_heads * head_dim
    assert attn.k_norm.weight.shape[0] == config.num_key_value_heads * head_dim


def test_olmo3_mixed_attention_types():
    """Test that model correctly assigns sliding/full attention types."""
    config = _create_small_config()
    model = Olmo3ForCausalLM(config)

    for idx, layer in enumerate(model.model.layers):
        expected_type = config.layer_types[idx]
        actual_type = layer.self_attn.attention_type
        assert actual_type == expected_type, (
            f"Layer {idx}: expected {expected_type}, got {actual_type}"
        )
        if expected_type == "sliding_attention":
            assert layer.self_attn.sliding_window == config.sliding_window
        else:
            assert layer.self_attn.sliding_window is None


def test_olmo3_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = Olmo3ForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )
