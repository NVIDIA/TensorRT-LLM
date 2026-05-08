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

"""Tests for Seed-OSS custom model implementation.

This module tests the custom Seed-OSS model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility. Seed-OSS uses GQA with standard RoPE and
attention_bias=True on Q/K/V projections.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.seed_oss.configuration_seed_oss import SeedOssConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_seed_oss import (
    SeedOssAttention,
    SeedOssDecoderLayer,
    SeedOssForCausalLM,
    SeedOssMLP,
    SeedOssRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> SeedOssConfig:
    """Create a small Seed-OSS config for testing."""
    return SeedOssConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 heads, 2 KV heads
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=True,
        attention_out_bias=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF SeedOssForCausalLM class."""
    try:
        from transformers.models.seed_oss.modeling_seed_oss import (
            SeedOssForCausalLM as HFSeedOssForCausalLM,
        )

        return HFSeedOssForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF SeedOssAttention class."""
    try:
        from transformers.models.seed_oss.modeling_seed_oss import (
            SeedOssAttention as HFSeedOssAttention,
        )

        return HFSeedOssAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF SeedOssMLP class."""
    try:
        from transformers.models.seed_oss.modeling_seed_oss import SeedOssMLP as HFSeedOssMLP

        return HFSeedOssMLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF SeedOssDecoderLayer class."""
    try:
        from transformers.models.seed_oss.modeling_seed_oss import (
            SeedOssDecoderLayer as HFSeedOssDecoderLayer,
        )

        return HFSeedOssDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF SeedOssRotaryEmbedding class."""
    try:
        from transformers.models.seed_oss.modeling_seed_oss import (
            SeedOssRotaryEmbedding as HFSeedOssRotaryEmbedding,
        )

        return HFSeedOssRotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_seed_oss_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have SeedOssMLP")

    device = "cuda"
    config = _create_small_config()

    # Create HF MLP
    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = SeedOssMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    # Create input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Run both
    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # MLP uses identical math, so use tight tolerance
    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_seed_oss_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have SeedOssAttention or SeedOssRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    # Force eager attention for HF model
    config._attn_implementation = "eager"

    # Create HF attention
    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights
    custom_attn = SeedOssAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    # Create input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)  # [B, S, head_dim]

    # Compute custom position embeddings (pre-sliced by position_ids)
    custom_rotary = SeedOssRotaryEmbedding(
        config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Construct causal mask for HF eager attention (which does NOT apply causal masking
    # when attention_mask=None — unlike our custom model which always uses is_causal=True)
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    # Run HF attention with explicit causal mask
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Run custom attention (position_embeddings are pre-sliced)
    custom_out = custom_attn(
        hidden_states=x,
        position_embeddings=(custom_cos, custom_sin),
    )

    # Attention uses custom ops, allow wider tolerance
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_seed_oss_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have SeedOssDecoderLayer or SeedOssRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF decoder layer
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = SeedOssDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    # Create input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings (pre-sliced by position_ids)
    custom_rotary = SeedOssRotaryEmbedding(
        config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Construct causal mask for HF eager attention
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    # Run HF decoder layer (returns tensor directly, but handle tuple for safety)
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Run custom decoder layer (position_embeddings are pre-sliced)
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
def test_seed_oss_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have SeedOssForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = SeedOssForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Run both
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
def test_seed_oss_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output to eager model
    3. Dynamic shapes work correctly
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = SeedOssForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get eager model output for comparison
    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    # Define dynamic shapes
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    # Export the model
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    # Move graph module to device
    move_to_device(gm, device)

    # Verify the exported model produces numerically equivalent output
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


def test_seed_oss_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "seed_oss"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "head_dim")
    assert hasattr(config, "attention_bias")
    assert hasattr(config, "attention_out_bias")


def test_seed_oss_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = SeedOssForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"

    # Verify Q/K projections have bias (Seed-OSS specific: attention_bias=True)
    assert attn.q_proj.bias is not None, "Q projection should have bias"
    assert attn.k_proj.bias is not None, "K projection should have bias"
    assert attn.v_proj.bias is not None, "V projection should have bias"
    # O projection has no bias (attention_out_bias=False)
    assert attn.o_proj.bias is None, "O projection should not have bias"


def test_seed_oss_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = SeedOssForCausalLM(config)
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

    # Verify O projection does NOT have bias key
    assert "model.layers.0.self_attn.o_proj.bias" not in state_dict, (
        "O projection should not have bias key in state_dict"
    )
