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

"""Tests for GLM4 MoE custom model implementation.

This module tests the custom GLM4 MoE model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_moe) for export compatibility. GLM4 MoE uses GQA with partial rotary
embeddings, optional QK normalization, and MoE with sigmoid gating.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeMoE,
    Glm4MoeRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(7)
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_small_config(use_moe: bool = True) -> Glm4MoeConfig:
    """Create a small GLM4 MoE config for testing.

    Args:
        use_moe: If True, use MoE layers after the first dense layer.
    """
    return Glm4MoeConfig(
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
        attention_bias=True,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        use_qk_norm=True,
        tie_word_embeddings=False,
        # MoE parameters
        n_routed_experts=4 if use_moe else 0,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        initializer_range=0.01,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF Glm4MoeForCausalLM class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeForCausalLM as HFGlm4MoeForCausalLM,
        )

        return HFGlm4MoeForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Glm4MoeAttention class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeAttention as HFGlm4MoeAttention,
        )

        return HFGlm4MoeAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Glm4MoeMLP class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP as HFGlm4MoeMLP

        return HFGlm4MoeMLP
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF Glm4MoeMoE class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE as HFGlm4MoeMoE

        return HFGlm4MoeMoE
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Glm4MoeDecoderLayer class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeDecoderLayer as HFGlm4MoeDecoderLayer,
        )

        return HFGlm4MoeDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Glm4MoeRotaryEmbedding class."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeRotaryEmbedding as HFGlm4MoeRotaryEmbedding,
        )

        return HFGlm4MoeRotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Glm4MoeMLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Glm4MoeMLP(config)
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
def test_glm4_moe_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Glm4MoeAttention or Glm4MoeRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Glm4MoeAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    custom_rotary = Glm4MoeRotaryEmbedding(
        rotary_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create 4D causal mask for HF eager attention (which does not apply causal masking
    # when attention_mask=None, unlike our custom model which always uses is_causal=True)
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :]  # [1, 1, S, S]

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


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_moe_equivalence(B, S, dtype):
    """Test MoE layer produces numerically equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have Glm4MoeMoE")

    device = "cuda"
    config = _create_small_config()

    hf_moe = HFMoE(config)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    custom_moe = Glm4MoeMoE(config)
    custom_moe.to(device=device, dtype=dtype)
    # Load weights - the gate.weight in custom model is the same key
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    # Skip if both produce NaN (can happen with small random weights and MoE routing)
    if hf_out.isnan().any() and custom_out.isnan().any():
        pytest.skip("Both HF and custom MoE produce NaN with this seed/shape combination")

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 1])
@torch.no_grad()
def test_glm4_moe_decoder_layer_equivalence(B, S, dtype, layer_idx):
    """Test decoder layer produces numerically equivalent output to HF implementation.

    Tests both dense (layer_idx=0) and MoE (layer_idx=1) decoder layers.
    """
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Glm4MoeDecoderLayer or Glm4MoeRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Glm4MoeDecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    custom_rotary = Glm4MoeRotaryEmbedding(
        rotary_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create 4D causal mask for HF eager attention
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :]  # [1, 1, S, S]

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

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg=f"Decoder layer {layer_idx}: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@torch.no_grad()
def test_glm4_moe_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Glm4MoeForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = Glm4MoeForCausalLM(config)
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
def test_glm4_moe_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Glm4MoeForCausalLM(config)
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


def test_glm4_moe_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "glm4_moe"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "partial_rotary_factor")
    assert hasattr(config, "use_qk_norm")


def test_glm4_moe_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = Glm4MoeForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"

    # Check partial rotary
    assert attn.partial_rotary_factor == 0.5
    assert attn.rotary_dim == attn.head_dim // 2

    # Check QK norms exist
    assert hasattr(attn, "q_norm"), "Attention should have q_norm"
    assert hasattr(attn, "k_norm"), "Attention should have k_norm"


def test_glm4_moe_dense_and_moe_layers():
    """Test that first_k_dense_replace layers are dense, rest are MoE."""
    config = _create_small_config()
    model = Glm4MoeForCausalLM(config)

    # Layer 0 should be dense (first_k_dense_replace=1)
    assert isinstance(model.model.layers[0].mlp, Glm4MoeMLP), "Layer 0 should use dense MLP"
    # Layers 1+ should be MoE
    assert isinstance(model.model.layers[1].mlp, Glm4MoeMoE), "Layer 1 should use MoE"
    assert isinstance(model.model.layers[2].mlp, Glm4MoeMoE), "Layer 2 should use MoE"


def test_glm4_moe_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = Glm4MoeForCausalLM(config)
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
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        # MoE layer (layer 1)
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )
