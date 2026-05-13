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

"""Tests for Starcoder2 custom model implementation.

This module tests the custom Starcoder2 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility. Starcoder2 uses GQA with a 4096-token sliding window,
standard RoPE, vanilla GELU MLP (no gating), and LayerNorm normalization.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.starcoder2.configuration_starcoder2 import Starcoder2Config

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2ForCausalLM,
    Starcoder2MLP,
    Starcoder2RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Starcoder2Config:
    """Create a small Starcoder2 config for testing."""
    return Starcoder2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=512,
        norm_epsilon=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        sliding_window=None,  # Disable sliding window for small test (seq < 4096 anyway)
        use_bias=True,
        residual_dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF Starcoder2ForCausalLM class."""
    try:
        from transformers.models.starcoder2.modeling_starcoder2 import (
            Starcoder2ForCausalLM as HFStarcoder2ForCausalLM,
        )

        return HFStarcoder2ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Starcoder2Attention class."""
    try:
        from transformers.models.starcoder2.modeling_starcoder2 import (
            Starcoder2Attention as HFStarcoder2Attention,
        )

        return HFStarcoder2Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Starcoder2MLP class."""
    try:
        from transformers.models.starcoder2.modeling_starcoder2 import (
            Starcoder2MLP as HFStarcoder2MLP,
        )

        return HFStarcoder2MLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Starcoder2DecoderLayer class."""
    try:
        from transformers.models.starcoder2.modeling_starcoder2 import (
            Starcoder2DecoderLayer as HFStarcoder2DecoderLayer,
        )

        return HFStarcoder2DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Starcoder2RotaryEmbedding class."""
    try:
        from transformers.models.starcoder2.modeling_starcoder2 import (
            Starcoder2RotaryEmbedding as HFStarcoder2RotaryEmbedding,
        )

        return HFStarcoder2RotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_starcoder2_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation.

    Starcoder2 MLP is a vanilla 2-layer GELU MLP (c_fc -> GELU -> c_proj).
    Uses identical math, so tight tolerance is expected.
    """
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Starcoder2MLP")

    device = "cuda"
    config = _create_small_config()

    # Create HF MLP
    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights (identical key names: c_fc, c_proj)
    custom_mlp = Starcoder2MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # Identical math → tight tolerance
    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_starcoder2_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation.

    Starcoder2 uses GQA (4 Q heads, 2 KV heads in test config) with standard RoPE.
    HF uses BNSD + repeat_kv; custom uses BSND + torch_attention (GQA-native).
    """
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Starcoder2Attention or Starcoder2RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF attention
    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights (identical key names)
    custom_attn = Starcoder2Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings: [B, S, head_dim]
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings (full table, slicing happens inside attention)
    head_dim = config.hidden_size // config.num_attention_heads
    custom_rotary = Starcoder2RotaryEmbedding(
        head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_parameters["rope_theta"],
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x)  # full tables [max_seq_len, head_dim]

    # Create causal attention mask in HF additive format [B, 1, S, S]:
    # 0 for attended positions (on/below diagonal), -inf for masked (above diagonal)
    causal_mask = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
    causal_mask = causal_mask.masked_fill(
        torch.ones(S, S, device=device, dtype=torch.bool).triu(diagonal=1),
        float("-inf"),
    )

    # Run HF attention (uses pre-sliced cos/sin from HF rotary: [B, S, head_dim])
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Run custom attention (receives full tables, slices by position_ids internally)
    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
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
def test_starcoder2_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Starcoder2DecoderLayer or Starcoder2RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF decoder layer
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = Starcoder2DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Compute HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Compute custom position embeddings (full table, slicing happens inside attention)
    head_dim = config.hidden_size // config.num_attention_heads
    custom_rotary = Starcoder2RotaryEmbedding(
        head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_parameters["rope_theta"],
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x)  # full tables

    # Create causal attention mask in HF additive format [B, 1, S, S]
    causal_mask = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
    causal_mask = causal_mask.masked_fill(
        torch.ones(S, S, device=device, dtype=torch.bool).triu(diagonal=1),
        float("-inf"),
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

    # Run custom decoder layer (full tables passed; position_ids used for slicing inside)
    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
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
def test_starcoder2_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Starcoder2ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = Starcoder2ForCausalLM(config)
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
def test_starcoder2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output to eager model
    3. Dynamic shapes work correctly
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Starcoder2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get eager model output for comparison
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
    assert logits2.shape == (B2, S2, config.vocab_size), (
        f"Dynamic shape test failed: expected {(B2, S2, config.vocab_size)}, got {logits2.shape}"
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


def test_starcoder2_config_recognition():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "starcoder2"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "sliding_window")


def test_starcoder2_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = Starcoder2ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == config.num_attention_heads
    assert attn.num_kv_heads == config.num_key_value_heads
    assert attn.num_kv_heads < attn.num_heads, "Should use GQA"


def test_starcoder2_state_dict_keys():
    """Test that state_dict keys match expected HF checkpoint format."""
    config = _create_small_config()
    model = Starcoder2ForCausalLM(config)
    state_dict = model.state_dict()

    expected_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.bias",
        "model.layers.0.mlp.c_fc.weight",
        "model.layers.0.mlp.c_fc.bias",
        "model.layers.0.mlp.c_proj.weight",
        "model.layers.0.mlp.c_proj.bias",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.input_layernorm.bias",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.post_attention_layernorm.bias",
        "model.norm.weight",
        "model.norm.bias",
        "lm_head.weight",
    ]

    for key in expected_keys:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict. Got keys: {list(state_dict.keys())[:10]}..."
        )
