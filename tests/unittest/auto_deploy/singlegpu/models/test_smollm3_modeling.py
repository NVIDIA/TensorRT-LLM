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

"""Tests for SmolLM3 custom model implementation.

SmolLM3 is Llama-like with GQA, SwiGLU MLP, and RMSNorm.
Its distinguishing feature is NoPE (No Position Embedding) layers:
certain layers skip RoPE entirely, controlled by no_rope_layers config.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_smollm3 import (
    SmolLM3Attention,
    SmolLM3DecoderLayer,
    SmolLM3ForCausalLM,
    SmolLM3MLP,
    SmolLM3RMSNorm,
    SmolLM3RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


def _create_causal_mask(B: int, S: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create the additive causal mask expected by HF eager attention."""
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)


# 8 layers: layers 3 and 7 are NoPE (every 4th layer), rest use RoPE
_NUM_TEST_LAYERS = 8
_NO_ROPE_LAYERS = [int((i + 1) % 4 != 0) for i in range(_NUM_TEST_LAYERS)]


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> SmolLM3Config:
    """Create a small SmolLM3 config for testing.

    Uses 8 layers to cover both RoPE and NoPE layer types.
    """
    return SmolLM3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=_NUM_TEST_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 heads, 2 KV heads
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        no_rope_layers=_NO_ROPE_LAYERS,
        no_rope_layer_interval=4,
        use_sliding_window=False,
        sliding_window=None,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    try:
        from transformers.models.smollm3.modeling_smollm3 import (
            SmolLM3ForCausalLM as HFSmolLM3ForCausalLM,
        )

        return HFSmolLM3ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    try:
        from transformers.models.smollm3.modeling_smollm3 import (
            SmolLM3Attention as HFSmolLM3Attention,
        )

        return HFSmolLM3Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    try:
        from transformers.models.smollm3.modeling_smollm3 import SmolLM3MLP as HFSmolLM3MLP

        return HFSmolLM3MLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    try:
        from transformers.models.smollm3.modeling_smollm3 import (
            SmolLM3DecoderLayer as HFSmolLM3DecoderLayer,
        )

        return HFSmolLM3DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    try:
        from transformers.models.smollm3.modeling_smollm3 import (
            SmolLM3RotaryEmbedding as HFSmolLM3RotaryEmbedding,
        )

        return HFSmolLM3RotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_smollm3_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces equivalent output to HF implementation."""
    try:
        from transformers.models.smollm3.modeling_smollm3 import SmolLM3RMSNorm as HFSmolLM3RMSNorm
    except ImportError:
        pytest.skip("transformers doesn't have SmolLM3RMSNorm")

    device = "cuda"
    config = _create_small_config()

    hf_norm = HFSmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
def test_smollm3_mlp_equivalence(B, S, dtype):
    """Test MLP produces equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have SmolLM3MLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = SmolLM3MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 3])  # layer 0 uses RoPE, layer 3 is NoPE
@torch.no_grad()
def test_smollm3_attention_equivalence(B, S, dtype, layer_idx):
    """Test Attention produces equivalent output to HF for both RoPE and NoPE layers."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have SmolLM3Attention or SmolLM3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=layer_idx)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = SmolLM3Attention(config, layer_idx=layer_idx)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = SmolLM3RotaryEmbedding(config)
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

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 3])  # RoPE layer and NoPE layer
@torch.no_grad()
def test_smollm3_decoder_layer_equivalence(B, S, dtype, layer_idx):
    """Test decoder layer produces equivalent output to HF for both layer types."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have SmolLM3DecoderLayer or SmolLM3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = SmolLM3DecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (pre-sliced by position_ids)
    custom_rotary = SmolLM3RotaryEmbedding(config)
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

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_smollm3_full_model_equivalence(B, S, dtype, device):
    """Test full model produces equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have SmolLM3ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = SmolLM3ForCausalLM(config)
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
def test_smollm3_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = SmolLM3ForCausalLM(config)
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


def test_smollm3_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "smollm3"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "no_rope_layers")


def test_smollm3_nope_layer_pattern():
    """Test that NoPE layers are correctly configured."""
    config = _create_small_config()
    model = SmolLM3ForCausalLM(config)

    for i in range(_NUM_TEST_LAYERS):
        attn = model.model.layers[i].self_attn
        expected_use_rope = _NO_ROPE_LAYERS[i]
        assert attn.use_rope == expected_use_rope, (
            f"Layer {i}: expected use_rope={expected_use_rope}, got {attn.use_rope}"
        )


def test_smollm3_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = SmolLM3ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_smollm3_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = SmolLM3ForCausalLM(config)
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
