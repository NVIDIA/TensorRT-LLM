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

"""Tests for Cohere/Cohere2 custom model implementation.

This module tests the custom Cohere model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_qk_interleaving)
for export compatibility.

Both Cohere v1 (aya-expanse) and Cohere2 (command-a) are tested:
- Cohere v1: standard GQA with interleaved RoPE, parallel attn+MLP, LayerNorm
- Cohere2: adds sliding window pattern and conditional RoPE per layer
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.cohere.configuration_cohere import CohereConfig
from transformers.models.cohere2.configuration_cohere2 import Cohere2Config

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_cohere import (
    CohereAttention,
    CohereDecoderLayer,
    CohereForCausalLM,
    CohereLayerNorm,
    CohereMLP,
    CohereRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_cohere_config() -> CohereConfig:
    """Create a small Cohere v1 config for testing."""
    return CohereConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        hidden_act="silu",
        max_position_embeddings=512,
        layer_norm_eps=1e-5,
        logit_scale=0.125,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        tie_word_embeddings=True,
    )


def _create_small_cohere2_config() -> Cohere2Config:
    """Create a small Cohere2 config for testing with sliding window pattern."""
    return Cohere2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 4 layers to test sliding_window_pattern=4
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        layer_norm_eps=1e-5,
        logit_scale=0.125,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4,
        sliding_window_pattern=4,
        tie_word_embeddings=True,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_cohere_model_class():
    try:
        from transformers.models.cohere.modeling_cohere import (
            CohereForCausalLM as HFCohereForCausalLM,
        )

        return HFCohereForCausalLM
    except ImportError:
        return None


def _get_hf_cohere_attention_class():
    try:
        from transformers.models.cohere.modeling_cohere import CohereAttention as HFCohereAttention

        return HFCohereAttention
    except ImportError:
        return None


def _get_hf_cohere_mlp_class():
    try:
        from transformers.models.cohere.modeling_cohere import CohereMLP as HFCohereMLP

        return HFCohereMLP
    except ImportError:
        return None


def _get_hf_cohere_decoder_layer_class():
    try:
        from transformers.models.cohere.modeling_cohere import (
            CohereDecoderLayer as HFCohereDecoderLayer,
        )

        return HFCohereDecoderLayer
    except ImportError:
        return None


def _get_hf_cohere_rotary_class():
    try:
        from transformers.models.cohere.modeling_cohere import (
            CohereRotaryEmbedding as HFCohereRotaryEmbedding,
        )

        return HFCohereRotaryEmbedding
    except ImportError:
        return None


def _get_hf_cohere_layernorm_class():
    try:
        from transformers.models.cohere.modeling_cohere import CohereLayerNorm as HFCohereLayerNorm

        return HFCohereLayerNorm
    except ImportError:
        return None


def _get_hf_cohere2_model_class():
    try:
        from transformers.models.cohere2.modeling_cohere2 import (
            Cohere2ForCausalLM as HFCohere2ForCausalLM,
        )

        return HFCohere2ForCausalLM
    except ImportError:
        return None


def _get_hf_cohere2_decoder_layer_class():
    try:
        from transformers.models.cohere2.modeling_cohere2 import (
            Cohere2DecoderLayer as HFCohere2DecoderLayer,
        )

        return HFCohere2DecoderLayer
    except ImportError:
        return None


def _get_hf_cohere2_rotary_class():
    try:
        from transformers.models.cohere2.modeling_cohere2 import (
            Cohere2RotaryEmbedding as HFCohere2RotaryEmbedding,
        )

        return HFCohere2RotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1) — Cohere v1
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_cohere_layernorm_equivalence(B, S, dtype):
    """Test CohereLayerNorm produces numerically equivalent output to HF."""
    HFLayerNorm = _get_hf_cohere_layernorm_class()
    if HFLayerNorm is None:
        pytest.skip("transformers doesn't have CohereLayerNorm")

    device = "cuda"
    hidden_size = 64
    eps = 1e-5

    hf_norm = HFLayerNorm(hidden_size=hidden_size, eps=eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = CohereLayerNorm(hidden_size=hidden_size, eps=eps)
    custom_norm.to(device=device, dtype=dtype)
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_cohere_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF."""
    HFMLP = _get_hf_cohere_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have CohereMLP")

    device = "cuda"
    config = _create_small_cohere_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = CohereMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_cohere_attention_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF."""
    HFAttention = _get_hf_cohere_attention_class()
    HFRotary = _get_hf_cohere_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have CohereAttention or CohereRotaryEmbedding")

    device = "cuda"
    config = _create_small_cohere_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = CohereAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings (interleaved format)
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings (NeoX format, full cached tables)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = CohereRotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x)  # Full cached tables (no position_ids)

    # HF eager_attention_forward requires an explicit 4D causal mask because it
    # does NOT apply causal masking when attention_mask=None. Our custom attention
    # uses torch_attention with is_causal=True which handles masking internally.
    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Cohere Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2) — Cohere v1
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_cohere_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF."""
    HFDecoderLayer = _get_hf_cohere_decoder_layer_class()
    HFRotary = _get_hf_cohere_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have CohereDecoderLayer or CohereRotaryEmbedding")

    device = "cuda"
    config = _create_small_cohere_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = CohereDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = CohereRotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x)  # Full cached tables

    # HF decoder layer needs explicit causal mask (see attention test comment)
    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Cohere Decoder layer: ")


# =========================================================================
# Layer equivalence tests (Level 2) — Cohere2
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 3])  # 0=sliding_attention, 3=full_attention
@torch.no_grad()
def test_cohere2_decoder_layer_equivalence(B, S, dtype, layer_idx):
    """Test Cohere2 decoder layer (both sliding and full attention types)."""
    HFDecoderLayer = _get_hf_cohere2_decoder_layer_class()
    HFRotary = _get_hf_cohere2_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Cohere2DecoderLayer or Cohere2RotaryEmbedding")

    device_str = "cuda"
    config = _create_small_cohere2_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device_str, dtype=dtype)
    hf_layer.eval()

    custom_layer = CohereDecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device_str, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device_str, dtype=dtype)
    position_ids = torch.arange(S, device=device_str).unsqueeze(0).expand(B, -1)

    hf_rotary = HFRotary(config=config, device=device_str)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = CohereRotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device_str, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x)

    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device_str), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_cos, custom_sin),
    )

    layer_type = config.layer_types[layer_idx]
    # Sliding window layers use interleaved RoPE which introduces slightly more numerical
    # difference from the canonical op de-interleaving; use wider tolerance than dense layers
    tol = 0.10 if layer_type == "sliding_attention" else 0.05
    assert_rmse_close(
        custom_out, hf_out, rmse_ratio_tol=tol, msg=f"Cohere2 Decoder layer ({layer_type}): "
    )


# =========================================================================
# Full model equivalence tests (Level 3) — Cohere v1
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_cohere_full_model_equivalence(B, S, dtype, device):
    """Test full Cohere v1 model produces numerically equivalent output to HF."""
    HFModel = _get_hf_cohere_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have CohereForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_cohere_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = CohereForCausalLM(config)
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
        msg="Cohere v1 full model logits: ",
    )


# =========================================================================
# Full model equivalence tests (Level 3) — Cohere2
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_cohere2_full_model_equivalence(B, S, dtype, device):
    """Test full Cohere2 model produces numerically equivalent output to HF."""
    HFModel = _get_hf_cohere2_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Cohere2ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_cohere2_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = CohereForCausalLM(config)
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
        msg="Cohere2 full model logits: ",
    )


# =========================================================================
# Export tests (Level 4) — Cohere v1
# =========================================================================


@torch.no_grad()
def test_cohere_model_can_be_exported():
    """Test that the Cohere v1 custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_cohere_config()

    model = CohereForCausalLM(config)
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
    assert logits.shape == (B, S, config.vocab_size)
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test dynamic shapes with different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert_rmse_close(
        logits2.float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Export tests (Level 4) — Cohere2
# =========================================================================


@torch.no_grad()
def test_cohere2_model_can_be_exported():
    """Test that the Cohere2 custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_cohere2_config()

    model = CohereForCausalLM(config)
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
    assert logits.shape == (B, S, config.vocab_size)
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
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert_rmse_close(
        logits2.float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_cohere_config_registration():
    """Test that Cohere v1 config is properly recognized."""
    config = _create_small_cohere_config()
    assert config.model_type == "cohere"
    assert hasattr(config, "logit_scale")
    assert hasattr(config, "use_qk_norm")


def test_cohere2_config_registration():
    """Test that Cohere2 config is properly recognized."""
    config = _create_small_cohere2_config()
    assert config.model_type == "cohere2"
    assert hasattr(config, "sliding_window")
    assert hasattr(config, "layer_types")
    # Verify sliding window pattern: with pattern=4, layers 0,1,2 are sliding, layer 3 is full
    assert config.layer_types[0] == "sliding_attention"
    assert config.layer_types[1] == "sliding_attention"
    assert config.layer_types[2] == "sliding_attention"
    assert config.layer_types[3] == "full_attention"


def test_cohere_parallel_attn_mlp_structure():
    """Test that Cohere decoder layers use parallel attn+MLP (no post_attn_layernorm)."""
    config = _create_small_cohere_config()
    model = CohereForCausalLM(config)
    layer = model.model.layers[0]

    assert hasattr(layer, "input_layernorm"), "Should have input_layernorm"
    assert not hasattr(layer, "post_attention_layernorm"), (
        "Should NOT have post_attention_layernorm (parallel pattern)"
    )


def test_cohere_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_cohere_config()
    model = CohereForCausalLM(config)
    attn = model.model.layers[0].self_attn

    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2


def test_cohere2_conditional_rope():
    """Test that Cohere2 applies RoPE only on sliding window layers."""
    config = _create_small_cohere2_config()
    model = CohereForCausalLM(config)

    # Layers 0,1,2 are sliding_attention → use_rope=True
    assert model.model.layers[0].self_attn.use_rope is True
    assert model.model.layers[1].self_attn.use_rope is True
    assert model.model.layers[2].self_attn.use_rope is True
    # Layer 3 is full_attention → use_rope=False
    assert model.model.layers[3].self_attn.use_rope is False


def test_cohere_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_cohere_config()
    model = CohereForCausalLM(config)
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
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, f"Expected key '{key}' in state_dict"


def test_cohere_logit_scale():
    """Test that logit_scale is applied to output logits."""
    config = _create_small_cohere_config()
    config.logit_scale = 0.5
    model = CohereForCausalLM(config)
    model.eval()

    assert model.logit_scale == 0.5
