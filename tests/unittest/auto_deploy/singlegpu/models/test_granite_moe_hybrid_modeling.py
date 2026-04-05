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

"""Tests for GraniteMoeHybrid custom model implementation.

Tests two configuration variants:
1. Attention-only (e.g. granite-4.0-micro): all-attention layers, no MoE, with RoPE.
2. Hybrid (e.g. granite-4.0-tiny): mixed Mamba/Attention layers, MoE, no RoPE ("nope").
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 - register AD ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_granite_moe_hybrid import (
    GraniteMoeHybridAttention,
    GraniteMoeHybridDecoderLayer,
    GraniteMoeHybridForCausalLM,
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridMLP,
    GraniteMoeHybridMoEBlock,
    GraniteMoeHybridRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


# =============================================================================
# HF class import helpers
# =============================================================================


def _get_hf_config_class():
    try:
        from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
            GraniteMoeHybridConfig,
        )

        return GraniteMoeHybridConfig
    except ImportError:
        return None


def _get_hf_model_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridForCausalLM as HFGraniteMoeHybridForCausalLM,
        )

        return HFGraniteMoeHybridForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridAttention as HFAttention,
        )

        return HFAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridMLP as HFMLP,
        )

        return HFMLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridDecoderLayer as HFDecoderLayer,
        )

        return HFDecoderLayer
    except ImportError:
        return None


def _get_hf_mamba_layer_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridMambaLayer as HFMambaLayer,
        )

        return HFMambaLayer
    except ImportError:
        return None


def _get_hf_moe_class():
    try:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
            GraniteMoeHybridMoE as HFMoE,
        )

        return HFMoE
    except ImportError:
        return None


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Config factories
# =============================================================================


def _create_small_config():
    """Create a small attention-only config (granite-4.0-micro style).

    The model has: attention-only layers, no MoE, muP scaling, RoPE.
    """
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridConfig")

    config = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        shared_intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # muP multipliers
        embedding_multiplier=12.0,
        attention_multiplier=0.015625,
        residual_multiplier=0.22,
        logits_scaling=10.0,
        # No MoE
        num_local_experts=0,
        num_experts_per_tok=0,
        # All attention layers
        layer_types=["attention"] * 3,
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        position_embedding_type="rope",
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    # Required by HF's attention implementation dispatch
    config._attn_implementation = "eager"
    return config


def _create_small_hybrid_config():
    """Create a small hybrid Mamba/Attention config (granite-4.0-tiny style).

    The model has: mixed Mamba/Attention layers, MoE, muP scaling, no positional encoding.
    """
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridConfig")

    config = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # muP multipliers
        embedding_multiplier=12.0,
        attention_multiplier=0.015625,
        residual_multiplier=0.22,
        logits_scaling=10.0,
        # MoE
        num_local_experts=4,
        num_experts_per_tok=2,
        # Mixed layers: 3 mamba + 1 attention
        layer_types=["mamba", "mamba", "mamba", "attention"],
        # No positional encoding
        position_embedding_type="nope",
        rope_theta=10000.0,
        rope_scaling=None,
        # Mamba SSM params
        mamba_d_conv=4,
        mamba_d_state=16,
        mamba_d_head=8,
        mamba_n_heads=8,
        mamba_n_groups=1,
        mamba_expand=1,
        mamba_chunk_size=8,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def _build_causal_mask(B, S, device, dtype):
    """Build a 4D causal mask for HF eager attention."""
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)


# =============================================================================
# Level 1: Block equivalence tests (attention-only / micro variant)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_mlp_equivalence(B, S, dtype):
    """Test MLP produces equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridMLP")

    device = "cpu"
    config = _create_small_config()

    # Create HF MLP
    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = GraniteMoeHybridMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    # Input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_attention_equivalence(B, S, dtype):
    """Test attention produces equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    if HFAttention is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridAttention")

    device = "cpu"
    config = _create_small_config()

    # Create HF attention
    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights
    custom_attn = GraniteMoeHybridAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    # Input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get full RoPE table
    rotary_emb = GraniteMoeHybridRotaryEmbedding(config).to(device=device)
    full_cos_sin = rotary_emb(x)

    # Custom forward: receives full table, slices internally
    # Note: custom attention uses torch_attention with is_causal=True
    custom_out = custom_attn(x, position_ids, full_cos_sin)

    # HF forward: expects pre-sliced cos/sin + causal mask to match custom
    cos_sliced = full_cos_sin[0][position_ids]
    sin_sliced = full_cos_sin[1][position_ids]
    causal_mask = _build_causal_mask(B, S, device, dtype)
    hf_out, _ = hf_attn(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(cos_sliced, sin_sliced),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10)


# =============================================================================
# Level 1: Block equivalence tests (hybrid variant)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_attention_nope_equivalence(B, S, dtype):
    """Test attention without RoPE (position_embedding_type='nope')."""
    HFAttention = _get_hf_attention_class()
    if HFAttention is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridAttention")

    device = "cpu"
    config = _create_small_hybrid_config()

    hf_attn = HFAttention(config, layer_idx=3)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = GraniteMoeHybridAttention(config, layer_idx=3)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Custom: position_embeddings=None means no RoPE
    custom_out = custom_attn(x, position_ids, position_embeddings=None)

    # HF: position_embeddings=None skips RoPE, but needs causal mask
    causal_mask = _build_causal_mask(B, S, device, dtype)
    hf_out, _ = hf_attn(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=None,
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_mamba_equivalence(B, S, dtype):
    """Test Mamba layer produces equivalent output to HF torch_forward."""
    HFMambaLayer = _get_hf_mamba_layer_class()
    if HFMambaLayer is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridMambaLayer")

    device = "cpu"
    config = _create_small_hybrid_config()

    hf_mamba = HFMambaLayer(config, layer_idx=0)
    hf_mamba.to(device=device, dtype=dtype)
    hf_mamba.eval()

    custom_mamba = GraniteMoeHybridMambaLayer(config, layer_idx=0)
    custom_mamba.to(device=device, dtype=dtype)
    custom_mamba.load_state_dict(hf_mamba.state_dict())
    custom_mamba.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # HF forward on CPU dispatches to torch_forward (uncached)
    hf_out = hf_mamba(x)
    custom_out = custom_mamba(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_moe_equivalence(B, S, dtype):
    """Test MoE block produces equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridMoE")

    device = "cpu"
    config = _create_small_hybrid_config()

    hf_moe = HFMoE(config)
    # HF ParallelExperts stores expert weights in torch.empty() buffers.
    # Reinitialize to avoid comparing NaN-filled reference outputs.
    for _, param in hf_moe.named_parameters():
        torch.nn.init.normal_(param, std=0.02)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    custom_moe = GraniteMoeHybridMoEBlock(config)
    custom_moe.to(device=device, dtype=dtype)
    # Pre-hook unfuses input_linear/output_linear into per-expert weights
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out, _ = hf_moe(x)
    custom_out = custom_moe(x)

    # Guard against degenerate all-zero outputs (0/0 RMSE ratio → NaN)
    if torch.allclose(custom_out, hf_out, atol=1e-6):
        return
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05)


# =============================================================================
# Level 2: Layer equivalence tests (attention-only / micro variant)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_decoder_layer_equivalence(B, S, dtype):
    """Test attention decoder layer (micro variant) produces equivalent output."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridDecoderLayer")

    device = "cpu"
    config = _create_small_config()

    # Create HF decoder layer
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = GraniteMoeHybridDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    # Input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get full RoPE table
    rotary_emb = GraniteMoeHybridRotaryEmbedding(config).to(device=device)
    full_cos_sin = rotary_emb(x)

    # Custom forward: receives full table + position_ids
    custom_out = custom_layer(x, position_ids, full_cos_sin)

    # HF forward: expects pre-sliced position_embeddings + causal mask
    cos_sliced = full_cos_sin[0][position_ids]
    sin_sliced = full_cos_sin[1][position_ids]
    causal_mask = _build_causal_mask(B, S, device, dtype)
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(cos_sliced, sin_sliced),
    )
    hf_hidden = hf_out[0]

    assert_rmse_close(custom_out, hf_hidden, rmse_ratio_tol=0.05)


# =============================================================================
# Level 2: Layer equivalence tests (hybrid variant)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_mamba_decoder_layer_equivalence(B, S, dtype):
    """Test Mamba decoder layer (with MoE) produces equivalent output."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridDecoderLayer")

    device = "cpu"
    config = _create_small_hybrid_config()

    # Layer 0 is a mamba layer
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    # HF ParallelExperts uses torch.empty() — reinitialize all weights to avoid NaN
    for name, p in hf_layer.named_parameters():
        torch.nn.init.normal_(p, std=0.02)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = GraniteMoeHybridDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # HF: mamba layers don't use position_embeddings or causal mask
    hf_out = hf_layer(hidden_states=x)
    hf_hidden = hf_out[0]

    # Custom: mamba layers ignore position_ids and position_embeddings
    custom_out = custom_layer(x)

    assert_rmse_close(custom_out, hf_hidden, rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_attention_decoder_layer_nope_equivalence(B, S, dtype):
    """Test attention decoder layer (with MoE, no RoPE) produces equivalent output."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridDecoderLayer")

    device = "cpu"
    config = _create_small_hybrid_config()

    # Layer 3 is an attention layer
    hf_layer = HFDecoderLayer(config, layer_idx=3)
    # HF ParallelExperts uses torch.empty() — reinitialize all weights to avoid NaN
    for name, p in hf_layer.named_parameters():
        torch.nn.init.normal_(p, std=0.02)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = GraniteMoeHybridDecoderLayer(config, layer_idx=3)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF: no RoPE (nope), but needs causal mask for eager attention
    causal_mask = _build_causal_mask(B, S, device, dtype)
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=None,
    )
    hf_hidden = hf_out[0]

    # Custom: position_embeddings=None means no RoPE
    custom_out = custom_layer(x, position_ids, position_embeddings=None)

    assert_rmse_close(custom_out, hf_hidden, rmse_ratio_tol=0.05)


# =============================================================================
# Level 3: Full model equivalence tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_full_model_equivalence(B, S, dtype):
    """Test full attention-only model produces equivalent logits."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridForCausalLM")

    device = "cpu"
    config = _create_small_config()

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = GraniteMoeHybridForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    # Input
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Run both
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits.float(), hf_out.logits.float(), rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_granite_moe_hybrid_full_model_hybrid_equivalence(B, S, dtype):
    """Test full hybrid model (Mamba+Attention+MoE) produces equivalent logits."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have GraniteMoeHybridForCausalLM")

    device = "cpu"
    config = _create_small_hybrid_config()

    hf_model = HFModel(config)
    # HF ParallelExperts uses torch.empty() — reinitialize all weights to avoid NaN
    for name, p in hf_model.named_parameters():
        torch.nn.init.normal_(p, std=0.02)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = GraniteMoeHybridForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits.float(), hf_out.logits.float(), rmse_ratio_tol=0.05)


# =============================================================================
# Level 4: Export tests
# =============================================================================


def test_granite_moe_hybrid_model_can_be_exported():
    """Test that the attention-only model can be exported with torch_export_to_gm."""
    device = "cpu"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = GraniteMoeHybridForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get eager reference output
    with torch.no_grad():
        eager_out = model(input_ids=input_ids, position_ids=position_ids)

    # Define dynamic shapes
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    # Export
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    # Verify exported output matches eager output
    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert_rmse_close(logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05)

    # Test with different shape to verify dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.no_grad():
        eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert_rmse_close(logits2.float(), eager_out2.logits.float(), rmse_ratio_tol=0.05)


def test_granite_moe_hybrid_model_hybrid_can_be_exported():
    """Test that the hybrid Mamba+Attention+MoE model can be exported."""
    device = "cpu"
    dtype = torch.bfloat16
    config = _create_small_hybrid_config()

    model = GraniteMoeHybridForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
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

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert_rmse_close(logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05)

    # Verify with different shape
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.no_grad():
        eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert_rmse_close(logits2.float(), eager_out2.logits.float(), rmse_ratio_tol=0.05)
