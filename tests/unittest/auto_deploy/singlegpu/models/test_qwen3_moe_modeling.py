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

"""Tests for Qwen3 MoE custom model implementation.

This module tests the custom Qwen3 MoE model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_moe) for export compatibility. Qwen3 MoE uses GQA with per-head Q/K
normalization and softmax top-k MoE routing.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Qwen3MoeConfig:
    """Create a small Qwen3 MoE config for testing."""
    return Qwen3MoeConfig(
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
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=False,
        # MoE settings
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    """Get the HF Qwen3MoeForCausalLM class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM as HFQwen3MoeForCausalLM,
        )

        return HFQwen3MoeForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Qwen3MoeAttention class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeAttention as HFQwen3MoeAttention,
        )

        return HFQwen3MoeAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Qwen3MoeMLP class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP as HFQwen3MoeMLP

        return HFQwen3MoeMLP
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF Qwen3MoeSparseMoeBlock class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
        )

        return HFQwen3MoeSparseMoeBlock
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Qwen3MoeDecoderLayer class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeDecoderLayer as HFQwen3MoeDecoderLayer,
        )

        return HFQwen3MoeDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Qwen3MoeRotaryEmbedding class."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeRotaryEmbedding as HFQwen3MoeRotaryEmbedding,
        )

        return HFQwen3MoeRotaryEmbedding
    except ImportError:
        return None


def _init_hf_moe_params_in_place(module: torch.nn.Module, std: float = 0.02) -> None:
    """Manually initialize standalone HF Qwen3MoE submodule parameters.

    These would normally be filled by ``Qwen3MoePreTrainedModel._init_weights``
    via ``post_init()``.

    transformers 5.x's ``Qwen3MoeExperts`` allocates ``gate_up_proj`` /
    ``down_proj`` with ``torch.empty(...)`` (uninitialized memory) and
    ``Qwen3MoeTopKRouter`` allocates ``weight`` with ``torch.zeros(...)``.
    When these are constructed standalone (not inside a ``PreTrainedModel``
    subclass), ``post_init()`` is never run, so matmuls produce NaN.
    Walk every submodule and apply the same ``normal_(0, std)`` that the
    upstream ``_init_weights`` would.
    """
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeExperts,
            Qwen3MoeTopKRouter,
        )
    except ImportError:
        return
    for m in module.modules():
        if isinstance(m, Qwen3MoeExperts):
            torch.nn.init.normal_(m.gate_up_proj, mean=0.0, std=std)
            torch.nn.init.normal_(m.down_proj, mean=0.0, std=std)
        elif isinstance(m, Qwen3MoeTopKRouter):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_moe_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Qwen3MoeMLP")

    device = "cuda"
    config = _create_small_config()

    # Create HF MLP
    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = Qwen3MoeMLP(config)
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
def test_qwen3_moe_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Qwen3MoeAttention or Qwen3MoeRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    # Force eager attention for HF model
    config._attn_implementation = "eager"

    # Create HF attention
    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights
    custom_attn = Qwen3MoeAttention(config, layer_idx=0)
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
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = Qwen3MoeRotaryEmbedding(
        head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_parameters["rope_theta"],
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    q_len = S
    causal_mask = (
        torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

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


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_moe_sparse_moe_block_equivalence(B, S, dtype):
    """Test MoE block produces numerically equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have Qwen3MoeSparseMoeBlock")

    device = "cuda"
    config = _create_small_config()

    # Create HF MoE block. Standalone construction skips PreTrainedModel.post_init(),
    # so Qwen3MoeExperts (torch.empty) and Qwen3MoeTopKRouter (torch.zeros) need
    # manual initialization to avoid NaN matmuls on uninitialized memory.
    hf_moe = HFMoE(config)
    hf_moe.to(device=device, dtype=dtype)
    _init_hf_moe_params_in_place(hf_moe, std=config.initializer_range)
    hf_moe.eval()

    # Create custom MoE block and load same weights
    custom_moe = Qwen3MoeSparseMoeBlock(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    # Create input
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Run both. transformers 5.x's HFMoE.forward returns a single tensor
    # despite its ``tuple[Tensor, Tensor]`` annotation; older versions
    # returned ``(output, router_logits)``.
    hf_out = hf_moe(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]
    custom_out, _ = custom_moe(x)

    # MoE uses fused routing, use relaxed tolerance
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE block: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_moe_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Qwen3MoeDecoderLayer or Qwen3MoeRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF decoder layer. Standalone construction skips
    # PreTrainedModel.post_init(), so the inner Qwen3MoeExperts /
    # Qwen3MoeTopKRouter parameters (torch.empty / torch.zeros) need manual
    # initialization to avoid NaN matmuls.
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    _init_hf_moe_params_in_place(hf_layer, std=config.initializer_range)
    hf_layer.eval()

    # Create custom decoder layer and load same weights
    custom_layer = Qwen3MoeDecoderLayer(config, layer_idx=0)
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
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = Qwen3MoeRotaryEmbedding(
        head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_parameters["rope_theta"],
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    q_len = S
    causal_mask = (
        torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        .unsqueeze(0)
        .unsqueeze(0)
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
def test_qwen3_moe_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Qwen3MoeForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = Qwen3MoeForCausalLM(config)
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
def test_qwen3_moe_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output to eager model
    3. Dynamic shapes work correctly
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Qwen3MoeForCausalLM(config)
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


def test_qwen3_moe_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "qwen3_moe"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "num_experts")
    assert hasattr(config, "num_experts_per_tok")


def test_qwen3_moe_structure():
    """Test model structure: GQA, Q/K norms, MoE layers."""
    config = _create_small_config()
    model = Qwen3MoeForCausalLM(config)

    # Check attention
    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"
    assert hasattr(attn, "q_norm"), "Attention should have q_norm"
    assert hasattr(attn, "k_norm"), "Attention should have k_norm"

    # All layers should be MoE (decoder_sparse_step=1, no mlp_only_layers)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, Qwen3MoeSparseMoeBlock), (
            f"Layer {i} should use MoE, got {type(layer.mlp)}"
        )


def test_qwen3_moe_dense_layer_config():
    """Test that mlp_only_layers correctly creates dense MLP layers."""
    config = _create_small_config()
    config.mlp_only_layers = [0]  # First layer should be dense

    model = Qwen3MoeForCausalLM(config)

    # Layer 0 should be dense MLP
    assert isinstance(model.model.layers[0].mlp, Qwen3MoeMLP), (
        f"Layer 0 should use dense MLP, got {type(model.model.layers[0].mlp)}"
    )
    # Layer 1 should be MoE
    assert isinstance(model.model.layers[1].mlp, Qwen3MoeSparseMoeBlock), (
        f"Layer 1 should use MoE, got {type(model.model.layers[1].mlp)}"
    )


def test_qwen3_moe_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = Qwen3MoeForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        # MoE-specific keys
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )
