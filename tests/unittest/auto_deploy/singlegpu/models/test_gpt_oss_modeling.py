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

"""Tests for GPT-OSS custom model implementation.

This module tests the custom GPT-OSS model implementation which uses
auto_deploy custom ops (torch_attention with sinks, torch_moe_router,
torch_rope_with_explicit_cos_sin) for export compatibility.

GPT-OSS is a MoE model with:
- GQA attention with learnable per-head attention sinks
- YaRN RoPE
- MoE with clamped SwiGLU activation and biased expert projections
- Alternating sliding-window and full-attention layers
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssYarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config():
    """Create a small GptOssConfig for testing."""
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    except ImportError:
        pytest.skip("transformers doesn't have GptOssConfig")

    return GptOssConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        attention_bias=True,
        attention_dropout=0.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        experts_per_token=2,
        sliding_window=4,
        swiglu_limit=7.0,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_model_class():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssForCausalLM as HFGptOssForCausalLM,
        )

        return HFGptOssForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssAttention as HFGptOssAttention,
        )

        return HFGptOssAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP as HFGptOssMLP

        return HFGptOssMLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssDecoderLayer as HFGptOssDecoderLayer,
        )

        return HFGptOssDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssRotaryEmbedding as HFGptOssRotaryEmbedding,
        )

        return HFGptOssRotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_gpt_oss_experts_equivalence(B, S, dtype):
    """Test expert block produces numerically equivalent output to HF implementation."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts as HFGptOssExperts
    except ImportError:
        pytest.skip("transformers doesn't have GptOssExperts")

    device = "cuda"
    config = _create_small_config()

    # Create HF experts and initialize weights (torch.empty leaves them uninitialized)
    hf_experts = HFGptOssExperts(config)
    with torch.no_grad():
        hf_experts.gate_up_proj.normal_(std=0.02)
        hf_experts.gate_up_proj_bias.zero_()
        hf_experts.down_proj.normal_(std=0.02)
        hf_experts.down_proj_bias.zero_()
    hf_experts.to(device=device, dtype=dtype)
    hf_experts.eval()

    # Create custom experts with same weights
    custom_experts = GptOssExperts(config)
    custom_experts.to(device=device, dtype=dtype)
    custom_experts.load_state_dict(hf_experts.state_dict())
    custom_experts.eval()

    # Create input and routing
    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    T = B * S

    # Create routing weights (sparse, only top-k non-zero per token)
    router_logits = torch.randn(T, config.num_local_experts, device=device, dtype=dtype)
    top_values, top_indices = torch.topk(router_logits, config.num_experts_per_tok, dim=-1)
    top_values = torch.softmax(top_values, dim=-1)
    routing_weights = torch.zeros_like(router_logits).scatter_(1, top_indices, top_values)

    # Run HF experts (expects router_indices and routing_weights separately)
    hf_out = hf_experts(x, router_indices=top_indices, routing_weights=routing_weights)

    # Run custom experts (expects sparse routing_weights)
    custom_out = custom_experts(x, routing_weights=routing_weights)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="Experts: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_gpt_oss_mlp_equivalence(B, S, dtype):
    """Test MLP (router + experts) produces numerically equivalent output to HF."""
    HFMlp = _get_hf_mlp_class()
    if HFMlp is None:
        pytest.skip("transformers doesn't have GptOssMLP")

    device = "cuda"
    config = _create_small_config()

    # Create HF MLP and initialize weights (torch.empty leaves them uninitialized)
    hf_mlp = HFMlp(config)
    with torch.no_grad():
        hf_mlp.router.weight.normal_(std=0.02)
        hf_mlp.router.bias.normal_(std=0.02)
        hf_mlp.experts.gate_up_proj.normal_(std=0.02)
        hf_mlp.experts.gate_up_proj_bias.zero_()
        hf_mlp.experts.down_proj.normal_(std=0.02)
        hf_mlp.experts.down_proj_bias.zero_()
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = GptOssMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # HF returns (output, router_scores)
    hf_out, _ = hf_mlp(x)
    custom_out = custom_mlp(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MLP (MoE): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_gpt_oss_attention_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have GptOssAttention or GptOssRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Use layer_idx=1 (full_attention) to avoid sliding window mask complexity
    hf_attn = HFAttention(config, layer_idx=1)
    with torch.no_grad():
        hf_attn.sinks.normal_(std=0.02)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention and load same weights
    custom_attn = GptOssAttention(config, layer_idx=1)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings
    custom_rotary = GptOssYarnRotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create causal attention mask for HF (additive mask: 0 for attend, -inf for masked)
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    causal_mask = (
        causal_mask.unsqueeze(0).unsqueeze(0).expand(B, config.num_attention_heads, -1, -1)
    )

    # HF attention
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Custom attention
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
def test_gpt_oss_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have GptOssDecoderLayer or GptOssRotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Use layer_idx=1 (full_attention) to avoid sliding window mask complexity in block test.
    # Sliding window behavior is implicitly tested in full model equivalence tests.
    hf_layer = HFDecoderLayer(config, layer_idx=1)
    with torch.no_grad():
        hf_layer.self_attn.sinks.normal_(std=0.02)
        hf_layer.mlp.router.weight.normal_(std=0.02)
        hf_layer.mlp.router.bias.normal_(std=0.02)
        hf_layer.mlp.experts.gate_up_proj.normal_(std=0.02)
        hf_layer.mlp.experts.gate_up_proj_bias.zero_()
        hf_layer.mlp.experts.down_proj.normal_(std=0.02)
        hf_layer.mlp.experts.down_proj_bias.zero_()
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = GptOssDecoderLayer(config, layer_idx=1)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings
    custom_rotary = GptOssYarnRotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create causal attention mask for HF (additive: 0 attend, -inf masked)
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    causal_mask = (
        causal_mask.unsqueeze(0).unsqueeze(0).expand(B, config.num_attention_heads, -1, -1)
    )

    # HF decoder layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Custom decoder layer
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
def test_gpt_oss_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have GptOssForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load same weights
    custom_model = GptOssForCausalLM(config)
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
def test_gpt_oss_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = GptOssForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    # Dynamic shapes
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

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test dynamic shapes
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


def test_gpt_oss_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "gpt_oss"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "num_local_experts")
    assert hasattr(config, "swiglu_limit")


def test_gpt_oss_gqa_structure():
    """Test that attention uses GQA."""
    config = _create_small_config()
    model = GptOssForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert hasattr(attn, "sinks")
    assert attn.sinks.shape == (4,)


def test_gpt_oss_sliding_window_layers():
    """Test that layers correctly alternate between sliding and full attention."""
    config = _create_small_config()
    model = GptOssForCausalLM(config)

    # Layer 0: sliding_attention -> has sliding_window
    assert model.model.layers[0].self_attn.sliding_window == config.sliding_window
    # Layer 1: full_attention -> no sliding_window
    assert model.model.layers[1].self_attn.sliding_window is None
    # Layer 2: sliding_attention -> has sliding_window
    assert model.model.layers[2].self_attn.sliding_window == config.sliding_window


def test_gpt_oss_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = GptOssForCausalLM(config)
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
        "model.layers.0.self_attn.sinks",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.down_proj_bias",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_keys:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:20]}..."
        )


def test_gpt_oss_expert_weight_shapes():
    """Test expert weight shapes match expected stacked format."""
    config = _create_small_config()
    model = GptOssForCausalLM(config)

    experts = model.model.layers[0].mlp.experts
    E = config.num_local_experts
    H = config.hidden_size
    D = config.intermediate_size

    assert experts.gate_up_proj.shape == (E, H, 2 * D)
    assert experts.gate_up_proj_bias.shape == (E, 2 * D)
    assert experts.down_proj.shape == (E, D, H)
    assert experts.down_proj_bias.shape == (E, H)
