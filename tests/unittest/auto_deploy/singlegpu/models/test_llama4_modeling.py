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

"""Tests for Llama 4 custom model implementation.

This module tests the custom Llama 4 model implementation which uses
auto_deploy custom ops for export compatibility. Llama 4 features:
* GQA with complex-frequency RoPE
* NoPE layers (interleaved layers that skip RoPE)
* L2 QK norm on RoPE layers
* Attention temperature tuning on NoPE layers
* MoE with sigmoid router + shared expert
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_llama4 import (
    Llama4Attention,
    Llama4DecoderLayer,
    Llama4ForCausalLM,
    Llama4L2Norm,
    Llama4MLP,
    Llama4MoE,
    Llama4RMSNorm,
    Llama4RotaryEmbedding,
)

# Note: Llama4MoE uses stacked expert weights (nn.Parameter + bmm) matching
# HF checkpoint format. The AD MatchBmmMoePattern transform handles conversion
# to torch_moe at deployment time.
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Llama4TextConfig:
    """Create a small Llama4 text config for testing.

    Key features tested:
    - 3 layers: layer 0,1 are MoE with RoPE, layer 2 is MoE NoPE
    - GQA: 4 Q heads, 2 KV heads
    - 4 experts, top-1 routing
    - QK L2 norm on RoPE layers
    - Attention temperature tuning on NoPE layers
    """
    return Llama4TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,  # MoE expert intermediate size
        intermediate_size_mlp=128,  # Dense MLP intermediate size
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        num_experts_per_tok=1,
        num_local_experts=4,
        interleave_moe_layer_step=1,  # All layers are MoE
        use_qk_norm=True,
        attn_temperature_tuning=True,
        floor_scale=8192,
        attn_scale=0.1,
        no_rope_layer_interval=3,  # Every 3rd layer is NoPE (layer 2 is NoPE)
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_causal_lm_class():
    """Get the HF Llama4ForCausalLM class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4ForCausalLM as HFLlama4ForCausalLM,
        )

        return HFLlama4ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Llama4TextAttention class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextAttention as HFLlama4TextAttention,
        )

        return HFLlama4TextAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Llama4TextMLP class."""
    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP as HFLlama4TextMLP

        return HFLlama4TextMLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Llama4TextDecoderLayer class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextDecoderLayer as HFLlama4TextDecoderLayer,
        )

        return HFLlama4TextDecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Llama4TextRotaryEmbedding class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextRotaryEmbedding as HFLlama4TextRotaryEmbedding,
        )

        return HFLlama4TextRotaryEmbedding
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF Llama4TextMoe class."""
    try:
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe as HFLlama4TextMoe

        return HFLlama4TextMoe
    except ImportError:
        return None


def _get_hf_rmsnorm_class():
    """Get the HF Llama4TextRMSNorm class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextRMSNorm as HFLlama4TextRMSNorm,
        )

        return HFLlama4TextRMSNorm
    except ImportError:
        return None


def _get_hf_l2norm_class():
    """Get the HF Llama4TextL2Norm class."""
    try:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4TextL2Norm as HFLlama4TextL2Norm,
        )

        return HFLlama4TextL2Norm
    except ImportError:
        return None


# Note: No weight conversion needed — custom model uses stacked expert format
# (nn.Parameter + bmm) matching HF directly. AD MatchBmmMoePattern transform
# handles conversion to torch_moe at deployment time.


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_llama4_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces equivalent output to HF implementation."""
    HFRMSNorm = _get_hf_rmsnorm_class()
    if HFRMSNorm is None:
        pytest.skip("transformers doesn't have Llama4TextRMSNorm")

    device = "cuda"
    config = _create_small_config()

    hf_norm = HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = Llama4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
def test_llama4_l2norm_equivalence(B, S, dtype):
    """Test L2Norm produces equivalent output to HF implementation."""
    HFL2Norm = _get_hf_l2norm_class()
    if HFL2Norm is None:
        pytest.skip("transformers doesn't have Llama4TextL2Norm")

    device = "cuda"
    config = _create_small_config()

    hf_norm = HFL2Norm(config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = Llama4L2Norm(config.rms_norm_eps)
    custom_norm.to(device=device, dtype=dtype)
    custom_norm.eval()

    # Test with attention-like input [B, S, N, head_dim]
    x = torch.randn(B, S, config.num_attention_heads, config.head_dim, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_llama4_mlp_equivalence(B, S, dtype):
    """Test MLP produces equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Llama4TextMLP")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config, intermediate_size=config.intermediate_size_mlp)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Llama4MLP(config, intermediate_size=config.intermediate_size_mlp)
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
def test_llama4_attention_rope_equivalence(B, S, dtype):
    """Test Attention on a RoPE layer produces equivalent output to HF."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Llama4TextAttention or RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Layer 0 is a RoPE layer
    layer_idx = 0
    assert config.no_rope_layers[layer_idx] == 1, "Layer 0 should use RoPE"

    hf_attn = HFAttention(config, layer_idx=layer_idx)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Llama4Attention(config, layer_idx=layer_idx)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF: compute freqs_cis dynamically
    hf_rotary = HFRotary(config=config, device=device)
    hf_freqs_cis = hf_rotary(x, position_ids)

    # Custom: use precomputed cache
    custom_rotary = Llama4RotaryEmbedding(config)
    custom_rotary.to(device=device)
    custom_freqs_cis = custom_rotary(x)

    # Create causal mask for HF (torch_attention uses is_causal=True internally)
    cache_position = torch.arange(S, device=device)
    causal_mask = torch.zeros(1, 1, S, S, device=device, dtype=dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )

    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=hf_freqs_cis,
        attention_mask=causal_mask,
        cache_position=cache_position,
    )

    # Custom forward
    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
        freqs_cis=custom_freqs_cis,
    )

    # QK L2 norm amplifies numerical differences between attention implementations
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.20, msg="Attention (RoPE): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_llama4_attention_nope_equivalence(B, S, dtype):
    """Test Attention on a NoPE layer produces equivalent output to HF."""
    HFAttention = _get_hf_attention_class()
    if HFAttention is None:
        pytest.skip("transformers doesn't have Llama4TextAttention")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    # Find a NoPE layer index (no_rope_layers[idx] == 0)
    nope_layer_idx = None
    for i in range(config.num_hidden_layers):
        if config.no_rope_layers[i] == 0:
            nope_layer_idx = i
            break

    if nope_layer_idx is None:
        pytest.skip("No NoPE layers in test config")

    hf_attn = HFAttention(config, layer_idx=nope_layer_idx)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Llama4Attention(config, layer_idx=nope_layer_idx)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(S, device=device)

    # Create causal mask for HF
    causal_mask = torch.zeros(1, 1, S, S, device=device, dtype=dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )

    # HF forward (NoPE doesn't use freqs_cis, but still needs position_embeddings arg)
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=None,
        attention_mask=causal_mask,
        cache_position=cache_position,
    )

    # Custom forward
    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
        freqs_cis=None,
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention (NoPE): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_llama4_moe_equivalence(B, S, dtype):
    """Test MoE layer produces equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have Llama4TextMoe")

    device = "cuda"
    config = _create_small_config()

    hf_moe = HFMoE(config)
    # Llama4TextExperts uses torch.empty (uninitialized) — must manually init
    std = config.initializer_range
    hf_moe.experts.gate_up_proj.data.normal_(mean=0.0, std=std)
    hf_moe.experts.down_proj.data.normal_(mean=0.0, std=std)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    custom_moe = Llama4MoE(config)
    custom_moe.to(device=device, dtype=dtype)
    # Weight format matches HF directly — no conversion needed
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out, _ = hf_moe(x)
    custom_out = custom_moe(x)

    # Both return [B, S, H] since custom reshapes back to orig_shape
    # HF returns flattened [B*S, H] — reshape to match
    hf_out = hf_out.view(B, S, -1)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layer_idx", [0, 2])  # 0=RoPE+MoE, 2=NoPE+MoE
@torch.no_grad()
def test_llama4_decoder_layer_equivalence(B, S, dtype, layer_idx):
    """Test decoder layer produces equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Llama4TextDecoderLayer")

    device = "cuda"
    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    # Llama4TextExperts uses torch.empty (uninitialized) — must manually init
    if hasattr(hf_layer.feed_forward, "experts"):
        std = config.initializer_range
        hf_layer.feed_forward.experts.gate_up_proj.data.normal_(mean=0.0, std=std)
        hf_layer.feed_forward.experts.down_proj.data.normal_(mean=0.0, std=std)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Llama4DecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    # Weight format matches HF directly — no conversion needed
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(S, device=device)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_freqs_cis = hf_rotary(x, position_ids)

    # Custom position embeddings
    custom_rotary = Llama4RotaryEmbedding(config)
    custom_rotary.to(device=device)
    custom_freqs_cis = custom_rotary(x)

    # Create causal mask for HF
    causal_mask = torch.zeros(1, 1, S, S, device=device, dtype=dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=hf_freqs_cis,
        cache_position=cache_position,
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Custom forward
    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        freqs_cis=custom_freqs_cis,
    )

    # QK L2 norm amplifies numerical differences between attention implementations
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Decoder layer: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_llama4_dense_decoder_layer_equivalence(B, S, dtype):
    """Test dense (non-MoE) decoder layer equivalence."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Llama4TextDecoderLayer")

    device = "cuda"
    # Config with some dense layers (interleave_moe_layer_step=2 → layer 0 is dense)
    config = Llama4TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        intermediate_size_mlp=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_experts_per_tok=1,
        num_local_experts=4,
        interleave_moe_layer_step=2,  # Layers 1 are MoE, layer 0,2 are dense
        use_qk_norm=True,
        attn_temperature_tuning=True,
        no_rope_layer_interval=3,
    )
    config._attn_implementation = "eager"

    # Layer 0 should be dense (not in moe_layers)
    layer_idx = 0
    assert layer_idx not in config.moe_layers, f"Layer {layer_idx} should be dense"

    hf_layer = HFDecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Llama4DecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(S, device=device)

    hf_rotary = HFRotary(config=config, device=device)
    hf_freqs_cis = hf_rotary(x, position_ids)

    custom_rotary = Llama4RotaryEmbedding(config)
    custom_rotary.to(device=device)
    custom_freqs_cis = custom_rotary(x)

    causal_mask = torch.zeros(1, 1, S, S, device=device, dtype=dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=hf_freqs_cis,
        cache_position=cache_position,
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        freqs_cis=custom_freqs_cis,
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Dense decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_llama4_full_model_equivalence(B, S, dtype, device):
    """Test full model produces equivalent output to HF implementation."""
    HFModel = _get_hf_causal_lm_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Llama4ForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()
    config._attn_implementation = "eager"

    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = Llama4ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    # Use load_state_dict with the pre-hook handling MoE conversion
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    # Llama 4's QK L2 norm amplifies numerical differences between attention
    # implementations (HF eager vs AD torch_attention). Empirically verified:
    # float32 gives RMSE ratio < 1e-6 (confirming algorithmic correctness),
    # bfloat16 gives ~0.13-0.16 due to L2 norm sensitivity to precision.
    # Without QK norm, bfloat16 gives < 0.006.
    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.20,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_llama4_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Llama4ForCausalLM(config)
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
        rmse_ratio_tol=0.10,
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


def test_llama4_config_registration():
    """Test that the config is properly recognized."""
    config = _create_small_config()
    assert config.model_type == "llama4_text"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "head_dim")
    assert hasattr(config, "moe_layers")
    assert hasattr(config, "no_rope_layers")


def test_llama4_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = Llama4ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_llama4_moe_structure():
    """Test MoE layer structure."""
    config = _create_small_config()
    model = Llama4ForCausalLM(config)

    # All layers should be MoE with interleave_moe_layer_step=1
    for layer_idx in range(config.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        assert layer.is_moe_layer, f"Layer {layer_idx} should be MoE"
        assert isinstance(layer.feed_forward, Llama4MoE)
        assert layer.feed_forward.experts.num_experts == config.num_local_experts


def test_llama4_nope_layers():
    """Test NoPE/RoPE layer configuration."""
    config = _create_small_config()
    model = Llama4ForCausalLM(config)

    for layer_idx in range(config.num_hidden_layers):
        attn = model.model.layers[layer_idx].self_attn
        expected_rope = bool(config.no_rope_layers[layer_idx])
        assert attn.use_rope == expected_rope, (
            f"Layer {layer_idx}: expected use_rope={expected_rope}, got {attn.use_rope}"
        )
        # QK norm should only be on RoPE layers
        has_qk_norm = hasattr(attn, "qk_norm")
        assert has_qk_norm == expected_rope, (
            f"Layer {layer_idx}: expected qk_norm={expected_rope}, got {has_qk_norm}"
        )


def test_llama4_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = Llama4ForCausalLM(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.qk_norm.eps",  # L2Norm has eps (as buffer or attr)
        "model.layers.0.feed_forward.router.weight",
        "model.layers.0.feed_forward.experts.gate_up_proj",
        "model.layers.0.feed_forward.experts.down_proj",
        "model.layers.0.feed_forward.shared_expert.gate_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for key in expected_key_patterns:
        if "qk_norm.eps" in key:
            # L2Norm has no learnable parameters, just check the module exists
            continue
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {sorted(state_dict.keys())[:20]}..."
        )
