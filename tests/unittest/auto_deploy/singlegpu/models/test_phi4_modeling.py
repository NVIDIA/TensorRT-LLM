"""Tests for Phi-4 custom model implementation.

This module tests the custom Phi-4 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility.

Hierarchical test levels:
1. Block equivalence (MLP, Attention, RMSNorm)
2. Layer equivalence (DecoderLayer)
3. Full model equivalence (ForCausalLM) — on both CPU and CUDA
4. Export test (torch_export_to_gm with dynamic shapes, eager vs graph comparison)
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_phi4 import (
    Phi4Attention,
    Phi4DecoderLayer,
    Phi4ForCausalLM,
    Phi4MLP,
    Phi4RMSNorm,
    Phi4RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config():
    """Create a small Phi3Config for testing."""
    from transformers.models.phi3.configuration_phi3 import Phi3Config

    return Phi3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        hidden_act="silu",
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        pad_token_id=0,
        attention_dropout=0.0,
        resid_pdrop=0.0,
    )


# =============================================================================
# HF Reference Helpers
# =============================================================================


def _create_hf_config():
    """Create a HF Phi3Config with _attn_implementation set for direct instantiation.

    When creating HF models directly (not through from_pretrained), the
    _attn_implementation attribute may not be set. We set it to "eager"
    explicitly to avoid KeyError in the HF attention forward pass.
    """
    config = _create_small_config()
    config._attn_implementation = "eager"
    return config


def _make_causal_mask(S, device, dtype):
    """Create a causal attention mask for HF attention.

    HF eager attention expects mask of shape [B, 1, S, S] with 0 for allowed
    positions and large negative values for masked positions.
    """
    mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def _get_hf_model_class():
    """Get the HF Phi3ForCausalLM class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

        return Phi3ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Phi3Attention class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3Attention

        return Phi3Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Phi3MLP class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3MLP

        return Phi3MLP
    except ImportError:
        return None


def _get_hf_norm_class():
    """Get the HF Phi3RMSNorm class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm

        return Phi3RMSNorm
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Phi3DecoderLayer class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

        return Phi3DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_class():
    """Get the HF Phi3RotaryEmbedding class."""
    try:
        from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

        return Phi3RotaryEmbedding
    except ImportError:
        return None


# =============================================================================
# Structural Tests
# =============================================================================


def test_phi4_config_uses_phi3():
    """Test that the model correctly uses Phi3Config."""
    config = _create_small_config()
    assert config.model_type == "phi3"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "partial_rotary_factor")


def test_phi4_gqa_structure():
    """Test that GQA is set up correctly (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = Phi4ForCausalLM(config)

    attn = model.model.layers[0].self_attn
    assert attn.num_heads == config.num_attention_heads
    assert attn.num_key_value_heads == config.num_key_value_heads
    assert attn.num_key_value_groups == config.num_attention_heads // config.num_key_value_heads

    # Check QKV proj output size
    expected_qkv_size = (
        config.num_attention_heads * attn.head_dim + 2 * config.num_key_value_heads * attn.head_dim
    )
    assert attn.qkv_proj.out_features == expected_qkv_size


# =============================================================================
# Numerical Equivalence Tests (Level 1: Block Equivalence)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_phi4_rmsnorm_numerical_equivalence(B, S, dtype):
    """Test RMSNorm produces identical output to HF implementation."""
    HFNorm = _get_hf_norm_class()
    if HFNorm is None:
        pytest.skip("transformers doesn't have Phi3RMSNorm")

    device = "cuda"
    config = _create_small_config()
    H = config.hidden_size

    hf_norm = HFNorm(H, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
    custom_norm = Phi4RMSNorm(H, eps=config.rms_norm_eps).to(device=device, dtype=dtype)

    # Load same weights
    custom_norm.load_state_dict(hf_norm.state_dict())

    x = torch.randn(B, S, H, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_phi4_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces identical output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Phi3MLP")

    device = "cuda"
    config = _create_small_config()
    H = config.hidden_size

    hf_mlp = HFMLP(config).to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Phi4MLP(config).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, H, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_phi4_attention_numerical_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF implementation."""
    HFAttention = _get_hf_attention_class()
    HFRotary = _get_hf_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have Phi3Attention or Phi3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()
    H = config.hidden_size

    # Create HF attention and rotary
    hf_attn = HFAttention(hf_config, layer_idx=0).to(device=device, dtype=dtype)
    hf_attn.eval()
    hf_rotary = HFRotary(hf_config).to(device=device, dtype=dtype)

    # Create custom attention and rotary
    custom_attn = Phi4Attention(config, layer_idx=0).to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    head_dim = H // config.num_attention_heads
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    custom_rotary = Phi4RotaryEmbedding(
        rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Create causal mask for HF (our custom attention uses is_causal=True internally)
    causal_mask = _make_causal_mask(S, device, dtype)

    # HF forward
    hf_position_embeddings = hf_rotary(x, position_ids)
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=hf_position_embeddings,
        attention_mask=causal_mask,
    )

    # Custom forward
    custom_position_embeddings = custom_rotary(x)
    custom_out = custom_attn(x, position_ids, custom_position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention output mismatch")


# =============================================================================
# Numerical Equivalence Tests (Level 2: Layer Equivalence)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_phi4_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF implementation."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    HFRotary = _get_hf_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have Phi3DecoderLayer or Phi3RotaryEmbedding")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()
    H = config.hidden_size

    # Create HF layer and rotary
    hf_layer = HFDecoderLayer(hf_config, layer_idx=0).to(device=device, dtype=dtype)
    hf_layer.eval()
    hf_rotary = HFRotary(hf_config).to(device=device, dtype=dtype)

    # Create custom layer and rotary
    custom_layer = Phi4DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)

    # Load weights from HF layer - filter out dropout layers we don't have
    hf_sd = hf_layer.state_dict()
    custom_sd = {k: v for k, v in hf_sd.items() if "dropout" not in k}
    custom_layer.load_state_dict(custom_sd)
    custom_layer.eval()

    head_dim = H // config.num_attention_heads
    rotary_dim = int(head_dim * config.partial_rotary_factor)
    custom_rotary = Phi4RotaryEmbedding(
        rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device=device, dtype=dtype)

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Create causal mask for HF (our custom attention uses is_causal=True internally)
    causal_mask = _make_causal_mask(S, device, dtype)

    # HF forward
    hf_position_embeddings = hf_rotary(x, position_ids)
    hf_out = hf_layer(
        hidden_states=x,
        position_embeddings=hf_position_embeddings,
        attention_mask=causal_mask,
    )

    # Custom forward
    custom_position_embeddings = custom_rotary(x)
    custom_out = custom_layer(x, position_ids, custom_position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer output mismatch")


# =============================================================================
# Numerical Equivalence Tests (Level 3: Full Model Equivalence)
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_phi4_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF on CUDA."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Phi3ForCausalLM")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF model
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load weights
    custom_model = Phi4ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    # Convert HF state dict - filter out dropout-related keys
    hf_sd = hf_model.state_dict()
    custom_sd = {k: v for k, v in hf_sd.items() if "dropout" not in k}
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Run both
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits mismatch",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.float32])
@torch.no_grad()
def test_phi4_full_model_numerical_equivalence_cpu(B, S, dtype):
    """Test full model produces numerically equivalent output to HF on CPU.

    Confirms no GPU-only ops in the model's forward path.
    """
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Phi3ForCausalLM")

    device = "cpu"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF model
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load weights
    custom_model = Phi4ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    hf_sd = hf_model.state_dict()
    custom_sd = {k: v for k, v in hf_sd.items() if "dropout" not in k}
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits mismatch (CPU)",
    )


# =============================================================================
# Export Test (Level 4)
# =============================================================================


def test_phi4_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. Export succeeds without graph breaks
    2. Exported graph output matches eager model output (assert_rmse_close)
    3. Dynamic shapes work with a second input shape
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Phi4ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get eager model output for comparison
    with torch.inference_mode():
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

    move_to_device(gm, device)

    # Verify exported graph output matches eager model output
    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all(), "Logits should not contain NaN or Inf"

    assert_rmse_close(
        logits.float(),
        eager_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Exported graph output does not match eager model output",
    )

    # Test with different input shape to verify dynamic shapes work
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()
