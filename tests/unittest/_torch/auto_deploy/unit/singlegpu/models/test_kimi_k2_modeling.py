"""Tests for Kimi-K2 / Kimi-K2.5 custom model implementation.

This module tests the custom Kimi-K2 model implementation (DeepSeek-V3 variant)
which uses auto_deploy custom ops (torch_mla, torch_moe, etc.) for export
compatibility.

Hierarchical test levels:
1. Block equivalence — MLP, MoE, Attention individually
2. Layer equivalence — Full decoder layer (dense and MoE)
3. Full model equivalence — End-to-end logits comparison
4. Export test — torch_export_to_gm with dynamic shapes
"""

import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_kimi_k2 import (
    KimiK2Attention,
    KimiK2Config,
    KimiK2DecoderLayer,
    KimiK2ForCausalLM,
    KimiK2MLP,
    KimiK2MoE,
    KimiK2RotaryEmbedding,
    KimiK25Config,
    KimiK25ForConditionalGeneration,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Helpers
# =============================================================================


def _create_small_text_config() -> KimiK2Config:
    """Create a small Kimi-K2 text config for testing."""
    return KimiK2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,  # Layer 0 dense, layers 1-2 MoE
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # MLA params (scaled down)
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        # MoE params (scaled down)
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )


def _create_small_vlm_config() -> KimiK25Config:
    """Create a small KimiK25 config wrapping the text config."""
    text_config = _create_small_text_config()
    return KimiK25Config(
        text_config=text_config,
        vision_config=None,
        pad_token_id=0,
    )


def _create_moe_layer(config: KimiK2Config) -> KimiK2MoE:
    """Create a MoE layer from config with reproducible gate weights."""
    moe = KimiK2MoE(config)
    moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
    return moe


# =============================================================================
# Export Tests
# =============================================================================


@torch.no_grad()
def test_kimi_k2_text_model_can_be_exported():
    """Test that KimiK2ForCausalLM can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output to eager
    3. Dynamic shapes work correctly with different input sizes
    """
    device = "cpu"
    dtype = torch.bfloat16
    config = _create_small_text_config()

    model = KimiK2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

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

    # Eager reference output
    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    # Exported graph output
    out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    torch.testing.assert_close(logits.float(), eager_out.logits.float(), rtol=1e-3, atol=1e-3)

    # Test with different input shape to verify dynamic shapes work
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)
    out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    expected_shape = (B2, S2, config.vocab_size)
    assert logits2.shape == expected_shape, (
        f"Dynamic shape test failed: expected {expected_shape}, got {logits2.shape}"
    )
    torch.testing.assert_close(logits2.float(), eager_out2.logits.float(), rtol=1e-3, atol=1e-3)


# =============================================================================
# Structural Tests
# =============================================================================


def test_kimi_k2_config_registration():
    """Test that configs are properly instantiated with correct model_type."""
    text_config = _create_small_text_config()
    assert text_config.model_type == "kimi_k2"
    assert hasattr(text_config, "hidden_size")
    assert hasattr(text_config, "n_routed_experts")
    assert hasattr(text_config, "kv_lora_rank")
    assert hasattr(text_config, "qk_rope_head_dim")
    assert hasattr(text_config, "moe_layer_freq")

    vlm_config = _create_small_vlm_config()
    assert vlm_config.model_type == "kimi_k25"
    assert isinstance(vlm_config.text_config, KimiK2Config)


def test_kimi_k2_layer_types():
    """Test that layer 0 uses dense MLP and later layers use MoE."""
    config = _create_small_text_config()
    model = KimiK2ForCausalLM(config)

    layer0_mlp = model.model.layers[0].mlp
    assert type(layer0_mlp).__name__ == "KimiK2MLP", (
        f"Layer 0 should use KimiK2MLP, got {type(layer0_mlp).__name__}"
    )

    for i in range(1, config.num_hidden_layers):
        layer_mlp = model.model.layers[i].mlp
        assert type(layer_mlp).__name__ == "KimiK2MoE", (
            f"Layer {i} should use KimiK2MoE, got {type(layer_mlp).__name__}"
        )


def test_kimi_k2_expert_structure():
    """Test that experts have correct structure for checkpoint loading."""
    config = _create_small_text_config()
    moe = KimiK2MoE(config)

    assert isinstance(moe.experts, torch.nn.ModuleList), "experts should be nn.ModuleList"
    assert len(moe.experts) == config.n_routed_experts, (
        f"Expected {config.n_routed_experts} experts, got {len(moe.experts)}"
    )

    for i, expert in enumerate(moe.experts):
        assert hasattr(expert, "gate_proj"), f"Expert {i} missing gate_proj"
        assert hasattr(expert, "up_proj"), f"Expert {i} missing up_proj"
        assert hasattr(expert, "down_proj"), f"Expert {i} missing down_proj"

    state_dict = moe.state_dict()
    expected_keys = [
        "experts.0.gate_proj.weight",
        "experts.0.up_proj.weight",
        "experts.0.down_proj.weight",
    ]
    for key in expected_keys:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )


def test_kimi_k25_weight_layout():
    """Test that VLM wrapper has correct weight prefix for checkpoint compatibility."""
    config = _create_small_vlm_config()
    model = KimiK25ForConditionalGeneration(config)

    state_dict = model.state_dict()
    # Check that weights are under language_model.* prefix
    assert any(k.startswith("language_model.model.") for k in state_dict), (
        "Expected weights under 'language_model.model.*' prefix"
    )
    assert any(k.startswith("language_model.lm_head.") for k in state_dict), (
        "Expected weights under 'language_model.lm_head.*' prefix"
    )


def test_kimi_k2_shared_experts():
    """Test that shared experts are present when n_shared_experts > 0."""
    config = _create_small_text_config()
    moe = KimiK2MoE(config)

    assert moe.shared_experts is not None, "shared_experts should be present"
    assert isinstance(moe.shared_experts, KimiK2MLP), "shared_experts should be KimiK2MLP"

    # Shared expert intermediate size = moe_intermediate_size * n_shared_experts
    expected_intermediate = config.moe_intermediate_size * config.n_shared_experts
    assert moe.shared_experts.intermediate_size == expected_intermediate, (
        f"Expected shared expert intermediate_size={expected_intermediate}, "
        f"got {moe.shared_experts.intermediate_size}"
    )


# =============================================================================
# HF Reference Import Helpers (for numerical equivalence tests)
# =============================================================================


def _get_hf_config_class():
    """Get the HF DeepseekV3Config class.

    Returns None if transformers doesn't have deepseek_v3 (requires v4.57+).
    """
    try:
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        return DeepseekV3Config
    except ImportError:
        return None


def _get_hf_model_class():
    """Get the HF DeepseekV3ForCausalLM class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

        return DeepseekV3ForCausalLM
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF DeepseekV3MoE class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

        return DeepseekV3MoE
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF DeepseekV3MLP class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP

        return DeepseekV3MLP
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF DeepseekV3Attention class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

        return DeepseekV3Attention
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF DeepseekV3DecoderLayer class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer

        return DeepseekV3DecoderLayer
    except ImportError:
        return None


# =============================================================================
# HF Config and Weight Conversion Helpers
# =============================================================================


def _create_hf_config():
    """Create HF DeepseekV3Config matching our small test config.

    Returns None if DeepseekV3Config is not available.
    """
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        return None

    config = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # MLA params
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        # MoE params
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=True,  # HF default: interleaved RoPE format
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )

    # Use eager attention for deterministic comparison
    config._attn_implementation = "eager"

    return config


def _deinterleave_attention_weights(state_dict, config, prefix=""):
    """De-interleave RoPE weight columns for attention weights.

    Applies the same transformation as mla_rope_utils._rope_deinterleave_load_hook
    but for a single layer's attention weights. The prefix parameter allows reuse
    for both block-level (prefix="") and layer-level (prefix="self_attn.") tests.
    """
    d = config.qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = config.qk_nope_head_dim + d

    # --- q_b_proj.weight ---
    q_key = f"{prefix}q_b_proj.weight"
    if q_key in state_dict:
        w = state_dict[q_key]
        w = w.view(config.num_attention_heads, qk_head_dim, -1)
        w_nope = w[:, : config.qk_nope_head_dim, :]
        w_rope = w[:, config.qk_nope_head_dim :, :]
        w_rope = w_rope[:, perm, :]
        w = torch.cat([w_nope, w_rope], dim=1)
        state_dict[q_key] = w.view(-1, w.shape[-1])

    # --- kv_a_proj_with_mqa.weight ---
    kv_key = f"{prefix}kv_a_proj_with_mqa.weight"
    if kv_key in state_dict:
        w = state_dict[kv_key]
        w_kv = w[: config.kv_lora_rank, :]
        w_pe = w[config.kv_lora_rank :, :]
        w_pe = w_pe[perm, :]
        state_dict[kv_key] = torch.cat([w_kv, w_pe], dim=0)

    # --- kv_a_proj_with_mqa.bias (if present) ---
    kv_bias_key = f"{prefix}kv_a_proj_with_mqa.bias"
    if kv_bias_key in state_dict:
        b = state_dict[kv_bias_key]
        b_kv = b[: config.kv_lora_rank]
        b_pe = b[config.kv_lora_rank :]
        b_pe = b_pe[perm]
        state_dict[kv_bias_key] = torch.cat([b_kv, b_pe])

    return state_dict


def _create_causal_mask(B, S, device, dtype):
    """Create a 4D causal attention mask for HF eager attention.

    Returns a [B, 1, S, S] mask with 0 for attended positions and -inf for masked.
    """
    mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)


# =============================================================================
# Numerical Equivalence Tests
# These tests compare our custom Kimi-K2 implementation against the HF
# DeepseekV3 reference with identical weights and inputs.
# =============================================================================

# --- Level 1: Block Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF DeepseekV3MLP."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF MLP
    hf_mlp = HFMLP(hf_config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights (identical structure)
    custom_mlp = KimiK2MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    # Create input
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    # Run both
    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # Compare — identical math, tight tolerance
    rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_moe_numerical_equivalence(B, S, dtype):
    """Test MoE produces numerically equivalent output to HF DeepseekV3MoE."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF MoE and initialize gate weights for reproducibility
    hf_moe = HFMoE(hf_config)
    hf_moe.gate.weight = torch.nn.Parameter(torch.randn_like(hf_moe.gate.weight))
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    # Create custom MoE and load same weights
    # State dict keys match: gate.weight, gate.e_score_correction_bias,
    # experts.{i}.{gate,up,down}_proj.weight, shared_experts.*
    custom_moe = KimiK2MoE(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    # Create input
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    # Run both
    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    # Compare — fused noaux_tc routing vs Python routing, wider tolerance
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_attention_numerical_equivalence(B, S, dtype):
    """Test attention produces numerically equivalent output to HF DeepseekV3Attention.

    HF uses rope_interleave=True (interleaved RoPE format in weights).
    Our model uses NeoX format. Weights are de-interleaved before loading.
    """
    HFAttn = _get_hf_attention_class()
    if HFAttn is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF attention
    hf_attn = HFAttn(hf_config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom attention with de-interleaved weights
    custom_attn = KimiK2Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)

    # Copy weights, de-interleaving RoPE dimensions
    hf_state_dict = dict(hf_attn.state_dict())
    _deinterleave_attention_weights(hf_state_dict, config)
    custom_attn.load_state_dict(hf_state_dict)
    custom_attn.eval()

    # Create inputs
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Create position embeddings using our rotary embedding (full table)
    rotary_emb = KimiK2RotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    full_cos, full_sin = rotary_emb(x)  # [max_seq_len, head_dim]

    # HF expects position-indexed cos/sin: [B, S, head_dim]
    hf_cos = full_cos[position_ids]
    hf_sin = full_sin[position_ids]

    # Causal mask for HF eager attention: [B, 1, S, S]
    causal_mask = _create_causal_mask(B, S, device, dtype)

    # Run HF attention
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Run custom attention (takes full table + position_ids)
    custom_out = custom_attn(x, position_ids, (full_cos, full_sin))

    # Compare — RoPE format conversion + fused MLA vs eager attention
    rtol, atol = 0.02, 0.02
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


# --- Level 2: Layer Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_dense_layer_numerical_equivalence(B, S, dtype):
    """Test dense decoder layer (layer 0) matches HF DeepseekV3DecoderLayer."""
    HFLayer = _get_hf_decoder_layer_class()
    if HFLayer is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF layer (layer 0 = dense MLP)
    hf_layer = HFLayer(hf_config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom layer with de-interleaved attention weights
    custom_layer = KimiK2DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)

    hf_state_dict = dict(hf_layer.state_dict())
    _deinterleave_attention_weights(hf_state_dict, config, prefix="self_attn.")
    custom_layer.load_state_dict(hf_state_dict)
    custom_layer.eval()

    # Create inputs
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Position embeddings
    rotary_emb = KimiK2RotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    full_cos, full_sin = rotary_emb(x)
    hf_cos = full_cos[position_ids]
    hf_sin = full_sin[position_ids]

    causal_mask = _create_causal_mask(B, S, device, dtype)

    # Run HF layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(hf_cos, hf_sin),
    )

    # Run custom layer
    custom_out = custom_layer(x, position_ids, (full_cos, full_sin))

    # Compare
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_moe_layer_numerical_equivalence(B, S, dtype):
    """Test MoE decoder layer (layer 1) matches HF DeepseekV3DecoderLayer."""
    HFLayer = _get_hf_decoder_layer_class()
    if HFLayer is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF layer (layer 1 = MoE)
    hf_layer = HFLayer(hf_config, layer_idx=1)
    # Initialize gate weights for reproducibility
    hf_layer.mlp.gate.weight = torch.nn.Parameter(torch.randn_like(hf_layer.mlp.gate.weight))
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom layer with de-interleaved attention weights
    custom_layer = KimiK2DecoderLayer(config, layer_idx=1)
    custom_layer.to(device=device, dtype=dtype)

    hf_state_dict = dict(hf_layer.state_dict())
    _deinterleave_attention_weights(hf_state_dict, config, prefix="self_attn.")
    custom_layer.load_state_dict(hf_state_dict)
    custom_layer.eval()

    # Create inputs
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    rotary_emb = KimiK2RotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    full_cos, full_sin = rotary_emb(x)
    hf_cos = full_cos[position_ids]
    hf_sin = full_sin[position_ids]

    causal_mask = _create_causal_mask(B, S, device, dtype)

    # Run HF layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(hf_cos, hf_sin),
    )

    # Run custom layer
    custom_out = custom_layer(x, position_ids, (full_cos, full_sin))

    # Compare — includes MoE routing differences
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


# --- Level 3: Full Model Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_kimi_k2_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent logits to HF DeepseekV3ForCausalLM.

    Weight conversion: the load_state_dict_pre_hook on KimiK2ForCausalLM
    automatically de-interleaves RoPE weights. All other weights have matching keys.
    """
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have DeepseekV3 (requires v4.57+)")

    device = "cpu"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF model
    hf_model = HFModel(hf_config)
    # Initialize all gate weights for reproducibility
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model — the pre-hook handles RoPE de-interleaving
    custom_model = KimiK2ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    hf_state_dict = hf_model.state_dict()
    custom_model.load_state_dict(hf_state_dict)
    custom_model.eval()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Run both
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    # Compare logits — cast to float32 since HF may return bfloat16
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rtol=rtol,
        atol=atol,
    )
