"""Tests for Qwen3.5 (dense) custom model implementation.

This module tests the custom Qwen3.5 dense model implementation which uses
auto_deploy custom ops for export compatibility. Qwen3.5 is a hybrid model
with linear attention (GatedDeltaNet) and full attention layers, without MoE.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))

# Full attention layer index for a 3-layer model: [linear, linear, full]
_FULL_ATTN_LAYER_IDX = 2


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_text_config() -> Qwen3_5TextConfig:
    """Create a small Qwen3.5 text config for testing (3 layers)."""
    return Qwen3_5TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 2 linear + 1 full
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=32,
        attn_output_gate=True,
        # linear attention params (scaled down)
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        # RoPE
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        # layer types: 2 linear + 1 full
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        pad_token_id=0,
    )


def _create_small_composite_config() -> Qwen3_5Config:
    """Create a small Qwen3.5 composite config for testing."""
    return Qwen3_5Config(
        text_config=_create_small_text_config().to_dict(),
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
        },
        tie_word_embeddings=True,
    )


# =============================================================================
# HF Reference Helpers
# =============================================================================


def _get_hf_model_class():
    """Get the HF Qwen3Next model class (Qwen3.5 dense shares Qwen3Next arch)."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM as HFCls

        return HFCls
    except ImportError:
        return None


def _get_hf_config_class():
    """Get the HF Qwen3Next config class."""
    try:
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig as HFCls

        return HFCls
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Qwen3Next attention class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention as HFCls

        return HFCls
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF Qwen3Next MLP class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextMLP as HFCls

        return HFCls
    except ImportError:
        return None


def _get_hf_gated_deltanet_class():
    """Get the HF Qwen3Next GatedDeltaNet class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextGatedDeltaNet as HFCls,
        )

        return HFCls
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF Qwen3Next decoder layer class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDecoderLayer as HFCls,
        )

        return HFCls
    except ImportError:
        return None


def _create_hf_config():
    """Create an HF Qwen3Next config matching our test config."""
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        return None

    return HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=32,
        # linear attention params
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        # No MoE for dense model
        decoder_sparse_step=0,
        num_experts=0,
        mlp_only_layers=list(range(3)),
        # layer types
        layer_types=["linear_attention", "linear_attention", "full_attention"],
    )


# =============================================================================
# Weight Conversion Helpers
# =============================================================================


def _convert_hf_gdn_state_dict(hf_state_dict: dict, config) -> dict:
    """Convert HF GatedDeltaNet state dict to our custom format.

    HF has:
      - in_proj_qkvz.weight: [key_dim*2 + value_dim*2, hidden_size]
      - in_proj_ba.weight: [num_v_heads*2, hidden_size]

    Custom has:
      - in_proj_qkv.weight: [key_dim*2 + value_dim, hidden_size]
      - in_proj_z.weight: [value_dim, hidden_size]
      - in_proj_b.weight: [num_v_heads, hidden_size]
      - in_proj_a.weight: [num_v_heads, hidden_size]
    """
    custom_sd = {}
    num_k_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    head_k_dim = config.linear_key_head_dim
    head_v_dim = config.linear_value_head_dim

    for key, value in hf_state_dict.items():
        if "in_proj_qkvz.weight" in key:
            prefix = key.replace("in_proj_qkvz.weight", "")
            # HF packs as interleaved groups:
            # [num_k_heads, 2*head_k_dim + 2*(num_v_heads//num_k_heads)*head_v_dim, H]
            new_shape = (
                num_k_heads,
                2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads,
                config.hidden_size,
            )
            grouped = value.view(new_shape)
            split_sizes = [
                head_k_dim,
                head_k_dim,
                (num_v_heads // num_k_heads) * head_v_dim,
                (num_v_heads // num_k_heads) * head_v_dim,
            ]
            q, k, v, z = torch.split(grouped, split_sizes, dim=1)
            q = q.reshape(-1, config.hidden_size)
            k = k.reshape(-1, config.hidden_size)
            v = v.reshape(-1, config.hidden_size)
            z = z.reshape(-1, config.hidden_size)

            qkv = torch.cat([q, k, v], dim=0)
            custom_sd[prefix + "in_proj_qkv.weight"] = qkv
            custom_sd[prefix + "in_proj_z.weight"] = z

        elif "in_proj_ba.weight" in key:
            prefix = key.replace("in_proj_ba.weight", "")
            grouped = value.view(num_k_heads, 2 * (num_v_heads // num_k_heads), config.hidden_size)
            b_per_group = num_v_heads // num_k_heads
            b, a = torch.split(grouped, [b_per_group, b_per_group], dim=1)
            b = b.reshape(-1, config.hidden_size)
            a = a.reshape(-1, config.hidden_size)
            custom_sd[prefix + "in_proj_b.weight"] = b
            custom_sd[prefix + "in_proj_a.weight"] = a

        else:
            custom_sd[key] = value

    return custom_sd


def _convert_hf_full_model_state_dict(hf_state_dict: dict, config) -> dict:
    """Convert full HF model state dict to custom format."""
    custom_sd = {}

    # Collect GDN layer keys and non-GDN keys
    gdn_layer_keys = {}
    for key in hf_state_dict:
        if ".linear_attn." in key:
            parts = key.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is not None:
                prefix = f"model.layers.{layer_idx}.linear_attn."
                gdn_key = key[len(prefix) :]
                if prefix not in gdn_layer_keys:
                    gdn_layer_keys[prefix] = {}
                gdn_layer_keys[prefix][gdn_key] = hf_state_dict[key]
        else:
            custom_sd[key] = hf_state_dict[key]

    # Convert each GDN layer
    for prefix, gdn_sd in gdn_layer_keys.items():
        converted = _convert_hf_gdn_state_dict(gdn_sd, config)
        for k, v in converted.items():
            custom_sd[prefix + k] = v

    return custom_sd


def _convert_hf_decoder_layer_state_dict(hf_state_dict: dict, config, layer_type: str) -> dict:
    """Convert HF decoder layer state dict to custom format."""
    if layer_type == "full_attention":
        return dict(hf_state_dict)  # No conversion needed for full attention layers
    else:
        return _convert_hf_gdn_state_dict(hf_state_dict, config)


# =============================================================================
# Structural Tests
# =============================================================================


def test_qwen3_5_config_registration():
    """Test that the config is properly registered with correct model_type."""
    text_config = _create_small_text_config()
    assert text_config.model_type == "qwen3_5_text"
    assert hasattr(text_config, "hidden_size")
    assert hasattr(text_config, "linear_key_head_dim")
    assert hasattr(text_config, "layer_types")

    composite_config = _create_small_composite_config()
    assert composite_config.model_type == "qwen3_5"
    assert hasattr(composite_config, "text_config")
    assert hasattr(composite_config, "vision_config")


def test_qwen3_5_layer_types():
    """Test that layers have correct types (linear vs full attention)."""
    config = _create_small_text_config()
    model = Qwen3_5ForCausalLM(config)

    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        expected_type = config.layer_types[i]
        if expected_type == "linear_attention":
            assert hasattr(layer, "linear_attn"), f"Layer {i} should have linear_attn"
        else:
            assert hasattr(layer, "self_attn"), f"Layer {i} should have self_attn"


def test_qwen3_5_tied_embeddings():
    """Test that embeddings are tied when tie_word_embeddings=True."""
    config = _create_small_text_config()
    assert config.tie_word_embeddings is True
    model = Qwen3_5ForCausalLM(config)
    assert model.lm_head.weight is model.model.embed_tokens.weight


# =============================================================================
# Block Equivalence Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    hf_mlp = HFMLP(hf_config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Qwen3_5MLP(config)
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
def test_qwen3_5_rmsnorm_offset_hook(B, S, dtype):
    """Test that the RMSNorm load hook properly offsets weights by +1."""
    device = "cuda"
    config = _create_small_text_config()

    norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.to(device=device, dtype=dtype)

    # Simulate loading from checkpoint: checkpoint stores zero-initialized weights
    checkpoint_state = {"weight": torch.zeros(config.hidden_size, device=device, dtype=dtype)}
    norm.load_state_dict(checkpoint_state)

    # After loading, weight should be ones (0 + 1)
    torch.testing.assert_close(
        norm.weight,
        torch.ones(config.hidden_size, device=device, dtype=dtype),
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_gated_deltanet_numerical_equivalence(B, S, dtype):
    """Test GatedDeltaNet produces numerically equivalent output to HF reference."""
    HFGatedDeltaNet = _get_hf_gated_deltanet_class()
    if HFGatedDeltaNet is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF GatedDeltaNet
    hf_gdn = HFGatedDeltaNet(hf_config, layer_idx=0)
    hf_gdn.to(device=device, dtype=dtype)
    hf_gdn.eval()

    # Create custom GatedDeltaNet and load converted weights
    custom_gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    custom_gdn.to(device=device, dtype=dtype)

    hf_sd = hf_gdn.state_dict()
    custom_sd = _convert_hf_gdn_state_dict(hf_sd, config)
    custom_gdn.load_state_dict(custom_sd)
    custom_gdn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # HF GatedDeltaNet returns just the output tensor (no cache)
    hf_out = hf_gdn(x)

    custom_out = custom_gdn(x)

    assert_rmse_close(
        custom_out,
        hf_out,
        rmse_ratio_tol=0.10,
        msg="GatedDeltaNet output diverges from HF reference",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_attention_numerical_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF reference."""
    HFAttention = _get_hf_attention_class()
    if HFAttention is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF Attention
    hf_attn = HFAttention(hf_config, layer_idx=_FULL_ATTN_LAYER_IDX)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom Attention with same weights (same weight format, no conversion needed)
    custom_attn = Qwen3_5Attention(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Compute position embeddings using our rotary embedding
    rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    rotary_emb.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x, position_ids)

    # HF attention forward: returns (attn_output, attn_weights)
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )

    custom_out = custom_attn(x, position_embeddings=position_embeddings)

    assert_rmse_close(
        custom_out,
        hf_out,
        rmse_ratio_tol=0.10,
        msg="Attention output diverges from HF reference",
    )


# =============================================================================
# Layer Equivalence Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_linear_decoder_layer_equivalence(B, S, dtype):
    """Test linear attention decoder layer equivalence against HF reference."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF decoder layer (linear_attention, layer 0)
    hf_layer = HFDecoderLayer(hf_config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer and load converted weights
    custom_layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)

    hf_sd = hf_layer.state_dict()
    custom_sd = _convert_hf_decoder_layer_state_dict(hf_sd, config, "linear_attention")
    custom_layer.load_state_dict(custom_sd)
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    rotary_emb.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x, position_ids)

    # HF decoder layer returns just hidden_states for non-attention layers
    hf_out = hf_layer(x, position_embeddings=position_embeddings)

    custom_out = custom_layer(x, position_embeddings=position_embeddings)

    assert_rmse_close(
        custom_out,
        hf_out,
        rmse_ratio_tol=0.05,
        msg="Linear decoder layer output diverges from HF reference",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_full_attention_decoder_layer_equivalence(B, S, dtype):
    """Test full attention decoder layer equivalence against HF reference."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF decoder layer (full_attention)
    hf_layer = HFDecoderLayer(hf_config, layer_idx=_FULL_ATTN_LAYER_IDX)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer with same weights (no GDN conversion needed)
    custom_layer = Qwen3_5DecoderLayer(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    custom_layer.to(device=device, dtype=dtype)

    hf_sd = hf_layer.state_dict()
    custom_sd = _convert_hf_decoder_layer_state_dict(hf_sd, config, "full_attention")
    custom_layer.load_state_dict(custom_sd)
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    rotary_emb.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary_emb(x, position_ids)

    hf_out = hf_layer(x, position_embeddings=position_embeddings)

    custom_out = custom_layer(x, position_embeddings=position_embeddings)

    assert_rmse_close(
        custom_out,
        hf_out,
        rmse_ratio_tol=0.05,
        msg="Full attention decoder layer output diverges from HF reference",
    )


# =============================================================================
# Full Model Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create HF model
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load converted weights
    custom_model = Qwen3_5ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    hf_state_dict = hf_model.state_dict()
    custom_state_dict = _convert_hf_full_model_state_dict(hf_state_dict, config)
    custom_model.load_state_dict(custom_state_dict)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    # Check shape and finiteness
    assert custom_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(custom_out.logits).all(), "Custom model logits contain NaN/Inf"

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits diverge from HF reference",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_5_conditional_model(B, S, dtype):
    """Test that the conditional (multimodal) model produces valid output for text-only.

    This is a smoke test for the vision+language wrapper pipeline. No HF equivalent
    exists for the composite Qwen3_5ForConditionalGeneration, so we verify the text
    CausalLM output (tested with HF equivalence above) is consistent when called through
    the multimodal wrapper with text-only inputs.
    """
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have qwen3_next (requires newer version)")

    device = "cuda"
    config = _create_small_composite_config()
    text_config = _create_small_text_config()
    hf_config = _create_hf_config()

    # Create the text-only CausalLM as reference (with HF weights)
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    ref_model = Qwen3_5ForCausalLM(text_config)
    ref_model.to(device=device, dtype=dtype)
    hf_sd = hf_model.state_dict()
    ref_sd = _convert_hf_full_model_state_dict(hf_sd, text_config)
    ref_model.load_state_dict(ref_sd)
    ref_model.eval()

    # Create the conditional model and copy the language_model weights from the ref model
    cond_model = Qwen3_5ForConditionalGeneration(config)
    cond_model.to(device=device, dtype=dtype)

    # Map ref_model weights into the conditional model's language_model path
    cond_sd = cond_model.state_dict()
    for key, val in ref_model.state_dict().items():
        # ref_model: model.layers.* -> cond_model: model.language_model.layers.*
        cond_key = key.replace("model.", "model.language_model.", 1)
        if cond_key in cond_sd:
            cond_sd[cond_key] = val
    # lm_head weight
    if "lm_head.weight" in ref_model.state_dict():
        cond_sd["lm_head.weight"] = ref_model.state_dict()["lm_head.weight"]
    cond_model.load_state_dict(cond_sd)
    cond_model.eval()

    B_val, S_val = B, S
    input_ids = torch.randint(0, config.text_config.vocab_size, (B_val, S_val), device=device)
    position_ids = torch.arange(S_val, device=device).unsqueeze(0).expand(B_val, -1)

    ref_out = ref_model(input_ids=input_ids, position_ids=position_ids)
    cond_out = cond_model(input_ids=input_ids)

    assert cond_out.logits.shape == (B_val, S_val, config.text_config.vocab_size)

    assert_rmse_close(
        cond_out.logits.float(),
        ref_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Conditional model text-only logits diverge from CausalLM reference",
    )


# =============================================================================
# Export Test
# =============================================================================


def test_qwen3_5_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_text_config()

    model = Qwen3_5ForCausalLM(config)
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

    # Get eager model output for numerical comparison
    with torch.inference_mode():
        eager_out = model(input_ids=input_ids, position_ids=position_ids)
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all()

    # Compare exported graph output against eager model output
    assert_rmse_close(
        logits.float(),
        eager_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Exported model logits diverge from eager model",
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
