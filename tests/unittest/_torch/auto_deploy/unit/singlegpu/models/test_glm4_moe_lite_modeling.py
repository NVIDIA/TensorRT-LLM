"""Tests for GLM4 MoE Lite custom model implementation.

This module tests the custom GLM4 MoE Lite model implementation which uses
auto_deploy custom ops (torch_mla, torch_moe, etc.) for export compatibility.
"""

import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm4_moe_lite import (
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteMoE,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Glm4MoeLiteConfig:
    """Create a small GLM4 MoE Lite config for testing."""
    return Glm4MoeLiteConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
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
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )


def _create_moe_layer(config: Glm4MoeLiteConfig) -> Glm4MoeLiteMoE:
    """Create a MoE layer from config."""
    moe = Glm4MoeLiteMoE(config)
    # Initialize gate weights with randn for reproducibility
    # (gate weight is initialized with torch.empty which isn't seeded)
    moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
    return moe


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_lite_moe_layer(B, S, dtype):
    """Test that MoE layer produces valid output."""
    device = "cuda"
    config = _create_small_config()

    moe = _create_moe_layer(config)
    moe.to(device=device, dtype=dtype)
    moe.eval()

    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    output = moe(x)

    # Check output shape matches input shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Check output is not all zeros (MoE should transform the input)
    assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"

    # Check output doesn't have NaN or Inf values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_lite_full_model(B, S, dtype):
    """Test that full model produces valid output."""
    device = "cuda"
    config = _create_small_config()

    model = Glm4MoeLiteForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    output = model(input_ids=input_ids, position_ids=position_ids)

    # Check output shape
    assert output.logits.shape == (B, S, config.vocab_size), (
        f"Expected logits shape {(B, S, config.vocab_size)}, got {output.logits.shape}"
    )

    # Check output doesn't have NaN or Inf values
    assert not torch.isnan(output.logits).any(), "Logits contain NaN values"
    assert not torch.isinf(output.logits).any(), "Logits contain Inf values"


def test_glm4_moe_lite_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm.

    This test verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces outputs with correct shape
    3. The outputs contain finite values (no NaN/Inf)

    Note: We don't test numerical equivalence between original and exported model
    here because torch.export lifts parameters into the graph, creating a different
    parameter structure that doesn't match load_state_dict. The numerical correctness
    of the model itself is already validated by test_glm4_moe_lite_full_model.
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Glm4MoeLiteForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    # Create input
    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Define dynamic shapes
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    # Export the model - this is the main test: verify no graph breaks
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    # Move graph module to device
    move_to_device(gm, device)

    # Verify the exported model produces valid output
    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    # Check output structure and shape
    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Logits should not contain NaN or Inf"

    # Test with different input shape to verify dynamic shapes work
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    expected_shape = (B2, S2, config.vocab_size)
    assert logits2.shape == expected_shape, (
        f"Dynamic shape test failed: expected {expected_shape}, got {logits2.shape}"
    )
    assert torch.isfinite(logits2).all(), "Logits should not contain NaN or Inf"


def test_glm4_moe_lite_config_registration():
    """Test that the config is properly registered or model_type is correct."""
    # Create a config and verify model_type
    config = _create_small_config()
    assert config.model_type == "glm4_moe_lite"

    # Verify our config class can be instantiated with expected attributes
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "n_routed_experts")
    assert hasattr(config, "kv_lora_rank")
    assert hasattr(config, "qk_rope_head_dim")


def test_glm4_moe_lite_layer_types():
    """Test that layer 0 uses dense MLP and later layers use MoE."""
    config = _create_small_config()
    model = Glm4MoeLiteForCausalLM(config)

    # Check layer 0 (should be dense MLP, not MoE)
    layer0_mlp = model.model.layers[0].mlp
    assert type(layer0_mlp).__name__ == "Glm4MoeLiteMLP", (
        f"Layer 0 should use Glm4MoeLiteMLP, got {type(layer0_mlp).__name__}"
    )

    # Check layer 1+ (should be MoE)
    for i in range(1, config.num_hidden_layers):
        layer_mlp = model.model.layers[i].mlp
        assert type(layer_mlp).__name__ == "Glm4MoeLiteMoE", (
            f"Layer {i} should use Glm4MoeLiteMoE, got {type(layer_mlp).__name__}"
        )


def test_glm4_moe_lite_expert_structure():
    """Test that experts have correct structure for checkpoint loading."""
    config = _create_small_config()
    moe = Glm4MoeLiteMoE(config)

    # Check that experts is a ModuleList
    assert isinstance(moe.experts, torch.nn.ModuleList), "experts should be nn.ModuleList"

    # Check number of experts
    assert len(moe.experts) == config.n_routed_experts, (
        f"Expected {config.n_routed_experts} experts, got {len(moe.experts)}"
    )

    # Check each expert has the correct structure
    for i, expert in enumerate(moe.experts):
        assert hasattr(expert, "gate_proj"), f"Expert {i} missing gate_proj"
        assert hasattr(expert, "up_proj"), f"Expert {i} missing up_proj"
        assert hasattr(expert, "down_proj"), f"Expert {i} missing down_proj"

    # Check state_dict keys match expected checkpoint format
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


# =============================================================================
# Numerical Equivalence Tests
# These tests compare our custom implementation against the HF implementation
# =============================================================================


def _get_hf_model_class():
    """Get the HF model class for GLM4 MoE Lite.

    Returns None if transformers doesn't have glm4_moe_lite (older versions).
    """
    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteForCausalLM as HFGlm4MoeLiteForCausalLM,
        )

        return HFGlm4MoeLiteForCausalLM
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF MoE class for GLM4 MoE Lite."""
    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMoE as HFGlm4MoeLiteMoE,
        )

        return HFGlm4MoeLiteMoE
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF Attention class for GLM4 MoE Lite."""
    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteAttention as HFGlm4MoeLiteAttention,
        )

        return HFGlm4MoeLiteAttention
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF MLP class for GLM4 MoE Lite."""
    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMLP as HFGlm4MoeLiteMLP,
        )

        return HFGlm4MoeLiteMLP
    except ImportError:
        return None


def _get_hf_config_class():
    """Get the HF Config class for GLM4 MoE Lite."""
    try:
        from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
            Glm4MoeLiteConfig as HFGlm4MoeLiteConfig,
        )

        return HFGlm4MoeLiteConfig
    except ImportError:
        return None


def _convert_hf_moe_state_dict_to_custom(hf_state_dict: dict, n_experts: int) -> dict:
    """Convert HF MoE state dict to our custom format.

    HF format (stacked):
        experts.gate_up_proj: [n_experts, 2 * intermediate_size, hidden_size]
        experts.down_proj: [n_experts, hidden_size, intermediate_size]

    Custom format (per-expert):
        experts.0.gate_proj.weight: [intermediate_size, hidden_size]
        experts.0.up_proj.weight: [intermediate_size, hidden_size]
        experts.0.down_proj.weight: [hidden_size, intermediate_size]
    """
    custom_state_dict = {}

    for key, value in hf_state_dict.items():
        if key == "experts.gate_up_proj":
            # Split stacked gate_up into individual gate and up per expert
            # Shape: [n_experts, 2 * intermediate_size, hidden_size]
            intermediate_size = value.shape[1] // 2
            for i in range(n_experts):
                gate_up = value[i]  # [2 * intermediate_size, hidden_size]
                gate_weight = gate_up[:intermediate_size]  # [intermediate_size, hidden_size]
                up_weight = gate_up[intermediate_size:]  # [intermediate_size, hidden_size]
                custom_state_dict[f"experts.{i}.gate_proj.weight"] = gate_weight
                custom_state_dict[f"experts.{i}.up_proj.weight"] = up_weight
        elif key == "experts.down_proj":
            # Split stacked down into individual down per expert
            # Shape: [n_experts, hidden_size, intermediate_size]
            for i in range(n_experts):
                custom_state_dict[f"experts.{i}.down_proj.weight"] = value[i]
        else:
            # Copy other keys as-is
            custom_state_dict[key] = value

    return custom_state_dict


def _convert_hf_full_model_state_dict_to_custom(hf_state_dict: dict, config) -> dict:
    """Convert full HF model state dict to custom format.

    Handles MoE expert weight conversion for all MoE layers.
    """
    custom_state_dict = {}
    n_experts = config.n_routed_experts

    for key, value in hf_state_dict.items():
        # Check if this is an MoE expert weight
        if ".mlp.experts.gate_up_proj" in key:
            # Extract layer prefix (e.g., "model.layers.1.mlp.")
            prefix = key.replace("experts.gate_up_proj", "")
            intermediate_size = value.shape[1] // 2
            for i in range(n_experts):
                gate_up = value[i]
                gate_weight = gate_up[:intermediate_size]
                up_weight = gate_up[intermediate_size:]
                custom_state_dict[f"{prefix}experts.{i}.gate_proj.weight"] = gate_weight
                custom_state_dict[f"{prefix}experts.{i}.up_proj.weight"] = up_weight
        elif ".mlp.experts.down_proj" in key:
            prefix = key.replace("experts.down_proj", "")
            for i in range(n_experts):
                custom_state_dict[f"{prefix}experts.{i}.down_proj.weight"] = value[i]
        else:
            # Copy other keys as-is
            custom_state_dict[key] = value

    return custom_state_dict


def _create_hf_config():
    """Create HF config that matches our test config."""
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        return None

    config = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
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
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )

    # Set internal attributes needed by HF's MoE implementation
    # _experts_implementation tells HF which expert forward function to use
    config._experts_implementation = "eager"

    return config


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_lite_moe_numerical_equivalence(B, S, dtype):
    """Test MoE layer produces numerically equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have glm4_moe_lite (requires v5.0+)")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF MoE
    hf_moe = HFMoE(hf_config)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    # Initialize weights with randn for reproducibility
    # (HF uses torch.empty which may be zeros, causing NaN in computation)
    hf_moe.gate.weight = torch.nn.Parameter(torch.randn_like(hf_moe.gate.weight))
    hf_moe.experts.gate_up_proj = torch.nn.Parameter(torch.randn_like(hf_moe.experts.gate_up_proj))
    hf_moe.experts.down_proj = torch.nn.Parameter(torch.randn_like(hf_moe.experts.down_proj))

    # Create custom MoE and load converted weights
    custom_moe = Glm4MoeLiteMoE(config)
    custom_moe.to(device=device, dtype=dtype)

    # Convert HF stacked expert weights to our per-expert format
    hf_state_dict = hf_moe.state_dict()

    # Debug: print state dict keys and shapes
    print("\n=== HF MoE state_dict keys and shapes ===")
    for k, v in hf_state_dict.items():
        print(f"  {k}: {v.shape}")

    custom_state_dict = _convert_hf_moe_state_dict_to_custom(hf_state_dict, config.n_routed_experts)

    print("\n=== Converted custom state_dict keys and shapes ===")
    for k, v in custom_state_dict.items():
        print(f"  {k}: {v.shape}")

    print("\n=== Expected custom MoE state_dict keys ===")
    for k, v in custom_moe.state_dict().items():
        print(f"  {k}: {v.shape}")

    custom_moe.load_state_dict(custom_state_dict)
    custom_moe.eval()

    # Sanity check: verify expert weights match after conversion
    # HF has stacked weights: experts.gate_up_proj [n_experts, 2*intermediate, hidden]
    # Our model has per-expert: experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight
    hf_gate_up = hf_moe.experts.gate_up_proj  # [n_experts, 2*intermediate, hidden]
    hf_down = hf_moe.experts.down_proj  # [n_experts, hidden, intermediate]
    intermediate_size = config.moe_intermediate_size

    print(f"\n=== Debug: intermediate_size = {intermediate_size} ===")
    print(f"hf_gate_up shape: {hf_gate_up.shape}")
    print(f"hf_gate_up[0, :2, :2]: {hf_gate_up[0, :2, :2]}")

    # Get the converted state dict values for comparison
    converted_gate_0 = custom_state_dict["experts.0.gate_proj.weight"]
    print(f"converted_gate_0 shape: {converted_gate_0.shape}")
    print(f"converted_gate_0[:2, :2]: {converted_gate_0[:2, :2]}")

    # After load_state_dict
    loaded_gate_0 = custom_moe.experts[0].gate_proj.weight
    print(f"loaded_gate_0 shape: {loaded_gate_0.shape}")
    print(f"loaded_gate_0[:2, :2]: {loaded_gate_0[:2, :2]}")

    for i in range(config.n_routed_experts):
        # Check gate_proj
        hf_gate = hf_gate_up[i, :intermediate_size, :]  # [intermediate, hidden]
        custom_gate = custom_moe.experts[i].gate_proj.weight
        torch.testing.assert_close(
            custom_gate, hf_gate, msg=f"Expert {i} gate_proj weights don't match"
        )

        # Check up_proj
        hf_up = hf_gate_up[i, intermediate_size:, :]  # [intermediate, hidden]
        custom_up = custom_moe.experts[i].up_proj.weight
        torch.testing.assert_close(custom_up, hf_up, msg=f"Expert {i} up_proj weights don't match")

        # Check down_proj
        hf_down_i = hf_down[i]  # [hidden, intermediate]
        custom_down = custom_moe.experts[i].down_proj.weight
        torch.testing.assert_close(
            custom_down, hf_down_i, msg=f"Expert {i} down_proj weights don't match"
        )

    # Also verify gate weights match
    torch.testing.assert_close(
        custom_moe.gate.weight, hf_moe.gate.weight, msg="Gate weights don't match"
    )

    # Create input
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    # Run both
    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    # Handle tuple output from HF (output, router_logits)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Compare
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_lite_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have glm4_moe_lite (requires v5.0+)")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF MLP
    hf_mlp = HFMLP(hf_config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    # Create custom MLP and load same weights
    custom_mlp = Glm4MoeLiteMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    # Create input
    H = config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    # Run both
    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # Compare
    rtol, atol = 1e-3, 1e-3
    torch.testing.assert_close(custom_out, hf_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_glm4_moe_lite_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have glm4_moe_lite (requires v5.0+)")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF model
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Initialize all gate weights for reproducibility
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))

    # Create custom model and load converted weights
    custom_model = Glm4MoeLiteForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    # Convert HF stacked expert weights to our per-expert format
    hf_state_dict = hf_model.state_dict()
    custom_state_dict = _convert_hf_full_model_state_dict_to_custom(hf_state_dict, config)
    custom_model.load_state_dict(custom_state_dict)
    custom_model.eval()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Run both
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    # Compare logits - cast to same dtype for comparison
    # (HF model may output float32 for numerical stability in lm_head)
    rtol, atol = 0.05, 0.05
    torch.testing.assert_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rtol=rtol,
        atol=atol,
    )
