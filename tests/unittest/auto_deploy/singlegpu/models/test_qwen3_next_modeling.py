"""Tests for Qwen3-Next (MoE) custom model implementation.

This module tests the custom Qwen3-Next model implementation which uses
auto_deploy custom ops for export compatibility. Qwen3-Next is a hybrid model
with linear attention (GatedDeltaNet) and full attention layers, plus MoE.
"""

from functools import lru_cache

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet as HFQwen3NextGatedDeltaNet,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule as hf_torch_chunk_gated_delta_rule,
)

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.fla.fla_gated_delta import _l2norm
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))

# Full attention layer index for a 4-layer model: [linear, linear, linear, full]
_FULL_ATTN_LAYER_IDX = 3


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config():
    """Create a small Qwen3Next config for testing (4 layers: 3 linear + 1 full)."""
    HFConfig = _get_hf_config_class()
    if HFConfig is not None:
        config = HFConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,  # 3 linear + 1 full
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
            # MoE params
            decoder_sparse_step=1,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            norm_topk_prob=True,
            mlp_only_layers=[],
            # layer types
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        # Required by HF's attention implementation dispatch (avoids KeyError: None)
        config._attn_implementation = "eager"
        return config
    pytest.skip("transformers doesn't have qwen3_next")


# =============================================================================
# HF Reference Helpers
# =============================================================================


def _get_hf_config_class():
    """Get the HF Qwen3Next config class."""
    try:
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig as HFCls

        return HFCls
    except ImportError:
        return None


def _get_hf_model_class():
    """Get the HF Qwen3Next model class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM as HFCls

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


def _get_hf_moe_class():
    """Get the HF Qwen3Next MoE class."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextSparseMoeBlock as HFCls,
        )

        return HFCls
    except ImportError:
        return None


# =============================================================================
# Weight Conversion Helpers
# =============================================================================


def _convert_hf_full_model_state_dict(hf_state_dict: dict, config) -> dict:
    """Convert full HF model state dict to custom format.

    HF MoE uses per-expert format (same as ours), so no expert weight conversion needed.
    The weight names match directly between HF and custom for:
      - experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, experts.{i}.down_proj.weight
      - shared_expert.gate_proj.weight, etc.
      - gate.weight, shared_expert_gate.weight

    No GDN conversion is needed either since both use fused projections
    (in_proj_qkvz, in_proj_ba).
    """
    return dict(hf_state_dict)


@lru_cache(maxsize=1)
def _get_qwen3_next_gdn_state():
    """Build and cache a tiny HF Qwen3Next GDN block state."""
    config = AutoConfig.for_model("qwen3_next")
    config.num_hidden_layers = 1
    config.use_cache = False
    config.hidden_size = 32
    config.intermediate_size = 16
    config.moe_intermediate_size = 16
    config.shared_expert_intermediate_size = 16
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.head_dim = 8
    config.decoder_sparse_step = 1
    config.norm_topk_prob = True
    config.layer_types = ["linear_attention"]
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 4
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.linear_conv_kernel_dim = 4

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    layer_name = "model.layers.0.linear_attn"
    module = dict(model.named_modules()).get(layer_name)
    assert module is not None, f"Layer '{layer_name}' not found in the model"
    assert isinstance(module, HFQwen3NextGatedDeltaNet), (
        f"Expected HF Qwen3NextGatedDeltaNet, got {type(module)}"
    )

    state_dict = {
        name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()
    }
    return config, state_dict


def _load_qwen3_next_gdn_layer():
    """Build a tiny HF Qwen3Next GDN block from cached HF weights."""
    config, state_dict = _get_qwen3_next_gdn_state()
    module = HFQwen3NextGatedDeltaNet(config, layer_idx=0)
    module.load_state_dict(state_dict)
    module.eval()
    return module, config


def _force_torch_fallbacks(module):
    """Force the GDN module to use pure-torch fallbacks instead of external kernels."""
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule as hf_chunk,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_recurrent_gated_delta_rule as hf_recurrent,
    )

    module.causal_conv1d_fn = None
    module.chunk_gated_delta_rule = hf_chunk
    module.recurrent_gated_delta_rule = hf_recurrent


# =============================================================================
# Structural Tests
# =============================================================================


def test_torch_gated_delta_rule_op():
    """Verify the raw-input GDN op matches HF's preprocessed reference."""
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 128
    num_heads = 4
    k_head_dim = 16
    v_head_dim = 16

    q_raw = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    k_raw = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.float32)
    a = torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32)
    b = torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32)
    A_log = torch.randn(num_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, dtype=torch.float32)

    q_norm = _l2norm(q_raw.float())
    k_norm = _l2norm(k_raw.float())
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()

    with torch.no_grad():
        ref_output, _ = hf_torch_chunk_gated_delta_rule(
            q_norm, k_norm, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False
        )

    with torch.no_grad():
        test_output = torch.ops.auto_deploy.torch_gated_delta_rule(
            q_raw, k_raw, v, a, b, A_log, dt_bias
        )

    torch.testing.assert_close(ref_output, test_output, atol=1e-4, rtol=1e-4)


def test_torch_gated_delta_rule_op_bfloat16():
    """Verify the raw-input GDN op works with bfloat16 inputs."""
    torch.manual_seed(123)

    batch_size = 1
    seq_len = 64
    num_heads = 2
    k_head_dim = 8
    v_head_dim = 8

    q_raw = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.bfloat16)
    k_raw = torch.randn(batch_size, seq_len, num_heads, k_head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, v_head_dim, dtype=torch.bfloat16)
    a = torch.randn(batch_size, seq_len, num_heads, dtype=torch.bfloat16)
    b = torch.randn(batch_size, seq_len, num_heads, dtype=torch.bfloat16)
    A_log = torch.randn(num_heads, dtype=torch.bfloat16)
    dt_bias = torch.randn(num_heads, dtype=torch.bfloat16)

    q_norm = _l2norm(q_raw.float()).to(torch.bfloat16)
    k_norm = _l2norm(k_raw.float()).to(torch.bfloat16)
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()

    with torch.no_grad():
        ref_output, _ = hf_torch_chunk_gated_delta_rule(
            q_norm,
            k_norm,
            v,
            g=g.to(torch.bfloat16),
            beta=beta.to(torch.bfloat16),
            use_qk_l2norm_in_kernel=False,
        )

    with torch.no_grad():
        test_output = torch.ops.auto_deploy.torch_gated_delta_rule(
            q_raw, k_raw, v, a, b, A_log, dt_bias
        )

    torch.testing.assert_close(ref_output, test_output, atol=1e-2, rtol=1e-2)


def test_qwen3_next_gdn_patch():
    """Verify the raw-input GDN op path matches the original HF implementation."""
    torch.manual_seed(42)

    module, config = _load_qwen3_next_gdn_layer()
    _force_torch_fallbacks(module)
    module = module.to(torch.bfloat16)

    inputs = torch.randn(2, 16, 32, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = type(module).forward(module, inputs)

    custom_gdn = Qwen3NextGatedDeltaNet(config, layer_idx=0).to(torch.bfloat16)
    custom_gdn.load_state_dict(module.state_dict())
    custom_gdn.eval()

    with torch.no_grad():
        test_output = custom_gdn(inputs)

    torch.testing.assert_close(ref_output, test_output, atol=1e-2, rtol=1e-2)


def test_qwen3_next_gdn_patch_float32():
    """Same as above but in float32 for tighter tolerances."""
    torch.manual_seed(42)

    module, config = _load_qwen3_next_gdn_layer()
    _force_torch_fallbacks(module)
    module = module.to(torch.float32)

    inputs = torch.randn(2, 16, 32, dtype=torch.float32)

    with torch.no_grad():
        ref_output = type(module).forward(module, inputs)

    custom_gdn = Qwen3NextGatedDeltaNet(config, layer_idx=0).to(torch.float32)
    custom_gdn.load_state_dict(module.state_dict())
    custom_gdn.eval()

    with torch.no_grad():
        test_output = custom_gdn(inputs)

    torch.testing.assert_close(ref_output, test_output, atol=1e-4, rtol=1e-4)


def test_qwen3_next_layer_types():
    """Test that layers have correct types (linear vs full attention) and MoE."""
    config = _create_small_config()
    model = Qwen3NextForCausalLM(config)

    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        expected_type = config.layer_types[i]
        if expected_type == "linear_attention":
            assert hasattr(layer, "linear_attn"), f"Layer {i} should have linear_attn"
        else:
            assert hasattr(layer, "self_attn"), f"Layer {i} should have self_attn"

        # All layers should have MoE (decoder_sparse_step=1)
        assert isinstance(layer.mlp, Qwen3NextSparseMoeBlock), (
            f"Layer {i} should have MoE, got {type(layer.mlp).__name__}"
        )


def test_qwen3_next_tied_embeddings():
    """Test that embeddings are tied when tie_word_embeddings=True."""
    config = _create_small_config()
    assert config.tie_word_embeddings is True
    model = Qwen3NextForCausalLM(config)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qwen3_next_expert_structure():
    """Test that experts have correct structure for checkpoint loading."""
    config = _create_small_config()
    model = Qwen3NextForCausalLM(config)
    moe = model.model.layers[0].mlp

    assert isinstance(moe.experts, torch.nn.ModuleList)
    assert len(moe.experts) == config.num_experts

    for i, expert in enumerate(moe.experts):
        assert hasattr(expert, "gate_proj"), f"Expert {i} missing gate_proj"
        assert hasattr(expert, "up_proj"), f"Expert {i} missing up_proj"
        assert hasattr(expert, "down_proj"), f"Expert {i} missing down_proj"


# =============================================================================
# Block Equivalence Tests
# =============================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_next_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF implementation."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Qwen3NextMLP(config)
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
def test_qwen3_next_rmsnorm_offset_hook(B, S, dtype):
    """Test that the RMSNorm load hook properly offsets weights by +1."""
    device = "cuda"
    config = _create_small_config()

    norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
def test_qwen3_next_moe_numerical_equivalence(B, S, dtype):
    """Test MoE layer produces numerically equivalent output to HF implementation."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF MoE
    hf_moe = HFMoE(config)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.eval()

    # Create custom MoE with same weights (same format, no conversion needed)
    custom_moe = Qwen3NextSparseMoeBlock(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    # HF returns (hidden_states, router_logits)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    custom_out = custom_moe(x)

    assert_rmse_close(
        custom_out,
        hf_out,
        rmse_ratio_tol=0.02,
        msg="MoE output diverges from HF reference",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_qwen3_next_gated_deltanet_numerical_equivalence(B, S, dtype):
    """Test GatedDeltaNet produces numerically equivalent output to HF reference."""
    HFGatedDeltaNet = _get_hf_gated_deltanet_class()
    if HFGatedDeltaNet is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF GatedDeltaNet
    hf_gdn = HFGatedDeltaNet(config, layer_idx=0)
    hf_gdn.to(device=device, dtype=dtype)
    hf_gdn.eval()

    # Create custom GatedDeltaNet with same weights (same fused projection format)
    custom_gdn = Qwen3NextGatedDeltaNet(config, layer_idx=0)
    custom_gdn.to(device=device, dtype=dtype)
    custom_gdn.load_state_dict(hf_gdn.state_dict())
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
def test_qwen3_next_attention_numerical_equivalence(B, S, dtype):
    """Test Attention produces numerically equivalent output to HF reference."""
    HFAttention = _get_hf_attention_class()
    if HFAttention is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF Attention
    hf_attn = HFAttention(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Create custom Attention with same weights
    custom_attn = Qwen3NextAttention(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Compute position embeddings (sliced by position_ids) — shared by HF and custom
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextRotaryEmbedding as HFRoPE,
    )

    hf_rope = HFRoPE(config)
    hf_rope.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = hf_rope(x, position_ids)  # (cos, sin), each (B, S, rotary_dim)

    # HF eager attention is non-causal when attention_mask=None, but the
    # AutoDeploy custom attention uses is_causal=True.
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    # HF attention forward: returns (attn_output, attn_weights)
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
    )

    # Custom attention also gets pre-sliced position embeddings (slicing done at model level)
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
def test_qwen3_next_linear_decoder_layer_equivalence(B, S, dtype):
    """Test linear attention decoder layer equivalence against HF reference."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF decoder layer (linear_attention, layer 0)
    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer with same weights
    custom_layer = Qwen3NextDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Compute position embeddings (sliced) — shared by HF and custom
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextRotaryEmbedding as HFRoPE,
    )

    hf_rope = HFRoPE(config)
    hf_rope.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = hf_rope(x, position_ids)

    # HF decoder layer returns hidden_states (may return a tuple)
    hf_out = hf_layer(x, position_embeddings=position_embeddings)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Custom decoder layer gets same pre-sliced position embeddings
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
def test_qwen3_next_full_attention_decoder_layer_equivalence(B, S, dtype):
    """Test full attention decoder layer equivalence against HF reference."""
    HFDecoderLayer = _get_hf_decoder_layer_class()
    if HFDecoderLayer is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF decoder layer (full_attention)
    hf_layer = HFDecoderLayer(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    # Create custom decoder layer with same weights
    custom_layer = Qwen3NextDecoderLayer(config, layer_idx=_FULL_ATTN_LAYER_IDX)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Compute position embeddings (sliced) — shared by HF and custom
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextRotaryEmbedding as HFRoPE,
    )

    hf_rope = HFRoPE(config)
    hf_rope.to(device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = hf_rope(x, position_ids)

    # HF eager attention needs an explicit causal mask to match custom model's is_causal=True
    causal_mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)

    hf_out = hf_layer(x, position_embeddings=position_embeddings, attention_mask=causal_mask)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Custom gets same pre-sliced position embeddings
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
def test_qwen3_next_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF implementation."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have qwen3_next")

    device = "cuda"
    config = _create_small_config()

    # Create HF model
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create custom model and load weights (same format, no conversion)
    custom_model = Qwen3NextForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)

    hf_state_dict = hf_model.state_dict()
    custom_state_dict = _convert_hf_full_model_state_dict(hf_state_dict, config)
    custom_model.load_state_dict(custom_state_dict)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert custom_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(custom_out.logits).all(), "Custom model logits contain NaN/Inf"

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits diverge from HF reference",
    )


# =============================================================================
# Export Test
# =============================================================================


def test_qwen3_next_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cpu"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Qwen3NextForCausalLM(config)
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
