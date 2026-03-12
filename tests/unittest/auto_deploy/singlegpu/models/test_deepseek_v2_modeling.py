"""Tests for DeepSeekV2 custom model implementation for auto_deploy export.

Covers the DeepSeek-V2 model family (model_type="deepseek_v2"):
- DeepSeek-Coder-V2-Instruct (group_limited_greedy routing, q_lora_rank=1536)
- DeepSeek-Coder-V2-Lite-Instruct (greedy routing, q_lora_rank=None)
- DeepSeek-V2.5 (group_limited_greedy routing, q_lora_rank=1536)
"""

import pytest
import torch
from torch.export import Dim
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v2 import (
    DeepSeekV2Attention,
    DeepSeekV2DecoderLayer,
    DeepSeekV2ForCausalLM,
    DeepSeekV2MLP,
    DeepSeekV2MoE,
    DeepSeekV2MoEGate,
    DeepSeekV2RMSNorm,
    DeepSeekV2RotaryEmbedding,
    DeepSeekV2YarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Config helpers
# =============================================================================


class MockDeepSeekV2Config(PretrainedConfig):
    """Mock config for DeepSeek-V2 with greedy routing (like V2-Lite)."""

    model_type = "deepseek_v2"

    def __init__(self, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 64)
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)
        self.v_head_dim = kwargs.get("v_head_dim", 64)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 128)
        self.q_lora_rank = kwargs.get("q_lora_rank", None)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 512)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.intermediate_size = kwargs.get("intermediate_size", 512)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.n_routed_experts = kwargs.get("n_routed_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", 256)
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)
        self.routed_scaling_factor = kwargs.get("routed_scaling_factor", 1.0)
        self.n_group = kwargs.get("n_group", 1)
        self.topk_group = kwargs.get("topk_group", 1)
        self.topk_method = kwargs.get("topk_method", "greedy")
        self.norm_topk_prob = kwargs.get("norm_topk_prob", False)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 2)
        self.first_k_dense_replace = kwargs.get("first_k_dense_replace", 1)
        self.moe_layer_freq = kwargs.get("moe_layer_freq", 1)
        self.vocab_size = kwargs.get("vocab_size", 1000)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.initializer_range = kwargs.get("initializer_range", 0.02)


def _create_greedy_config():
    """Create config with greedy routing (like DeepSeek-Coder-V2-Lite)."""
    return MockDeepSeekV2Config(
        topk_method="greedy",
        routed_scaling_factor=1.0,
        q_lora_rank=None,
        n_group=1,
        topk_group=1,
    )


def _create_group_limited_greedy_config():
    """Create config with group_limited_greedy routing (like DeepSeek-Coder-V2)."""
    return MockDeepSeekV2Config(
        topk_method="group_limited_greedy",
        routed_scaling_factor=16.0,
        q_lora_rank=128,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
    )


def _create_small_config():
    """Create a small config for full model tests."""
    return MockDeepSeekV2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # Layer 0 dense, layers 1-2 MoE
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        topk_method="greedy",
        norm_topk_prob=False,
        first_k_dense_replace=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        pad_token_id=0,
    )


# =============================================================================
# RMSNorm Tests
# =============================================================================


class TestDeepSeekV2RMSNorm:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        hidden_size = 256
        norm = DeepSeekV2RMSNorm(hidden_size).to(self.device, self.dtype)
        x = torch.randn(2, 4, hidden_size, dtype=self.dtype, device=self.device)
        output = norm(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_output_normalized(self):
        hidden_size = 256
        norm = DeepSeekV2RMSNorm(hidden_size).to(self.device, torch.float32)
        x = torch.randn(2, 4, hidden_size, dtype=torch.float32, device=self.device)
        output = norm(x)
        rms = torch.sqrt((output**2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


# =============================================================================
# Rotary Embedding Tests
# =============================================================================


class TestDeepSeekV2RotaryEmbedding:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_base_rope_shape(self):
        dim = 32
        max_pos = 512
        B, S = 2, 4
        rope = DeepSeekV2RotaryEmbedding(dim, max_pos).to(self.device)
        x = torch.randn(B, S, 8, dim, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, position_ids)
        # Position-indexed: [B, S, dim]
        assert cos.shape == (B, S, dim)
        assert sin.shape == (B, S, dim)

    def test_yarn_rope_shape(self):
        dim = 32
        max_pos = 512
        B, S = 2, 4
        rope = DeepSeekV2YarnRotaryEmbedding(
            dim,
            max_pos,
            scaling_factor=2.0,
            original_max_position_embeddings=256,
        ).to(self.device)
        x = torch.randn(B, S, 8, dim, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (B, S, dim)
        assert sin.shape == (B, S, dim)


# =============================================================================
# MLP Tests
# =============================================================================


class TestDeepSeekV2MLP:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        config = _create_greedy_config()
        mlp = DeepSeekV2MLP(config).to(self.device, self.dtype)
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = mlp(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))


# =============================================================================
# MoE Gate Tests
# =============================================================================


class TestDeepSeekV2MoEGate:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_greedy_routing(self):
        config = _create_greedy_config()
        gate = DeepSeekV2MoEGate(config).to(self.device, self.dtype)
        gate.weight = torch.nn.Parameter(torch.randn_like(gate.weight))

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        topk_idx, topk_weight = gate(x)

        assert topk_idx.shape == (8, config.num_experts_per_tok)  # 2*4 tokens
        assert topk_weight.shape == (8, config.num_experts_per_tok)
        assert torch.isfinite(topk_weight).all()

    def test_group_limited_greedy_routing(self):
        config = _create_group_limited_greedy_config()
        gate = DeepSeekV2MoEGate(config).to(self.device, self.dtype)
        gate.weight = torch.nn.Parameter(torch.randn_like(gate.weight))

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        topk_idx, topk_weight = gate(x)

        assert topk_idx.shape == (8, config.num_experts_per_tok)
        assert topk_weight.shape == (8, config.num_experts_per_tok)
        assert torch.isfinite(topk_weight).all()

    def test_scaling_factor(self):
        config = _create_group_limited_greedy_config()
        gate = DeepSeekV2MoEGate(config).to(self.device, self.dtype)
        gate.weight = torch.nn.Parameter(torch.randn_like(gate.weight))

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        _, topk_weight = gate(x)

        # With routed_scaling_factor=16.0, weights should be scaled up
        # Softmax gives values in (0,1), top-k of 2 experts, then *16
        assert topk_weight.max() > 1.0


# =============================================================================
# MoE Tests
# =============================================================================


class TestDeepSeekV2MoE:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def _create_moe(self, config):
        moe = DeepSeekV2MoE(config).to(self.device, self.dtype)
        # Initialize gate weights for reproducibility
        moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
        return moe

    def test_forward_shape_greedy(self):
        config = _create_greedy_config()
        moe = self._create_moe(config)
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_forward_shape_group_limited_greedy(self):
        config = _create_group_limited_greedy_config()
        moe = self._create_moe(config)
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_with_shared_experts(self):
        config = _create_greedy_config()
        config.n_shared_experts = 2
        moe = self._create_moe(config)
        assert moe.shared_experts is not None
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape

    def test_without_shared_experts(self):
        config = _create_greedy_config()
        config.n_shared_experts = None
        moe = self._create_moe(config)
        assert moe.shared_experts is None
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape

    def test_expert_structure(self):
        """Verify checkpoint-compatible expert structure."""
        config = _create_greedy_config()
        moe = DeepSeekV2MoE(config)
        assert isinstance(moe.experts, torch.nn.ModuleList)
        assert len(moe.experts) == config.n_routed_experts
        for i, expert in enumerate(moe.experts):
            assert hasattr(expert, "gate_proj"), f"Expert {i} missing gate_proj"
            assert hasattr(expert, "up_proj"), f"Expert {i} missing up_proj"
            assert hasattr(expert, "down_proj"), f"Expert {i} missing down_proj"
        state_dict = moe.state_dict()
        assert "experts.0.gate_proj.weight" in state_dict
        assert "experts.0.up_proj.weight" in state_dict
        assert "experts.0.down_proj.weight" in state_dict


# =============================================================================
# Attention Tests
# =============================================================================


class TestDeepSeekV2Attention:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape_no_qlora(self):
        """Test attention without Q LoRA (like V2-Lite)."""
        config = _create_greedy_config()
        assert config.q_lora_rank is None
        attn = DeepSeekV2Attention(config, layer_idx=0).to(self.device, self.dtype)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = attn(hidden_states, position_ids)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_forward_shape_with_qlora(self):
        """Test attention with Q LoRA (like V2 full)."""
        config = _create_group_limited_greedy_config()
        assert config.q_lora_rank is not None
        attn = DeepSeekV2Attention(config, layer_idx=0).to(self.device, self.dtype)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = attn(hidden_states, position_ids)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_different_batch_sizes(self):
        config = _create_greedy_config()
        attn = DeepSeekV2Attention(config, layer_idx=0).to(self.device, self.dtype)
        for B in [1, 2, 4]:
            S = 4
            hidden_states = torch.randn(
                B, S, config.hidden_size, dtype=self.dtype, device=self.device
            )
            position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
            output = attn(hidden_states, position_ids)
            assert output.shape == (B, S, config.hidden_size)

    def test_different_sequence_lengths(self):
        config = _create_greedy_config()
        attn = DeepSeekV2Attention(config, layer_idx=0).to(self.device, self.dtype)
        for S in [1, 4, 16]:
            B = 2
            hidden_states = torch.randn(
                B, S, config.hidden_size, dtype=self.dtype, device=self.device
            )
            position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
            output = attn(hidden_states, position_ids)
            assert output.shape == (B, S, config.hidden_size)


# =============================================================================
# Decoder Layer Tests
# =============================================================================


class TestDeepSeekV2DecoderLayer:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_dense_layer(self):
        config = _create_greedy_config()
        layer = DeepSeekV2DecoderLayer(config, layer_idx=0).to(self.device, self.dtype)
        assert isinstance(layer.mlp, DeepSeekV2MLP)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
        output = layer(hidden_states, position_ids)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

    def test_moe_layer(self):
        config = _create_greedy_config()
        layer = DeepSeekV2DecoderLayer(config, layer_idx=1).to(self.device, self.dtype)
        assert isinstance(layer.mlp, DeepSeekV2MoE)
        # Initialize gate weights for reproducibility (torch.empty may cause NaN)
        layer.mlp.gate.weight = torch.nn.Parameter(torch.randn_like(layer.mlp.gate.weight))

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
        output = layer(hidden_states, position_ids)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()


# =============================================================================
# Full Model Tests
# =============================================================================


class TestDeepSeekV2ForCausalLM:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward(self):
        config = _create_small_config()
        model = DeepSeekV2ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = model(input_ids=input_ids, position_ids=position_ids)
        assert output.logits.shape == (B, S, config.vocab_size)

    def test_output_dtype(self):
        config = _create_small_config()
        model = DeepSeekV2ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = model(input_ids=input_ids, position_ids=position_ids)
        assert output.logits.dtype == torch.float32

    def test_position_ids_required(self):
        config = _create_small_config()
        model = DeepSeekV2ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)

        with pytest.raises(AssertionError, match="position_ids is required"):
            model(input_ids=input_ids)

    def test_layer_types(self):
        config = _create_small_config()
        model = DeepSeekV2ForCausalLM(config)
        # Layer 0 should be dense
        assert type(model.model.layers[0].mlp).__name__ == "DeepSeekV2MLP"
        # Layer 1+ should be MoE
        for i in range(1, config.num_hidden_layers):
            assert type(model.model.layers[i].mlp).__name__ == "DeepSeekV2MoE"


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_full_model(B, S, dtype):
    """Test full model produces valid output."""
    device = "cuda"
    config = _create_small_config()
    model = DeepSeekV2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    output = model(input_ids=input_ids, position_ids=position_ids)

    assert output.logits.shape == (B, S, config.vocab_size)
    assert not torch.isnan(output.logits).any()
    assert not torch.isinf(output.logits).any()
    assert not torch.allclose(output.logits, torch.zeros_like(output.logits))


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_moe_layer(B, S, dtype):
    """Test MoE layer produces valid output with both routing methods."""
    device = "cuda"

    for topk_method in ["greedy", "group_limited_greedy"]:
        if topk_method == "greedy":
            config = _create_greedy_config()
        else:
            config = _create_group_limited_greedy_config()

        moe = DeepSeekV2MoE(config).to(device=device, dtype=dtype)
        moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
        moe.eval()

        x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
        output = moe(x)
        assert output.shape == x.shape, f"Failed for {topk_method}"
        assert not torch.isnan(output).any(), f"NaN in output for {topk_method}"
        assert not torch.isinf(output).any(), f"Inf in output for {topk_method}"
        assert not torch.allclose(output, torch.zeros_like(output)), (
            f"Zero output for {topk_method}"
        )


# =============================================================================
# Export Test
# =============================================================================


def test_deepseek_v2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = DeepSeekV2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

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

    # Export the model
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
    assert torch.isfinite(logits).all()

    # Test with different input shape (dynamic shapes)
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()


# =============================================================================
# Registration Test
# =============================================================================


def test_deepseek_v2_model_registration():
    """Test that DeepSeekV2ForCausalLM is registered with the factory."""
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    assert "DeepseekV2Config" in AutoModelForCausalLMFactory._custom_model_mapping
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["DeepseekV2Config"]
        == DeepSeekV2ForCausalLM
    )


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================


def _get_hf_model_class():
    """Get the HF DeepseekV2ForCausalLM class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2ForCausalLM as HFDeepseekV2ForCausalLM,
        )

        return HFDeepseekV2ForCausalLM
    except ImportError:
        return None


def _get_hf_moe_class():
    """Get the HF DeepseekV2MoE class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2MoE as HFDeepseekV2MoE,
        )

        return HFDeepseekV2MoE
    except ImportError:
        return None


def _get_hf_mlp_class():
    """Get the HF DeepseekV2MLP class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2MLP as HFDeepseekV2MLP,
        )

        return HFDeepseekV2MLP
    except ImportError:
        return None


def _get_hf_config_class():
    """Get the HF DeepseekV2Config class."""
    try:
        from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
            DeepseekV2Config as HFDeepseekV2Config,
        )

        return HFDeepseekV2Config
    except ImportError:
        return None


def _create_hf_config():
    """Create HF config matching our small test config."""
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
        rms_norm_eps=1e-6,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        topk_method="greedy",
        norm_topk_prob=False,
        first_k_dense_replace=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        pad_token_id=0,
    )
    return config


def _get_hf_rmsnorm_class():
    """Get the HF DeepseekV2RMSNorm class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2RMSNorm as HFDeepseekV2RMSNorm,
        )

        return HFDeepseekV2RMSNorm
    except ImportError:
        return None


def _get_hf_attention_class():
    """Get the HF DeepseekV2Attention class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2Attention as HFDeepseekV2Attention,
        )

        return HFDeepseekV2Attention
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    """Get the HF DeepseekV2DecoderLayer class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2DecoderLayer as HFDeepseekV2DecoderLayer,
        )

        return HFDeepseekV2DecoderLayer
    except ImportError:
        return None


def _get_hf_rotary_emb_class():
    """Get the HF DeepseekV2RotaryEmbedding class."""
    try:
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2RotaryEmbedding as HFDeepseekV2RotaryEmbedding,
        )

        return HFDeepseekV2RotaryEmbedding
    except ImportError:
        return None


def _convert_hf_moe_state_dict_to_custom(hf_state_dict: dict, n_experts: int) -> dict:
    """Convert HF MoE state dict (per-expert format) to our custom format.

    HF V2 MoE uses the same per-expert nn.ModuleList format as our custom model,
    so the state dict keys should match directly. However, HF also stores ep_rank
    and experts_per_rank as buffers in some versions.
    """
    custom_state_dict = {}
    for key, value in hf_state_dict.items():
        # Skip HF-specific buffers
        if key in ("ep_rank", "experts_per_rank"):
            continue
        custom_state_dict[key] = value
    return custom_state_dict


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_rmsnorm_numerical_equivalence(B, S, dtype):
    """Test RMSNorm produces numerically equivalent output to HF."""
    HFRMSNorm = _get_hf_rmsnorm_class()
    if HFRMSNorm is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    hidden_size = 64

    hf_norm = HFRMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = DeepSeekV2RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, hidden_size, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_attention_numerical_equivalence(B, S, dtype):
    """Test attention (MLA) produces numerically equivalent output to HF.

    Uses the full model to load weights (via the model-level de-interleaving hook),
    then compares attention layer outputs directly.
    """
    HFModel = _get_hf_model_class()
    HFRotaryEmb = _get_hf_rotary_emb_class()
    if HFModel is None or HFRotaryEmb is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF model and extract attention + rope
    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    hf_attn = hf_model.model.layers[0].self_attn
    hf_rope = hf_model.model.rotary_emb

    # Create custom model, load HF weights (hook handles de-interleaving), extract attention
    custom_model = DeepSeekV2ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = hf_model.state_dict()
    custom_sd = {
        k: v for k, v in hf_sd.items() if "ep_rank" not in k and "experts_per_rank" not in k
    }
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()
    custom_attn = custom_model.model.layers[0].self_attn

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF attention needs position_embeddings
    pos_emb = hf_rope(x, position_ids)
    hf_out, _ = hf_attn(x, position_embeddings=pos_emb, position_ids=position_ids)
    custom_out = custom_attn(x, position_ids)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test decoder layer (dense, layer_idx=0) produces numerically equivalent output to HF.

    Uses the full model to load weights (via the model-level de-interleaving hook),
    then compares layer outputs directly.
    """
    HFModel = _get_hf_model_class()
    HFRotaryEmb = _get_hf_rotary_emb_class()
    if HFModel is None or HFRotaryEmb is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Create HF model and extract layer 0 (dense) + rope
    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    hf_layer = hf_model.model.layers[0]
    hf_rope = hf_model.model.rotary_emb

    # Create custom model, load HF weights, extract layer 0
    custom_model = DeepSeekV2ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = hf_model.state_dict()
    custom_sd = {
        k: v for k, v in hf_sd.items() if "ep_rank" not in k and "experts_per_rank" not in k
    }
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()
    custom_layer = custom_model.model.layers[0]

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    pos_emb = hf_rope(x, position_ids)
    hf_out = hf_layer(x, position_ids=position_ids, position_embeddings=pos_emb)
    custom_out = custom_layer(x, position_ids)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer (dense): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_moe_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test MoE decoder layer (layer_idx=1) produces numerically equivalent output to HF."""
    HFModel = _get_hf_model_class()
    HFRotaryEmb = _get_hf_rotary_emb_class()
    if HFModel is None or HFRotaryEmb is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    # Initialize gate weights for reproducibility
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))

    hf_layer = hf_model.model.layers[1]  # MoE layer
    hf_rope = hf_model.model.rotary_emb

    custom_model = DeepSeekV2ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = hf_model.state_dict()
    custom_sd = {
        k: v for k, v in hf_sd.items() if "ep_rank" not in k and "experts_per_rank" not in k
    }
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()
    custom_layer = custom_model.model.layers[1]

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    pos_emb = hf_rope(x, position_ids)
    hf_out = hf_layer(x, position_ids=position_ids, position_embeddings=pos_emb)
    custom_out = custom_layer(x, position_ids)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer (MoE): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_mlp = HFMLP(hf_config).to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = DeepSeekV2MLP(config).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_moe_numerical_equivalence(B, S, dtype):
    """Test MoE produces numerically equivalent output to HF."""
    HFMoE = _get_hf_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_moe = HFMoE(hf_config).to(device=device, dtype=dtype)
    hf_moe.eval()

    # Initialize gate weights for reproducibility
    hf_moe.gate.weight = torch.nn.Parameter(torch.randn_like(hf_moe.gate.weight))

    custom_moe = DeepSeekV2MoE(config).to(device=device, dtype=dtype)
    hf_sd = hf_moe.state_dict()
    custom_sd = _convert_hf_moe_state_dict_to_custom(hf_sd, config.n_routed_experts)
    custom_moe.load_state_dict(custom_sd)
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]
    custom_out = custom_moe(x)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v2_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have deepseek_v2 modeling")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()

    # Initialize gate weights for reproducibility
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))

    # Create custom model and load converted weights
    custom_model = DeepSeekV2ForCausalLM(config).to(device=device, dtype=dtype)

    hf_sd = hf_model.state_dict()
    # HF V2 uses per-expert ModuleList like our custom model, so keys should match
    # But we need to filter any HF-specific buffers
    custom_sd = {}
    for key, value in hf_sd.items():
        if "ep_rank" in key or "experts_per_rank" in key:
            continue
        custom_sd[key] = value
    custom_model.load_state_dict(custom_sd)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model: ",
    )
