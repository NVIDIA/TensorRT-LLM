"""Testing custom DeepSeekV3 model implementation for auto_deploy export."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek import (
    DeepSeekV3Attention,
    DeepSeekV3DecoderLayer,
    DeepSeekV3ForCausalLM,
    DeepSeekV3MLP,
    DeepSeekV3Model,
    DeepSeekV3MoE,
    DeepSeekV3RMSNorm,
    DeepSeekV3RotaryEmbedding,
    DeepSeekV3YarnRotaryEmbedding,
)


class MockDeepSeekConfig(PretrainedConfig):
    """Mock DeepSeek config for testing the custom model components."""

    model_type = "deepseek_v3"

    def __init__(self):
        super().__init__()
        # Attention config
        self.num_attention_heads = 8
        self.qk_nope_head_dim = 64
        self.qk_rope_head_dim = 32
        self.v_head_dim = 64
        self.kv_lora_rank = 128
        self.q_lora_rank = None  # No LoRA for Q in tests
        self.hidden_size = 256
        self.rope_theta = 10000.0
        self.max_position_embeddings = 512
        self.attention_bias = False
        self.rope_scaling = None
        self.rms_norm_eps = 1e-6

        # MLP config
        self.intermediate_size = 512
        self.hidden_act = "silu"

        # MoE config
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 256
        self.n_shared_experts = 1
        self.routed_scaling_factor = 1.0
        self.n_group = 1
        self.topk_group = 1

        # Model config
        self.num_hidden_layers = 2
        self.first_k_dense_replace = 1  # First layer is dense, second is MoE
        self.moe_layer_freq = 1
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.initializer_range = 0.02


class TestDeepSeekV3RMSNorm:
    """Test DeepSeekV3RMSNorm implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test that RMSNorm preserves input shape."""
        hidden_size = 256
        norm = DeepSeekV3RMSNorm(hidden_size).to(self.device, self.dtype)

        x = torch.randn(2, 4, hidden_size, dtype=self.dtype, device=self.device)
        output = norm(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_output_normalized(self):
        """Test that output has approximately unit variance."""
        hidden_size = 256
        norm = DeepSeekV3RMSNorm(hidden_size).to(self.device, torch.float32)

        x = torch.randn(2, 4, hidden_size, dtype=torch.float32, device=self.device)
        output = norm(x)

        # RMS should be close to 1 after normalization (scaled by weight)
        rms = torch.sqrt((output**2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestDeepSeekV3RotaryEmbedding:
    """Test DeepSeekV3 Rotary Embedding implementations."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_base_rope_shape(self):
        """Test base rotary embedding output shape."""
        dim = 32
        max_pos = 512
        rope = DeepSeekV3RotaryEmbedding(dim, max_pos).to(self.device)

        x = torch.randn(2, 4, 8, dim, dtype=self.dtype, device=self.device)
        cos, sin = rope(x)

        # Should return full cached values
        assert cos.shape == (max_pos, dim)
        assert sin.shape == (max_pos, dim)

    def test_yarn_rope_shape(self):
        """Test YaRN rotary embedding output shape."""
        dim = 32
        max_pos = 512
        rope = DeepSeekV3YarnRotaryEmbedding(
            dim,
            max_pos,
            scaling_factor=2.0,
            original_max_position_embeddings=256,
        ).to(self.device)

        x = torch.randn(2, 4, 8, dim, dtype=self.dtype, device=self.device)
        cos, sin = rope(x)

        assert cos.shape == (max_pos, dim)
        assert sin.shape == (max_pos, dim)


class TestDeepSeekV3MLP:
    """Test DeepSeekV3MLP implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test MLP output shape."""
        config = MockDeepSeekConfig()
        mlp = DeepSeekV3MLP(config).to(self.device, self.dtype)

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = mlp(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()


class TestDeepSeekV3Attention:
    """Test DeepSeekV3Attention (MLA) implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test attention output shape."""
        config = MockDeepSeekConfig()
        attn = DeepSeekV3Attention(config, layer_idx=0).to(self.device, self.dtype)

        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
        )
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        output = attn(hidden_states, position_ids)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

    def test_different_batch_sizes(self):
        """Test attention with different batch sizes."""
        config = MockDeepSeekConfig()
        attn = DeepSeekV3Attention(config, layer_idx=0).to(self.device, self.dtype)

        for batch_size in [1, 2, 4]:
            seq_len = 4
            hidden_states = torch.randn(
                batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
            )
            position_ids = (
                torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            )

            output = attn(hidden_states, position_ids)
            assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        config = MockDeepSeekConfig()
        attn = DeepSeekV3Attention(config, layer_idx=0).to(self.device, self.dtype)

        for seq_len in [1, 4, 16]:
            batch_size = 2
            hidden_states = torch.randn(
                batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
            )
            position_ids = (
                torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            )

            output = attn(hidden_states, position_ids)
            assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestDeepSeekV3MoE:
    """Test DeepSeekV3MoE implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test MoE output shape."""
        config = MockDeepSeekConfig()
        moe = DeepSeekV3MoE(config).to(self.device, self.dtype)

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_with_shared_experts(self):
        """Test MoE with shared experts."""
        config = MockDeepSeekConfig()
        config.n_shared_experts = 2
        moe = DeepSeekV3MoE(config).to(self.device, self.dtype)

        assert moe.shared_experts is not None

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)

        assert output.shape == x.shape

    def test_without_shared_experts(self):
        """Test MoE without shared experts."""
        config = MockDeepSeekConfig()
        config.n_shared_experts = None
        moe = DeepSeekV3MoE(config).to(self.device, self.dtype)

        assert moe.shared_experts is None

        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)

        assert output.shape == x.shape


class TestDeepSeekV3DecoderLayer:
    """Test DeepSeekV3DecoderLayer implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_dense_layer(self):
        """Test decoder layer with dense MLP."""
        config = MockDeepSeekConfig()
        # Layer 0 should be dense (before first_k_dense_replace)
        layer = DeepSeekV3DecoderLayer(config, layer_idx=0).to(self.device, self.dtype)

        assert isinstance(layer.mlp, DeepSeekV3MLP)

        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
        )
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        output = layer(hidden_states, position_ids)

        assert output.shape == hidden_states.shape

    def test_moe_layer(self):
        """Test decoder layer with MoE."""
        config = MockDeepSeekConfig()
        # Layer 1 should be MoE (at first_k_dense_replace)
        layer = DeepSeekV3DecoderLayer(config, layer_idx=1).to(self.device, self.dtype)

        assert isinstance(layer.mlp, DeepSeekV3MoE)

        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
        )
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        output = layer(hidden_states, position_ids)

        assert output.shape == hidden_states.shape


class TestDeepSeekV3Model:
    """Test DeepSeekV3Model implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward_with_input_ids(self):
        """Test model forward with input_ids."""
        config = MockDeepSeekConfig()
        model = DeepSeekV3Model(config).to(self.device, self.dtype)

        batch_size, seq_len = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

        output = model(input_ids=input_ids)

        assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_forward_with_inputs_embeds(self):
        """Test model forward with inputs_embeds."""
        config = MockDeepSeekConfig()
        model = DeepSeekV3Model(config).to(self.device, self.dtype)

        batch_size, seq_len = 2, 4
        inputs_embeds = torch.randn(
            batch_size, seq_len, config.hidden_size, dtype=self.dtype, device=self.device
        )

        output = model(inputs_embeds=inputs_embeds)

        assert output.last_hidden_state.shape == inputs_embeds.shape


class TestDeepSeekV3ForCausalLM:
    """Test DeepSeekV3ForCausalLM implementation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        torch.manual_seed(42)

    def test_forward(self):
        """Test causal LM forward pass."""
        config = MockDeepSeekConfig()
        model = DeepSeekV3ForCausalLM(config).to(self.device, self.dtype)

        batch_size, seq_len = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

        output = model(input_ids=input_ids)

        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_output_dtype(self):
        """Test that logits are float32."""
        config = MockDeepSeekConfig()
        model = DeepSeekV3ForCausalLM(config).to(self.device, self.dtype)

        batch_size, seq_len = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)

        output = model(input_ids=input_ids)

        # Logits should be float32 for numerical stability
        assert output.logits.dtype == torch.float32


class TestMLAOpRegistration:
    """Test that MLA ops are properly registered."""

    def test_torch_mla_registered(self):
        """Test that torch_mla op is registered."""
        assert hasattr(torch.ops.auto_deploy, "torch_mla"), "torch_mla op should be registered"

    def test_torch_cached_mla_registered(self):
        """Test that torch_cached_mla_with_cache op is registered."""
        assert hasattr(torch.ops.auto_deploy, "torch_cached_mla_with_cache"), (
            "torch_cached_mla_with_cache op should be registered"
        )

    def test_torch_mla_callable(self):
        """Test that torch_mla op is callable."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        batch_size, seq_len, num_heads = 1, 2, 4
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank = 128
        v_head_dim = 64

        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=dtype, device=device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=dtype, device=device
        )
        compressed_kv = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=dtype, device=device)
        kpe = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, dtype=dtype, device=device)
        kv_b_proj_weight = torch.randn(
            num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank, dtype=dtype, device=device
        )

        # Should not raise
        output = torch.ops.auto_deploy.torch_mla(
            q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight, True, None, "bsnd"
        )
        assert output is not None

    def test_torch_cached_mla_callable(self):
        """Test that torch_cached_mla_with_cache op is callable."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        batch_size, seq_len, num_heads = 1, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        kv_lora_rank = 64
        v_head_dim = 32
        max_seq_len = 32

        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=dtype, device=device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=dtype, device=device
        )
        compressed_kv = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=dtype, device=device)
        kpe = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, dtype=dtype, device=device)
        kv_b_proj_weight = torch.randn(
            num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank, dtype=dtype, device=device
        )

        batch_info_host = torch.tensor([0, 0, batch_size], dtype=torch.int32, device=device)
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int32, device=device)
        input_pos = torch.tensor([0], dtype=torch.int32, device=device)
        cache_loc = torch.tensor([0], dtype=torch.int32, device=device)
        cu_seqlen = torch.tensor([0], dtype=torch.int32, device=device)

        mla_cache = torch.zeros(
            batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )

        # Should not raise
        output = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            batch_info_host,
            seq_len_tensor,
            input_pos,
            cache_loc,
            cu_seqlen,
            mla_cache,
            None,
            kv_lora_rank,
        )
        assert output is not None


class TestMoEOpRegistration:
    """Test that MoE ops are properly registered."""

    def test_torch_moe_registered(self):
        """Test that torch_moe op is registered."""
        assert hasattr(torch.ops.auto_deploy, "torch_moe"), "torch_moe op should be registered"


class TestCustomModelRegistration:
    """Test that custom model is properly registered."""

    def test_model_registered(self):
        """Test that DeepSeekV3ForCausalLM is registered with the factory."""
        from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

        # Check that the model is registered in the custom model mapping
        assert "DeepseekV3Config" in AutoModelForCausalLMFactory._custom_model_mapping
        assert (
            AutoModelForCausalLMFactory._custom_model_mapping["DeepseekV3Config"]
            == DeepSeekV3ForCausalLM
        )
