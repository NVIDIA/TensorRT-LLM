import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import (RopeParams,
                                                             RotaryScalingType)
from tensorrt_llm._torch.modules.rotary_embedding import (MRotaryEmbedding,
                                                          RotaryEmbedding)
from tensorrt_llm.functional import RopeEmbeddingUtils


class TestRotaryEmbedding:
    """Test suite for RotaryEmbedding module."""

    @pytest.fixture
    def basic_rope_params(self):
        """Create basic RopeParams for testing."""
        return RopeParams(
            dim=128,
            theta=1000000.0,
            alpha=1.0,
            scale_type=RotaryScalingType.none,
            scale=1.0,
            max_positions=32768,
            original_max_positions=1024,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=0.0,
            short_factor=None,
            long_factor=None,
            max_seq_len=None,
            duplicate_data=True,
        )

    def test_rotary_embedding_sanity(self, basic_rope_params):
        """Test RotaryEmbedding with sanity test."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"

        head_dim = 128
        seq_len = 1000
        num_heads = 8

        rope = RotaryEmbedding(rope_params=basic_rope_params,
                               head_dim=head_dim,
                               is_neox=True)

        # Move rotary_cos_sin to the correct device
        rope.rotary_cos_sin = rope.rotary_cos_sin.to(device)

        # Create test inputs in remove_input_padding format
        q = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float16,
                        device=device)
        k = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float16,
                        device=device)
        position_ids = torch.arange(seq_len, device=device)

        # Forward pass
        result = rope.forward(position_ids, [q, k])
        assert len(result) == 2
        assert result[0].shape == q.shape
        assert result[1].shape == k.shape


class TestMRotaryEmbedding:
    """Test suite for MRotaryEmbedding module."""

    @pytest.fixture
    def basic_rope_params(self):
        """Create basic RopeParams for testing."""
        return RopeParams(
            dim=128,
            theta=1000000.0,
            alpha=1.0,
            scale_type=RotaryScalingType.none,
            scale=1.0,
            max_positions=32768,
            original_max_positions=1024,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=0.0,
            short_factor=None,
            long_factor=None,
            max_seq_len=None,
            duplicate_data=True,
        )

    def test_mrotary_embedding_sanity_3d_position_ids(self, basic_rope_params):
        """Test MRotaryEmbedding forward pass with 3D position_ids."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"
        head_dim = 128
        seq_len = 1000
        num_heads = 8
        mrope_section = [16, 24, 24]

        mrope = MRotaryEmbedding(rope_params=basic_rope_params,
                                 head_dim=head_dim,
                                 mrope_section=mrope_section,
                                 is_neox=True)

        # Create test inputs with 3D position_ids (for mrope)
        q = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float16,
                        device=device)
        k = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float16,
                        device=device)
        position_ids = torch.stack([
            torch.arange(seq_len).unsqueeze(0),
            torch.arange(seq_len).unsqueeze(0),
            torch.arange(seq_len).unsqueeze(0)
        ],
                                   dim=0)  # Shape: [3, batch_size, seq_len]

        # Forward pass
        result = mrope.forward(position_ids, [q, k])

        assert len(result) == 2
        assert result[0].shape == q.shape
        assert result[1].shape == k.shape

    def test_mrotary_embedding_sanity_1d_position_ids(self, basic_rope_params):
        """Test MRotaryEmbedding forward pass with 1D position_ids (fallback to regular RoPE)."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"

        head_dim = 128
        seq_len = 1000
        num_heads = 8
        mrope_section = [16, 24, 24]

        mrope = MRotaryEmbedding(rope_params=basic_rope_params,
                                 head_dim=head_dim,
                                 mrope_section=mrope_section,
                                 is_neox=True)

        # Create test inputs with 1D position_ids (fallback case)
        q = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float32,
                        device=device)
        k = torch.randn(seq_len,
                        num_heads * head_dim,
                        dtype=torch.float32,
                        device=device)
        position_ids = torch.arange(seq_len, device=device)

        # Forward pass
        result = mrope.forward(position_ids, [q, k])

        assert len(result) == 2
        assert result[0].shape == q.shape
        assert result[1].shape == k.shape


class TestDuplicateData:
    """Tests for duplicate_data support in RoPE table creation."""

    @pytest.mark.parametrize("dim,max_positions,theta", [
        (64, 4096, 10000.0),
        (64, 4096, 1000000.0),
        (128, 8192, 10000.0),
    ])
    def test_attention_plugin_duplicate_matches_manual_cat(
            self, dim, max_positions, theta):
        """duplicate_data=True should equal creating without duplication then
        manually concatenating, which is what _normalize_mla_rotary_cache_layout did."""
        _, cs_no = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=max_positions,
            dim=dim,
            theta=theta,
            duplicate_data=False,
        )
        _, cs_yes = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=max_positions,
            dim=dim,
            theta=theta,
            duplicate_data=True,
        )

        t = torch.tensor(cs_no).view(max_positions, -1, 2)
        expected = torch.cat([t, t], dim=1).reshape(1, -1).contiguous()
        actual = torch.tensor(cs_yes)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)

    def test_create_rope_const_params_mla_table_size(self):
        """With duplicate_data=True and interleave=True, the table should have
        dim*2 floats per position (doubled)."""
        rp = RopeParams(dim=64,
                        theta=10000.0,
                        max_positions=2048,
                        duplicate_data=True)
        _, cos_sin = rp.create_rope_const_params(interleave=True)
        floats_per_pos = cos_sin.numel() // rp.max_positions
        assert floats_per_pos == rp.dim * 2

    def test_create_rope_const_params_normal_table_size(self):
        """With duplicate_data=False and interleave=True, the table should have
        dim floats per position (normal)."""
        rp = RopeParams(dim=64,
                        theta=10000.0,
                        max_positions=2048,
                        duplicate_data=False)
        _, cos_sin = rp.create_rope_const_params(interleave=True)
        floats_per_pos = cos_sin.numel() // rp.max_positions
        assert floats_per_pos == rp.dim


class TestFromConfigDuplicateData:
    """Tests for RopeParams.from_config setting duplicate_data based on qk_rope_head_dim."""

    @staticmethod
    def _make_config(**overrides):
        defaults = dict(
            hidden_size=4096,
            num_attention_heads=32,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            model_type="llama",
        )
        defaults.update(overrides)

        class _Cfg:
            pass

        cfg = _Cfg()
        for k, v in defaults.items():
            setattr(cfg, k, v)
        return cfg

    def test_no_qk_rope_head_dim(self):
        rp = RopeParams.from_config(self._make_config())
        assert rp.duplicate_data is False

    def test_with_qk_rope_head_dim(self):
        rp = RopeParams.from_config(self._make_config(qk_rope_head_dim=64))
        assert rp.duplicate_data is True
        assert rp.dim == 64


class TestUnfusedRopeOwnership:
    """With rope_fusion=False the Python rotary module owns RoPE; the backend
    must receive no position-embedding params. yarn is not listed in
    PositionEmbeddingType.is_rope(), which used to leak the params through and
    made the attention kernel rotate a second time (double RoPE)."""

    def test_unfused_yarn_rope_is_applied_exactly_once(self):
        from tensorrt_llm._torch.attention_backend.interface import \
            PositionalEmbeddingParams
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.attention import Attention
        from tensorrt_llm.functional import PositionEmbeddingType

        yarn_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=32,
                theta=150000,
                scale_type=RotaryScalingType.yarn,
                scale=32.0,
                max_positions=1024,
                original_max_positions=256,
                beta_fast=32,
                beta_slow=1,
                duplicate_data=False,
            ),
            is_neox=True,
        )
        attn = Attention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=1024,
            bias=False,
            pos_embd_params=yarn_params,
            rope_fusion=False,
            layer_idx=0,
            dtype=torch.bfloat16,
            config=ModelConfig(),
        )

        assert attn.rotary_emb is not None
        # 0 means the kernel side received no position embedding.
        assert attn.attn.position_embedding_type == 0

    def test_unfused_yarn_rotation_matches_fused_kernel_convention(self):
        """The unfused (Python) yarn rotation must match the NeoX rotate-half
        convention the fused kernel applies with the same cos/sin table."""
        head_dim = 64
        num_pos, num_heads = 64, 4
        rope_params = RopeParams(
            dim=head_dim,
            theta=150000,
            scale_type=RotaryScalingType.yarn,
            scale=32.0,
            max_positions=1024,
            original_max_positions=256,
            beta_fast=32,
            beta_slow=1,
            duplicate_data=False,
        )
        emb = RotaryEmbedding(rope_params, head_dim=head_dim, is_neox=True)
        torch.manual_seed(0)
        q = torch.randn(num_pos, num_heads * head_dim)
        positions = torch.arange(num_pos).cuda()
        # Single-target call takes the pure-torch path.
        q_unfused = emb(positions, [q.cuda()])[0].cpu()

        # Reference: NeoX rotate-half with the exact table the kernel reads.
        table = emb.rotary_cos_sin.cpu()  # (max_pos, 2, head_dim/2)
        cos = table[:num_pos, 0, :].unsqueeze(1)
        sin = table[:num_pos, 1, :].unsqueeze(1)
        qh = q.view(num_pos, num_heads, head_dim)
        q1, q2 = qh[..., :head_dim // 2], qh[..., head_dim // 2:]
        q_ref = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin),
                          dim=-1).reshape(num_pos, -1)

        torch.testing.assert_close(q_unfused, q_ref, rtol=1e-5, atol=1e-5)
