import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import (RopeParams,
                                                             RotaryScalingType)
from tensorrt_llm._torch.modules.rotary_embedding import (MRotaryEmbedding,
                                                          RotaryEmbedding)


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
