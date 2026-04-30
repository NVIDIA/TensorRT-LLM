import pytest
import torch

import tensorrt_llm  # noqa: F401 - registers torch ops
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


def _python_mla_rope_ref(data, position_ids, cos_sin_cache, nope_dim, rope_dim,
                         inverse, is_neox):
    """Python reference for MLA RoPE (neox or interleaved style)."""
    num_tokens, num_heads, _ = data.shape
    half_rope = rope_dim // 2
    out = data.clone()
    for t in range(num_tokens):
        pos = position_ids[t].item()
        cos_val = cos_sin_cache[pos, 0, :]  # [half_rope]
        sin_val = cos_sin_cache[pos, 1, :]  # [half_rope]
        for h in range(num_heads):
            if is_neox:
                x1 = data[t, h, nope_dim:nope_dim + half_rope].float()
                x2 = data[t, h,
                          nope_dim + half_rope:nope_dim + rope_dim].float()
            else:
                rope_slice = data[t, h, nope_dim:nope_dim + rope_dim].float()
                x1 = rope_slice[0::2]
                x2 = rope_slice[1::2]
            if inverse:
                o1 = x1 * cos_val + x2 * sin_val
                o2 = x2 * cos_val - x1 * sin_val
            else:
                o1 = x1 * cos_val - x2 * sin_val
                o2 = x2 * cos_val + x1 * sin_val
            if is_neox:
                out[t, h, nope_dim:nope_dim + half_rope] = o1.to(out.dtype)
                out[t, h,
                    nope_dim + half_rope:nope_dim + rope_dim] = o2.to(out.dtype)
            else:
                out[t, h, nope_dim:nope_dim + rope_dim:2] = o1.to(out.dtype)
                out[t, h, nope_dim + 1:nope_dim + rope_dim:2] = o2.to(out.dtype)
    return out


class TestMLARoPEInplace:
    """Test suite for the fused mla_rope_inplace torch op."""

    @pytest.fixture
    def rope_params(self):
        return RopeParams(
            dim=64,
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

    @pytest.mark.parametrize("is_neox", [True, False])
    @pytest.mark.parametrize("inverse", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_mla_rope_inplace_matches_reference(self, rope_params, inverse,
                                                dtype, is_neox):
        """Test mla_rope_inplace op matches Python reference implementation."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"
        num_tokens = 32
        num_heads = 8
        nope_dim = 128
        rope_dim = 64

        rope = RotaryEmbedding(rope_params=rope_params,
                               head_dim=rope_dim,
                               is_neox=is_neox,
                               inverse=inverse)
        cos_sin_cache = rope.rotary_cos_sin.to(device)  # [max_pos, 2, half]

        data = torch.randn(num_tokens,
                           num_heads,
                           nope_dim + rope_dim,
                           dtype=dtype,
                           device=device)
        position_ids = torch.randint(0,
                                     1024, (num_tokens, ),
                                     dtype=torch.int32,
                                     device=device)

        # Reference
        expected = _python_mla_rope_ref(data, position_ids, cos_sin_cache,
                                        nope_dim, rope_dim, inverse, is_neox)

        # Op under test (in-place)
        torch.ops.trtllm.mla_rope_inplace(data, position_ids, cos_sin_cache,
                                          num_heads, nope_dim, rope_dim,
                                          inverse, is_neox)

        # nope portion should be unchanged
        torch.testing.assert_close(data[:, :, :nope_dim],
                                   expected[:, :, :nope_dim],
                                   atol=0,
                                   rtol=0)
        # rope portion should match reference
        torch.testing.assert_close(data[:, :, nope_dim:],
                                   expected[:, :, nope_dim:],
                                   atol=1e-2,
                                   rtol=1e-2)

    @pytest.mark.parametrize("is_neox", [True, False])
    def test_mla_rope_inplace_roundtrip(self, rope_params, is_neox):
        """Test that forward followed by inverse RoPE recovers original data."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"
        num_tokens = 16
        num_heads = 4
        nope_dim = 128
        rope_dim = 64
        dtype = torch.bfloat16

        rope = RotaryEmbedding(rope_params=rope_params,
                               head_dim=rope_dim,
                               is_neox=is_neox)
        cos_sin_cache = rope.rotary_cos_sin.to(device)

        data = torch.randn(num_tokens,
                           num_heads,
                           nope_dim + rope_dim,
                           dtype=dtype,
                           device=device)
        original = data.clone()
        position_ids = torch.arange(num_tokens,
                                    dtype=torch.int32,
                                    device=device)

        # Forward then inverse should recover original
        torch.ops.trtllm.mla_rope_inplace(data, position_ids, cos_sin_cache,
                                          num_heads, nope_dim, rope_dim, False,
                                          is_neox)
        torch.ops.trtllm.mla_rope_inplace(data, position_ids, cos_sin_cache,
                                          num_heads, nope_dim, rope_dim, True,
                                          is_neox)

        torch.testing.assert_close(data, original, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("num_tokens,num_heads", [(1, 4), (1, 64),
                                                      (16, 64)])
    def test_mla_rope_inplace_edge_cases(self, rope_params, num_tokens,
                                         num_heads):
        """Test edge cases: single token, many heads (exercises grid.y > 1)."""
        assert torch.cuda.is_available(), "This test requires CUDA"
        device = "cuda"
        nope_dim = 128
        rope_dim = 64
        dtype = torch.bfloat16

        rope = RotaryEmbedding(rope_params=rope_params,
                               head_dim=rope_dim,
                               is_neox=True)
        cos_sin_cache = rope.rotary_cos_sin.to(device)

        data = torch.randn(num_tokens,
                           num_heads,
                           nope_dim + rope_dim,
                           dtype=dtype,
                           device=device)
        expected = _python_mla_rope_ref(
            data, torch.arange(num_tokens, dtype=torch.int32, device=device),
            cos_sin_cache, nope_dim, rope_dim, False, True)

        torch.ops.trtllm.mla_rope_inplace(
            data, torch.arange(num_tokens, dtype=torch.int32, device=device),
            cos_sin_cache, num_heads, nope_dim, rope_dim, False, True)

        torch.testing.assert_close(data[:, :, :nope_dim],
                                   expected[:, :, :nope_dim],
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(data[:, :, nope_dim:],
                                   expected[:, :, nope_dim:],
                                   atol=1e-2,
                                   rtol=1e-2)
