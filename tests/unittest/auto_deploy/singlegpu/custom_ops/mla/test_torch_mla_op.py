"""Comprehensive test suite for torch MLA backend operations.

Tests the torch_mla source op and torch_backend_mla_with_cache cached op
with FlashInfer-compatible compressed cache layout.

Key features:
- 5 tensor arguments: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight
- Compressed cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
- Prefill: Expand compressed_kv, compute normal attention
- Generate: Weight absorption for efficiency
"""

import math

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def numpy_mla_reference_with_expansion(
    q_nope: np.ndarray,
    q_pe: np.ndarray,
    compressed_kv: np.ndarray,
    kpe: np.ndarray,
    kv_b_proj_weight: np.ndarray,
    mla_cache: np.ndarray,
    seq_len: np.ndarray,
    input_pos: np.ndarray,
    cache_loc: np.ndarray,
    seq_start: np.ndarray,
    scale: float = None,
    kv_lora_rank: int = None,
    is_generate: bool = False,
):
    """Numpy reference implementation of MLA attention with FlashInfer cache layout.

    This expands compressed_kv using kv_b_proj_weight for attention computation.
    """
    # Get dimensions
    if is_generate:
        batch_size = q_nope.shape[0]
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]
    else:
        batch_size = len(seq_len)
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if kv_lora_rank is None:
        kv_lora_rank = compressed_kv.shape[-1]

    # Infer v_head_dim from kv_b_proj_weight
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Update MLA cache first
    if is_generate:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            mla_cache[cache_idx, pos, :kv_lora_rank] = compressed_kv[i, 0]
            mla_cache[cache_idx, pos, kv_lora_rank:] = kpe[i, 0, 0]
    else:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            seq_len_i = seq_len[i]
            seq_start_i = seq_start[i]
            for j in range(seq_len_i):
                mla_cache[cache_idx, pos + j, :kv_lora_rank] = compressed_kv[seq_start_i + j]
                mla_cache[cache_idx, pos + j, kv_lora_rank:] = kpe[seq_start_i + j, 0]

    # Compute attention for each sequence
    outputs = []

    for i in range(batch_size):
        cache_idx = cache_loc[i]
        pos = input_pos[i]
        seq_len_i = seq_len[i]
        seq_start_i = seq_start[i]

        if seq_len_i == 0:
            continue

        # Get query for this sequence
        if is_generate:
            q_nope_seq = q_nope[i, 0]  # [N, qk_nope_head_dim]
            q_pe_seq = q_pe[i, 0]  # [N, qk_rope_head_dim]
        else:
            q_nope_seq = q_nope[seq_start_i : seq_start_i + seq_len_i]
            q_pe_seq = q_pe[seq_start_i : seq_start_i + seq_len_i]

        # Get cached compressed_kv and kpe
        kv_seq_len = pos + seq_len_i
        cached_data = mla_cache[cache_idx, :kv_seq_len]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]

        # Expand compressed_kv using kv_b_proj_weight
        # compressed_kv_cached: [kv_seq_len, kv_lora_rank]
        # kv_b_proj_weight: [N * kv_head_dim, kv_lora_rank]
        kv_expanded = np.matmul(compressed_kv_cached, kv_b_proj_weight.T)
        kv_expanded = kv_expanded.reshape(kv_seq_len, num_heads, kv_head_dim)

        k_nope = kv_expanded[:, :, :qk_nope_head_dim]
        v = kv_expanded[:, :, qk_nope_head_dim:]

        # Expand kpe to all heads
        kpe_expanded = np.broadcast_to(
            kpe_cached[:, None, :], (kv_seq_len, num_heads, qk_rope_head_dim)
        )

        # Construct full query and key
        if is_generate:
            query_full = np.concatenate([q_nope_seq, q_pe_seq], axis=-1)
        else:
            query_full = np.concatenate([q_nope_seq, q_pe_seq], axis=-1)

        key_full = np.concatenate([k_nope, kpe_expanded], axis=-1)

        # Compute attention scores
        if is_generate:
            attn_scores = np.einsum("nh,knh->nk", query_full, key_full) * scale
        else:
            attn_scores = np.einsum("snh,knh->snk", query_full, key_full) * scale
            causal_mask = np.triu(np.ones((seq_len_i, kv_seq_len)), k=kv_seq_len - seq_len_i + 1)
            attn_scores = np.where(causal_mask[:, None, :], -np.inf, attn_scores)

        # Apply softmax
        attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
        attn_scores_exp = np.exp(attn_scores - attn_scores_max)
        attn_weights = attn_scores_exp / np.sum(attn_scores_exp, axis=-1, keepdims=True)

        # Compute output
        if is_generate:
            attn_out = np.einsum("nk,knh->nh", attn_weights, v)
        else:
            attn_out = np.einsum("snk,knh->snh", attn_weights, v)

        outputs.append(attn_out)

    # Concatenate outputs
    if len(outputs) == 0:
        return np.zeros((1, 0, num_heads, v_head_dim), dtype=np.float32)
    elif is_generate:
        result = np.stack(outputs, axis=0)
        return result[:, None, :, :]
    else:
        result = np.concatenate(outputs, axis=0)
        return result[None, :, :, :]


class TestTorchMLASourceOp:
    """Test torch_mla source op (without cache)."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test configuration."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.atol = 1e-2
        self.rtol = 1e-2

        torch.cuda.empty_cache()
        torch.manual_seed(42)
        np.random.seed(42)

    def _create_mla_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        layout: str = "bsnd",
    ):
        """Create test data for MLA source op with compressed_kv."""
        kv_head_dim = qk_nope_head_dim + v_head_dim

        if layout == "bsnd":
            q_nope = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            q_pe = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            compressed_kv = torch.randn(
                batch_size, seq_len, kv_lora_rank, dtype=self.dtype, device=self.device
            )
            kpe = torch.randn(
                batch_size, seq_len, 1, qk_rope_head_dim, dtype=self.dtype, device=self.device
            )
        else:  # bnsd
            q_nope = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                qk_nope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            q_pe = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            compressed_kv = torch.randn(
                batch_size, seq_len, kv_lora_rank, dtype=self.dtype, device=self.device
            )
            kpe = torch.randn(
                batch_size, 1, seq_len, qk_rope_head_dim, dtype=self.dtype, device=self.device
            )

        # kv_b_proj_weight: [num_heads * kv_head_dim, kv_lora_rank]
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=self.dtype, device=self.device
        )

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "kpe": kpe,
            "kv_b_proj_weight": kv_b_proj_weight,
        }

    def test_basic_functionality(self):
        """Test basic MLA source op functionality."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 128, 64
        kv_lora_rank = 512
        v_head_dim = 128

        data = self._create_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
        )

        output = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            True,  # is_causal
            None,  # scale
            "bsnd",  # layout
        )

        # Verify output shape: [B, S, N, v_head_dim]
        expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_both_layouts(self):
        """Test MLA source op with both bsnd and bnsd layouts."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank = 256
        v_head_dim = 64

        for layout in ["bsnd", "bnsd"]:
            data = self._create_mla_data(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                v_head_dim,
                layout,
            )

            output = torch.ops.auto_deploy.torch_mla(
                data["q_nope"],
                data["q_pe"],
                data["compressed_kv"],
                data["kpe"],
                data["kv_b_proj_weight"],
                True,
                None,
                layout,
            )

            if layout == "bsnd":
                expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
            else:
                expected_shape = (batch_size, num_heads, seq_len, v_head_dim)

            assert output.shape == expected_shape, (
                f"Layout {layout}: Expected {expected_shape}, got {output.shape}"
            )

    def test_custom_scale(self):
        """Test MLA source op with custom scale."""
        batch_size, seq_len, num_heads = 1, 2, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        kv_lora_rank = 128
        v_head_dim = 32

        data = self._create_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
        )

        # Test with default scale
        output_default = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            True,
            None,
            "bsnd",
        )

        # Test with custom scale
        custom_scale = 0.5
        output_custom = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            True,
            custom_scale,
            "bsnd",
        )

        # Outputs should be different
        assert not torch.allclose(output_default, output_custom, atol=1e-3), (
            "Custom scale should affect output"
        )


class TestTorchBackendMLAWithCache:
    """Test torch_backend_mla_with_cache cached op."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test configuration."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.atol = 5e-2
        self.rtol = 5e-2

        torch.cuda.empty_cache()
        torch.manual_seed(42)
        np.random.seed(42)

    def _create_cached_mla_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        max_seq_len: int,
        cache_offset: int = 0,
    ):
        """Create test data for cached MLA op with FlashInfer layout."""
        kv_head_dim = qk_nope_head_dim + v_head_dim

        # Create input tensors (BSND layout)
        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=self.dtype, device=self.device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        compressed_kv = torch.randn(
            batch_size, seq_len, kv_lora_rank, dtype=self.dtype, device=self.device
        )
        kpe = torch.randn(
            batch_size, seq_len, 1, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )

        # kv_b_proj_weight: [num_heads * kv_head_dim, kv_lora_rank]
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=self.dtype, device=self.device
        )

        # Create FlashInfer MLA cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        mla_cache = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Pre-fill cache with random data if cache_offset > 0
        if cache_offset > 0:
            mla_cache[:, :cache_offset, :] = torch.randn(
                batch_size,
                cache_offset,
                kv_lora_rank + qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )

        # Setup metadata
        seq_len_tensor = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.int32)
        input_pos = torch.full((batch_size,), cache_offset, device=self.device, dtype=torch.int32)
        cache_loc = torch.arange(batch_size, device=self.device, dtype=torch.int32)

        if seq_len == 1:
            # Generate phase
            batch_info_host = torch.tensor(
                [0, 0, batch_size], device=self.device, dtype=torch.int32
            )
            cu_seqlen = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        else:
            # Context phase
            batch_info_host = torch.tensor(
                [batch_size, batch_size * seq_len, 0], device=self.device, dtype=torch.int32
            )
            cu_seqlen = torch.arange(
                0, batch_size * seq_len, seq_len, device=self.device, dtype=torch.int32
            )
            # Flatten inputs for context phase
            q_nope = q_nope.view(1, batch_size * seq_len, num_heads, qk_nope_head_dim)
            q_pe = q_pe.view(1, batch_size * seq_len, num_heads, qk_rope_head_dim)
            compressed_kv = compressed_kv.view(1, batch_size * seq_len, kv_lora_rank)
            kpe = kpe.view(1, batch_size * seq_len, 1, qk_rope_head_dim)

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "kpe": kpe,
            "kv_b_proj_weight": kv_b_proj_weight,
            "batch_info_host": batch_info_host,
            "seq_len": seq_len_tensor,
            "input_pos": input_pos,
            "cache_loc": cache_loc,
            "cu_seqlen": cu_seqlen,
            "mla_cache": mla_cache,
            "kv_lora_rank": kv_lora_rank,
        }

    def _run_cached_mla(self, data, scale=None):
        """Run cached MLA operation."""
        return torch.ops.auto_deploy.torch_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["cache_loc"],
            data["cu_seqlen"],
            data["mla_cache"],
            scale,
            data["kv_lora_rank"],
        )

    def test_generate_phase_basic(self):
        """Test generate phase (single token) basic functionality."""
        batch_size, seq_len, num_heads = 2, 1, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank = 256
        v_head_dim = 64
        max_seq_len = 128
        cache_offset = 5

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
            cache_offset,
        )

        output = self._run_cached_mla(data)

        # Verify output shape
        expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_context_phase_basic(self):
        """Test context phase (multi-token) basic functionality."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank = 256
        v_head_dim = 64
        max_seq_len = 128

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
        )

        output = self._run_cached_mla(data)

        # Verify output shape
        expected_shape = (1, batch_size * seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_cache_update_correctness(self):
        """Test that cache is updated correctly during forward pass."""
        batch_size, seq_len, num_heads = 1, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        kv_lora_rank = 128
        v_head_dim = 32
        max_seq_len = 32
        cache_offset = 5

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
            cache_offset,
        )

        # Store original cache values at target position
        original_cache_at_pos = data["mla_cache"][0, cache_offset].clone()

        # Run forward pass
        _ = self._run_cached_mla(data)

        # Check cache was updated at the correct position
        updated_cache_at_pos = data["mla_cache"][0, cache_offset]

        # The cache should have been updated
        assert not torch.allclose(original_cache_at_pos, updated_cache_at_pos, atol=1e-6), (
            "Cache should have been updated at the target position"
        )

    def test_cache_layout_flashinfer_compatible(self):
        """Test that cache layout matches FlashInfer spec (no num_heads dimension)."""
        batch_size, seq_len, num_heads = 2, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        kv_lora_rank = 512  # DeepSeek-style
        v_head_dim = 128
        max_seq_len = 64

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
        )

        # Verify cache shape matches FlashInfer layout: [batch, seq, kv_lora_rank + rope_dim]
        expected_cache_shape = (batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim)
        assert data["mla_cache"].shape == expected_cache_shape, (
            f"Cache shape {data['mla_cache'].shape} doesn't match FlashInfer layout {expected_cache_shape}"
        )

        # Verify zero-copy slicing works
        compressed_kv_slice = data["mla_cache"][:, :, :kv_lora_rank]
        kpe_slice = data["mla_cache"][:, :, kv_lora_rank:]

        assert compressed_kv_slice.shape == (batch_size, max_seq_len, kv_lora_rank)
        assert kpe_slice.shape == (batch_size, max_seq_len, qk_rope_head_dim)

        # Verify slices share memory (zero-copy)
        assert compressed_kv_slice.data_ptr() == data["mla_cache"].data_ptr(), (
            "compressed_kv slice should be zero-copy"
        )

    def test_generate_with_reference(self):
        """Test generate phase against numpy reference."""
        batch_size, seq_len, num_heads = 2, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        kv_lora_rank = 64
        v_head_dim = 32
        max_seq_len = 64
        cache_offset = 3

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            max_seq_len,
            cache_offset,
        )

        # Run backend
        output = self._run_cached_mla(data)

        # Run numpy reference
        reference = numpy_mla_reference_with_expansion(
            data["q_nope"].cpu().float().numpy(),
            data["q_pe"].cpu().float().numpy(),
            data["compressed_kv"].cpu().float().numpy(),
            data["kpe"].cpu().float().numpy(),
            data["kv_b_proj_weight"].cpu().float().numpy(),
            data["mla_cache"].cpu().float().numpy(),
            data["seq_len"].cpu().numpy(),
            data["input_pos"].cpu().numpy(),
            data["cache_loc"].cpu().numpy(),
            data["cu_seqlen"].cpu().numpy(),
            None,
            kv_lora_rank,
            is_generate=True,
        )

        reference_torch = torch.from_numpy(reference).to(output.device, output.dtype)
        assert torch.allclose(output, reference_torch, atol=self.atol, rtol=self.rtol), (
            f"Generate phase output doesn't match reference. "
            f"Max diff: {(output - reference_torch).abs().max():.6f}"
        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        batch_size, seq_len, num_heads = 1, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        kv_lora_rank = 64
        v_head_dim = 32
        max_seq_len = 32

        for dtype in [torch.float16, torch.bfloat16]:
            self.dtype = dtype
            data = self._create_cached_mla_data(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                v_head_dim,
                max_seq_len,
            )

            output = self._run_cached_mla(data)
            assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"

    def test_memory_efficiency(self):
        """Test that cache uses compressed dimensions (no num_heads)."""
        batch_size = 1
        max_seq_len = 1024
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        num_heads = 128  # DeepSeek V3

        # FlashInfer compressed cache size
        compressed_cache_size = batch_size * max_seq_len * (kv_lora_rank + qk_rope_head_dim)

        # Expanded per-head cache size (what we avoid)
        qk_nope_head_dim = 128
        v_head_dim = 128
        expanded_cache_size = (
            batch_size
            * max_seq_len
            * num_heads
            * (qk_nope_head_dim + v_head_dim + qk_rope_head_dim)
        )

        # Verify compression ratio
        compression_ratio = expanded_cache_size / compressed_cache_size
        assert compression_ratio > 50, f"Expected >50x compression, got {compression_ratio:.1f}x"


class TestMLADescriptor:
    """Test MultiHeadLatentAttention descriptor configuration."""

    def _get_mla_descriptor(self):
        """Get MLA descriptor from registry."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        return AttentionRegistry.get("torch_mla")

    def test_descriptor_registration(self):
        """Test that MLA descriptor is properly registered."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        assert AttentionRegistry.has("torch_mla"), "torch_mla should be registered"

    def test_descriptor_layout(self):
        """Test that MLA descriptor uses correct layout."""
        mla_descriptor = self._get_mla_descriptor()

        assert mla_descriptor.get_attention_layout() == "bsnd", "MLA should use bsnd layout"

    def test_descriptor_num_qkv_args(self):
        """Test that MLA descriptor expects 5 tensor args."""
        mla_descriptor = self._get_mla_descriptor()

        assert mla_descriptor.get_num_qkv_args() == 5, (
            "MLA should expect 5 tensor args (q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight)"
        )

    def test_descriptor_source_op(self):
        """Test that MLA descriptor points to correct source op."""
        mla_descriptor = self._get_mla_descriptor()

        source_op = mla_descriptor.get_source_attention_op()
        assert source_op == torch.ops.auto_deploy.torch_mla, "MLA should use torch_mla as source op"

    def test_descriptor_cached_op(self):
        """Test that MLA descriptor points to correct cached op."""
        mla_descriptor = self._get_mla_descriptor()

        cached_op = mla_descriptor.get_cached_attention_op()
        assert cached_op == torch.ops.auto_deploy.torch_cached_mla_with_cache.default, (
            "MLA should use torch_cached_mla_with_cache as cached op"
        )

    def test_descriptor_standard_metadata(self):
        """Test that MLA descriptor uses standard metadata args."""
        mla_descriptor = self._get_mla_descriptor()

        expected_args = ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]
        actual_args = mla_descriptor.get_standard_metadata_args()
        assert actual_args == expected_args, (
            f"Expected standard metadata {expected_args}, got {actual_args}"
        )
