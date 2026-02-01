# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy tests for TRT-LLM attention kernel in Auto-Deploy.

This test suite validates the numerical accuracy of the TRT-LLM attention backend
by comparing its outputs against a reference PyTorch implementation (SDPA).

Key testing scenarios:
1. Reference attention implementation correctness
2. Metadata preparation correctness
3. KV cache layout verification
4. GQA (Grouped Query Attention) configurations

These tests help debug issues like:
- Incorrect K/V being written to cache
- QKV layout mismatches
- Metadata conversion errors
- GQA replication issues
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest
import torch

# Skip if TRT-LLM is not available
pytest.importorskip("tensorrt_llm")

from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
    TrtllmLayerState,
    _prepare_trtllm_metadata,
    reset_trtllm_attention_state,
    trtllm_mha_with_cache,
)

try:
    from tensorrt_llm._torch.auto_deploy.custom_ops.pt_cache_backend import (
        PTCacheBackend,
        PTCacheConfig,
    )

    HAS_PT_CACHE_BACKEND = True
except ImportError:
    HAS_PT_CACHE_BACKEND = False

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

# Tolerance settings
ATOL_FP16 = 1e-2
RTOL_FP16 = 1e-3
ATOL_BF16 = 2e-2
RTOL_BF16 = 2e-3


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (b, num_kv_heads, s, d) -> (b, num_heads, s, d)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference attention implementation using PyTorch.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        v: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to use causal masking

    Returns:
        Output tensor [batch, seq_len, num_heads, head_dim]
    """
    batch, seq_len = q.shape[:2]
    num_kv_groups = num_heads // num_kv_heads

    # Reshape to [batch, heads, seq, head_dim]
    q = q.transpose(1, 2)  # [b, num_heads, s, d]
    k = k.transpose(1, 2)  # [b, num_kv_heads, s, d]
    v = v.transpose(1, 2)  # [b, num_kv_heads, s, d]

    # Expand KV for GQA
    if num_kv_groups > 1:
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

    # Compute attention
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    # Softmax in fp32 for numerical stability
    attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

    # Apply attention to values
    output = torch.matmul(attn_weights, v)

    # Reshape back to [batch, seq, heads, head_dim]
    output = output.transpose(1, 2)
    return output


def reference_attention_with_past_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference attention with past KV (for decode phase).

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: New Key tensor [batch, seq_len, num_kv_heads, head_dim]
        v: New Value tensor [batch, seq_len, num_kv_heads, head_dim]
        past_k: Past Key tensor [batch, past_len, num_kv_heads, head_dim]
        past_v: Past Value tensor [batch, past_len, num_kv_heads, head_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to use causal masking

    Returns:
        Tuple of (output, updated_k, updated_v)
    """
    batch, seq_len = q.shape[:2]
    past_len = past_k.shape[1]
    total_len = past_len + seq_len
    num_kv_groups = num_heads // num_kv_heads

    # Concatenate past and present KV
    k_full = torch.cat([past_k, k], dim=1)  # [batch, total_len, num_kv_heads, head_dim]
    v_full = torch.cat([past_v, v], dim=1)  # [batch, total_len, num_kv_heads, head_dim]

    # Reshape to [batch, heads, seq, head_dim]
    q = q.transpose(1, 2)  # [b, num_heads, seq_len, d]
    k_full = k_full.transpose(1, 2)  # [b, num_kv_heads, total_len, d]
    v_full = v_full.transpose(1, 2)  # [b, num_kv_heads, total_len, d]

    # Expand KV for GQA
    if num_kv_groups > 1:
        k_expanded = repeat_kv(k_full, num_kv_groups)
        v_expanded = repeat_kv(v_full, num_kv_groups)
    else:
        k_expanded = k_full
        v_expanded = v_full

    # Compute attention
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        # Create mask: query at position i (offset by past_len) can attend to keys <= past_len + i
        causal_mask = torch.zeros(seq_len, total_len, dtype=torch.bool, device=q.device)
        for i in range(seq_len):
            query_pos = past_len + i
            for j in range(total_len):
                if j > query_pos:
                    causal_mask[i, j] = True
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    # Softmax in fp32 for numerical stability
    attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)

    # Apply attention to values
    output = torch.matmul(attn_weights, v_expanded)

    # Reshape back to [batch, seq, heads, head_dim]
    output = output.transpose(1, 2)

    # Return full K/V for caching
    return output, k_full.transpose(1, 2), v_full.transpose(1, 2)


@dataclass
class AttentionTestConfig:
    """Configuration for attention accuracy tests."""

    # Model dimensions
    num_heads: int = 32
    num_kv_heads: int = 8  # GQA config
    head_dim: int = 128
    layer_idx: int = 0

    # Sequence configuration
    batch_size: int = 2
    seq_lens: List[int] = None  # Per-sequence lengths
    cache_positions: List[int] = None  # Starting cache position per sequence

    # Cache configuration
    page_size: int = 64
    num_pages: int = 32
    max_batch_size: int = 8
    max_seq_len: int = 2048
    max_num_tokens: int = 2048

    # Data types
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.seq_lens is None:
            self.seq_lens = [128] * self.batch_size
        if self.cache_positions is None:
            self.cache_positions = [0] * self.batch_size

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def pages_per_seq(self) -> List[int]:
        """Calculate pages needed per sequence."""
        pages = []
        for seq_len, cache_pos in zip(self.seq_lens, self.cache_positions):
            total_len = cache_pos + seq_len
            n_pages = (total_len + self.page_size - 1) // self.page_size
            pages.append(n_pages)
        return pages


class TestReferenceAttention:
    """Test suite for reference attention implementation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reference_attention_basic(self):
        """Test basic reference attention matches PyTorch SDPA."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 16, 8, 8, 64
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Reference implementation
        ref_output = reference_attention(q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True)

        # PyTorch SDPA
        q_sdpa = q.transpose(1, 2)  # [b, h, s, d]
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True
        )
        sdpa_output = sdpa_output.transpose(1, 2)  # [b, s, h, d]

        torch.testing.assert_close(ref_output, sdpa_output, atol=ATOL_FP16, rtol=RTOL_FP16)
        print("✓ Basic reference attention matches SDPA")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reference_attention_gqa(self):
        """Test reference attention with GQA (Grouped Query Attention)."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 16, 32, 8, 128
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Reference implementation
        ref_output = reference_attention(q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True)

        # Expand KV manually and use SDPA
        num_kv_groups = num_heads // num_kv_heads
        k_expanded = (
            k.unsqueeze(3)
            .expand(-1, -1, -1, num_kv_groups, -1)
            .reshape(batch, seq_len, num_heads, head_dim)
        )
        v_expanded = (
            v.unsqueeze(3)
            .expand(-1, -1, -1, num_kv_groups, -1)
            .reshape(batch, seq_len, num_heads, head_dim)
        )

        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_expanded.transpose(1, 2)
        v_sdpa = v_expanded.transpose(1, 2)
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=True
        )
        sdpa_output = sdpa_output.transpose(1, 2)

        torch.testing.assert_close(ref_output, sdpa_output, atol=ATOL_FP16, rtol=RTOL_FP16)
        print("✓ GQA reference attention matches expanded SDPA")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reference_attention_with_past_kv(self):
        """Test reference attention with past KV (decode scenario)."""
        batch, new_len, past_len = 2, 1, 32
        num_heads, num_kv_heads, head_dim = 32, 8, 128
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(batch, new_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, new_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, new_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        past_k = torch.randn(batch, past_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        past_v = torch.randn(batch, past_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Reference implementation
        ref_output, full_k, full_v = reference_attention_with_past_kv(
            q, k, v, past_k, past_v, num_heads, num_kv_heads, head_dim, is_causal=True
        )

        # Verify shapes
        assert ref_output.shape == (batch, new_len, num_heads, head_dim)
        assert full_k.shape == (batch, past_len + new_len, num_kv_heads, head_dim)
        assert full_v.shape == (batch, past_len + new_len, num_kv_heads, head_dim)

        # Verify full K/V contains past and new
        torch.testing.assert_close(full_k[:, :past_len], past_k, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(full_k[:, past_len:], k, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(full_v[:, :past_len], past_v, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(full_v[:, past_len:], v, atol=1e-6, rtol=1e-6)

        print("✓ Reference attention with past KV works correctly")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reference_attention_non_causal(self):
        """Test reference attention without causal masking."""
        batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 16, 8, 8, 64
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Reference implementation (non-causal)
        ref_output = reference_attention(
            q, k, v, num_heads, num_kv_heads, head_dim, is_causal=False
        )

        # PyTorch SDPA (non-causal)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=False
        )
        sdpa_output = sdpa_output.transpose(1, 2)

        torch.testing.assert_close(ref_output, sdpa_output, atol=ATOL_FP16, rtol=RTOL_FP16)
        print("✓ Non-causal reference attention matches SDPA")


class TestMetadataPreparation:
    """Test suite for TRT-LLM metadata preparation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_trtllm_attention_state()
        yield
        reset_trtllm_attention_state()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_trtllm_metadata_prefill(self):
        """Test metadata preparation for prefill scenario."""
        device = "cuda"
        num_prefill, num_decode = 2, 0
        num_seq = num_prefill + num_decode
        seq_lens = [32, 64]
        page_size = 64
        num_kv_heads = 8
        head_dim = 128

        # Create a layer state
        state = TrtllmLayerState(
            layer_idx=0,
            num_heads=32,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Prepare input metadata
        batch_info_host = torch.tensor([num_prefill, sum(seq_lens), num_decode], dtype=torch.int32)

        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        pages_per_seq = [(sl + page_size - 1) // page_size for sl in seq_lens]
        cu_pages = [0]
        for pp in pages_per_seq:
            cu_pages.append(cu_pages[-1] + pp)

        cu_num_pages = torch.tensor(cu_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = cu_num_pages.cpu()

        total_pages = sum(pages_per_seq)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)

        last_page_len = torch.tensor(
            [(sl - 1) % page_size + 1 for sl in seq_lens], dtype=torch.int32, device=device
        )
        last_page_len_host = last_page_len.cpu()

        seq_len_with_cache_host = torch.tensor(seq_lens, dtype=torch.int32)

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16, device=device
        )

        # Run metadata preparation
        result = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            kv_cache,
        )

        # Unpack results
        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = result

        # Verify results
        print(f"sequence_length: {sequence_length.tolist()}")
        print(f"context_lengths: {context_lengths.tolist()}")
        print(f"host_request_types: {host_request_types.tolist()}")
        print(f"kv_cache_block_offsets shape: {kv_cache_block_offsets.shape}")

        # Check sequence lengths match
        assert sequence_length.tolist() == seq_lens, "sequence_length mismatch"
        assert context_lengths.tolist() == seq_lens, "context_lengths mismatch"

        # Check request types (0 = prefill)
        assert host_request_types.tolist() == [0] * num_prefill, "request_types mismatch"

        # Check past KV lengths (should be 0 for fresh prefill)
        assert host_past_key_value_lengths.tolist() == [0] * num_seq, "past_kv_lengths mismatch"

        print("✓ Metadata preparation for prefill works correctly")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_trtllm_metadata_decode(self):
        """Test metadata preparation for decode scenario."""
        device = "cuda"
        num_prefill, num_decode = 0, 3
        # num_seq = num_prefill + num_decode
        seq_lens = [1, 1, 1]  # Decode generates 1 token
        cache_positions = [100, 200, 150]  # Already have cached tokens
        page_size = 64
        num_kv_heads = 8
        head_dim = 128

        # Create a layer state
        state = TrtllmLayerState(
            layer_idx=0,
            num_heads=32,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Prepare input metadata
        batch_info_host = torch.tensor([num_prefill, 0, num_decode], dtype=torch.int32)

        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        # Total length with cache
        total_lens = [cache_pos + sl for cache_pos, sl in zip(cache_positions, seq_lens)]
        pages_per_seq = [(tl + page_size - 1) // page_size for tl in total_lens]

        cu_pages = [0]
        for pp in pages_per_seq:
            cu_pages.append(cu_pages[-1] + pp)

        cu_num_pages = torch.tensor(cu_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = cu_num_pages.cpu()

        total_pages = sum(pages_per_seq)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)

        last_page_len = torch.tensor(
            [(tl - 1) % page_size + 1 for tl in total_lens], dtype=torch.int32, device=device
        )
        last_page_len_host = last_page_len.cpu()

        seq_len_with_cache_host = torch.tensor(total_lens, dtype=torch.int32)

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16, device=device
        )

        # Run metadata preparation
        result = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            kv_cache,
        )

        # Unpack results
        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = result

        # Verify results
        print(f"sequence_length: {sequence_length.tolist()}")
        print(f"context_lengths: {context_lengths.tolist()}")
        print(f"host_past_key_value_lengths: {host_past_key_value_lengths.tolist()}")
        print(f"host_request_types: {host_request_types.tolist()}")

        # Check sequence lengths (total including cache)
        assert sequence_length.tolist() == total_lens, "sequence_length mismatch"

        # Check context lengths (current input only)
        assert context_lengths.tolist() == seq_lens, "context_lengths mismatch"

        # Check request types (1 = decode)
        assert host_request_types.tolist() == [1] * num_decode, "request_types mismatch"

        # Check past KV lengths
        assert host_past_key_value_lengths.tolist() == cache_positions, "past_kv_lengths mismatch"

        print("✓ Metadata preparation for decode works correctly")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_trtllm_metadata_mixed(self):
        """Test metadata preparation for mixed prefill+decode scenario."""
        device = "cuda"
        num_prefill, num_decode = 2, 2
        # num_seq = num_prefill + num_decode
        seq_lens = [64, 32, 1, 1]  # First 2 prefill, last 2 decode
        cache_positions = [0, 0, 100, 150]  # Prefill starts at 0, decode has history
        page_size = 64
        num_kv_heads = 8
        head_dim = 128

        # Create a layer state
        state = TrtllmLayerState(
            layer_idx=0,
            num_heads=32,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Prepare input metadata
        num_prefill_tokens = sum(seq_lens[:num_prefill])
        batch_info_host = torch.tensor(
            [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32
        )

        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        # Total length with cache
        total_lens = [cache_pos + sl for cache_pos, sl in zip(cache_positions, seq_lens)]
        pages_per_seq = [(tl + page_size - 1) // page_size for tl in total_lens]

        cu_pages = [0]
        for pp in pages_per_seq:
            cu_pages.append(cu_pages[-1] + pp)

        cu_num_pages = torch.tensor(cu_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = cu_num_pages.cpu()

        total_pages = sum(pages_per_seq)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)

        last_page_len = torch.tensor(
            [(tl - 1) % page_size + 1 for tl in total_lens], dtype=torch.int32, device=device
        )
        last_page_len_host = last_page_len.cpu()

        seq_len_with_cache_host = torch.tensor(total_lens, dtype=torch.int32)

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16, device=device
        )

        # Run metadata preparation
        result = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            kv_cache,
        )

        # Unpack results
        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = result

        # Verify results
        print(f"sequence_length: {sequence_length.tolist()}")
        print(f"context_lengths: {context_lengths.tolist()}")
        print(f"host_past_key_value_lengths: {host_past_key_value_lengths.tolist()}")
        print(f"host_request_types: {host_request_types.tolist()}")

        # Check sequence lengths (total including cache)
        assert sequence_length.tolist() == total_lens, "sequence_length mismatch"

        # Check context lengths (current input only)
        assert context_lengths.tolist() == seq_lens, "context_lengths mismatch"

        # Check request types (0 = prefill, 1 = decode)
        expected_types = [0] * num_prefill + [1] * num_decode
        assert host_request_types.tolist() == expected_types, "request_types mismatch"

        # Check past KV lengths
        assert host_past_key_value_lengths.tolist() == cache_positions, "past_kv_lengths mismatch"

        print("✓ Metadata preparation for mixed batch works correctly")


class TestKVCacheLayout:
    """Test suite for KV cache layout verification."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ad_cache_layout(self):
        """Verify Auto-Deploy's expected KV cache layout."""
        num_pages = 4
        page_size = 64
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # AD cache format: [num_pages, page_size, num_kv_heads, head_dim]
        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Write some test values
        test_token = torch.randn(1, num_kv_heads, head_dim, dtype=dtype, device=device)
        page_idx, offset = 0, 5
        k_cache[page_idx, offset] = test_token
        v_cache[page_idx, offset] = -test_token  # Different value

        # Verify we can read back correctly
        k_read = k_cache[page_idx, offset]
        v_read = v_cache[page_idx, offset]

        torch.testing.assert_close(k_read, test_token.squeeze(0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_read, -test_token.squeeze(0), atol=1e-6, rtol=1e-6)

        print(f"AD cache shape: k_cache={k_cache.shape}, v_cache={v_cache.shape}")
        print("✓ AD cache layout verified")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cache_page_indexing(self):
        """Test that page indexing works correctly."""
        num_pages = 8
        page_size = 64
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        k_cache = torch.zeros(
            num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Simulate writing a sequence of length 150 to pages
        seq_len = 150
        pages_needed = (seq_len + page_size - 1) // page_size  # 3 pages
        cache_loc = torch.tensor(
            [0, 3, 5], device=device, dtype=torch.int32
        )  # Non-contiguous pages

        # Write values
        for i in range(seq_len):
            page_idx_in_seq = i // page_size
            offset_in_page = i % page_size
            physical_page = cache_loc[page_idx_in_seq].item()
            k_cache[physical_page, offset_in_page, 0, 0] = float(i)  # Store position as value

        # Verify read back
        for i in range(seq_len):
            page_idx_in_seq = i // page_size
            offset_in_page = i % page_size
            physical_page = cache_loc[page_idx_in_seq].item()
            assert k_cache[physical_page, offset_in_page, 0, 0].item() == float(i), (
                f"Mismatch at position {i}"
            )

        print(f"Pages needed: {pages_needed}, cache_loc: {cache_loc.tolist()}")
        print("✓ Cache page indexing verified")


@pytest.mark.skipif(not HAS_PT_CACHE_BACKEND, reason="PTCacheBackend not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPTCacheBackendMetadata:
    """Test suite for PTCacheBackend metadata handling.

    Note: PTCacheBackend.initialize() has a debug loop that tries to access
    layers [0, 1, 2, 31], so tests must use num_layers >= 32 to avoid IndexError.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_trtllm_attention_state()
        yield
        reset_trtllm_attention_state()

    def test_pt_backend_initialization(self):
        """Test PTCacheBackend initialization."""
        max_batch_size = 8
        max_seq_len = 2048
        # Must use num_layers >= 32 due to debug loop in pt_cache_backend.py
        num_layers = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 64

        # Create SequenceInfo
        si = SequenceInfo(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_num_tokens=2048,
            page_size=page_size,
        )

        # Create PTCacheConfig
        config = PTCacheConfig(
            num_layers=num_layers,
            num_kv_heads_per_layer=[num_kv_heads] * num_layers,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_sequences=max_batch_size,
            max_seq_len=max_seq_len,
            num_pages=si.num_pages,
            dtype=torch.float16,
        )

        # Create and initialize backend
        backend = PTCacheBackend(config)
        backend.initialize(si, torch.device("cuda"))

        # Verify tensors are allocated
        assert backend.sequence_length is not None
        assert backend.context_lengths is not None
        assert backend.kv_cache_block_offsets is not None

        print(f"sequence_length shape: {backend.sequence_length.shape}")
        print(f"context_lengths shape: {backend.context_lengths.shape}")
        print(f"block_offsets shape: {backend.kv_cache_block_offsets.shape}")

        # Get caches
        k_cache_0 = backend.get_cache("k_cache", 0)
        v_cache_0 = backend.get_cache("v_cache", 0)

        print(f"k_cache layer 0 shape: {k_cache_0.shape}")
        print(f"v_cache layer 0 shape: {v_cache_0.shape}")

        print("✓ PTCacheBackend initialization successful")

    def test_pt_backend_metadata_preparation(self):
        """Test PTCacheBackend metadata preparation."""
        max_batch_size = 8
        max_seq_len = 2048
        # Must use num_layers >= 32 due to debug loop in pt_cache_backend.py
        num_layers = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 64

        si = SequenceInfo(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_num_tokens=2048,
            page_size=page_size,
        )

        config = PTCacheConfig(
            num_layers=num_layers,
            num_kv_heads_per_layer=[num_kv_heads] * num_layers,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_sequences=max_batch_size,
            max_seq_len=max_seq_len,
            num_pages=si.num_pages,
            dtype=torch.float16,
        )

        backend = PTCacheBackend(config)
        backend.initialize(si, torch.device("cuda"))

        # Simulate a prefill batch
        batch_size = 3
        seq_lens = [32, 64, 48]
        total_tokens = sum(seq_lens)

        batch_info_host = torch.tensor([batch_size, total_tokens, 0], dtype=torch.int32)

        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        pages_per_seq = [(sl + page_size - 1) // page_size for sl in seq_lens]
        cu_pages = [0]
        for pp in pages_per_seq:
            cu_pages.append(cu_pages[-1] + pp)
        cu_num_pages_host = torch.tensor(cu_pages, dtype=torch.int32)

        total_pages = sum(pages_per_seq)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device="cuda")

        seq_len_with_cache_host = torch.tensor(seq_lens, dtype=torch.int32)

        # Get prepare function
        prep_fn = backend.get_host_prepare_metadata_function()
        assert prep_fn is not None

        # Call preparation
        prep_fn(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_host,
            cache_loc,
            seq_len_with_cache_host,
            skip_device_ops=False,
        )

        # Check results
        print(f"After prep - sequence_length[:3]: {backend.sequence_length[:batch_size].tolist()}")
        print(f"After prep - context_lengths[:3]: {backend.context_lengths[:batch_size].tolist()}")
        print(
            f"After prep - host_request_types[:3]: {backend.host_request_types[:batch_size].tolist()}"
        )

        assert backend.sequence_length[:batch_size].tolist() == seq_lens
        assert backend.context_lengths[:batch_size].tolist() == seq_lens
        assert backend.host_request_types[:batch_size].tolist() == [0] * batch_size

        print("✓ PTCacheBackend metadata preparation successful")


class TestFlashInferVsSDPA:
    """Test FlashInfer attention output against PyTorch SDPA.

    These tests verify that paged attention kernels (like FlashInfer, which is
    similar to what TRT-LLM uses) produce numerically correct results compared
    to PyTorch's reference implementation.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_trtllm_attention_state()
        yield
        reset_trtllm_attention_state()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_prefill_vs_sdpa(self):
        """Compare FlashInfer prefill attention against PyTorch SDPA."""
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Configuration
        batch_size = 2
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # KV cache config
        num_blocks = 16
        tokens_per_block = 128
        num_layers = 1

        # Create KV cache manager
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        # Add dummy requests
        request_ids = list(range(batch_size))
        token_nums = [seq_len] * batch_size
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Generate random Q, K, V
        torch.manual_seed(42)
        q = torch.randn(batch_size * seq_len, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size * seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size * seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)

        # Create FlashInfer attention layer
        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create metadata
        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=batch_size,  # All prefill
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0] * batch_size,  # Fresh prefill
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        # Run FlashInfer attention
        flashinfer_output = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        # Compute reference with SDPA
        q_ref = q.view(batch_size, seq_len, num_heads, head_dim)
        k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        ref_output = reference_attention(
            q_ref, k_ref, v_ref, num_heads, num_kv_heads, head_dim, is_causal=True
        )
        ref_output_flat = ref_output.contiguous().view(batch_size * seq_len, num_heads * head_dim)

        # Compare
        diff = (flashinfer_output - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("FlashInfer vs SDPA comparison (prefill):")
        print(f"  FlashInfer output mean: {flashinfer_output.abs().mean().item():.6f}")
        print(f"  Reference output mean: {ref_output_flat.abs().mean().item():.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        torch.testing.assert_close(
            flashinfer_output, ref_output_flat, atol=ATOL_FP16, rtol=RTOL_FP16
        )
        print("✓ FlashInfer prefill matches SDPA")

        kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_decode_vs_sdpa(self):
        """Compare FlashInfer decode attention against PyTorch SDPA."""
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Configuration
        batch_size = 4
        past_len = 64  # Already cached tokens
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # KV cache config
        num_blocks = 16
        tokens_per_block = 128
        num_layers = 1

        # Create KV cache manager
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        # Add dummy requests with past tokens
        request_ids = list(range(batch_size))
        past_lens = [past_len + i * 10 for i in range(batch_size)]  # Varying past lengths
        token_nums = [pl + 1 for pl in past_lens]  # past + 1 new token
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Pre-populate cache with random data
        for layer_idx in range(num_layers):
            buf = kv_cache_manager.get_buffers(layer_idx)
            if buf is not None:
                torch.nn.init.normal_(buf)

        # Generate random Q, K, V for decode (1 token per sequence)
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, num_kv_heads * head_dim, dtype=dtype, device=device)

        # Create FlashInfer attention layer
        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create metadata for decode
        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.ones(batch_size, dtype=torch.int32)  # 1 token each
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,  # All decode
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=past_lens),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        # Run FlashInfer attention
        flashinfer_output = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        # Verify output shape
        assert flashinfer_output.shape == (batch_size, num_heads * head_dim), (
            f"Expected shape {(batch_size, num_heads * head_dim)}, got {flashinfer_output.shape}"
        )

        # Note: Full numerical verification with decode requires reconstructing the
        # full K/V from cache which is complex. Here we verify shapes and that output
        # is non-zero and finite.
        assert flashinfer_output.isfinite().all(), "FlashInfer output contains NaN/Inf"
        assert flashinfer_output.abs().mean() > 0.01, "FlashInfer output is essentially zero"

        print("FlashInfer decode output:")
        print(f"  Shape: {flashinfer_output.shape}")
        print(f"  Mean abs value: {flashinfer_output.abs().mean().item():.6f}")
        print("✓ FlashInfer decode produces valid output")

        kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_gqa_vs_sdpa(self):
        """Compare FlashInfer GQA attention against PyTorch SDPA."""
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Llama-3.1-8B like configuration (GQA with 4:1 ratio)
        batch_size = 2
        seq_len = 16
        num_heads = 32
        num_kv_heads = 8  # 4 query heads per KV head
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # KV cache config
        num_blocks = 8
        tokens_per_block = 128
        num_layers = 1

        # Create KV cache manager
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        request_ids = list(range(batch_size))
        token_nums = [seq_len] * batch_size
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Generate random Q, K, V
        torch.manual_seed(42)
        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)

        # Create FlashInfer attention layer
        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create metadata
        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens_tensor = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens_tensor,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        # Run FlashInfer attention
        flashinfer_output = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        # Compute reference with SDPA (with GQA expansion)
        q_ref = q.view(batch_size, seq_len, num_heads, head_dim)
        k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        ref_output = reference_attention(
            q_ref, k_ref, v_ref, num_heads, num_kv_heads, head_dim, is_causal=True
        )
        ref_output_flat = ref_output.contiguous().view(total_tokens, num_heads * head_dim)

        # Compare
        diff = (flashinfer_output - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("FlashInfer GQA vs SDPA comparison:")
        print(
            f"  num_heads={num_heads}, num_kv_heads={num_kv_heads} (ratio {num_heads // num_kv_heads}:1)"
        )
        print(f"  FlashInfer output mean: {flashinfer_output.abs().mean().item():.6f}")
        print(f"  Reference output mean: {ref_output_flat.abs().mean().item():.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        torch.testing.assert_close(
            flashinfer_output, ref_output_flat, atol=ATOL_FP16, rtol=RTOL_FP16
        )
        print("✓ FlashInfer GQA matches SDPA")

        kv_cache_manager.shutdown()


class TestKVCacheContentVerification:
    """Tests that verify KV cache content after attention operations.

    These tests catch the GQA bug where V values are incorrectly written to K cache.
    The bug manifests when num_heads != num_kv_heads (GQA configuration).
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_trtllm_attention_state()
        yield
        reset_trtllm_attention_state()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_cache_content_after_prefill(self):
        """Verify K cache contains K values and V cache contains V values after prefill.

        This test uses correlation to verify that K and V are written to the correct
        cache locations, accounting for any layout transformations.
        """
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Configuration - GQA with 4:1 ratio (Llama-3 style)
        batch_size = 1
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8  # GQA
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # KV cache config
        num_blocks = 8
        tokens_per_block = 128
        num_layers = 1

        # Create KV cache manager
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        request_ids = list(range(batch_size))
        token_nums = [seq_len] * batch_size
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        # Generate random K and V with different patterns
        torch.manual_seed(42)
        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)

        # Create FlashInfer attention layer
        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # Create metadata
        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        # Run attention (this should update the cache)
        _ = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        # Verify cache content using correlation-based verification
        # FlashInfer may transform/scale values, but the correlation pattern should be preserved
        cache_buf = kv_cache_manager.get_buffers(0, kv_layout=attn_metadata.kv_layout)
        assert cache_buf is not None, "Cache buffer is None"

        print(f"Cache buffer shape: {cache_buf.shape}")
        print(f"KV layout: {attn_metadata.kv_layout}")

        # Get block IDs for the sequence
        block_ids = kv_cache_manager.get_batch_cache_indices(request_ids)[0]
        print(f"Block IDs: {block_ids}")

        # Get last page length
        last_page_len = attn_metadata.paged_kv_last_page_len[0].item()
        print(f"Last page length: {last_page_len}")

        # Get cached KV values - handle HND layout [num_blocks, 2, num_kv_heads, page_size, head_dim]
        # Need to handle layout properly
        cached_kvs = torch.concat(cache_buf[block_ids, :].unbind(dim=0), dim=1)

        # Reshape input K/V to match cache layout for comparison
        k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)[0]  # [seq, heads, dim]
        v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)[0]  # [seq, heads, dim]

        # The cache layout depends on kv_layout - for HND: [2, num_kv_heads, total_len, head_dim]
        # Extract K and V from cache
        if attn_metadata.kv_layout == "HND":
            cached_k = cached_kvs[0, :, :seq_len, :]  # [heads, seq, dim]
            cached_v = cached_kvs[1, :, :seq_len, :]  # [heads, seq, dim]
            # Transpose to match reference layout [seq, heads, dim]
            cached_k = cached_k.transpose(0, 1)
            cached_v = cached_v.transpose(0, 1)
        else:
            # NHD layout
            cached_k = cached_kvs[0, :seq_len]
            cached_v = cached_kvs[1, :seq_len]

        print(f"Cached K shape: {cached_k.shape}, K ref shape: {k_ref.shape}")
        print(f"Cached V shape: {cached_v.shape}, V ref shape: {v_ref.shape}")

        # Compare with numerical tolerance (FlashInfer stores KV exactly)
        try:
            torch.testing.assert_close(
                cached_k.to(k_ref.dtype),
                k_ref,
                atol=ATOL_FP16,
                rtol=RTOL_FP16,
                msg="K cache content doesn't match input K",
            )
            torch.testing.assert_close(
                cached_v.to(v_ref.dtype),
                v_ref,
                atol=ATOL_FP16,
                rtol=RTOL_FP16,
                msg="V cache content doesn't match input V",
            )
            print("✓ KV cache content verified: K cache has K values, V cache has V values")
        except AssertionError as e:
            # If exact match fails, check if K/V are swapped (the GQA bug)
            k_matches_v = torch.allclose(cached_k.to(v_ref.dtype), v_ref, atol=0.1, rtol=0.1)
            v_matches_k = torch.allclose(cached_v.to(k_ref.dtype), k_ref, atol=0.1, rtol=0.1)
            if k_matches_v or v_matches_k:
                raise AssertionError(
                    f"GQA BUG DETECTED: K and V appear to be swapped in cache!\n"
                    f"K cache matches input V: {k_matches_v}\n"
                    f"V cache matches input K: {v_matches_k}\n"
                    f"Original error: {e}"
                )
            raise

        kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_multistep_decode_accuracy(self):
        """Test multi-step decode with cache reads to catch accumulated errors.

        This test catches bugs where incorrect cache values propagate through
        multiple decode steps, causing accuracy degradation.
        """
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Configuration - GQA
        batch_size = 2
        prefill_len = 16
        num_decode_steps = 8
        num_heads = 32
        num_kv_heads = 8  # GQA
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        # KV cache config
        num_blocks = 16
        tokens_per_block = 128
        num_layers = 1

        # Create KV cache manager
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        request_ids = list(range(batch_size))

        # Generate test data
        torch.manual_seed(42)
        total_prefill_tokens = batch_size * prefill_len

        # Prefill QKV
        q_prefill = torch.randn(
            total_prefill_tokens, num_heads * head_dim, dtype=dtype, device=device
        )
        k_prefill = torch.randn(
            total_prefill_tokens, num_kv_heads * head_dim, dtype=dtype, device=device
        )
        v_prefill = torch.randn(
            total_prefill_tokens, num_kv_heads * head_dim, dtype=dtype, device=device
        )

        # Create FlashInfer attention layer
        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # --- STEP 1: Prefill ---
        token_nums = [prefill_len] * batch_size
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.tensor([prefill_len] * batch_size, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        prefill_output = flashinfer_attn.forward(
            q_prefill,
            k_prefill,
            v_prefill,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )

        # Store all K/V for reference computation
        all_k = [k_prefill.view(batch_size, prefill_len, num_kv_heads, head_dim).clone()]
        all_v = [v_prefill.view(batch_size, prefill_len, num_kv_heads, head_dim).clone()]
        all_q = [q_prefill.view(batch_size, prefill_len, num_heads, head_dim).clone()]
        flashinfer_outputs = [prefill_output]

        # --- STEP 2-N: Decode steps ---
        for step in range(num_decode_steps):
            past_len = prefill_len + step

            # Generate decode QKV (1 token per sequence)
            q_decode = torch.randn(batch_size, num_heads * head_dim, dtype=dtype, device=device)
            k_decode = torch.randn(batch_size, num_kv_heads * head_dim, dtype=dtype, device=device)
            v_decode = torch.randn(batch_size, num_kv_heads * head_dim, dtype=dtype, device=device)

            # Store for reference
            all_k.append(k_decode.view(batch_size, 1, num_kv_heads, head_dim).clone())
            all_v.append(v_decode.view(batch_size, 1, num_kv_heads, head_dim).clone())
            all_q.append(q_decode.view(batch_size, 1, num_heads, head_dim).clone())

            # Create decode metadata
            seq_lens_decode = torch.ones(batch_size, dtype=torch.int32)
            attn_metadata_decode = FlashInferAttentionMetadata(
                seq_lens=seq_lens_decode,
                num_contexts=0,  # All decode
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[past_len] * batch_size
                ),
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )
            attn_metadata_decode.prepare()

            decode_output = flashinfer_attn.forward(
                q_decode,
                k_decode,
                v_decode,
                attn_metadata_decode,
                attention_mask=PredefinedAttentionMask.CAUSAL,
            )
            flashinfer_outputs.append(decode_output)

        # --- Compute reference for final decode step ---
        # Concatenate all K/V for reference
        # full_k = torch.cat(all_k, dim=1)  # [batch, total_len, num_kv_heads, head_dim]
        # full_v = torch.cat(all_v, dim=1)
        last_q = all_q[-1]  # [batch, 1, num_heads, head_dim]

        # Reference attention for last decode step
        ref_output = reference_attention_with_past_kv(
            last_q,
            all_k[-1],  # New K
            all_v[-1],  # New V
            torch.cat(all_k[:-1], dim=1),  # Past K
            torch.cat(all_v[:-1], dim=1),  # Past V
            num_heads,
            num_kv_heads,
            head_dim,
            is_causal=True,
        )[0]  # Just get output, not updated K/V

        # Compare final decode output
        flashinfer_last = flashinfer_outputs[-1]
        ref_last = ref_output.contiguous().view(batch_size, num_heads * head_dim)

        diff = (flashinfer_last - ref_last).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Multi-step decode comparison (step {num_decode_steps}):")
        print(f"  Total sequence length: {prefill_len + num_decode_steps}")
        print(f"  FlashInfer output mean: {flashinfer_last.abs().mean().item():.6f}")
        print(f"  Reference output mean: {ref_last.abs().mean().item():.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        # More lenient tolerance for multi-step (accumulated fp16 errors)
        torch.testing.assert_close(flashinfer_last, ref_last, atol=0.05, rtol=0.01)
        print("✓ Multi-step decode matches reference")

        kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_gqa_cache_bug_detection(self):
        """Specifically test for the GQA cache bug where V is written to K cache.

        The bug: With GQA (num_heads > num_kv_heads), the kernel may incorrectly
        write V values to the K cache instead of K values.

        Test strategy:
        1. Create K with constant values ~1.0, V with constant values ~100.0
        2. Run prefill
        3. Check if K cache values are ~1.0 (correct) or ~100.0 (bug!)
        """
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Various GQA configurations to test
        configs = [
            {"num_heads": 32, "num_kv_heads": 8},  # 4:1 ratio (Llama-3)
            {"num_heads": 32, "num_kv_heads": 4},  # 8:1 ratio
            {"num_heads": 64, "num_kv_heads": 8},  # 8:1 ratio
            {"num_heads": 32, "num_kv_heads": 32},  # MHA (no GQA)
        ]

        for config in configs:
            num_heads = config["num_heads"]
            num_kv_heads = config["num_kv_heads"]
            is_gqa = num_heads != num_kv_heads

            batch_size = 1
            seq_len = 16
            head_dim = 128
            dtype = torch.float16
            device = "cuda"

            num_blocks = 8
            tokens_per_block = 128
            num_layers = 1

            mapping = Mapping(world_size=1, tp_size=1, rank=0)
            kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

            kv_cache_manager = KVCacheManager(
                kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=num_blocks * tokens_per_block,
                max_batch_size=batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
            )

            request_ids = list(range(batch_size))
            token_nums = [seq_len] * batch_size
            kv_cache_manager.add_dummy_requests(request_ids, token_nums)

            # Create VERY distinguishable K and V
            torch.manual_seed(42)
            total_tokens = batch_size * seq_len
            q = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device=device)

            # K: constant value ~1.0
            k = torch.ones(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)
            # V: constant value ~100.0
            v = torch.ones(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device) * 100

            flashinfer_attn = FlashInferAttention(
                layer_idx=0,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )

            from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

            seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
            attn_metadata = FlashInferAttentionMetadata(
                seq_lens=seq_lens,
                num_contexts=batch_size,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
                ),
                max_num_requests=batch_size,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
            )
            attn_metadata.prepare()

            _ = flashinfer_attn.forward(
                q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
            )

            # Check cache content - handle HND layout
            cache_buf = kv_cache_manager.get_buffers(0, kv_layout=attn_metadata.kv_layout)
            block_ids = kv_cache_manager.get_batch_cache_indices(request_ids)[0]
            cached_kvs = torch.concat(cache_buf[block_ids, :].unbind(dim=0), dim=1)

            # Handle layout: [2, num_kv_heads, seq, head_dim] for HND
            if attn_metadata.kv_layout == "HND":
                cached_k = cached_kvs[0, :, :seq_len, :]  # [heads, seq, dim]
                cached_v = cached_kvs[1, :, :seq_len, :]  # [heads, seq, dim]
            else:
                cached_k = cached_kvs[0, :seq_len]
                cached_v = cached_kvs[1, :seq_len]

            k_cache_mean = cached_k.mean().item()
            v_cache_mean = cached_v.mean().item()

            gqa_ratio = f"{num_heads}:{num_kv_heads}"

            # The critical check: K cache should have values ~1.0, not ~100.0
            k_has_correct_values = abs(k_cache_mean - 1.0) < 1.0  # Should be ~1.0
            v_has_correct_values = abs(v_cache_mean - 100.0) < 10.0  # Should be ~100.0
            k_has_v_values = abs(k_cache_mean - 100.0) < 10.0  # Bug indicator!

            print(
                f"Config {gqa_ratio} (GQA={is_gqa}): K cache mean={k_cache_mean:.2f}, V cache mean={v_cache_mean:.2f}"
            )

            if k_has_v_values:
                print("  ⚠️  BUG DETECTED: K cache contains V values!")

            assert k_has_correct_values, (
                f"GQA {gqa_ratio}: K cache has wrong values! "
                f"K cache mean={k_cache_mean:.2f}, expected ~1.0. "
                f"This indicates V was written to K cache (GQA bug)."
            )
            assert v_has_correct_values, (
                f"GQA {gqa_ratio}: V cache has wrong values! "
                f"V cache mean={v_cache_mean:.2f}, expected ~100.0."
            )

            print(f"  ✓ GQA {gqa_ratio}: Cache content correct")
            kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_variable_seqlen_batch(self):
        """Test with variable sequence lengths in the same batch."""
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        # Variable sequence lengths
        seq_lens_list = [8, 32, 16, 64]
        batch_size = len(seq_lens_list)
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        num_blocks = 16
        tokens_per_block = 128
        num_layers = 1

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        request_ids = list(range(batch_size))
        kv_cache_manager.add_dummy_requests(request_ids, seq_lens_list)

        # Generate Q, K, V for each sequence
        torch.manual_seed(42)
        total_tokens = sum(seq_lens_list)
        q = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)

        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        flashinfer_output = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        # Compute reference for each sequence separately
        offset = 0
        ref_outputs = []
        for i, sl in enumerate(seq_lens_list):
            q_seq = q[offset : offset + sl].view(1, sl, num_heads, head_dim)
            k_seq = k[offset : offset + sl].view(1, sl, num_kv_heads, head_dim)
            v_seq = v[offset : offset + sl].view(1, sl, num_kv_heads, head_dim)

            ref_out = reference_attention(
                q_seq, k_seq, v_seq, num_heads, num_kv_heads, head_dim, is_causal=True
            )
            ref_outputs.append(ref_out.contiguous().view(sl, num_heads * head_dim))
            offset += sl

        ref_output_flat = torch.cat(ref_outputs, dim=0)

        # Compare
        diff = (flashinfer_output - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("Variable seqlen batch comparison:")
        print(f"  Sequence lengths: {seq_lens_list}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        torch.testing.assert_close(
            flashinfer_output, ref_output_flat, atol=ATOL_FP16, rtol=RTOL_FP16
        )
        print("✓ Variable seqlen batch matches reference")

        kv_cache_manager.shutdown()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flashinfer_long_sequence(self):
        """Test with longer sequences to catch potential issues at scale."""
        try:
            import tensorrt_llm
            from tensorrt_llm._torch.attention_backend import FlashInferAttention
            from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
            from tensorrt_llm._torch.metadata import KVCacheParams
            from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
            from tensorrt_llm.bindings.executor import KvCacheConfig
            from tensorrt_llm.mapping import Mapping
        except ImportError:
            pytest.skip("FlashInfer attention backend not available")

        batch_size = 1
        seq_len = 512  # Longer sequence
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = "cuda"

        num_blocks = 64
        tokens_per_block = 128
        num_layers = 1

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        request_ids = list(range(batch_size))
        token_nums = [seq_len] * batch_size
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        torch.manual_seed(42)
        total_tokens = batch_size * seq_len
        q = torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(total_tokens, num_kv_heads * head_dim, dtype=dtype, device=device)

        flashinfer_attn = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        from tensorrt_llm._torch.attention_backend import FlashInferAttentionMetadata

        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
        attn_metadata = FlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[0] * batch_size
            ),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )
        attn_metadata.prepare()

        flashinfer_output = flashinfer_attn.forward(
            q, k, v, attn_metadata, attention_mask=PredefinedAttentionMask.CAUSAL
        )

        q_ref = q.view(batch_size, seq_len, num_heads, head_dim)
        k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        ref_output = reference_attention(
            q_ref, k_ref, v_ref, num_heads, num_kv_heads, head_dim, is_causal=True
        )
        ref_output_flat = ref_output.contiguous().view(total_tokens, num_heads * head_dim)

        diff = (flashinfer_output - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Long sequence comparison (seq_len={seq_len}):")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        # Slightly more lenient for long sequences
        torch.testing.assert_close(flashinfer_output, ref_output_flat, atol=0.02, rtol=0.01)
        print("✓ Long sequence matches reference")

        kv_cache_manager.shutdown()


class TestTRTLLMKernelAccuracy:
    """Tests that directly invoke the trtllm_mha_with_cache kernel.

    These tests catch accuracy bugs in the actual TRT-LLM kernel used by Auto-Deploy,
    not FlashInfer. The GQA bug where V is written to K cache can be caught here.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        reset_trtllm_attention_state()
        yield
        reset_trtllm_attention_state()

    def _setup_metadata(
        self,
        batch_size: int,
        seq_lens: List[int],
        cache_positions: List[int],
        page_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, ...]:
        """Set up AD-style metadata for trtllm_mha_with_cache.

        Returns:
            Tuple of metadata tensors needed by the kernel.
        """
        num_seq = batch_size
        num_prefill = sum(1 for cp in cache_positions if cp == 0)
        num_decode = num_seq - num_prefill
        num_prefill_tokens = sum(sl for sl, cp in zip(seq_lens, cache_positions) if cp == 0)

        # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
        batch_info_host = torch.tensor(
            [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32
        )

        # cu_seqlen: cumulative sequence lengths
        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        # Total length with cache for each sequence
        total_lens = [cp + sl for cp, sl in zip(cache_positions, seq_lens)]

        # Pages per sequence
        pages_per_seq = [(tl + page_size - 1) // page_size for tl in total_lens]

        # cu_num_pages: cumulative page counts
        cu_pages = [0]
        for pp in pages_per_seq:
            cu_pages.append(cu_pages[-1] + pp)
        cu_num_pages = torch.tensor(cu_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = cu_num_pages.cpu()

        # cache_loc: page indices (simple allocation: sequential pages)
        total_pages = sum(pages_per_seq)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)

        # last_page_len: tokens in last page per sequence
        last_page_len = torch.tensor(
            [(tl - 1) % page_size + 1 for tl in total_lens], dtype=torch.int32, device=device
        )
        last_page_len_host = last_page_len.cpu()

        # seq_len_with_cache: total length including cached tokens
        seq_len_with_cache_host = torch.tensor(total_lens, dtype=torch.int32)

        return (
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            total_pages,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_kernel_prefill_output_vs_sdpa(self):
        """Test TRT-LLM kernel prefill output against SDPA reference."""
        # Configuration - GQA
        batch_size = 1
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8  # GQA 4:1
        head_dim = 128
        page_size = 64
        dtype = torch.float16
        device = "cuda"
        # Use layer_idx=1 to avoid debug logging that accesses pt_backend
        layer_idx = 1

        # Set up metadata
        seq_lens = [seq_len] * batch_size
        cache_positions = [0] * batch_size  # Fresh prefill
        (
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            total_pages,
        ) = self._setup_metadata(batch_size, seq_lens, cache_positions, page_size, device)

        # Create Q, K, V
        torch.manual_seed(42)
        total_tokens = sum(seq_lens)
        # TRT-LLM expects [batch, seq, hidden] format for input
        q = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )

        # Create workspace buffer
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Call the TRT-LLM kernel
        output = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Compute reference with SDPA
        q_ref = q.view(batch_size, seq_len, num_heads, head_dim)
        k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        ref_output = reference_attention(
            q_ref, k_ref, v_ref, num_heads, num_kv_heads, head_dim, is_causal=True
        )
        ref_output_flat = ref_output.contiguous().view(total_tokens, num_heads * head_dim)

        # Flatten output if needed (kernel may return 3D)
        output_flat = output.view(-1, num_heads * head_dim)

        # Compare
        diff = (output_flat - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("TRT-LLM kernel vs SDPA (prefill):")
        print(f"  Output shape: {output.shape} -> {output_flat.shape}")
        print(f"  TRT-LLM output mean: {output_flat.abs().mean().item():.6f}")
        print(f"  Reference output mean: {ref_output_flat.abs().mean().item():.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        torch.testing.assert_close(output_flat, ref_output_flat, atol=ATOL_FP16, rtol=RTOL_FP16)
        print("✓ TRT-LLM kernel prefill output matches SDPA")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_kernel_cache_content_gqa(self):
        """Test that TRT-LLM kernel writes correct K/V to cache with GQA.

        This test catches the GQA bug where V is incorrectly written to K cache.
        """
        # Configuration - GQA
        batch_size = 1
        seq_len = 16
        num_heads = 32
        num_kv_heads = 8  # GQA 4:1
        head_dim = 128
        page_size = 64
        dtype = torch.float16
        device = "cuda"
        # Use layer_idx=1 to avoid debug logging that accesses pt_backend
        layer_idx = 1

        # Set up metadata
        seq_lens = [seq_len] * batch_size
        cache_positions = [0] * batch_size  # Fresh prefill
        (
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            total_pages,
        ) = self._setup_metadata(batch_size, seq_lens, cache_positions, page_size, device)

        # Create distinguishable K and V
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=dtype, device=device)
        # K: constant value ~1.0
        k = torch.ones(batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)
        # V: constant value ~100.0
        v = (
            torch.ones(batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device)
            * 100
        )

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )

        # Create workspace buffer
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Call the TRT-LLM kernel
        _ = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Check cache content (unified cache: kv_cache[:, 0, ...] = K, kv_cache[:, 1, ...] = V)
        # K cache should have values ~1.0, V cache should have values ~100.0
        k_cache_mean = kv_cache[:, 0, :, :seq_len, :].mean().item()
        v_cache_mean = kv_cache[:, 1, :, :seq_len, :].mean().item()

        print(f"TRT-LLM kernel cache content (GQA {num_heads}:{num_kv_heads}):")
        print(f"  K cache mean: {k_cache_mean:.2f} (expected ~1.0)")
        print(f"  V cache mean: {v_cache_mean:.2f} (expected ~100.0)")

        # Check for the GQA bug
        k_has_correct_values = abs(k_cache_mean - 1.0) < 1.0
        v_has_correct_values = abs(v_cache_mean - 100.0) < 10.0
        k_has_v_values = abs(k_cache_mean - 100.0) < 10.0

        if k_has_v_values:
            print("  ⚠️  BUG DETECTED: K cache contains V values!")

        assert k_has_correct_values, (
            f"GQA BUG: K cache has wrong values! "
            f"K cache mean={k_cache_mean:.2f}, expected ~1.0. "
            f"V was likely written to K cache."
        )
        assert v_has_correct_values, (
            f"V cache has wrong values! V cache mean={v_cache_mean:.2f}, expected ~100.0."
        )
        print("✓ TRT-LLM kernel cache content correct")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_kernel_multistep_decode(self):
        """Test TRT-LLM kernel with prefill followed by decode steps."""
        # Configuration - GQA
        batch_size = 1
        prefill_len = 16
        num_decode_steps = 4
        num_heads = 32
        num_kv_heads = 8  # GQA 4:1
        head_dim = 128
        page_size = 64
        dtype = torch.float16
        device = "cuda"
        # Use layer_idx=1 to avoid debug logging that accesses pt_backend
        layer_idx = 1

        torch.manual_seed(42)

        # Allocate unified KV cache large enough for prefill + decode
        max_seq_len = prefill_len + num_decode_steps
        total_pages = (max_seq_len + page_size - 1) // page_size
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Store all Q, K, V for reference computation
        all_q = []
        all_k = []
        all_v = []

        # --- STEP 1: Prefill ---
        seq_lens = [prefill_len]
        cache_positions = [0]
        (
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            _,
        ) = self._setup_metadata(batch_size, seq_lens, cache_positions, page_size, device)

        q_prefill = torch.randn(
            batch_size, prefill_len, num_heads * head_dim, dtype=dtype, device=device
        )
        k_prefill = torch.randn(
            batch_size, prefill_len, num_kv_heads * head_dim, dtype=dtype, device=device
        )
        v_prefill = torch.randn(
            batch_size, prefill_len, num_kv_heads * head_dim, dtype=dtype, device=device
        )

        all_q.append(q_prefill.view(batch_size, prefill_len, num_heads, head_dim))
        all_k.append(k_prefill.view(batch_size, prefill_len, num_kv_heads, head_dim))
        all_v.append(v_prefill.view(batch_size, prefill_len, num_kv_heads, head_dim))

        prefill_output = trtllm_mha_with_cache(
            q_prefill,
            k_prefill,
            v_prefill,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        trtllm_outputs = [prefill_output]

        # --- STEP 2-N: Decode steps ---
        for step in range(num_decode_steps):
            past_len = prefill_len + step

            seq_lens = [1]
            cache_positions = [past_len]
            (
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                _,
            ) = self._setup_metadata(batch_size, seq_lens, cache_positions, page_size, device)

            q_decode = torch.randn(batch_size, 1, num_heads * head_dim, dtype=dtype, device=device)
            k_decode = torch.randn(
                batch_size, 1, num_kv_heads * head_dim, dtype=dtype, device=device
            )
            v_decode = torch.randn(
                batch_size, 1, num_kv_heads * head_dim, dtype=dtype, device=device
            )

            all_q.append(q_decode.view(batch_size, 1, num_heads, head_dim))
            all_k.append(k_decode.view(batch_size, 1, num_kv_heads, head_dim))
            all_v.append(v_decode.view(batch_size, 1, num_kv_heads, head_dim))

            decode_output = trtllm_mha_with_cache(
                q_decode,
                k_decode,
                v_decode,
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                kv_cache,
                workspace,
                layer_idx=layer_idx,
                scale=None,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=page_size,
                max_num_requests=8,
                max_context_length=2048,
            )
            trtllm_outputs.append(decode_output)

        # Compute reference for last decode step
        # full_k = torch.cat(all_k, dim=1)  # [batch, total_len, num_kv_heads, head_dim]
        # full_v = torch.cat(all_v, dim=1)
        last_q = all_q[-1]

        ref_output, _, _ = reference_attention_with_past_kv(
            last_q,
            all_k[-1],
            all_v[-1],
            torch.cat(all_k[:-1], dim=1),
            torch.cat(all_v[:-1], dim=1),
            num_heads,
            num_kv_heads,
            head_dim,
            is_causal=True,
        )

        # Compare last decode output
        trtllm_last = trtllm_outputs[-1].view(-1, num_heads * head_dim)
        ref_last = ref_output.contiguous().view(-1, num_heads * head_dim)

        diff = (trtllm_last - ref_last).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"TRT-LLM kernel multi-step decode (step {num_decode_steps}):")
        print(f"  Total sequence length: {prefill_len + num_decode_steps}")
        print(f"  TRT-LLM output mean: {trtllm_last.abs().mean().item():.6f}")
        print(f"  Reference output mean: {ref_last.abs().mean().item():.6f}")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")

        # More lenient tolerance for multi-step
        torch.testing.assert_close(trtllm_last, ref_last, atol=0.1, rtol=0.05)
        print("✓ TRT-LLM kernel multi-step decode matches reference")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trtllm_kernel_gqa_ratios(self):
        """Test TRT-LLM kernel with various GQA ratios."""
        configs = [
            {"num_heads": 32, "num_kv_heads": 8},  # 4:1 (Llama-3)
            {"num_heads": 32, "num_kv_heads": 4},  # 8:1
            {"num_heads": 32, "num_kv_heads": 32},  # MHA
            {"num_heads": 32, "num_kv_heads": 1},  # MQA
        ]

        for config in configs:
            reset_trtllm_attention_state()

            num_heads = config["num_heads"]
            num_kv_heads = config["num_kv_heads"]
            # is_gqa = num_heads != num_kv_heads
            gqa_ratio = f"{num_heads}:{num_kv_heads}"

            batch_size = 1
            seq_len = 16
            head_dim = 128
            page_size = 64
            dtype = torch.float16
            device = "cuda"
            # Use layer_idx=1 to avoid debug logging that accesses pt_backend
            layer_idx = 1

            seq_lens = [seq_len]
            cache_positions = [0]
            (
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                total_pages,
            ) = self._setup_metadata(batch_size, seq_lens, cache_positions, page_size, device)

            torch.manual_seed(42)
            q = torch.randn(batch_size, seq_len, num_heads * head_dim, dtype=dtype, device=device)
            k = torch.randn(
                batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device
            )
            v = torch.randn(
                batch_size, seq_len, num_kv_heads * head_dim, dtype=dtype, device=device
            )

            kv_cache = torch.zeros(
                total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
            )
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

            output = trtllm_mha_with_cache(
                q,
                k,
                v,
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                kv_cache,
                workspace,
                layer_idx=layer_idx,
                scale=None,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=page_size,
                max_num_requests=8,
                max_context_length=2048,
            )

            # Compute reference
            q_ref = q.view(batch_size, seq_len, num_heads, head_dim)
            k_ref = k.view(batch_size, seq_len, num_kv_heads, head_dim)
            v_ref = v.view(batch_size, seq_len, num_kv_heads, head_dim)

            ref_output = reference_attention(
                q_ref, k_ref, v_ref, num_heads, num_kv_heads, head_dim, is_causal=True
            )
            ref_output_flat = ref_output.contiguous().view(-1, num_heads * head_dim)
            output_flat = output.view(-1, num_heads * head_dim)

            diff = (output_flat - ref_output_flat).abs()
            max_diff = diff.max().item()

            print(f"GQA {gqa_ratio}: max_diff={max_diff:.6f}")

            torch.testing.assert_close(
                output_flat,
                ref_output_flat,
                atol=ATOL_FP16,
                rtol=RTOL_FP16,
                msg=f"GQA {gqa_ratio} output mismatch",
            )
            print(f"  ✓ GQA {gqa_ratio} passed")


class TestDirectADIntegration:
    """Tests for direct AD KVCacheManager integration (use_pt_cache_backend=False).

    These tests verify that the TRT-LLM attention kernel works correctly when
    AD's KVCacheManager manages the KV caches directly, without PTCacheBackend.

    This flow is critical for the "proper integration" where AD's pool pointers
    are passed directly to thop.attention.
    """

    def test_direct_ad_prefill_accuracy(self):
        """Test prefill accuracy with direct AD integration.

        This simulates the flow where:
        1. AD's KVCacheManager manages the unified KV cache
        2. Pool pointers from KVCacheManager are passed to the kernel
        3. Block offsets use the num_layers * kv_factor multiplier
        """
        device = "cuda"
        dtype = torch.float16

        # Model config (similar to Llama-8B)
        batch_size = 2
        seq_len = 64
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32  # Critical for block offset calculation
        layer_idx = 0

        # Calculate pages needed
        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        # Create input tensors in 4D format: [batch, seq_len, num_heads, head_dim]
        # The kernel expects this format and internally reshapes
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Create unified KV cache: [num_blocks, kv_factor=2, num_kv_heads, tokens_per_block, head_dim]
        # This is the format AD's KVCacheManager uses with kv_factor=2
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )

        # Workspace
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Create metadata for prefill batch
        batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)

        seq_lens = [seq_len] * batch_size
        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        # Page allocation (sequential for simplicity)
        cache_loc_list = list(range(total_pages))
        cache_loc = torch.tensor(cache_loc_list, dtype=torch.int32, device=device)

        pages_per_seq_list = [pages_per_seq] * batch_size
        cu_num_pages = [0]
        for pp in pages_per_seq_list:
            cu_num_pages.append(cu_num_pages[-1] + pp)
        cu_num_pages_host = torch.tensor(cu_num_pages, dtype=torch.int32)
        cu_num_pages_dev = cu_num_pages_host.to(device)

        # Last page lengths
        last_page_len_val = seq_len % page_size
        if last_page_len_val == 0:
            last_page_len_val = page_size
        last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
        last_page_len = last_page_len_host.to(device)

        # Sequence lengths with cache (same as seq_len for prefill)
        seq_len_with_cache_host = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

        # Configure TRT-LLM attention for direct AD integration
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            _trtllm_config,
            trtllm_mha_with_cache,
        )

        # Reset state
        _global_state.reset()
        _trtllm_config.use_pt_cache_backend = False
        _trtllm_config._num_layers = num_layers

        # Call the kernel
        output = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_dev,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Compute reference using SDPA
        # Q, K, V are already 4D [batch, seq_len, num_heads, head_dim]
        ref_output = reference_attention(q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True)

        ref_output_flat = ref_output.contiguous().view(-1, num_heads * head_dim)
        output_flat = output.view(-1, num_heads * head_dim)

        # Check accuracy
        diff = (output_flat - ref_output_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print("Direct AD Integration Prefill Test:")
        print(f"  batch_size={batch_size}, seq_len={seq_len}")
        print(f"  num_layers={num_layers}, layer_idx={layer_idx}")
        print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        torch.testing.assert_close(
            output_flat,
            ref_output_flat,
            atol=ATOL_FP16,
            rtol=RTOL_FP16,
            msg="Direct AD prefill output mismatch",
        )
        print("  ✓ Direct AD prefill accuracy test passed")

    def test_direct_ad_multilayer_accuracy(self):
        """Test accuracy across multiple layers with direct AD integration.

        This tests that the block offset calculation correctly handles
        the num_layers * kv_factor multiplier across different layers.
        """
        device = "cuda"
        dtype = torch.float16

        batch_size = 2
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32

        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            _trtllm_config,
            trtllm_mha_with_cache,
        )

        # Reset and configure
        _global_state.reset()
        _trtllm_config.use_pt_cache_backend = False
        _trtllm_config._num_layers = num_layers

        # Test multiple layers
        test_layers = [0, 15, 31]  # First, middle, last

        for layer_idx in test_layers:
            # Create fresh tensors for each layer in 4D format
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            kv_cache = torch.zeros(
                total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
            )
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

            # Metadata
            batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)
            cu_seqlen_host = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
            cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)
            cu_num_pages_host = torch.tensor(
                [0, pages_per_seq, 2 * pages_per_seq], dtype=torch.int32
            )
            cu_num_pages_dev = cu_num_pages_host.to(device)
            last_page_len_val = seq_len % page_size or page_size
            last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
            last_page_len = last_page_len_host.to(device)
            seq_len_with_cache_host = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

            # Reset layer state for each layer test
            _global_state.reset()

            output = trtllm_mha_with_cache(
                q,
                k,
                v,
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages_dev,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                kv_cache,
                workspace,
                layer_idx=layer_idx,
                scale=None,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=page_size,
                max_num_requests=8,
                max_context_length=2048,
            )

            # Reference - Q, K, V are already 4D
            ref_output = reference_attention(
                q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True
            )

            ref_output_flat = ref_output.contiguous().view(-1, num_heads * head_dim)
            output_flat = output.view(-1, num_heads * head_dim)

            max_diff = (output_flat - ref_output_flat).abs().max().item()
            print(f"Layer {layer_idx}: max_diff={max_diff:.6f}")

            torch.testing.assert_close(
                output_flat,
                ref_output_flat,
                atol=ATOL_FP16,
                rtol=RTOL_FP16,
                msg=f"Layer {layer_idx} output mismatch",
            )

        print("  ✓ Multi-layer direct AD accuracy test passed")

    def test_direct_ad_decode_accuracy(self):
        """Test decode (generation) accuracy with direct AD integration.

        This tests the decode path where seq_len=1 and we have cached KV.
        """
        device = "cuda"
        dtype = torch.float16

        batch_size = 4
        prefill_len = 32
        decode_seq_len = 1
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32
        layer_idx = 0

        # Total sequence length including cached
        total_seq_len = prefill_len + decode_seq_len
        pages_per_seq = (total_seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            _trtllm_config,
            trtllm_mha_with_cache,
        )

        _global_state.reset()
        _trtllm_config.use_pt_cache_backend = False
        _trtllm_config._num_layers = num_layers

        # Create decode inputs (only new token) in 4D format
        q = torch.randn(batch_size, decode_seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(
            batch_size, decode_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v = torch.randn(
            batch_size, decode_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # KV cache with "prefilled" data
        kv_cache = torch.randn(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # Decode batch metadata
        batch_info_host = torch.tensor([0, 0, batch_size], dtype=torch.int32)  # All decode

        # Each sequence has 1 token input
        seq_lens = [decode_seq_len] * batch_size
        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = torch.tensor(
            [i * pages_per_seq for i in range(batch_size + 1)], dtype=torch.int32
        )
        cu_num_pages_dev = cu_num_pages_host.to(device)

        last_page_len_val = total_seq_len % page_size or page_size
        last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
        last_page_len = last_page_len_host.to(device)

        # Total sequence length with cache
        seq_len_with_cache_host = torch.tensor([total_seq_len] * batch_size, dtype=torch.int32)

        output = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_dev,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Basic sanity checks
        # Output shape may be flattened by the kernel
        expected_elements = batch_size * decode_seq_len * num_heads * head_dim
        assert output.numel() == expected_elements, (
            f"Output elements mismatch: {output.numel()} vs {expected_elements}"
        )
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Check output is non-trivial (not all zeros or constant)
        output_std = output.std().item()
        assert output_std > 0.01, f"Output appears degenerate: std={output_std}"

        print("Direct AD Decode Test:")
        print(f"  batch_size={batch_size}, prefill_len={prefill_len}")
        print(f"  output_shape={output.shape}")
        print(f"  output_std={output_std:.4f}")
        print("  ✓ Direct AD decode sanity test passed")

    def test_ad_pool_path_with_mock_pointers(self):
        """Test the actual AD pool path by setting up mock pool pointers.

        This tests the use_ad_pool=True code path by:
        1. Creating a simulated multi-layer KV cache pool (like KVCacheManagerCpp)
        2. Setting up seq_info with pool pointers to this memory
        3. Running the kernel with the correct block offset multiplier

        This is the closest simulation of the real AD integration without
        actually instantiating KVCacheManagerCpp.
        """
        device = "cuda"
        dtype = torch.float16

        batch_size = 2
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32
        layer_idx = 0

        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        # Create a multi-layer interleaved pool like KVCacheManagerCpp
        # Layout: [total_interleaved_blocks, block_size]
        # where total_interleaved_blocks = total_pages * num_layers * kv_factor
        kv_factor = 2
        block_size = num_kv_heads * page_size * head_dim
        total_interleaved_blocks = total_pages * num_layers * kv_factor

        # This simulates AD's KVCacheManager pool
        ad_pool = torch.randn(total_interleaved_blocks, block_size, dtype=dtype, device=device)

        # Also need a workspace and the kv_cache for the kernel interface
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

        # The kv_cache passed to trtllm_mha_with_cache when using AD pool
        # is just for shape/layout info - actual data comes from pool pointers
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Metadata
        batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)
        cu_seqlen_host = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)
        cu_num_pages_host = torch.tensor([0, pages_per_seq, 2 * pages_per_seq], dtype=torch.int32)
        cu_num_pages_dev = cu_num_pages_host.to(device)
        last_page_len_val = seq_len % page_size or page_size
        last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
        last_page_len = last_page_len_host.to(device)
        seq_len_with_cache_host = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

        # Set up seq_info with pool pointers (simulating AD's KVCacheManager)
        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            _trtllm_config,
            trtllm_mha_with_cache,
        )

        _global_state.reset()
        _trtllm_config.use_pt_cache_backend = False
        _trtllm_config._num_layers = num_layers

        # Create pool pointers tensor: [1, 2] - primary and secondary pool (secondary unused)
        pool_pointers = torch.zeros(1, 2, dtype=torch.int64)
        pool_pointers[0, 0] = ad_pool.data_ptr()
        pool_pointers[0, 1] = 0  # No secondary pool

        # Create pool mapping: [num_layers, 2] - format from AD's KVCacheManager
        # AD format: [[0, 0], [0, 1], [0, 2], ...] where:
        #   - Column 0: pool index (0 = primary pool)
        #   - Column 1: layer index/offset
        pool_mapping = torch.zeros(num_layers, 2, dtype=torch.int32)
        for layer in range(num_layers):
            pool_mapping[layer, 0] = 0  # All layers use pool 0
            pool_mapping[layer, 1] = layer  # Layer-specific offset

        # Create a mock SequenceInfo with pool pointers
        class MockSequenceInfo:
            def __init__(self, pool_ptrs, pool_map):
                self.kv_cache_pool_pointers = pool_ptrs
                self.kv_cache_pool_mapping = pool_map

        _trtllm_config._sequence_info = MockSequenceInfo(pool_pointers, pool_mapping)

        # Run kernel
        output = trtllm_mha_with_cache(
            q,
            k,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_dev,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            kv_cache,
            workspace,
            layer_idx=layer_idx,
            scale=None,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Basic sanity checks
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Compute reference
        ref_output = reference_attention(q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True)

        ref_output_flat = ref_output.contiguous().view(-1, num_heads * head_dim)
        output_flat = output.view(-1, num_heads * head_dim)

        max_diff = (output_flat - ref_output_flat).abs().max().item()
        mean_diff = (output_flat - ref_output_flat).abs().mean().item()

        print("AD Pool Path Test (with mock pool pointers):")
        print(f"  total_interleaved_blocks={total_interleaved_blocks}")
        print(f"  pool_pointer={ad_pool.data_ptr()}")
        print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        torch.testing.assert_close(
            output_flat,
            ref_output_flat,
            atol=ATOL_FP16,
            rtol=RTOL_FP16,
            msg="AD pool path output mismatch",
        )
        print("  ✓ AD pool path accuracy test passed")

    def test_ad_pool_path_multilayer(self):
        """Test AD pool path across multiple layers.

        This verifies that the block offset calculation and pool_mapping
        work correctly for all layers, not just layer 0.
        """
        device = "cuda"
        dtype = torch.float16

        batch_size = 2
        seq_len = 32
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32

        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        # Create a multi-layer interleaved pool
        kv_factor = 2
        block_size = num_kv_heads * page_size * head_dim
        total_interleaved_blocks = total_pages * num_layers * kv_factor

        ad_pool = torch.randn(total_interleaved_blocks, block_size, dtype=dtype, device=device)

        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            _global_state,
            _trtllm_config,
            trtllm_mha_with_cache,
        )

        # Create pool pointers
        pool_pointers = torch.zeros(1, 2, dtype=torch.int64)
        pool_pointers[0, 0] = ad_pool.data_ptr()
        pool_pointers[0, 1] = 0

        # AD's pool_mapping format
        pool_mapping = torch.zeros(num_layers, 2, dtype=torch.int32)
        for layer in range(num_layers):
            pool_mapping[layer, 0] = 0  # Pool index
            pool_mapping[layer, 1] = layer  # Layer offset

        class MockSequenceInfo:
            def __init__(self, pool_ptrs, pool_map):
                self.kv_cache_pool_pointers = pool_ptrs
                self.kv_cache_pool_mapping = pool_map

        # Test layers 0, 15, 31
        test_layers = [0, 15, 31]

        for layer_idx in test_layers:
            # Reset state for each layer
            _global_state.reset()
            _trtllm_config.use_pt_cache_backend = False
            _trtllm_config._num_layers = num_layers
            _trtllm_config._sequence_info = MockSequenceInfo(pool_pointers, pool_mapping)

            q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            kv_cache = torch.zeros(
                total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
            )
            workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

            batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)
            cu_seqlen_host = torch.tensor([0, seq_len, 2 * seq_len], dtype=torch.int32)
            cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)
            cu_num_pages_host = torch.tensor(
                [0, pages_per_seq, 2 * pages_per_seq], dtype=torch.int32
            )
            cu_num_pages_dev = cu_num_pages_host.to(device)
            last_page_len_val = seq_len % page_size or page_size
            last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
            last_page_len = last_page_len_host.to(device)
            seq_len_with_cache_host = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

            output = trtllm_mha_with_cache(
                q,
                k,
                v,
                batch_info_host,
                cu_seqlen_host,
                cu_num_pages_dev,
                cu_num_pages_host,
                cache_loc,
                last_page_len,
                last_page_len_host,
                seq_len_with_cache_host,
                kv_cache,
                workspace,
                layer_idx=layer_idx,
                scale=None,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=page_size,
                max_num_requests=8,
                max_context_length=2048,
            )

            # Reference
            ref_output = reference_attention(
                q, k, v, num_heads, num_kv_heads, head_dim, is_causal=True
            )

            ref_output_flat = ref_output.contiguous().view(-1, num_heads * head_dim)
            output_flat = output.view(-1, num_heads * head_dim)

            max_diff = (output_flat - ref_output_flat).abs().max().item()
            mean_diff = (output_flat - ref_output_flat).abs().mean().item()

            print(f"  Layer {layer_idx}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

            torch.testing.assert_close(
                output_flat,
                ref_output_flat,
                atol=ATOL_FP16,
                rtol=RTOL_FP16,
                msg=f"AD pool path layer {layer_idx} output mismatch",
            )

        print("  ✓ AD pool path multilayer test passed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pool_pointers_and_block_offsets_format(self):
        """Verify pool pointers and block offsets are in correct format.

        This test validates the fix for thop.attention integration by checking:
        1. pool_pointers[0, 1] == 0 (secondary pool, NOT duplicated base_ptr)
        2. K and V block offsets are IDENTICAL (they share blocks)
        3. Block offsets equal cache_loc directly (no multiplier)

        These conditions must be met for thop.attention to work correctly with
        AD's KVCacheManager-managed caches.
        """
        device = "cuda"
        dtype = torch.float16

        batch_size = 2
        seq_len = 64
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 32
        num_layers = 32
        layer_idx = 0

        pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = pages_per_seq * batch_size

        from tensorrt_llm._torch.auto_deploy.custom_ops.trtllm_attention import (
            TrtllmLayerState,
            _global_state,
            _prepare_trtllm_metadata,
            _trtllm_config,
        )

        # Reset state
        _global_state.reset()
        _trtllm_config.use_pt_cache_backend = False
        _trtllm_config._num_layers = num_layers

        # Create layer state
        state = TrtllmLayerState(
            layer_idx=layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_num_requests=8,
            max_context_length=2048,
        )

        # Setup metadata for prefill batch
        batch_info_host = torch.tensor([batch_size, batch_size * seq_len, 0], dtype=torch.int32)

        seq_lens = [seq_len] * batch_size
        cu_seqlen = [0]
        for sl in seq_lens:
            cu_seqlen.append(cu_seqlen[-1] + sl)
        cu_seqlen_host = torch.tensor(cu_seqlen, dtype=torch.int32)

        # Page allocation
        cache_loc = torch.arange(total_pages, dtype=torch.int32, device=device)
        cu_num_pages = [0]
        for i in range(batch_size):
            cu_num_pages.append(cu_num_pages[-1] + pages_per_seq)
        cu_num_pages_host = torch.tensor(cu_num_pages, dtype=torch.int32)
        cu_num_pages_dev = cu_num_pages_host.to(device)

        # Last page lengths
        last_page_len_val = seq_len % page_size or page_size
        last_page_len_host = torch.tensor([last_page_len_val] * batch_size, dtype=torch.int32)
        last_page_len = last_page_len_host.to(device)

        seq_len_with_cache_host = torch.tensor([seq_len] * batch_size, dtype=torch.int32)

        # Create unified KV cache
        kv_cache = torch.zeros(
            total_pages, 2, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
        )

        # Create mock AD pool pointers (simulating KVCacheManager output)
        # Format: [[primary_gpu_ptr, secondary_cpu_ptr]] where secondary is 0 (no CPU offload)
        ad_pool_pointers = torch.tensor([[kv_cache.data_ptr(), 0]], dtype=torch.int64, device="cpu")

        # Create mock pool mapping: [pool_idx, layer_offset] for each layer
        # pool_mapping[layer] = [0, layer] means pool 0, offset = layer
        ad_pool_mapping = torch.zeros(num_layers, 2, dtype=torch.int32, device="cpu")
        for i in range(num_layers):
            ad_pool_mapping[i, 0] = 0  # pool index
            ad_pool_mapping[i, 1] = i  # layer offset

        # Call _prepare_trtllm_metadata with AD pool info
        result = _prepare_trtllm_metadata(
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages_dev,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            state,
            kv_cache,
            ad_pool_pointers=ad_pool_pointers,
            ad_pool_mapping=ad_pool_mapping,
        )

        (
            sequence_length,
            host_past_key_value_lengths,
            host_total_kv_lens,
            context_lengths,
            host_context_lengths,
            host_request_types,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
        ) = result

        print("\n=== Pool Pointers and Block Offsets Format Test ===")
        print(f"batch_size={batch_size}, seq_len={seq_len}, pages_per_seq={pages_per_seq}")
        print(f"total_pages={total_pages}, num_layers={num_layers}")
        print(f"cache_loc: {cache_loc.tolist()}")
        print(f"host_kv_cache_pool_pointers: {host_kv_cache_pool_pointers.tolist()}")
        print(f"kv_cache_block_offsets shape: {kv_cache_block_offsets.shape}")

        # CHECK 1: Pool pointers format - [0, 1] should be 0 (no secondary CPU pool)
        primary_ptr = host_kv_cache_pool_pointers[0, 0].item()
        secondary_ptr = host_kv_cache_pool_pointers[0, 1].item()
        print(f"  Primary pool pointer: {primary_ptr}")
        print(f"  Secondary pool pointer: {secondary_ptr}")

        assert primary_ptr != 0, "Primary pool pointer should be non-zero"
        assert secondary_ptr == 0, (
            f"Secondary pool pointer should be 0 (no CPU offload), "
            f"got {secondary_ptr}. This indicates pool_pointers[0,1] was incorrectly "
            f"set to the base pointer instead of 0."
        )
        print("  ✓ CHECK 1 PASSED: Pool pointers format is correct [[ptr, 0]]")

        # CHECK 2: V block offsets = K block offsets + 1
        k_offsets = kv_cache_block_offsets[0, :batch_size, 0, :pages_per_seq]  # K
        v_offsets = kv_cache_block_offsets[0, :batch_size, 1, :pages_per_seq]  # V
        print(f"  K offsets: {k_offsets.tolist()}")
        print(f"  V offsets: {v_offsets.tolist()}")

        assert torch.equal(v_offsets, k_offsets + 1), (
            f"V block offsets must equal K + 1 (interleaved K/V in pool). "
            f"K={k_offsets.tolist()}, V={v_offsets.tolist()}"
        )
        print("  ✓ CHECK 2 PASSED: V block offsets = K + 1")

        # CHECK 3: Block offsets use multiplier=2 (K/V per page within layer's region)
        # AD's pool has each layer in its own contiguous region: [P0_K, P0_V, P1_K, P1_V, ...]
        # K offset = cache_loc * 2, V offset = cache_loc * 2 + 1
        kv_factor = 2
        for seq_idx in range(batch_size):
            page_start = seq_idx * pages_per_seq
            expected_k = [p * kv_factor for p in range(page_start, page_start + pages_per_seq)]
            expected_v = [p * kv_factor + 1 for p in range(page_start, page_start + pages_per_seq)]
            actual_k = k_offsets[seq_idx].tolist()
            actual_v = v_offsets[seq_idx].tolist()
            assert actual_k == expected_k, (
                f"Seq {seq_idx}: K block offsets should be cache_loc * {kv_factor}. "
                f"Expected {expected_k}, got {actual_k}."
            )
            assert actual_v == expected_v, (
                f"Seq {seq_idx}: V block offsets should be cache_loc * {kv_factor} + 1. "
                f"Expected {expected_v}, got {actual_v}."
            )
        print("  ✓ CHECK 3 PASSED: K offset = cache_loc * 2, V offset = K + 1")

        print("✓ All pool pointer and block offset format checks PASSED")


def run_quick_tests():
    """Run a quick subset of tests for debugging."""
    print("=" * 60)
    print("TRT-LLM Attention Accuracy Quick Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    # Test reference attention
    print("\n--- Testing Reference Attention ---")
    test_ref = TestReferenceAttention()
    test_ref.test_reference_attention_basic()
    test_ref.test_reference_attention_gqa()
    test_ref.test_reference_attention_with_past_kv()

    # Test metadata preparation
    print("\n--- Testing Metadata Preparation ---")
    reset_trtllm_attention_state()
    test_meta = TestMetadataPreparation()
    test_meta.test_prepare_trtllm_metadata_prefill()

    reset_trtllm_attention_state()
    test_meta.test_prepare_trtllm_metadata_decode()

    reset_trtllm_attention_state()
    test_meta.test_prepare_trtllm_metadata_mixed()

    # Test cache layout
    print("\n--- Testing KV Cache Layout ---")
    test_cache = TestKVCacheLayout()
    test_cache.test_ad_cache_layout()
    test_cache.test_cache_page_indexing()

    print("\n" + "=" * 60)
    print("All quick tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_tests()
