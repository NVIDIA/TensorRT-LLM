# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Triton Paged Attention.

Tests the Triton paged attention kernels and compares against FlashInfer for correctness.
"""

import math

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def create_paged_kv_cache(
    num_blocks: int,
    page_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> torch.Tensor:
    """Create an empty paged KV cache with HND layout.

    Shape: [num_blocks, 2, n_kv_heads, page_size, head_dim]
    """
    return torch.zeros(num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=dtype, device=device)


def create_page_table(
    batch_size: int,
    max_pages_per_seq: int,
    num_blocks: int,
    device: str = "cuda",
) -> tuple:
    """Create page table metadata.

    Returns:
        kv_indices: Flattened page indices
        kv_indptr: Cumulative page counts [batch_size + 1]
        kv_last_page_len: Valid tokens in last page [batch_size]
    """
    # Assign sequential pages to each sequence
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    all_indices = []

    for i in range(batch_size):
        num_pages = min(max_pages_per_seq, num_blocks - sum(kv_indptr[: i + 1].tolist()))
        pages = list(range(int(kv_indptr[i].item()), int(kv_indptr[i].item()) + num_pages))
        all_indices.extend(pages)
        kv_indptr[i + 1] = kv_indptr[i] + num_pages

    kv_indices = torch.tensor(all_indices, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    return kv_indices, kv_indptr, kv_last_page_len


class TestTritonPagedDecodeKernel:
    """Tests for the single-stage paged decode kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [64, 256, 512])
    def test_decode_kernel_basic(
        self, batch_size: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int
    ):
        """Test decode kernel produces valid output shapes and no NaNs."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
        )

        page_size = 16
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 10

        # Create inputs
        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)

        # Fill cache with random data
        kv_cache.normal_()

        # Create page table
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )[: batch_size + 1]
        kv_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        kv_last_page_len = torch.full(
            (batch_size,), seq_len % page_size or page_size, dtype=torch.int32, device="cuda"
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        # Run kernel
        output = triton_paged_decode(q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale)

        # Check output
        assert output.shape == q.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_decode_kernel_vs_pytorch_reference(self):
        """Test decode kernel against PyTorch SDPA reference."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        batch_size = 2
        n_heads = 8
        n_kv_heads = 8
        head_dim = 64
        seq_len = 32
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        # Create Q for decode (single token per sequence)
        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")

        # Create K, V for the full sequence
        k = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Flatten for cache update
        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)

        # Create metadata for cache update
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

        # Create page table
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )[: batch_size + 1]
        kv_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        last_token_in_page = seq_len % page_size
        kv_last_page_len = torch.full(
            (batch_size,),
            last_token_in_page if last_token_in_page > 0 else page_size,
            dtype=torch.int32,
            device="cuda",
        )

        # Create and fill cache
        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(
            k_flat, v_flat, batch_indices, positions, kv_cache, kv_indices, kv_indptr
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        # Run Triton kernel
        output_triton = triton_paged_decode(
            q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale
        )

        # Compute PyTorch reference
        # Q: [B, n_heads, head_dim] -> [B, n_heads, 1, head_dim]
        # K: [B, seq_len, n_kv_heads, head_dim] -> [B, n_kv_heads, seq_len, head_dim]
        # V: [B, seq_len, n_kv_heads, head_dim] -> [B, n_kv_heads, seq_len, head_dim]
        q_ref = q.unsqueeze(2)  # [B, n_heads, 1, head_dim]
        k_ref = k.transpose(1, 2)  # [B, n_kv_heads, seq_len, head_dim]
        v_ref = v.transpose(1, 2)  # [B, n_kv_heads, seq_len, head_dim]

        # Handle GQA by expanding K, V
        head_ratio = n_heads // n_kv_heads
        if head_ratio > 1:
            k_ref = k_ref.repeat_interleave(head_ratio, dim=1)
            v_ref = v_ref.repeat_interleave(head_ratio, dim=1)

        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=sm_scale, is_causal=False
        )
        output_ref = output_ref.squeeze(2)  # [B, n_heads, head_dim]

        # Compare
        torch.testing.assert_close(output_triton.float(), output_ref.float(), rtol=1e-2, atol=1e-2)


class TestTritonPagedContextKernel:
    """Tests for the context/prefill kernel."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_context_kernel_basic(self, batch_size: int, seq_len: int):
        """Test context kernel produces valid output."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        n_heads = 8
        n_kv_heads = 8
        head_dim = 64
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5
        total_tokens = batch_size * seq_len

        # Create inputs (flattened)
        q = torch.randn(total_tokens, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")

        # Create metadata
        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
        )[: batch_size + 1]
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )[: batch_size + 1]
        kv_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        last_token_in_page = seq_len % page_size
        kv_last_page_len = torch.full(
            (batch_size,),
            last_token_in_page if last_token_in_page > 0 else page_size,
            dtype=torch.int32,
            device="cuda",
        )
        seq_len_with_cache = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

        # Create batch_indices and positions for cache update
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

        # Create and fill cache
        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        sm_scale = 1.0 / math.sqrt(head_dim)

        # Run kernel
        output = triton_paged_context(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
        )

        # Check output
        assert output.shape == q.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestCacheUpdate:
    """Tests for the KV cache update kernel."""

    def test_cache_update_writes_correct_values(self):
        """Test that cache update writes K, V to correct locations."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            update_paged_kv_cache,
        )

        batch_size = 2
        seq_len = 8
        n_kv_heads = 4
        head_dim = 32
        page_size = 4
        num_blocks = 10

        # Create K, V with known values
        k = torch.arange(
            batch_size * seq_len * n_kv_heads * head_dim, dtype=torch.float16, device="cuda"
        ).reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v = k + 1000  # Offset to distinguish K from V

        # Create metadata
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

        kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

        # Create empty cache
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )

        # Run update
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        # Verify: check first token of sequence 0
        # Token 0 should be at page 0, offset 0
        expected_k = k[0]  # First token's K
        actual_k = kv_cache[0, 0, :, 0, :]  # Page 0, K (idx 0), all heads, offset 0
        torch.testing.assert_close(actual_k, expected_k)

        expected_v = v[0]  # First token's V
        actual_v = kv_cache[0, 1, :, 0, :]  # Page 0, V (idx 1), all heads, offset 0
        torch.testing.assert_close(actual_v, expected_v)


class TestFlashInferComparison:
    """Tests comparing Triton implementation against FlashInfer."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [64, 128])
    def test_decode_vs_flashinfer(self, batch_size: int, seq_len: int):
        """Compare decode output against FlashInfer."""
        import flashinfer

        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        n_heads = 32
        n_kv_heads = 8
        head_dim = 128
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 10

        # Create shared K, V data
        k = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        # Query for decode
        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")

        # Page table metadata
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )[: batch_size + 1]
        kv_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        last_token_in_page = seq_len % page_size
        kv_last_page_len = torch.full(
            (batch_size,),
            last_token_in_page if last_token_in_page > 0 else page_size,
            dtype=torch.int32,
            device="cuda",
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        # ===== Triton =====
        kv_cache_triton = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )
        update_paged_kv_cache(
            k_flat, v_flat, batch_indices, positions, kv_cache_triton, kv_indices, kv_indptr
        )
        output_triton = triton_paged_decode(
            q, kv_cache_triton, kv_indices, kv_indptr, kv_last_page_len, sm_scale
        )

        # ===== FlashInfer =====
        kv_cache_fi = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        # Use FlashInfer's cache append
        fi_batch_indices = batch_indices.clone()
        fi_positions = positions.clone()
        flashinfer.page.append_paged_kv_cache(
            append_key=k_flat,
            append_value=v_flat,
            batch_indices=fi_batch_indices,
            positions=fi_positions,
            paged_kv_cache=kv_cache_fi,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout="HND",
        )

        # Use FlashInfer decode
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "HND", use_tensor_cores=True
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size,
            q_data_type=q.dtype,
            kv_data_type=kv_cache_fi.dtype,
            sm_scale=sm_scale,
        )
        output_fi = wrapper.run(q, kv_cache_fi)

        # Compare
        torch.testing.assert_close(output_triton.float(), output_fi.float(), rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(
        not pytest.importorskip("flashinfer", reason="FlashInfer not installed"),
        reason="FlashInfer not installed",
    )
    def test_prefill_vs_flashinfer(self):
        """Compare prefill output against FlashInfer."""
        import flashinfer

        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        batch_size = 2
        seq_len = 64
        n_heads = 32
        n_kv_heads = 8
        head_dim = 128
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 10
        total_tokens = batch_size * seq_len

        # Create inputs
        q = torch.randn(total_tokens, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")

        # Metadata
        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
        )[: batch_size + 1]
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )[: batch_size + 1]
        kv_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        last_token_in_page = seq_len % page_size
        kv_last_page_len = torch.full(
            (batch_size,),
            last_token_in_page if last_token_in_page > 0 else page_size,
            dtype=torch.int32,
            device="cuda",
        )
        seq_len_with_cache = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        # ===== Triton =====
        kv_cache_triton = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(
            k, v, batch_indices, positions, kv_cache_triton, kv_indices, kv_indptr
        )
        output_triton = triton_paged_context(
            q,
            kv_cache_triton,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
        )

        # ===== FlashInfer =====
        kv_cache_fi = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        flashinfer.page.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices.clone(),
            positions=positions.clone(),
            paged_kv_cache=kv_cache_fi,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout="HND",
        )

        workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")
        wrapper.plan(
            qo_indptr.cpu(),
            kv_indptr.cpu(),
            kv_indices,
            kv_last_page_len.cpu(),
            n_heads,
            n_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=kv_cache_fi.dtype,
            sm_scale=sm_scale,
            seq_lens=seq_len_with_cache.cpu(),
        )
        output_fi = wrapper.run(q, kv_cache_fi)

        # Compare
        torch.testing.assert_close(output_triton.float(), output_fi.float(), rtol=1e-2, atol=1e-2)
