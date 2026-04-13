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
    """Tests for the FlashDecoding paged decode kernel (stage1 + stage2)."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [64, 256, 512])
    def test_decode_kernel_vs_pytorch_reference(
        self, batch_size: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int
    ):
        """Test decode kernel against PyTorch SDPA reference."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

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
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 512])
    def test_context_kernel_vs_pytorch_reference(
        self, batch_size: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int
    ):
        """Test context kernel against PyTorch SDPA reference."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

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

        assert output.shape == q.shape

        # PyTorch SDPA reference (causal)
        q_ref = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k_ref = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        v_ref = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

        head_ratio = n_heads // n_kv_heads
        if head_ratio > 1:
            k_ref = k_ref.repeat_interleave(head_ratio, dim=1)
            v_ref = v_ref.repeat_interleave(head_ratio, dim=1)

        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=sm_scale, is_causal=True
        )
        output_ref = output_ref.transpose(1, 2).reshape(total_tokens, n_heads, head_dim)

        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)


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


class TestTritonPagedMHAIntegration:
    """Integration tests for triton_paged_mha_with_cache and prepare_triton_paged_metadata.

    These test the full integration layer including BatchInfo parsing,
    metadata preparation, KV cache update, and mixed prefill/decode dispatch.
    This would have caught the batch_info_host 12-element format change.
    """

    @staticmethod
    def _make_batch_info(
        num_prefill: int,
        num_prefill_tokens: int,
        num_decode: int,
        max_context_length: int = 8192,
        max_blocks_per_seq: int = 256,
        max_batch_size: int = 8,
    ) -> torch.Tensor:
        """Create a 12-element batch_info_host tensor."""
        bi = torch.zeros(12, dtype=torch.int, pin_memory=True)
        bi[0] = num_prefill
        bi[1] = num_prefill_tokens
        bi[2] = 0  # num_extend
        bi[3] = 0  # num_extend_tokens
        bi[4] = num_decode
        bi[5] = num_decode  # num_decode_tokens = num_decode (1 token each)
        bi[6] = max_context_length
        bi[7] = max_blocks_per_seq
        bi[8] = 1  # block_offset_multiplier
        bi[9] = max_batch_size
        bi[10] = 0  # num_tokens_to_gather
        bi[11] = 0  # gather_required
        return bi

    def test_batch_info_12_element_format(self):
        """Test that triton_paged_mha_with_cache handles 12-element batch_info_host.

        Regression test: batch_info_host changed from 3-element to 12-element tensor.
        The old code did `num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()`
        which would crash with ValueError: too many values to unpack.
        """
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_mha_with_cache,
        )

        n_heads, n_kv_heads, head_dim, page_size = 32, 8, 128, 16
        seq_len = 32
        num_pages = (seq_len + page_size - 1) // page_size
        num_blocks = num_pages + 16

        # Prefill-only batch: 1 sequence, 32 tokens
        q = torch.randn(1, seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(1, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(1, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")

        batch_info_host = self._make_batch_info(
            num_prefill=1, num_prefill_tokens=seq_len, num_decode=0
        )
        cu_seqlen_host = torch.tensor([0, seq_len], dtype=torch.int32)
        cu_num_pages = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
        cu_num_pages_host = cu_num_pages.cpu()
        cache_loc = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        last_page_len = torch.tensor(
            [seq_len % page_size or page_size], dtype=torch.int32, device="cuda"
        )
        last_page_len_host = last_page_len.cpu()
        seq_len_with_cache_host = torch.tensor([seq_len], dtype=torch.int32)
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )

        # Prepare metadata (this also uses batch_info_host)
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            prepare_triton_paged_metadata,
        )

        position_ids = torch.arange(seq_len, device="cuda")
        batch_indices, positions = prepare_triton_paged_metadata(
            position_ids,
            batch_info_host,
            cu_seqlen_host.to("cuda", non_blocking=True),
            seq_len_with_cache_host.to("cuda", non_blocking=True),
        )

        # Run the full MHA with cache (should not crash)
        output = triton_paged_mha_with_cache(
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
            batch_indices,
            positions,
            kv_cache,
            scale=None,
        )

        assert output.shape == q.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_batch_info_with_extend_requests(self):
        """Test that extend requests are absorbed into prefill counts."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

        bi = torch.zeros(12, dtype=torch.int, pin_memory=True)
        bi[0] = 1  # num_prefill
        bi[1] = 32  # num_prefill_tokens
        bi[2] = 2  # num_extend
        bi[3] = 64  # num_extend_tokens
        bi[4] = 3  # num_decode

        batch_info = BatchInfo(bi)
        num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()

        # Extend should be absorbed into prefill
        assert num_prefill == 3  # 1 + 2
        assert num_prefill_tokens == 96  # 32 + 64
        assert num_decode == 3

    def test_prepare_metadata_with_12_element_batch_info(self):
        """Test prepare_triton_paged_metadata with 12-element batch_info_host."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            prepare_triton_paged_metadata,
        )

        batch_info_host = self._make_batch_info(num_prefill=1, num_prefill_tokens=7, num_decode=0)
        position_ids = torch.arange(7, device="cuda")
        cu_seqlen = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
        seq_len_with_cache = torch.tensor([7], dtype=torch.int32, device="cuda")

        # Should not raise ValueError
        batch_indices, positions = prepare_triton_paged_metadata(
            position_ids, batch_info_host, cu_seqlen, seq_len_with_cache
        )

        assert batch_indices.shape[0] == 7
        assert positions.shape[0] == 7
        assert (batch_indices == 0).all()
        assert (positions == torch.arange(7, device="cuda")).all()


class TestSlidingWindow:
    """Tests for sliding window attention support in Triton paged kernels."""

    @staticmethod
    def _sliding_window_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
    ) -> torch.Tensor:
        """Compute causal + sliding window attention with manual masking.

        Args:
            q: [B, n_heads, S_q, head_dim]
            k: [B, n_heads, S_k, head_dim]
            v: [B, n_heads, S_k, head_dim]

        Returns:
            [B, n_heads, S_q, head_dim]
        """
        s_q = q.shape[2]
        s_k = k.shape[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

        q_pos = torch.arange(s_k - s_q + s_q, device=q.device)  # absolute positions
        # For prefill: q_pos = [0..s_q-1], k_pos = [0..s_k-1]
        q_pos = torch.arange(s_k - s_q, s_k, device=q.device)  # [s_q]
        k_pos = torch.arange(s_k, device=q.device)  # [s_k]

        pos_diff = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)  # [s_q, s_k]
        causal_mask = pos_diff < 0
        window_mask = pos_diff >= sliding_window
        combined = causal_mask | window_mask
        attn.masked_fill_(combined.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256, 512])
    @pytest.mark.parametrize("sliding_window", [32, 64])
    def test_decode_sliding_window(
        self,
        batch_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        seq_len: int,
        sliding_window: int,
    ):
        """Test decode with sliding window against reference (seq_len > window)."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        assert seq_len > sliding_window, "Test requires seq_len > sliding_window"
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)

        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

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

        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(
            k_flat, v_flat, batch_indices, positions, kv_cache, kv_indices, kv_indptr
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        output_triton = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            sliding_window=sliding_window,
        )

        # Reference: only attend to last `sliding_window` tokens
        head_ratio = n_heads // n_kv_heads
        k_ref = k[:, -sliding_window:, :, :].transpose(1, 2)
        v_ref = v[:, -sliding_window:, :, :].transpose(1, 2)
        if head_ratio > 1:
            k_ref = k_ref.repeat_interleave(head_ratio, dim=1)
            v_ref = v_ref.repeat_interleave(head_ratio, dim=1)

        q_ref = q.unsqueeze(2)  # [B, n_heads, 1, head_dim]
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=sm_scale, is_causal=False
        ).squeeze(2)

        torch.testing.assert_close(output_triton.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [128, 256])
    @pytest.mark.parametrize("sliding_window", [32, 64])
    def test_context_sliding_window(
        self,
        batch_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        seq_len: int,
        sliding_window: int,
    ):
        """Test prefill with sliding window against manual reference (seq_len > window)."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        assert seq_len > sliding_window, "Test requires seq_len > sliding_window"
        page_size = 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5
        total_tokens = batch_size * seq_len

        q = torch.randn(total_tokens, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")

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

        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        sm_scale = 1.0 / math.sqrt(head_dim)

        output = triton_paged_context(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
            sliding_window=sliding_window,
        )

        # Reference: manual causal + sliding window attention
        head_ratio = n_heads // n_kv_heads
        q_ref = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k_ref = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        v_ref = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        if head_ratio > 1:
            k_ref = k_ref.repeat_interleave(head_ratio, dim=1)
            v_ref = v_ref.repeat_interleave(head_ratio, dim=1)

        output_ref = self._sliding_window_reference(q_ref, k_ref, v_ref, sm_scale, sliding_window)
        output_ref = output_ref.transpose(1, 2).reshape(total_tokens, n_heads, head_dim)

        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    def test_no_sliding_window_unchanged(self):
        """Verify that sliding_window=None produces the same output as before."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        batch_size, n_heads, n_kv_heads, head_dim = 2, 8, 8, 64
        seq_len, page_size = 128, 16

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)

        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )

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

        kv_cache = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(
            k_flat, v_flat, batch_indices, positions, kv_cache, kv_indices, kv_indptr
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        out_none = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            sliding_window=None,
        )
        out_zero = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            sliding_window=0,
        )

        torch.testing.assert_close(out_none, out_zero)


class TestFlashInferComparison:
    """Tests comparing Triton implementation against FlashInfer."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [64, 128, 512])
    def test_decode_vs_flashinfer(
        self, batch_size: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int
    ):
        """Compare decode output against FlashInfer."""
        import flashinfer

        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

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
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(8, 8), (32, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("seq_len", [64, 128, 512])
    def test_prefill_vs_flashinfer(
        self, batch_size: int, n_heads: int, n_kv_heads: int, head_dim: int, seq_len: int
    ):
        """Compare prefill output against FlashInfer."""
        import flashinfer

        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

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


def _reference_attention_with_logit_cap(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch reference: scaled dot-product attention with logit soft-capping.

    Args:
        q: [B, H, Sq, D]
        k: [B, H, Sk, D]
        v: [B, H, Sk, D]

    Returns:
        output: [B, H, Sq, D]
    """
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    # Apply logit soft-capping BEFORE masking (matches triton kernel behavior)
    scores = logit_cap * torch.tanh(scores / logit_cap)
    if causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float()).to(q.dtype)


class TestLogitSoftCap:
    """Tests for logit soft-capping support (Gemma-4: logit_cap=50.0)."""

    LOGIT_CAP = 50.0

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("head_dim", [64, 128, 176])
    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_decode_logit_cap(self, batch_size: int, head_dim: int, seq_len: int):
        """Decode kernel with logit_cap produces correct output vs PyTorch reference."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        n_heads, n_kv_heads, page_size = 16, 8, 16
        sm_scale = 1.0 / math.sqrt(head_dim)

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size, seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
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
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )
        update_paged_kv_cache(
            k_flat, v_flat, batch_indices, positions, kv_cache, kv_indices, kv_indptr
        )

        output_triton = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            logit_cap=self.LOGIT_CAP,
        )

        # PyTorch reference
        q_ref = q.unsqueeze(2)  # [B, H, 1, D]
        k_ref = k.transpose(1, 2)  # [B, Hkv, S, D]
        v_ref = v.transpose(1, 2)
        head_ratio = n_heads // n_kv_heads
        k_ref = k_ref.repeat_interleave(head_ratio, dim=1)
        v_ref = v_ref.repeat_interleave(head_ratio, dim=1)
        output_ref = _reference_attention_with_logit_cap(
            q_ref, k_ref, v_ref, sm_scale, self.LOGIT_CAP, causal=False
        ).squeeze(2)

        torch.testing.assert_close(output_triton.float(), output_ref.float(), rtol=1e-2, atol=2e-2)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("head_dim", [64, 128, 176])
    @pytest.mark.parametrize("seq_len", [32, 128, 256])
    def test_context_logit_cap(self, batch_size: int, head_dim: int, seq_len: int):
        """Context kernel with logit_cap produces correct output vs PyTorch reference."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        n_heads, n_kv_heads, page_size = 16, 8, 16
        sm_scale = 1.0 / math.sqrt(head_dim)

        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        q = torch.randn(batch_size * seq_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(
            batch_size * seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size * seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )

        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
        )
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
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
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        output_triton = triton_paged_context(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
            logit_cap=self.LOGIT_CAP,
        )

        # PyTorch reference: per-sequence causal attention with logit_cap
        q_4d = q.reshape(batch_size, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        k_4d = k.reshape(batch_size, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        v_4d = v.reshape(batch_size, seq_len, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        head_ratio = n_heads // n_kv_heads
        k_4d = k_4d.repeat_interleave(head_ratio, dim=1)
        v_4d = v_4d.repeat_interleave(head_ratio, dim=1)
        output_ref_4d = _reference_attention_with_logit_cap(
            q_4d, k_4d, v_4d, sm_scale, self.LOGIT_CAP, causal=True
        )
        output_ref = output_ref_4d.permute(0, 2, 1, 3).reshape(
            batch_size * seq_len, n_heads, head_dim
        )

        torch.testing.assert_close(output_triton.float(), output_ref.float(), rtol=1e-2, atol=2e-2)

    def test_decode_no_logit_cap_unchanged(self):
        """Verify that logit_cap=None produces same result as omitting logit_cap arg."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        batch_size, n_heads, n_kv_heads, head_dim, page_size, seq_len = 2, 8, 8, 64, 16, 128
        sm_scale = 1.0 / math.sqrt(head_dim)
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(
            batch_size * seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch_size * seq_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * num_pages_per_seq,
            num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
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
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
        )
        positions = torch.tile(
            torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,)
        )
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        out_no_cap = triton_paged_decode(
            q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale
        )
        out_cap_none = triton_paged_decode(
            q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale, logit_cap=None
        )
        torch.testing.assert_close(out_no_cap, out_cap_none)


class TestInactiveSplits:
    """Tests for the inactive split path in the decode kernel.

    Inactive splits occur when a batch element has fewer pages than the total
    number of splits determined by the longest sequence in the batch.  Those
    extra splits must write zeros / -inf LSE and return early, leaving the
    output untouched for elements with short sequences.

    The WRITE_DIRECT path (num_splits==1) is exercised by including a batch
    element with an empty KV cache (num_pages=0), which triggers
    ``page_split_start >= num_pages`` even for split_id=0.
    """

    def test_inactive_split_correctness(self):
        """Mixed-length batch: long seqs force num_splits>1; short seqs have inactive splits."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        page_size = 16
        n_heads, n_kv_heads, head_dim = 8, 2, 64
        sm_scale = 1.0 / math.sqrt(head_dim)

        # Two long seqs (many pages → num_splits > 1) and one short seq (1 page → inactive splits)
        seq_lens = [256, 256, 16]
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens)
        num_pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens]
        total_pages = sum(num_pages_per_seq)

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")

        kv_cache = torch.zeros(
            total_pages, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )

        # Build per-sequence page tables
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i, np in enumerate(num_pages_per_seq):
            kv_indptr[i + 1] = kv_indptr[i] + np
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [s % page_size if s % page_size != 0 else page_size for s in seq_lens],
            dtype=torch.int32,
            device="cuda",
        )

        # Populate KV cache for each sequence
        for b, sl in enumerate(seq_lens):
            k = torch.randn(sl, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            v = torch.randn(sl, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            batch_indices = torch.full((sl,), b, dtype=torch.int32, device="cuda")
            positions = torch.arange(sl, dtype=torch.int32, device="cuda")
            update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        # Run mixed-length batch decode (num_splits>1 due to long seqs; short seq has inactive splits)
        out = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            max_seq_len=max_seq_len,
        )

        # Independently compute expected output for each sequence
        for b, sl in enumerate(seq_lens):
            q_b = q[b : b + 1]  # [1, n_heads, head_dim]
            kv_indptr_b = torch.tensor([0, num_pages_per_seq[b]], dtype=torch.int32, device="cuda")
            kv_indices_b = kv_indices[kv_indptr[b] : kv_indptr[b + 1]]
            kv_last_b = kv_last_page_len[b : b + 1]
            out_ref = triton_paged_decode(
                q_b,
                kv_cache,
                kv_indices_b,
                kv_indptr_b,
                kv_last_b,
                sm_scale,
                max_seq_len=sl,
            )
            torch.testing.assert_close(
                out[b],
                out_ref[0],
                atol=1e-2,
                rtol=1e-2,
                msg=f"Output mismatch for batch element {b} (seq_len={sl})",
            )

    def test_write_direct_inactive_split_with_empty_sequence(self):
        """WRITE_DIRECT path with empty sequence.

        A batch element with empty KV cache (num_pages=0) hits
        page_split_start >= num_pages even for split_id=0. Its output slot
        must be zeroed rather than containing uninitialized memory.
        """
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_decode,
            update_paged_kv_cache,
        )

        page_size = 16
        n_heads, n_kv_heads, head_dim = 4, 4, 64
        sm_scale = 1.0 / math.sqrt(head_dim)

        # batch[0]: normal seq, batch[1]: empty (0 pages → triggers inactive split with WRITE_DIRECT)
        seq_lens = [16, 0]
        batch_size = len(seq_lens)
        num_pages_per_seq = [max((s + page_size - 1) // page_size, 0) for s in seq_lens]
        total_pages = max(sum(num_pages_per_seq), 1)  # at least 1 block allocated

        q = torch.randn(batch_size, n_heads, head_dim, dtype=torch.float16, device="cuda")
        kv_cache = torch.zeros(
            total_pages, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )

        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i, np_count in enumerate(num_pages_per_seq):
            kv_indptr[i + 1] = kv_indptr[i] + np_count
        kv_indices = torch.arange(max(total_pages, 1), dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(
            [page_size if sl == 0 else (sl % page_size or page_size) for sl in seq_lens],
            dtype=torch.int32,
            device="cuda",
        )

        # Populate cache for non-empty sequences only
        if seq_lens[0] > 0:
            k = torch.randn(seq_lens[0], n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            v = torch.randn(seq_lens[0], n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            batch_indices = torch.zeros(seq_lens[0], dtype=torch.int32, device="cuda")
            positions = torch.arange(seq_lens[0], dtype=torch.int32, device="cuda")
            update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

        # Should not raise; the empty-sequence slot must produce finite (zero) output
        out = triton_paged_decode(
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            sm_scale,
            max_seq_len=seq_lens[0] if seq_lens[0] > 0 else 1,
        )
        assert torch.isfinite(out).all(), "Output contains non-finite values for empty sequence"
        # Empty-sequence output should be all-zero (written by inactive-split zero-store)
        torch.testing.assert_close(
            out[1], torch.zeros_like(out[1]), msg="Empty-sequence output should be zero"
        )
