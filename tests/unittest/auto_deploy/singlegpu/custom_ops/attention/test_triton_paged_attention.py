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
from _torch_test_utils import fp8_compatible

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

        # Compare decode
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

        # Compare prefill
        torch.testing.assert_close(output_triton.float(), output_fi.float(), rtol=1e-2, atol=1e-2)


class TestSDPADispatch:
    """Tests for the adaptive SDPA dispatch path in triton_paged_context.

    Covers: large head_dim forcing SDPA, variable-length sequence fallback
    to paged kernel, FP8 KV cache dtype casting, and oversized kv_indices buffers.
    """

    @staticmethod
    def _run_context_and_reference(
        seq_lens: list[int],
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        kv_cache_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run triton_paged_context and a PyTorch SDPA reference, return both outputs.

        Supports variable-length sequences within a batch.
        """
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        total_tokens = sum(seq_lens)
        q_dtype = torch.float16

        q = torch.randn(total_tokens, n_heads, head_dim, dtype=q_dtype, device="cuda")
        k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=q_dtype, device="cuda")
        v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=q_dtype, device="cuda")

        # Build per-sequence metadata
        qo_indptr_list = [0]
        kv_indptr_list = [0]
        kv_last_page_len_list = []
        all_page_indices = []
        page_counter = 0

        for sl in seq_lens:
            qo_indptr_list.append(qo_indptr_list[-1] + sl)
            n_pages = (sl + page_size - 1) // page_size
            kv_indptr_list.append(kv_indptr_list[-1] + n_pages)
            all_page_indices.extend(range(page_counter, page_counter + n_pages))
            page_counter += n_pages
            last = sl % page_size
            kv_last_page_len_list.append(last if last > 0 else page_size)

        num_blocks = page_counter + 5
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device="cuda")
        kv_indices = torch.tensor(all_page_indices, dtype=torch.int32, device="cuda")
        kv_last_page_len = torch.tensor(kv_last_page_len_list, dtype=torch.int32, device="cuda")
        seq_len_with_cache = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

        # Build batch_indices / positions for cache update
        batch_indices = torch.cat(
            [
                torch.full((sl,), i, dtype=torch.int32, device="cuda")
                for i, sl in enumerate(seq_lens)
            ]
        )
        positions = torch.cat(
            [torch.arange(sl, dtype=torch.int32, device="cuda") for sl in seq_lens]
        )

        cache_dtype = kv_cache_dtype if kv_cache_dtype is not None else q_dtype
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=cache_dtype, device="cuda"
        )
        if cache_dtype == q_dtype:
            update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)
        else:
            # For FP8: write in compute dtype then cast the whole cache
            kv_cache_tmp = torch.zeros_like(kv_cache, dtype=q_dtype)
            update_paged_kv_cache(
                k, v, batch_indices, positions, kv_cache_tmp, kv_indices, kv_indptr
            )
            kv_cache.copy_(kv_cache_tmp)

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

        # PyTorch reference: per-sequence causal SDPA, then concatenate
        head_ratio = n_heads // n_kv_heads
        ref_parts = []
        offset = 0
        for sl in seq_lens:
            q_s = q[offset : offset + sl].unsqueeze(0).transpose(1, 2)
            k_s = k[offset : offset + sl].unsqueeze(0).transpose(1, 2)
            v_s = v[offset : offset + sl].unsqueeze(0).transpose(1, 2)
            if head_ratio > 1:
                k_s = k_s.repeat_interleave(head_ratio, dim=1)
                v_s = v_s.repeat_interleave(head_ratio, dim=1)
            o_s = torch.nn.functional.scaled_dot_product_attention(
                q_s, k_s, v_s, scale=sm_scale, is_causal=True
            )
            ref_parts.append(o_s.transpose(1, 2).reshape(sl, n_heads, head_dim))
            offset += sl
        output_ref = torch.cat(ref_parts, dim=0)

        return output, output_ref

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(4, 4), (8, 2)])
    @pytest.mark.parametrize("seq_len", [64, 512])
    def test_large_head_dim_forces_sdpa(
        self, batch_size: int, n_heads: int, n_kv_heads: int, seq_len: int
    ):
        """head_dim > 256 forces the SDPA path regardless of seq_len.

        Regression test for Blackwell tl.dot misaligned shared memory accesses.
        """
        head_dim = 512
        seq_lens = [seq_len] * batch_size

        output, output_ref = self._run_context_and_reference(
            seq_lens,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size=16,
        )

        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "seq_lens",
        [
            [924, 910, 923],
            [512, 600],
            [1024, 900, 1024],
        ],
    )
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(32, 8)])
    @pytest.mark.parametrize("head_dim", [128])
    def test_variable_length_sequences_no_crash(
        self, seq_lens: list[int], n_heads: int, n_kv_heads: int, head_dim: int
    ):
        """Variable-length prefill sequences must not crash.

        Regression test: the SDPA path does q.view(num_seq, max_q_len, ...) which
        fails when sequences have different lengths. The fix ensures fallback to the
        paged Triton kernel for non-uniform q_lens.
        """
        output, output_ref = self._run_context_and_reference(
            seq_lens,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size=64,
        )

        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    def test_uniform_sequences_still_use_sdpa_path(self):
        """Uniform-length sequences >= 512 should still hit the SDPA path.

        Ensures the all_same_q_len guard doesn't over-restrict: equal-length
        sequences that previously used SDPA must continue to do so.
        """
        from unittest.mock import patch

        seq_lens = [512, 512]
        sdpa_called = False
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def tracking_sdpa(*args, **kwargs):
            nonlocal sdpa_called
            sdpa_called = True
            return original_sdpa(*args, **kwargs)

        with patch.object(torch.nn.functional, "scaled_dot_product_attention", tracking_sdpa):
            output, output_ref = self._run_context_and_reference(
                seq_lens,
                n_heads=8,
                n_kv_heads=8,
                head_dim=128,
                page_size=16,
            )

        assert sdpa_called, "SDPA path was not taken for uniform 512-token sequences"
        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    def test_oversized_kv_indices_buffer(self):
        """kv_indices buffer larger than actual page count should still work.

        Tests the pages_uniform fallback via kv_indptr when kv_indices is pre-allocated.
        """
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        batch_size, seq_len = 2, 512
        n_heads, n_kv_heads, head_dim, page_size = 8, 8, 128, 16
        total_tokens = batch_size * seq_len
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 10

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

        # Actual pages needed
        actual_indices = torch.arange(
            0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda"
        )
        # Over-allocate: pad kv_indices with extra zeros
        kv_indices = torch.zeros(
            batch_size * num_pages_per_seq + 100, dtype=torch.int32, device="cuda"
        )
        kv_indices[: actual_indices.shape[0]] = actual_indices

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
        )

        # Reference
        q_ref = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k_ref = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        v_ref = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        output_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            scale=sm_scale,
            is_causal=True,
            enable_gqa=True,
        )
        output_ref = output_ref.transpose(1, 2).reshape(total_tokens, n_heads, head_dim)

        torch.testing.assert_close(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [512, 1024])
    def test_fp8_kv_cache_dtype_casting(self, batch_size: int, seq_len: int):
        """FP8 KV cache values are correctly cast to query dtype in both paths."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
            update_paged_kv_cache,
        )

        n_heads, n_kv_heads, head_dim, page_size = 8, 8, 128, 16
        total_tokens = batch_size * seq_len
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        num_blocks = batch_size * num_pages_per_seq + 5

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

        # Write fp16 into cache, then cast to fp8
        kv_cache_fp16 = create_paged_kv_cache(num_blocks, page_size, n_kv_heads, head_dim)
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache_fp16, kv_indices, kv_indptr)
        kv_cache_fp8 = kv_cache_fp16.to(torch.float8_e4m3fn)

        sm_scale = 1.0 / math.sqrt(head_dim)

        output = triton_paged_context(
            q,
            kv_cache_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
        )

        assert output.dtype == q.dtype, f"Output dtype {output.dtype} != query dtype {q.dtype}"
        assert not torch.isnan(output).any(), "Output contains NaN with FP8 KV cache"
        assert not torch.isinf(output).any(), "Output contains Inf with FP8 KV cache"
