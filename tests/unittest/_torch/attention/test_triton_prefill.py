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
"""Tests for Triton prefill attention kernel with custom mask and paged KV cache."""

import math

import pytest
import torch
import torch.nn.functional as F


def _import_triton_prefill():
    """Import triton_prefill module directly to avoid TRT-LLM C++ bindings."""
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "tensorrt_llm",
        "_torch",
        "attention_backend",
        "triton_prefill.py",
    )
    spec = importlib.util.spec_from_file_location("triton_prefill", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


triton_prefill = _import_triton_prefill()
triton_prefill_with_custom_mask = triton_prefill.triton_prefill_with_custom_mask


def _sdpa_reference(q, k, v, mask=None, sm_scale=None):
    """Reference attention using PyTorch SDPA.

    Args:
        q: [total_tokens, num_heads, head_dim]
        k: [total_tokens, num_kv_heads, head_dim]
        v: [total_tokens, num_kv_heads, head_dim]
        mask: list of [seq_len, seq_len] bool masks per sequence, or None
        sm_scale: softmax scale

    Returns:
        output: [total_tokens, num_heads, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    # Infer seq_lens from masks
    if mask is not None:
        seq_lens = [m.shape[0] for m in mask]
    else:
        seq_lens = [q.shape[0]]
    prefix_lens = [0] * len(seq_lens)
    return _sdpa_reference_with_prefix(
        q,
        k,
        v,
        prefix_k=None,
        prefix_v=None,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        mask=mask,
        sm_scale=sm_scale,
    )


def _sdpa_reference_with_prefix(
    q, k, v, prefix_k, prefix_v, seq_lens, prefix_lens, mask=None, sm_scale=None
):
    """Reference attention with optional prefix KV.

    Args:
        q: [total_extend_tokens, num_heads, head_dim]
        k: [total_extend_tokens, num_kv_heads, head_dim]
        v: [total_extend_tokens, num_kv_heads, head_dim]
        prefix_k: list of [prefix_len_i, num_kv_heads, head_dim] per seq, or None
        prefix_v: list of [prefix_len_i, num_kv_heads, head_dim] per seq, or None
        seq_lens: list of extend lengths per seq
        prefix_lens: list of prefix lengths per seq
        mask: list of per-seq masks, or None
        sm_scale: softmax scale
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    gqa_groups = num_heads // num_kv_heads

    outputs = []
    offset = 0

    if seq_lens is None:
        # Single sequence, no prefix
        seq_lens = [q.shape[0]]
        prefix_lens = [0]

    for i, (ext_len, pfx_len) in enumerate(zip(seq_lens, prefix_lens)):
        q_i = q[offset : offset + ext_len]  # [ext_len, H, D]
        k_i = k[offset : offset + ext_len]  # [ext_len, Hkv, D]
        v_i = v[offset : offset + ext_len]

        # Combine prefix + extend KV
        if pfx_len > 0 and prefix_k is not None:
            full_k = torch.cat([prefix_k[i], k_i], dim=0)  # [pfx+ext, Hkv, D]
            full_v = torch.cat([prefix_v[i], v_i], dim=0)
        else:
            full_k = k_i
            full_v = v_i

        # GQA expansion: [total_kv, Hkv, D] -> [total_kv, H, D]
        if gqa_groups > 1:
            full_k = full_k.repeat_interleave(gqa_groups, dim=1)
            full_v = full_v.repeat_interleave(gqa_groups, dim=1)

        # Compute attention: [ext_len, H, D] x [total_kv, H, D]
        # scores: [H, ext_len, total_kv]
        scores = torch.einsum("qhd,khd->hqk", q_i.float(), full_k.float()) * sm_scale

        # Apply mask
        if mask is not None and mask[i] is not None:
            mask_i = mask[i]  # [ext_len, total_kv] or [ext_len, ext_len]
            scores = scores.masked_fill(~mask_i.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out_i = torch.einsum("hqk,khd->qhd", attn_weights, full_v.float())
        outputs.append(out_i.to(q.dtype))
        offset += ext_len

    return torch.cat(outputs, dim=0)


def _make_causal_mask(seq_len):
    """Create a causal mask [seq_len, seq_len]."""
    return torch.arange(seq_len).unsqueeze(0) <= torch.arange(seq_len).unsqueeze(1)


def _make_bidirectional_mask(seq_len, image_start, image_end):
    """Create causal + bidirectional mask for image tokens."""
    mask = _make_causal_mask(seq_len)
    # Image tokens attend to each other bidirectionally
    mask[image_start:image_end, image_start:image_end] = True
    return mask


def _flatten_masks(masks):
    """Flatten and concatenate per-sequence masks."""
    return torch.cat([m.flatten() for m in masks], dim=0).contiguous()


# ===========================================================================
# Tests: No prefix, basic correctness
# ===========================================================================


class TestTritonPrefillNoPrefix:
    """Tests with prefix_lens=0 (only stage 2 runs)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def _run_no_prefix(self, q, k, v, custom_mask_tensor, sm_scale, device):
        """Run Triton kernel with no prefix."""
        num_tokens = q.shape[0]

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
        page_table_indptr = torch.zeros(2, dtype=torch.int32, device=device)
        page_table_indices = torch.zeros(1, dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=None,
            prefix_lens=prefix_lens,
            page_table_indptr=page_table_indptr,
            page_table_indices=page_table_indices,
            page_size=1,
            custom_mask=custom_mask_tensor,
            sm_scale=sm_scale,
        )
        return output

    def test_causal_single_seq_hd512(self, device):
        """head_dim=512, single sequence, causal mask."""
        torch.manual_seed(42)
        seq_len, num_heads, num_kv_heads, head_dim = 64, 8, 4, 512
        sm_scale = 1.0  # Gemma4 uses scaling=1.0

        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        mask = _make_causal_mask(seq_len).to(device)
        mask_flat = mask.flatten().contiguous()

        output = self._run_no_prefix(q, k, v, mask_flat, sm_scale, device)
        ref = _sdpa_reference(q, k, v, mask=[mask], sm_scale=sm_scale)

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_bidirectional_mask_hd512(self, device):
        """head_dim=512, bidirectional mask for image tokens."""
        torch.manual_seed(42)
        seq_len, num_heads, num_kv_heads, head_dim = 128, 8, 4, 512
        sm_scale = 1.0

        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Image tokens at positions 10-30 attend bidirectionally
        mask = _make_bidirectional_mask(seq_len, 10, 30).to(device)
        mask_flat = mask.flatten().contiguous()

        output = self._run_no_prefix(q, k, v, mask_flat, sm_scale, device)
        ref = _sdpa_reference(q, k, v, mask=[mask], sm_scale=sm_scale)

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_causal_no_custom_mask(self, device):
        """Causal attention with custom_mask=None (NVFP4 fallback path).

        When trtllm-gen context cubins don't support mixed Q/KV dtypes
        (BF16 Q + FP8 KV for NVFP4 models), the Triton prefill kernel is
        called with custom_mask=None and must generate a causal mask
        internally. Result must match explicit causal mask.
        """
        torch.manual_seed(42)
        seq_len, num_heads, num_kv_heads, head_dim = 64, 8, 4, 512
        sm_scale = 1.0

        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Run with None mask (auto-causal)
        out_none = self._run_no_prefix(q, k, v, None, sm_scale, device)

        # Run with explicit causal mask
        mask = _make_causal_mask(seq_len).to(device)
        out_explicit = self._run_no_prefix(q, k, v, mask.flatten().contiguous(), sm_scale, device)

        # Both should match
        torch.testing.assert_close(out_none, out_explicit, atol=1e-5, rtol=1e-5)

        # Also match SDPA reference
        ref = _sdpa_reference(q, k, v, mask=[mask], sm_scale=sm_scale)
        torch.testing.assert_close(out_none, ref, atol=1e-2, rtol=1e-2)

    def test_gqa_num_heads_8_kv_4(self, device):
        """GQA with num_heads=8, num_kv_heads=4 (Gemma4 config)."""
        torch.manual_seed(42)
        seq_len, num_heads, num_kv_heads, head_dim = 32, 8, 4, 512
        sm_scale = 1.0

        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        mask = _make_causal_mask(seq_len).to(device)
        mask_flat = mask.flatten().contiguous()

        output = self._run_no_prefix(q, k, v, mask_flat, sm_scale, device)
        ref = _sdpa_reference(q, k, v, mask=[mask], sm_scale=sm_scale)

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_variable_length_batch(self, device):
        """Multiple sequences with different lengths."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        seq_lens = [32, 64, 16]
        total = sum(seq_lens)

        q = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        masks = [_make_causal_mask(sl).to(device) for sl in seq_lens]
        mask_flat = _flatten_masks(masks)

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        num_ctx = len(seq_lens)
        prefix_lens = torch.zeros(num_ctx, dtype=torch.int32, device=device)
        page_table_indptr = torch.zeros(num_ctx + 1, dtype=torch.int32, device=device)
        page_table_indices = torch.zeros(1, dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=None,
            prefix_lens=prefix_lens,
            page_table_indptr=page_table_indptr,
            page_table_indices=page_table_indices,
            page_size=1,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        ref = _sdpa_reference(q, k, v, mask=masks, sm_scale=sm_scale)
        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_batch_bidirectional_mixed(self, device):
        """Batch with some seqs having image tokens, some text-only."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        seq_lens = [48, 64]
        total = sum(seq_lens)

        q = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(total, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Seq 0: pure causal, Seq 1: image tokens at 5-25
        masks = [
            _make_causal_mask(48).to(device),
            _make_bidirectional_mask(64, 5, 25).to(device),
        ]
        mask_flat = _flatten_masks(masks)

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor([0, 48, 112], dtype=torch.int32, device=device)
        num_ctx = 2
        prefix_lens = torch.zeros(num_ctx, dtype=torch.int32, device=device)
        page_table_indptr = torch.zeros(num_ctx + 1, dtype=torch.int32, device=device)
        page_table_indices = torch.zeros(1, dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=None,
            prefix_lens=prefix_lens,
            page_table_indptr=page_table_indptr,
            page_table_indices=page_table_indices,
            page_size=1,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        ref = _sdpa_reference(q, k, v, mask=masks, sm_scale=sm_scale)
        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_head_dim_256(self, device):
        """Also works for smaller head_dim (not just 512)."""
        torch.manual_seed(42)
        seq_len, num_heads, num_kv_heads, head_dim = 64, 8, 8, 256
        sm_scale = 1.0 / math.sqrt(head_dim)

        q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        mask = _make_causal_mask(seq_len).to(device)
        mask_flat = mask.flatten().contiguous()

        output = self._run_no_prefix(q, k, v, mask_flat, sm_scale, device)
        ref = _sdpa_reference(q, k, v, mask=[mask], sm_scale=sm_scale)

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Tests: With prefix (paged KV cache)
# ===========================================================================


class TestTritonPrefillWithPrefix:
    """Tests with prefix_lens > 0 (both stage 1 and stage 2 run)."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def _build_paged_cache(
        self, prefix_k_list, prefix_v_list, page_size, num_kv_heads, head_dim, device
    ):
        """Build a paged KV cache from prefix K,V tensors.

        Returns:
            kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim]
            page_table_indices: [total_pages] physical page IDs
            page_table_indptr: [batch+1] cumulative page counts
        """
        all_pages_k = []
        all_pages_v = []
        page_counts = [0]

        for pfx_k, pfx_v in zip(prefix_k_list, prefix_v_list):
            pfx_len = pfx_k.shape[0]
            n_pages = (pfx_len + page_size - 1) // page_size
            page_counts.append(page_counts[-1] + n_pages)

            for p in range(n_pages):
                start = p * page_size
                end = min(start + page_size, pfx_len)
                tokens_in_page = end - start

                page_k = torch.zeros(
                    num_kv_heads, page_size, head_dim, dtype=pfx_k.dtype, device=device
                )
                page_v = torch.zeros(
                    num_kv_heads, page_size, head_dim, dtype=pfx_v.dtype, device=device
                )
                page_k[:, :tokens_in_page, :] = pfx_k[start:end].transpose(0, 1)
                page_v[:, :tokens_in_page, :] = pfx_v[start:end].transpose(0, 1)
                all_pages_k.append(page_k)
                all_pages_v.append(page_v)

        if len(all_pages_k) == 0:
            # No prefix — create dummy cache
            kv_cache = torch.zeros(
                1, 2, num_kv_heads, page_size, head_dim, dtype=torch.bfloat16, device=device
            )
            page_table_indices = torch.zeros(1, dtype=torch.int32, device=device)
            page_table_indptr = torch.tensor(page_counts, dtype=torch.int32, device=device)
            return kv_cache, page_table_indices, page_table_indptr

        # Stack pages: [num_pages, num_kv_heads, page_size, head_dim]
        k_pages = torch.stack(all_pages_k, dim=0)
        v_pages = torch.stack(all_pages_v, dim=0)
        num_pages = k_pages.shape[0]

        # Build HND kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim]
        kv_cache = torch.stack([k_pages, v_pages], dim=1)

        # Simple identity page table (page i maps to page i)
        page_table_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        page_table_indptr = torch.tensor(page_counts, dtype=torch.int32, device=device)

        return kv_cache, page_table_indices, page_table_indptr

    def test_prefix_page_size_1(self, device):
        """Prefix with page_size=1."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        page_size = 1
        prefix_len, extend_len = 16, 32

        # Generate data
        prefix_k = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        prefix_v = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        q = torch.randn(extend_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Build paged cache
        kv_cache, pt_indices, pt_indptr = self._build_paged_cache(
            [prefix_k], [prefix_v], page_size, num_kv_heads, head_dim, device
        )

        # Mask: [extend_len, prefix_len + extend_len] — causal over full range
        total_len = prefix_len + extend_len
        # Query positions are prefix_len..total_len-1 (extend portion)
        q_pos = torch.arange(prefix_len, total_len, device=device)
        kv_pos = torch.arange(total_len, device=device)
        mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)  # [extend_len, total_len]
        mask_flat = mask.flatten().contiguous()

        # Run Triton
        output = torch.zeros_like(q)
        qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
        prefix_lens_t = torch.tensor([prefix_len], dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=kv_cache,
            prefix_lens=prefix_lens_t,
            page_table_indptr=pt_indptr,
            page_table_indices=pt_indices,
            page_size=page_size,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        # Reference
        ref = _sdpa_reference_with_prefix(
            q,
            k,
            v,
            prefix_k=[prefix_k],
            prefix_v=[prefix_v],
            seq_lens=[extend_len],
            prefix_lens=[prefix_len],
            mask=[mask],
            sm_scale=sm_scale,
        )

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_prefix_page_size_64(self, device):
        """Prefix with page_size=64."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        page_size = 64
        prefix_len, extend_len = 100, 32  # 100 tokens = 2 pages (64+36)

        prefix_k = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        prefix_v = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        q = torch.randn(extend_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        kv_cache, pt_indices, pt_indptr = self._build_paged_cache(
            [prefix_k], [prefix_v], page_size, num_kv_heads, head_dim, device
        )

        # Causal mask over full range
        total_len = prefix_len + extend_len
        q_pos = torch.arange(prefix_len, total_len, device=device)
        kv_pos = torch.arange(total_len, device=device)
        mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)
        mask_flat = mask.flatten().contiguous()

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
        prefix_lens_t = torch.tensor([prefix_len], dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=kv_cache,
            prefix_lens=prefix_lens_t,
            page_table_indptr=pt_indptr,
            page_table_indices=pt_indices,
            page_size=page_size,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        ref = _sdpa_reference_with_prefix(
            q,
            k,
            v,
            prefix_k=[prefix_k],
            prefix_v=[prefix_v],
            seq_lens=[extend_len],
            prefix_lens=[prefix_len],
            mask=[mask],
            sm_scale=sm_scale,
        )

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_prefix_bidirectional_mask(self, device):
        """Prefix contains image tokens with bidirectional attention."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        page_size = 16
        prefix_len, extend_len = 48, 16

        prefix_k = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        prefix_v = torch.randn(
            prefix_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        q = torch.randn(extend_len, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(extend_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        kv_cache, pt_indices, pt_indptr = self._build_paged_cache(
            [prefix_k], [prefix_v], page_size, num_kv_heads, head_dim, device
        )

        # Causal mask + bidirectional for image tokens at positions 10-30 in prefix
        total_len = prefix_len + extend_len
        q_pos = torch.arange(prefix_len, total_len, device=device)
        kv_pos = torch.arange(total_len, device=device)
        mask = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)
        # All extend queries can see image tokens 10-30 (they're in prefix, already visible via causal)
        # But image tokens 10-30 should also be visible to each other for queries at those positions
        # Since we're in extend phase, queries are at positions >= prefix_len, so causal already covers prefix
        # This test mainly verifies the mask works correctly through the paged path
        mask_flat = mask.flatten().contiguous()

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
        prefix_lens_t = torch.tensor([prefix_len], dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=kv_cache,
            prefix_lens=prefix_lens_t,
            page_table_indptr=pt_indptr,
            page_table_indices=pt_indices,
            page_size=page_size,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        ref = _sdpa_reference_with_prefix(
            q,
            k,
            v,
            prefix_k=[prefix_k],
            prefix_v=[prefix_v],
            seq_lens=[extend_len],
            prefix_lens=[prefix_len],
            mask=[mask],
            sm_scale=sm_scale,
        )

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)

    def test_batch_mixed_prefix_lengths(self, device):
        """Batch: seq 0 has no prefix, seq 1 has prefix."""
        torch.manual_seed(42)
        num_heads, num_kv_heads, head_dim = 8, 4, 512
        sm_scale = 1.0
        page_size = 16

        ext_lens = [32, 16]
        pfx_lens = [0, 32]
        total_ext = sum(ext_lens)

        q = torch.randn(total_ext, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(total_ext, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(total_ext, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Only seq 1 has prefix
        prefix_k_1 = torch.randn(32, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        prefix_v_1 = torch.randn(32, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Build paged cache for seq 1's prefix only
        kv_cache, pt_indices, pt_indptr_raw = self._build_paged_cache(
            [
                torch.zeros(0, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device),
                prefix_k_1,
            ],
            [
                torch.zeros(0, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device),
                prefix_v_1,
            ],
            page_size,
            num_kv_heads,
            head_dim,
            device,
        )

        # Masks
        masks = []
        for i, (ext_l, pfx_l) in enumerate(zip(ext_lens, pfx_lens)):
            total_l = pfx_l + ext_l
            q_pos = torch.arange(pfx_l, total_l, device=device)
            kv_pos = torch.arange(total_l, device=device)
            masks.append(q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0))

        mask_flat = _flatten_masks(masks)

        output = torch.zeros_like(q)
        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(ext_lens), 0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        prefix_lens_t = torch.tensor(pfx_lens, dtype=torch.int32, device=device)

        triton_prefill_with_custom_mask(
            q=q,
            k=k,
            v=v,
            output=output,
            qo_indptr=qo_indptr,
            kv_cache=kv_cache,
            prefix_lens=prefix_lens_t,
            page_table_indptr=pt_indptr_raw,
            page_table_indices=pt_indices,
            page_size=page_size,
            custom_mask=mask_flat,
            sm_scale=sm_scale,
        )

        # Reference: seq 0 has no prefix, seq 1 has prefix
        prefix_k_list = [None, prefix_k_1]
        prefix_v_list = [None, prefix_v_1]
        ref = _sdpa_reference_with_prefix(
            q,
            k,
            v,
            prefix_k=prefix_k_list,
            prefix_v=prefix_v_list,
            seq_lens=ext_lens,
            prefix_lens=pfx_lens,
            mask=masks,
            sm_scale=sm_scale,
        )

        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
