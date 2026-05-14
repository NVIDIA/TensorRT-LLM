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

"""SWA correctness tests for Triton paged attention.

Two regimes that the existing kernel tests don't exercise:

1. **Long prefill SWA** (no eviction yet): a single prefill longer than the
   sliding window. The kernel must apply the SW mask within the chunk itself.
   This is the lightweight regression that catches mask-math regressions
   without spinning up a real KVCacheManager.

2. **Front-eviction SWA** (eviction has happened): the shim hands the kernel
   a window-coherent local-coord view — sliced cache_loc, window-capped
   seq_len_with_cache. The kernel must produce the same output as SDPA on the
   live window only. Pre-fix kernels that mixed coordinate spaces silently
   dropped boundary pages and corrupted masks; under Option B (everything
   local) the kernel needs no changes, so these tests should pass before AND
   after the shim/transform fix. They are the regression guard against future
   coordinate-space mistakes.
"""

import math

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _sdpa_reference_with_sw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int,
    cache_len_before_q: int,
) -> torch.Tensor:
    """Reference SDPA with a per-query causal + sliding-window mask.

    Inputs are in [tokens, n_heads, head_dim] (flat). For each query token at
    local position `q_pos = cache_len_before_q + i` (i ∈ [0, q_len)), the
    valid keys are positions in [max(0, q_pos - W + 1), q_pos].
    """
    q_t = q.transpose(0, 1).unsqueeze(0)  # [1, n_heads, q_len, head_dim]
    k_t = k.transpose(0, 1).unsqueeze(0)  # [1, n_kv_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1).unsqueeze(0)

    n_heads = q_t.shape[1]
    n_kv_heads = k_t.shape[1]
    if n_heads != n_kv_heads:
        rep = n_heads // n_kv_heads
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)

    q_len = q_t.shape[2]
    kv_len = k_t.shape[2]

    head_dim = q_t.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [1, h, q_len, kv_len]

    # mask in [q_len, kv_len]; q_pos = cache_len_before_q + i
    q_positions = torch.arange(q_len, device=q.device) + cache_len_before_q
    kv_positions = torch.arange(kv_len, device=q.device)
    diff = q_positions.unsqueeze(1) - kv_positions.unsqueeze(0)
    causal_ok = diff >= 0
    window_ok = diff < sliding_window
    valid = causal_ok & window_ok
    scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_t)  # [1, h, q_len, head_dim]
    return out.squeeze(0).transpose(0, 1)  # [q_len, n_heads, head_dim]


def _write_kv_to_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
):
    """Write all of (k, v) sequentially into kv_cache at the listed pages.

    k, v are flat [tokens, n_kv_heads, head_dim]. page_table is a 1D int32
    tensor listing physical page ids in token-order. Assumes page_table
    covers exactly enough pages for k.shape[0] tokens.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
        update_paged_kv_cache,
    )

    n_tokens = k.shape[0]
    batch_indices = torch.zeros(n_tokens, dtype=torch.int32, device=k.device)
    positions = torch.arange(n_tokens, dtype=torch.int32, device=k.device)
    kv_indptr = torch.tensor([0, page_table.numel()], dtype=torch.int32, device=k.device)
    update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, page_table, kv_indptr)


class TestTritonPagedLongPrefillSWA:
    """Prefill longer than the sliding window — no eviction yet.

    Verifies the kernel correctly applies the SW mask within a single long
    prefill chunk. Covers BOTH the Phase-1 (full pages) and Phase-2 (boundary
    pages) code paths since prefill_len > W * 2 forces both.
    """

    @pytest.mark.parametrize("page_size", [16, 64])
    @pytest.mark.parametrize("sliding_window", [128, 256])
    @pytest.mark.parametrize("prefill_len_mult", [1, 2, 4])  # 1*W, 2*W, 4*W
    @pytest.mark.parametrize("n_heads,n_kv_heads", [(4, 4), (8, 2)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_long_prefill_sw_matches_sdpa(
        self,
        page_size: int,
        sliding_window: int,
        prefill_len_mult: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
        )

        torch.manual_seed(0)
        prefill_len = sliding_window * prefill_len_mult
        # Round prefill_len up so it spans whole + boundary pages naturally.
        num_pages = (prefill_len + page_size - 1) // page_size
        num_blocks = num_pages + 4

        q = torch.randn(prefill_len, n_heads, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(prefill_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(prefill_len, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")

        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )
        page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda")
        _write_kv_to_cache(k, v, kv_cache, page_table, page_size)

        qo_indptr = torch.tensor([0, prefill_len], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
        kv_indices = page_table.clone()
        last_token_in_page = prefill_len % page_size
        kv_last_page_len = torch.tensor(
            [last_token_in_page if last_token_in_page > 0 else page_size],
            dtype=torch.int32,
            device="cuda",
        )
        seq_len_with_cache = torch.tensor([prefill_len], dtype=torch.int32, device="cuda")

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

        ref = _sdpa_reference_with_sw(
            q.float(),
            k.float(),
            v.float(),
            sliding_window=sliding_window,
            cache_len_before_q=0,
        ).to(output.dtype)

        torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)


class TestTritonPagedSWAFrontEviction:
    """Front-eviction SWA — the shim hands the kernel a window-local view.

    The setup mirrors what `ad_executor.py` does after Phase 2: it slices
    `cache_loc` to the live (post-eviction) pages and caps
    `seq_len_with_cache` at the window. With Option B (everything local), the
    kernel needs no changes — these tests are the regression guard for that
    contract.
    """

    @pytest.mark.parametrize("front_removed", [0, 1, 4])
    @pytest.mark.parametrize("chunk", [1, 64, 128])
    @pytest.mark.parametrize("page_size", [16, 64])
    @pytest.mark.parametrize("with_sliding_window", [True, False])
    def test_front_evicted_context_matches_sdpa_on_live_window(
        self,
        front_removed: int,
        chunk: int,
        page_size: int,
        with_sliding_window: bool,
    ):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            triton_paged_context,
        )

        # Fixed simple geometry for clarity. Window W=256 → 16 pages at PAGE_SIZE=16,
        # 4 pages at PAGE_SIZE=64.
        torch.manual_seed(0)
        sliding_window = 256
        n_heads = 4
        n_kv_heads = 4
        head_dim = 64

        # Total cached tokens (post-fill, pre-chunk). Make this exactly equal to W
        # so the "live" window contains exactly W tokens regardless of front_removed.
        total_cache_tokens = sliding_window
        # The historical page list must include the front-evicted pages too.
        historical_pages = front_removed + (total_cache_tokens + page_size - 1) // page_size
        num_blocks = historical_pages + 4 + (chunk + page_size - 1) // page_size

        # Build a historical KV history: front_removed*page_size junk tokens
        # (will not be read) + total_cache_tokens real tokens, then `chunk` new
        # tokens being processed.
        live_cache_tokens = total_cache_tokens
        new_chunk = chunk

        # Allocate cache and a "historical" page table (front_removed pages
        # contain stale data; the rest contain live cached tokens).
        kv_cache = torch.zeros(
            num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=torch.float16, device="cuda"
        )

        # Real (live + new) KV data we want the kernel to attend to.
        live_k = torch.randn(
            live_cache_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        live_v = torch.randn(
            live_cache_tokens, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        new_k = torch.randn(new_chunk, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        new_v = torch.randn(new_chunk, n_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        # Stale data the kernel must NOT read (we'll keep these pages
        # zero-filled — if the kernel reads them it'll skew the output).

        # Page table layout (historical, full list):
        #   [stale_0, stale_1, ..., stale_{front_removed-1}, live_0, ..., new_pages...]
        live_pages_needed = (live_cache_tokens + page_size - 1) // page_size
        new_pages_needed = (new_chunk + page_size - 1) // page_size
        # Total physical pages used by this request, from page 0..end:
        all_pages = torch.arange(
            historical_pages + new_pages_needed, dtype=torch.int32, device="cuda"
        )

        # Write live KV to the live pages (positions [front_removed*page_size,
        # front_removed*page_size + live_cache_tokens)) of the global cache.
        live_page_table = all_pages[front_removed : front_removed + live_pages_needed]
        _write_kv_to_cache(live_k, live_v, kv_cache, live_page_table, page_size)
        # Write new chunk KV to the new pages (just after the live pages).
        new_page_table = all_pages[
            front_removed + live_pages_needed : front_removed + live_pages_needed + new_pages_needed
        ]
        # Pad to page boundary for the write helper.
        new_k_padded = torch.zeros(
            new_pages_needed * page_size, n_kv_heads, head_dim, dtype=torch.float16, device="cuda"
        )
        new_v_padded = torch.zeros_like(new_k_padded)
        # Page-aligned local offset for the chunk's first token: it sits right
        # after the live cache, in local coords that's live_cache_tokens.
        # Since live_cache_tokens is a multiple of page_size (we set it = W and
        # require page_size | W), the chunk starts at offset 0 within new_page_table.
        assert live_cache_tokens % page_size == 0, (
            "test geometry assumes live cache is page-aligned"
        )
        new_k_padded[:new_chunk] = new_k
        new_v_padded[:new_chunk] = new_v
        # We can't easily write a partial-page; only fill if new_chunk is multiple of page_size.
        # For partial pages, fall back to per-token write.
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
            update_paged_kv_cache,
        )

        b_idx = torch.zeros(new_chunk, dtype=torch.int32, device="cuda")
        positions = torch.arange(new_chunk, dtype=torch.int32, device="cuda")
        kv_indptr_for_write = torch.tensor([0, new_pages_needed], dtype=torch.int32, device="cuda")
        update_paged_kv_cache(
            new_k, new_v, b_idx, positions, kv_cache, new_page_table, kv_indptr_for_write
        )

        # Shim's view: slice the historical page table to live + new only.
        # This is exactly `all_indices[front_removed : front_removed + num_active]`
        # in ad_executor.py after Phase 2.
        live_view_pages = all_pages[
            front_removed : front_removed + live_pages_needed + new_pages_needed
        ].contiguous()
        num_kv_pages = live_view_pages.numel()

        # Q is the chunk's queries.
        q = torch.randn(new_chunk, n_heads, head_dim, dtype=torch.float16, device="cuda")

        qo_indptr = torch.tensor([0, new_chunk], dtype=torch.int32, device="cuda")
        kv_indptr = torch.tensor([0, num_kv_pages], dtype=torch.int32, device="cuda")
        # Window-capped seq_len_with_cache: live_cache_tokens + new_chunk
        # (NOT the global value which would include the evicted prefix).
        seq_len_with_cache = torch.tensor(
            [live_cache_tokens + new_chunk], dtype=torch.int32, device="cuda"
        )
        last_token_in_page = (live_cache_tokens + new_chunk) % page_size
        kv_last_page_len = torch.tensor(
            [last_token_in_page if last_token_in_page > 0 else page_size],
            dtype=torch.int32,
            device="cuda",
        )

        sm_scale = 1.0 / math.sqrt(head_dim)
        sw_arg = sliding_window if with_sliding_window else 0
        output = triton_paged_context(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            live_view_pages,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
            sliding_window=sw_arg,
        )

        # SDPA reference computed on the LIVE window only: keys/values are
        # [live_k; new_k], queries are `q`. Cache length before q is
        # live_cache_tokens. Sliding window applied if requested (passing
        # sw=0 corresponds to plain causal on the live window).
        full_k = torch.cat([live_k, new_k], dim=0)
        full_v = torch.cat([live_v, new_v], dim=0)
        ref = _sdpa_reference_with_sw(
            q.float(),
            full_k.float(),
            full_v.float(),
            # sw=0 in the kernel ⇒ no SW pruning; reproduce that by feeding a
            # window larger than the entire kv length.
            sliding_window=sliding_window
            if with_sliding_window
            else (live_cache_tokens + new_chunk + 1),
            cache_len_before_q=live_cache_tokens,
        ).to(output.dtype)

        torch.testing.assert_close(output.float(), ref.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
