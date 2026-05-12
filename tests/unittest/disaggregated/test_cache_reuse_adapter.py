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
"""Tests for CacheReuseAdapter and _align_kv_blocks prefix-offset logic."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import Sender

# ---------------------------------------------------------------------------
# _align_kv_blocks tests (the core block-alignment helper)
# ---------------------------------------------------------------------------


class TestAlignKvBlocks:
    """Verify Sender._align_kv_blocks handles prefix-cache offsets correctly."""

    TPB = 64  # tokens_per_block

    def _align(self, src, dst, src_start=0, dst_start=0):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=self.TPB,
        )

    def test_no_prefix_cache(self):
        """Neither side has prefix cache — identity."""
        src, dst = self._align([10, 11, 12], [20, 21, 22])
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_gen_has_prefix_cache(self):
        """Gen cached 2 blocks — only transfer suffix."""
        # ctx sends blocks for tokens [0, 320), gen needs only [128, 320)
        src, dst = self._align(
            [10, 11, 12, 13, 14],  # 5 blocks, full prompt
            [20, 21, 22],  # 3 blocks, suffix only
            src_start=0,
            dst_start=2 * self.TPB,  # gen cached 2 blocks
        )
        # src should skip first 2 blocks, transfer 3
        np.testing.assert_array_equal(src, [12, 13, 14])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_ctx_has_prefix_cache(self):
        """Ctx cached 1 block — ctx starts from block 1."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: 3 blocks starting from token 64
            [20, 21, 22, 23],  # gen: 4 blocks, full prompt
            src_start=1 * self.TPB,
            dst_start=0,
        )
        # dst should skip first block, transfer 3
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [21, 22, 23])

    def test_both_have_prefix_cache(self):
        """Both sides cached — transfer only the overlap."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: 3 blocks from token 64
            [20, 21],  # gen: 2 blocks from token 128
            src_start=1 * self.TPB,
            dst_start=2 * self.TPB,
        )
        # overlap starts at token 128 → src_skip=1, dst_skip=0, n=2
        np.testing.assert_array_equal(src, [11, 12])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_gen_full_cache_hit(self):
        """Gen has entire prompt cached — nothing to transfer."""
        src, dst = self._align(
            [10, 11, 12],  # ctx: full prompt
            [20, 21, 22],  # gen: full prompt (all cached)
            src_start=0,
            dst_start=3 * self.TPB,  # gen cached all 3 blocks
        )
        assert src.size == 0
        assert dst.size == 0

    def test_gen_prefix_with_draft_block(self):
        """Gen has prefix cache + 1 extra draft block — transfer suffix only."""
        src, dst = self._align(
            [10, 11, 12, 13],  # ctx: 4 blocks
            [20, 21, 22],  # gen: 2 suffix + 1 draft
            src_start=0,
            dst_start=2 * self.TPB,
        )
        # transfer min(4-2, 3-0) = 2 blocks
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [20, 21])


# ---------------------------------------------------------------------------
# KVSlice token_range with prefix offset
# ---------------------------------------------------------------------------


class TestTokenRangeWithPrefix:
    """Verify TokenRange is correctly set with prefix offsets."""

    def test_no_cache(self):
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        tr = TokenRange(start=0, end=256)
        assert tr.start == 0
        assert tr.end == 256

    def test_partial_cache(self):
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        tr = TokenRange(start=128, end=256)
        assert tr.start == 128
        assert tr.end == 256

    def test_full_cache_hit_raises(self):
        """When cached_tokens == prompt_len, start == end should raise."""
        from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange

        with pytest.raises(ValueError):
            TokenRange(start=256, end=256)


# ---------------------------------------------------------------------------
# Sliding window + cache reuse: block-ID filter
# ---------------------------------------------------------------------------


def _apply_window_cache_filter(block_ids, prompt_len, tpb, window_size, num_cached_blocks):
    """Replicate the block-ID filtering from KvCacheTransceiverV2._create_kv_slice.

    In V2 with valid_only=True the adapter already strips stale blocks, so
    ``block_ids`` starts at stale_end.  ``cached_in_window`` counts only the
    blocks that are both cached and inside the valid window, avoiding
    over-skipping when num_cached_blocks includes stale entries.
    """
    block_ids = np.array(block_ids, dtype=np.int64)
    total_blocks = (prompt_len + tpb - 1) // tpb
    stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
    expected_valid = total_blocks - stale_end
    if expected_valid <= 0:
        return np.array([], dtype=np.int64)
    elif block_ids.size > expected_valid:
        block_ids = block_ids[-expected_valid:]
    cached_in_window = max(0, num_cached_blocks - stale_end)
    if cached_in_window >= block_ids.size:
        return np.array([], dtype=np.int64)
    elif cached_in_window > 0:
        block_ids = block_ids[cached_in_window:]
    return block_ids


class TestWindowCacheBlockFilter:
    """Verify cached_in_window filtering for sliding window + cache reuse.

    Setup: tpb=8, prompt_len=32 (4 blocks), window_size=16.
      stale_end = max(0, (33-16)//8) = 2.
      valid_only returns blocks starting at stale_end → [G2, G3].
    """

    TPB = 8
    PROMPT_LEN = 32
    WINDOW = 16
    BLOCKS = [20, 21]  # physical IDs for the 2 valid window blocks

    def _filter(self, num_cached_blocks):
        return _apply_window_cache_filter(
            self.BLOCKS, self.PROMPT_LEN, self.TPB, self.WINDOW, num_cached_blocks
        )

    def test_no_cache(self):
        """Gen has no cached tokens → all valid window blocks needed."""
        result = self._filter(num_cached_blocks=0)
        np.testing.assert_array_equal(result, [20, 21])

    def test_cache_entirely_in_stale_zone(self):
        """Gen cached 2 blocks, both stale (cached_in_window=0).

        num_cached_blocks=2 == stale_end=2 → cached_in_window=0.
        All valid window blocks still needed.
        """
        result = self._filter(num_cached_blocks=2)
        np.testing.assert_array_equal(result, [20, 21])

    def test_cache_partially_in_window(self):
        """Gen cached 3 blocks (2 stale + 1 window): cached_in_window=1.

        stale_end=2, num_cached_blocks=3 → cached_in_window=1.
        Skip 1 block from the valid window list → only [G3] needed.
        """
        result = _apply_window_cache_filter(
            [20, 21], self.PROMPT_LEN, self.TPB, self.WINDOW, num_cached_blocks=3
        )
        np.testing.assert_array_equal(result, [21])

    def test_cache_covers_all_window_blocks(self):
        """Gen cached all 4 blocks: cached_in_window ≥ expected_valid → empty."""
        result = self._filter(num_cached_blocks=4)
        assert result.size == 0

    def test_cached_in_window_vs_naive_skip(self):
        """cached_in_window skips correctly when stale_end > 0.

        window_size=24, tpb=8, prompt_len=32 → stale_end=1.
        valid window blocks after stale removal: [G1, G2, G3].
        Gen cached 2 blocks → cached_in_window=max(0, 2-1)=1 → skip 1 → [G2, G3].

        A naive block_ids[num_cached_blocks:] skips 2 from [G1,G2,G3] and gives [G3] only.
        """
        blocks = [10, 11, 12]  # [G1, G2, G3] after stale removal (stale_end=1)
        prompt_len, tpb, window_size = 32, 8, 24
        num_cached_blocks = 2

        result = _apply_window_cache_filter(blocks, prompt_len, tpb, window_size, num_cached_blocks)
        np.testing.assert_array_equal(result, [11, 12])

        stale_end = max(0, (prompt_len + 1 - window_size) // tpb)  # = 1
        naive_result = np.array(blocks, dtype=np.int64)[num_cached_blocks:]
        assert naive_result.tolist() == [12]
        assert result.tolist() != naive_result.tolist()
        assert stale_end == 1

    def test_large_window_no_stale_blocks(self):
        """Window covers entire prompt: stale_end=0, cached_in_window=num_cached_blocks.

        prompt_len=32, window_size=32, tpb=8 → stale_end=0.
        Gen cached 1 block → cached_in_window=1.
        """
        result = _apply_window_cache_filter(
            [10, 11, 12, 13], prompt_len=32, tpb=8, window_size=32, num_cached_blocks=1
        )
        np.testing.assert_array_equal(result, [11, 12, 13])


# ---------------------------------------------------------------------------
# Sliding window + cache reuse: sender src_start / dst_start
# ---------------------------------------------------------------------------


class TestWindowSenderAlignment:
    """Verify src_start / dst_start for sliding window layers in Sender._do_send.

    After valid_only removal, both the src and dst block lists start at
    stale_end*tpb (not at 0).  src_start=max(stale_end*tpb, ctx_cached_tokens)
    and dst_start=stale_end*tpb tell _align_kv_blocks where each list begins
    so that it computes the correct block-skip counts.
    """

    TPB = 8

    @staticmethod
    def _compute_starts(prompt_len, tpb, window_size, ctx_cached_tokens):
        """Compute src_start/dst_start for a window layer group."""
        stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
        src_start = max(stale_end * tpb, ctx_cached_tokens)
        dst_start = stale_end * tpb
        return src_start, dst_start

    def _align(self, src, dst, src_start, dst_start):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=self.TPB,
        )

    def test_window_no_cache_stale_blocks_present(self):
        """Pure window transfer: no cache, stale blocks removed by valid_only.

        tpb=8, window_size=16, prompt_len=32 → stale_end=2.
        After valid_only: src=[C2,C3], dst=[G2,G3], both starting at token 16.
        src_start=dst_start=16 → identity alignment.
        """
        prompt_len, window_size = 32, 16
        src = [10, 11]  # C2, C3 (valid window blocks)
        dst = [20, 21]  # G2, G3
        src_start, dst_start = self._compute_starts(prompt_len, self.TPB, window_size, 0)
        assert src_start == 16
        assert dst_start == 16
        result_src, result_dst = self._align(src, dst, src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11])
        np.testing.assert_array_equal(result_dst, [20, 21])

    def test_stale_cache_alignment(self):
        """Window + stale-only cache: src_start=stale_end*tpb avoids over-skipping.

        tpb=8, window_size=16, prompt_len=32 → stale_end=2.
        Gen cached 2 blocks (all stale) → req_info.start_token_idx=16.
        valid_only src=[C2,C3] starts at token 16, not 0.

        Using src_start=0 and dst_start=16 treats src as starting at 0,
        causing _align to compute src_skip=2 and return empty arrays.
        Setting src_start=dst_start=16 gives the correct full transfer.
        """
        prompt_len, window_size = 32, 16
        src = [10, 11]  # C2, C3 (valid window blocks, stale removed)
        dst = [20, 21]  # G2, G3

        wrong_src, _ = self._align(src, dst, src_start=0, dst_start=16)
        assert wrong_src.size == 0

        src_start, dst_start = self._compute_starts(prompt_len, self.TPB, window_size, 0)
        result_src, result_dst = self._align(src, dst, src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11])
        np.testing.assert_array_equal(result_dst, [20, 21])

    def test_ctx_cache_in_window(self):
        """Ctx has cached tokens within the valid window.

        tpb=8, window_size=24, prompt_len=32 → stale_end=1, stale_end*tpb=8.
        Ctx cached 2 blocks (16 tokens): src_start=max(8,16)=16, dst_start=8.
        After valid_only: src=[C1,C2,C3], dst=[G1,G2,G3].
        overlap_start=max(16,8)=16 → src_skip=0, dst_skip=1, n=2.
        """
        prompt_len, window_size = 32, 24
        ctx_cached_tokens = 16
        src = [10, 11, 12]  # C1, C2, C3 after valid_only (stale_end=1)
        dst = [20, 21, 22]  # G1, G2, G3 after valid_only

        src_start, dst_start = self._compute_starts(
            prompt_len, self.TPB, window_size, ctx_cached_tokens
        )
        assert src_start == 16
        assert dst_start == 8

        result_src, result_dst = self._align(src, dst, src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11])
        np.testing.assert_array_equal(result_dst, [21, 22])

    def test_no_stale_no_cache(self):
        """Window covers entire prompt, no cache: src_start=dst_start=0, identity alignment.

        tpb=8, window_size=32, prompt_len=32 → stale_end=0.
        """
        prompt_len, window_size = 32, 32
        src = [10, 11, 12, 13]
        dst = [20, 21, 22, 23]
        src_start, dst_start = self._compute_starts(prompt_len, self.TPB, window_size, 0)
        assert src_start == 0
        assert dst_start == 0
        result_src, result_dst = self._align(src, dst, src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11, 12, 13])
        np.testing.assert_array_equal(result_dst, [20, 21, 22, 23])
