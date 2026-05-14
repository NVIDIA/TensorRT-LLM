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
"""Tests for CacheReuseAdapter, _create_kv_slice SWA trim, and Sender token-start derivation."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange
from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import CacheReuseAdapter
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup

# ---------------------------------------------------------------------------
# _align_kv_blocks: contract unchanged.
# ---------------------------------------------------------------------------


class TestAlignKvBlocks:
    """Verify Sender._align_kv_blocks handles src/dst token starts correctly."""

    TPB = 64

    def _align(self, src, dst, src_start=0, dst_start=0):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=self.TPB,
        )

    def test_no_offset(self):
        src, dst = self._align([10, 11, 12], [20, 21, 22])
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_dst_starts_later(self):
        # dst covers tokens [128, 320), src covers [0, 320) → trim src head by 2 blocks.
        src, dst = self._align(
            [10, 11, 12, 13, 14],
            [20, 21, 22],
            src_start=0,
            dst_start=2 * self.TPB,
        )
        np.testing.assert_array_equal(src, [12, 13, 14])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_src_starts_later(self):
        src, dst = self._align(
            [10, 11, 12],
            [20, 21, 22, 23],
            src_start=1 * self.TPB,
            dst_start=0,
        )
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [21, 22, 23])

    def test_both_offset(self):
        src, dst = self._align(
            [10, 11, 12],
            [20, 21],
            src_start=1 * self.TPB,
            dst_start=2 * self.TPB,
        )
        np.testing.assert_array_equal(src, [11, 12])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_no_overlap(self):
        # dst entirely past src.
        src, dst = self._align([10, 11, 12], [20, 21, 22], src_start=0, dst_start=3 * self.TPB)
        assert src.size == 0
        assert dst.size == 0

    def test_dst_extra_draft_block(self):
        src, dst = self._align(
            [10, 11, 12, 13],
            [20, 21, 22],
            src_start=0,
            dst_start=2 * self.TPB,
        )
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [20, 21])


# ---------------------------------------------------------------------------
# TokenRange dataclass invariants.
# ---------------------------------------------------------------------------


class TestTokenRange:
    def test_zero_start(self):
        tr = TokenRange(start=0, end=256)
        assert (tr.start, tr.end) == (0, 256)

    def test_nonzero_start(self):
        # Allowed: caller may pass a non-prompt-zero range (e.g., chunk).
        tr = TokenRange(start=128, end=256)
        assert (tr.start, tr.end) == (128, 256)

    def test_start_eq_end_rejected(self):
        with pytest.raises(ValueError):
            TokenRange(start=256, end=256)


# ---------------------------------------------------------------------------
# CacheReuseAdapter.get_cached_token_count_per_layer_group: SWA clamp.
# ---------------------------------------------------------------------------


class _StubAdapter(CacheReuseAdapter):
    """Adapter whose only knob is the cache-manager-reported scalar."""

    def __init__(self, scalar: int, tpb: int, enabled: bool = True):
        self._scalar = scalar
        self._tpb = tpb
        self._enabled = enabled

    @property
    def enable_block_reuse(self) -> bool:
        return self._enabled

    @property
    def tokens_per_block(self) -> int:
        return self._tpb

    def _global_cached_token_count(self, req):  # noqa: ARG002
        return self._scalar

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        return np.array([], dtype=np.int64)

    def commit_blocks_for_reuse(self, req):  # noqa: ARG002
        pass


class _FakeReq:
    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len


def _lg(window=None):
    return AttentionLayerGroup(pool_group_idx=0, sliding_window_size=window)


class TestAdapterPerLayerGroup:
    """Per-layer cached prefix: SWA groups are clamped up to stale_end*tpb."""

    TPB = 8

    def test_reuse_disabled(self):
        ad = _StubAdapter(scalar=128, tpb=self.TPB, enabled=False)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg(window=64)])
        assert out == [0, 0]

    def test_zero_scalar(self):
        ad = _StubAdapter(scalar=0, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg(window=64)])
        assert out == [0, 0]

    def test_full_attn_passthrough(self):
        # full-attn group: scalar is returned as-is.
        ad = _StubAdapter(scalar=64, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg()])
        assert out == [64, 64]

    def test_swa_scalar_inside_window_no_clamp(self):
        # prompt_len=32, window=16, tpb=8 → stale_end=2 → stale_end*tpb=16.
        # scalar=24 ≥ 16 → no clamp.
        ad = _StubAdapter(scalar=24, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [24]

    def test_swa_scalar_below_stale_clamped_up(self):
        # stale_end*tpb = 16; scalar=8 → clamped up to 16.
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [16]

    def test_mixed_groups(self):
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(
            _FakeReq(32), [_lg(), _lg(window=16), _lg(window=32)]
        )
        # full-attn=8, swa(window=16) clamped to 16, swa(window=32) stale_end=0 → 8.
        assert out == [8, 16, 8]


# ---------------------------------------------------------------------------
# _create_kv_slice SWA block trim: window-trim + cache-skip via per-layer cached.
# ---------------------------------------------------------------------------


def _swa_trim(block_ids, prompt_len, tpb, window_size, scalar_cached_tokens):
    """Replicate the SWA branch of KvCacheTransceiverV2._create_kv_slice.

    Inputs:
      block_ids: list possibly containing stale entries (V1 pre-eviction view).
      scalar_cached_tokens: the cache-manager scalar BEFORE adapter SWA clamp.
    """
    block_ids = np.array(block_ids, dtype=np.int64)
    total_blocks = (prompt_len + tpb - 1) // tpb
    stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
    expected_valid = max(0, total_blocks - stale_end)
    if block_ids.size > expected_valid:
        block_ids = (
            block_ids[-expected_valid:] if expected_valid > 0 else np.array([], dtype=np.int64)
        )
    # Adapter clamps cached_lg ≥ stale_end*tpb.
    cached_lg = max(scalar_cached_tokens, stale_end * tpb)
    cache_skip = cached_lg // tpb - stale_end
    if cache_skip > 0:
        block_ids = (
            block_ids[cache_skip:] if cache_skip < block_ids.size else np.array([], dtype=np.int64)
        )
    return block_ids


class TestSwaTrim:
    """Window-trim + cache-skip in _create_kv_slice's SWA path.

    Setup: tpb=8, prompt_len=32 → total_blocks=4; window=16 → stale_end=2.
    """

    TPB = 8
    PROMPT_LEN = 32
    WINDOW = 16

    def _trim(self, blocks, scalar):
        return _swa_trim(blocks, self.PROMPT_LEN, self.TPB, self.WINDOW, scalar)

    def test_no_cache(self):
        np.testing.assert_array_equal(self._trim([20, 21], scalar=0), [20, 21])

    def test_cache_entirely_stale(self):
        # scalar=16=stale_end*tpb → cached_lg=16, cache_skip=0.
        np.testing.assert_array_equal(self._trim([20, 21], scalar=16), [20, 21])

    def test_cache_one_block_in_window(self):
        # scalar=24 → cached_lg=24, cache_skip=24/8-2=1.
        np.testing.assert_array_equal(self._trim([20, 21], scalar=24), [21])

    def test_cache_covers_full_window(self):
        # scalar=32 → cache_skip=2, list size=2 → empty.
        assert self._trim([20, 21], scalar=32).size == 0

    def test_window_offset_skip_uses_clamped_value(self):
        # window=24 → stale_end=1; scalar=16 (2 blocks) → cached_lg=16, cache_skip=1.
        # Naive block_ids[scalar//tpb:] would skip 2 from a 3-block list and return 1 block.
        out = _swa_trim([10, 11, 12], prompt_len=32, tpb=8, window_size=24, scalar_cached_tokens=16)
        np.testing.assert_array_equal(out, [11, 12])

    def test_window_covers_all_no_stale(self):
        # window=prompt_len → stale_end=0; behaves like full-attn.
        out = _swa_trim(
            [10, 11, 12, 13], prompt_len=32, tpb=8, window_size=32, scalar_cached_tokens=8
        )
        np.testing.assert_array_equal(out, [11, 12, 13])

    def test_v1_pre_eviction_includes_stale(self):
        # Pre-eviction list has all 4 blocks; window-trim keeps last expected_valid=2.
        out = _swa_trim([10, 11, 12, 13], self.PROMPT_LEN, self.TPB, self.WINDOW, 0)
        np.testing.assert_array_equal(out, [12, 13])


# ---------------------------------------------------------------------------
# Sender token-start derivation: (total_blocks - n_blocks) * tpb + SWA clamp.
# ---------------------------------------------------------------------------


def _derive_starts(prompt_len, tpb, window_size, n_src, n_dst, slice_end=None):
    """Replicate _build_kv_write_meta's per-layer src/dst token-start derivation."""
    if slice_end is None:
        slice_end = prompt_len
    total_blocks = (slice_end + tpb - 1) // tpb
    src_start = (total_blocks - n_src) * tpb
    dst_start = (total_blocks - n_dst) * tpb
    if window_size is not None:
        stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
        src_start = max(stale_end * tpb, src_start)
        dst_start = max(stale_end * tpb, dst_start)
    return src_start, dst_start


class TestSenderTokenStarts:
    """Verify (total_blocks - n) * tpb + SWA clamp produces correct src/dst starts."""

    TPB = 8

    def _align(self, src, dst, src_start, dst_start):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=self.TPB,
        )

    def test_full_prompt_no_cache(self):
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=None, n_src=4, n_dst=4
        )
        assert (src_start, dst_start) == (0, 0)

    def test_full_prompt_dst_cached(self):
        # dst cached 2 blocks → dst sends 2 → dst_start=16.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=None, n_src=4, n_dst=2
        )
        assert (src_start, dst_start) == (0, 16)

    def test_full_prompt_src_cached(self):
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=None, n_src=2, n_dst=4
        )
        assert (src_start, dst_start) == (16, 0)

    def test_swa_no_cache_stale_present(self):
        # window=16 → stale_end=2 → stale_end*tpb=16; both sides 2 blocks.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=16, n_src=2, n_dst=2
        )
        assert (src_start, dst_start) == (16, 16)

    def test_swa_dst_cache_in_stale_region(self):
        # dst cached 2 blocks but all stale → dst still has 2 valid window blocks.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=16, n_src=2, n_dst=2
        )
        result_src, result_dst = self._align([10, 11], [20, 21], src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11])
        np.testing.assert_array_equal(result_dst, [20, 21])

    def test_swa_src_cache_inside_window(self):
        # window=24 → stale_end=1 → stale_end*tpb=8.
        # ctx cached 16 tokens (2 blocks), window-trim leaves 3 blocks, skip 1 → src has 2 blocks.
        # dst no cache → window-trim leaves 3 blocks.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=24, n_src=2, n_dst=3
        )
        # total_blocks = 4. src_start = (4-2)*8 = 16. dst_start = (4-3)*8 = 8. SWA clamp keeps both.
        assert (src_start, dst_start) == (16, 8)
        result_src, result_dst = self._align([10, 11], [20, 21, 22], src_start, dst_start)
        np.testing.assert_array_equal(result_src, [10, 11])
        np.testing.assert_array_equal(result_dst, [21, 22])

    def test_swa_window_covers_prompt_no_stale(self):
        # window=prompt_len → stale_end=0; SWA clamp is a no-op.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=32, n_src=4, n_dst=4
        )
        assert (src_start, dst_start) == (0, 0)

    def test_chunked_slice_end_below_prompt(self):
        # Non-final slice: slice_end=16, prompt_len=32, no window.
        # 2 blocks in slice; cache-free.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=None, n_src=2, n_dst=2, slice_end=16
        )
        assert (src_start, dst_start) == (0, 0)

    def test_chunked_slice_entirely_stale_for_swa(self):
        # slice_end=16 ≤ stale_end*tpb=16 → SWA layer sends 0 blocks; clamp pushes start to 16.
        src_start, dst_start = _derive_starts(
            prompt_len=32, tpb=self.TPB, window_size=16, n_src=0, n_dst=0, slice_end=16
        )
        # total_blocks for slice = 2 → raw start = 16; clamped = max(16, 16) = 16.
        assert (src_start, dst_start) == (16, 16)
