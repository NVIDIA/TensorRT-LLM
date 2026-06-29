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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import TokenRange
from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    _CacheReuseAdapterV1,
)
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, LocalLayer
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

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
# Packed 1-D beam block layout.
# ---------------------------------------------------------------------------


class TestPackedBeamBlockLayout:
    """Verify beam search block IDs stay 1-D with only final tail blocks appended."""

    def test_v1_adapter_uses_request_py_beam_width(self):
        class _FakeMgr:
            enable_block_reuse = True
            tokens_per_block = 32

            def __init__(self):
                self.beam_width = None

            def get_batch_cache_indices(self, request_ids, layer_idx=None, beam_width=1):
                self.beam_width = beam_width
                return [[10, 11, 12, 13]]

        req = _FakeReq(prompt_len=7)
        req.py_request_id = 1
        req.py_beam_width = 4
        req.sampling_config = _FakeSamplingConfig(beam_width=1)
        mgr = _FakeMgr()

        block_ids = _CacheReuseAdapterV1(mgr).get_block_ids(req, 0, _lg())

        assert mgr.beam_width == 4
        np.testing.assert_array_equal(block_ids, [10, 11, 12, 13])

    def test_pack_beam_cache_indices_single_block_prompt_keeps_all_beams(self):
        packed = KVCacheManager._pack_beam_cache_indices([[10], [10], [10], [10]])

        assert packed == [10]

    def test_pack_beam_cache_indices_appends_final_unshared_blocks(self):
        packed = KVCacheManager._pack_beam_cache_indices(
            [
                [10, 11, 12],
                [10, 11, 13],
                [10, 11, 14],
                [10, 11, 15],
            ]
        )

        assert packed == [10, 11, 12, 13, 14, 15]

    def test_pack_beam_cache_indices_skips_shared_final_blocks(self):
        packed = KVCacheManager._pack_beam_cache_indices(
            [
                [10, 11, 12],
                [10, 11, 12],
                [10, 11, 13],
            ]
        )

        assert packed == [10, 11, 12, 13]

    def test_beam0_block_count_for_full_packed_prompt(self):
        block_ids = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)

        assert Sender._beam0_block_count(block_ids, total_blocks=3, beam_width=4) == 3

    def test_beam0_block_count_after_cached_prefix_skip(self):
        block_ids = np.array([12, 13, 14, 15], dtype=np.int64)

        assert Sender._beam0_block_count(block_ids, total_blocks=3, beam_width=4) == 1

    def test_beam0_block_count_for_single_beam_unchanged(self):
        block_ids = np.array([10, 11, 12], dtype=np.int64)

        assert Sender._beam0_block_count(block_ids, total_blocks=3, beam_width=1) == 3

    def test_align_packed_single_block_prompt_keeps_all_beam_blocks(self):
        src_block_ids = np.array([10, 10, 10, 10], dtype=np.int64)
        dst_block_ids = np.array([20, 21, 22, 23], dtype=np.int64)
        total_blocks = 1
        tpb = 32
        src_start = (total_blocks - Sender._beam0_block_count(src_block_ids, total_blocks, 4)) * tpb
        dst_start = (total_blocks - Sender._beam0_block_count(dst_block_ids, total_blocks, 4)) * tpb

        src, dst = Sender._align_kv_blocks(
            src_block_ids,
            dst_block_ids,
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=tpb,
        )

        np.testing.assert_array_equal(src, [10, 10, 10, 10])
        np.testing.assert_array_equal(dst, [20, 21, 22, 23])

    def test_trim_single_block_prompt_preserves_packed_beam_tails(self):
        block_ids = np.array([10, 11, 12, 13], dtype=np.int64)

        trimmed = KvCacheTransceiverV2._trim_packed_beam_block_ids(
            block_ids,
            beam_width=4,
            total_blocks=1,
            expected_valid=1,
            cache_skip=0,
        )

        np.testing.assert_array_equal(trimmed, [10, 11, 12, 13])

    def test_trim_long_prompt_preserves_valid_beam0_and_tails(self):
        block_ids = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)

        trimmed = KvCacheTransceiverV2._trim_packed_beam_block_ids(
            block_ids,
            beam_width=4,
            total_blocks=3,
            expected_valid=3,
            cache_skip=0,
        )

        np.testing.assert_array_equal(trimmed, [10, 11, 12, 13, 14, 15])

    def test_trim_swa_prefix_preserves_packed_beam_tails(self):
        block_ids = np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.int64)

        trimmed = KvCacheTransceiverV2._trim_packed_beam_block_ids(
            block_ids,
            beam_width=4,
            total_blocks=4,
            expected_valid=2,
            cache_skip=0,
        )

        np.testing.assert_array_equal(trimmed, [12, 13, 14, 15, 16])

    def test_cache_skip_drops_tail_blocks_when_beam0_fully_cached(self):
        block_ids = np.array([10, 11, 12, 13], dtype=np.int64)

        trimmed = KvCacheTransceiverV2._trim_packed_beam_block_ids(
            block_ids,
            beam_width=4,
            total_blocks=1,
            expected_valid=1,
            cache_skip=1,
        )

        assert trimmed.size == 0


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
# _create_kv_slice: default TokenRange spans prompt_len + num_extra_kv_tokens
# so transferred KV matches what resize_context / _get_context_bytes allocate.
# ---------------------------------------------------------------------------


def _build_transceiver_for_kv_slice(num_extra_kv_tokens: int, prompt_len: int):
    """Stub a KvCacheTransceiverV2 so _create_kv_slice runs without dist setup.

    Wires only the attributes the method touches:
      - reuse adapter: tokens_per_block, per-layer-group cached count, block ids
      - page table:    layer groups
      - cache manager: num_extra_kv_tokens (read in this code path)
    """
    tokens_per_block = 8
    layer_group = AttentionLayerGroup(pool_group_idx=0, kv_head_num_per_rank=1)
    total_blocks = (prompt_len + num_extra_kv_tokens + tokens_per_block - 1) // tokens_per_block
    block_ids = np.arange(total_blocks, dtype=np.int64)

    reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_cached_token_count_per_layer_group=lambda req, layer_groups: [0] * len(layer_groups),
        get_block_ids=lambda req, idx, lg: block_ids,
    )
    page_table = SimpleNamespace(layer_groups=[layer_group])
    cache_manager = SimpleNamespace(num_extra_kv_tokens=num_extra_kv_tokens)

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._reuse_adapter = reuse_adapter
    transceiver._page_table = page_table
    transceiver._kv_cache_manager = cache_manager

    req = SimpleNamespace(
        prompt_len=prompt_len,
        py_request_id=0,
        py_beam_width=1,  # beam search disabled; _trim_packed_beam_block_ids is pass-through
        is_generation_only_request=lambda: False,
    )
    return transceiver, req


class TestCreateKvSliceTokenRange:
    """Default TokenRange built by _create_kv_slice must align with KV-cache allocation.

    KV cache allocation in resize_context (V2) and prepare_resources (V1) reserves
    prompt_len + num_extra_kv_tokens slots whenever speculative decoding (e.g.
    EAGLE3, MTP) consumes extra KV positions per request. The transferred token
    range must cover the same span, otherwise the receiver under-receives KV.
    """

    def test_includes_num_extra_kv_tokens(self):
        prompt_len = 17
        num_extra_kv_tokens = 7
        transceiver, req = _build_transceiver_for_kv_slice(num_extra_kv_tokens, prompt_len)

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.token_range is not None
        assert (kv_slice.token_range.start, kv_slice.token_range.end) == (
            0,
            prompt_len + num_extra_kv_tokens,
        )

    def test_defaults_to_prompt_len_when_no_extra(self):
        prompt_len = 17
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0, prompt_len=prompt_len
        )

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.token_range is not None
        assert (kv_slice.token_range.start, kv_slice.token_range.end) == (0, prompt_len)

    def test_respects_explicit_token_range(self):
        prompt_len = 17
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=7, prompt_len=prompt_len
        )
        explicit = TokenRange(start=0, end=8)

        kv_slice = transceiver._create_kv_slice(req, token_range=explicit)

        assert kv_slice.token_range is explicit


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


class _FakeSamplingConfig:
    def __init__(self, beam_width: int):
        self.beam_width = beam_width


def _lg(window=None):
    return AttentionLayerGroup(
        pool_group_idx=0,
        sliding_window_size=window,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=0)],
    )


class TestAdapterPerLayerGroup:
    """Per-layer cached prefix: adapter reports only the reuse-hit scalar."""

    TPB = 8

    def test_reuse_disabled(self):
        ad = _StubAdapter(scalar=128, tpb=self.TPB, enabled=False)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg(window=64)])
        assert out == [0, 0]

    def test_zero_scalar(self):
        # No reuse hit: every group reports 0 — SWA stale handling is the
        # transfer call site's concern, not the adapter's.
        ad = _StubAdapter(scalar=0, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg(window=64)])
        assert out == [0, 0]

    def test_full_attn_passthrough(self):
        ad = _StubAdapter(scalar=64, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg()])
        assert out == [64, 64]

    def test_swa_passthrough_above_stale(self):
        # SWA layer: adapter passes scalar through unchanged regardless of stale_end.
        ad = _StubAdapter(scalar=24, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [24]

    def test_swa_passthrough_below_stale(self):
        # scalar=8 is below stale_end*tpb=16; adapter still returns the raw
        # scalar — the call site reconciles with stale_end via max(0, ...).
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [8]

    def test_mixed_groups(self):
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(
            _FakeReq(32), [_lg(), _lg(window=16), _lg(window=32)]
        )
        # All groups see the same reuse-hit scalar.
        assert out == [8, 8, 8]


# ---------------------------------------------------------------------------
# _create_kv_slice SWA block trim: window-trim + cache-skip via per-layer cached.
# ---------------------------------------------------------------------------


def _swa_trim(block_ids, prompt_len, tpb, window_size, cached_tokens, is_gen_only=True):
    """Replicate the SWA branch of KvCacheTransceiverV2._create_kv_slice.

    Inputs:
      block_ids: list possibly containing stale entries (V1 pre-eviction view).
      cached_tokens: reuse-hit prefix reported by the adapter (token-aligned).
      is_gen_only: True mirrors the gen-side path; False mirrors the ctx-side
        path where ``cached_per_lg`` is synthetically 0.
    """
    block_ids = np.array(block_ids, dtype=np.int64)
    total_blocks = (prompt_len + tpb - 1) // tpb
    stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
    expected_valid = max(0, total_blocks - stale_end)
    if block_ids.size > expected_valid:
        block_ids = (
            block_ids[-expected_valid:] if expected_valid > 0 else np.array([], dtype=np.int64)
        )
    # Ctx side bypasses adapter (cached=0); gen side uses adapter scalar.
    cached_lg = cached_tokens if is_gen_only else 0
    # Reuse-hit blocks beyond the already-pruned stale region.
    cache_skip = max(0, cached_lg // tpb - stale_end)
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

    def test_window_offset_skip_subtracts_stale(self):
        # window=24 → stale_end=1; scalar=16 (2 blocks) → cache_skip=2-1=1.
        # Naive block_ids[scalar//tpb:] would skip 2 from a 3-block list and return 1 block.
        out = _swa_trim([10, 11, 12], prompt_len=32, tpb=8, window_size=24, cached_tokens=16)
        np.testing.assert_array_equal(out, [11, 12])

    def test_window_covers_all_no_stale(self):
        # window=prompt_len → stale_end=0; behaves like full-attn.
        out = _swa_trim([10, 11, 12, 13], prompt_len=32, tpb=8, window_size=32, cached_tokens=8)
        np.testing.assert_array_equal(out, [11, 12, 13])

    def test_v1_pre_eviction_includes_stale(self):
        # Pre-eviction list has all 4 blocks; window-trim keeps last expected_valid=2.
        out = _swa_trim([10, 11, 12, 13], self.PROMPT_LEN, self.TPB, self.WINDOW, 0)
        np.testing.assert_array_equal(out, [12, 13])

    def test_ctx_side_no_adapter_no_skip(self):
        # Ctx-side path: adapter not invoked, cached_per_lg synthetically 0.
        # cache_skip = max(0, 0 - stale_end) = 0 — full valid window is sent.
        out = _swa_trim([20, 21], self.PROMPT_LEN, self.TPB, self.WINDOW, 0, is_gen_only=False)
        np.testing.assert_array_equal(out, [20, 21])

    def test_ctx_side_v1_pre_eviction(self):
        # Ctx-side path with V1 pre-eviction list: window-trim still drops stale
        # blocks, cache_skip stays 0 so trimmed window is sent in full.
        out = _swa_trim(
            [10, 11, 12, 13], self.PROMPT_LEN, self.TPB, self.WINDOW, 0, is_gen_only=False
        )
        np.testing.assert_array_equal(out, [12, 13])

    def test_gen_side_reuse_inside_stale_no_skip(self):
        # gen side with reuse-hit fully inside the stale region: cache_skip = 0.
        # Regression for SWA + reuse-hit < stale_end*tpb (no adapter clamp).
        out = _swa_trim([20, 21], self.PROMPT_LEN, self.TPB, self.WINDOW, 8, is_gen_only=True)
        np.testing.assert_array_equal(out, [20, 21])


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


# ---------------------------------------------------------------------------
# KvCacheTransceiverV2 context-manager (__enter__/__exit__) + shutdown idempotency. (#14137)
# ---------------------------------------------------------------------------
class TestTransceiverContextManager:
    @staticmethod
    def _tc():
        # Bypass the heavy __init__ (cuda device, TransferWorker, dist broadcasts).
        tc = object.__new__(KvCacheTransceiverV2)
        tc._send_sessions = {}
        tc._recv_sessions = {}
        tc._send_reqs = {}
        tc._recv_reqs = {}
        tc._transfer_worker = MagicMock()
        return tc

    def test_enter_returns_self(self):
        tc = self._tc()
        with tc as ctx:
            assert ctx is tc

    def test_exit_calls_shutdown(self):
        tc = self._tc()
        with tc:
            pass
        tc._transfer_worker.shutdown.assert_called_once()
        assert tc._shutdown is True

    def test_exit_calls_shutdown_on_exception(self):
        tc = self._tc()
        with pytest.raises(RuntimeError, match="boom"):
            with tc:
                raise RuntimeError("boom")
        # __exit__ still ran shutdown despite the in-block exception.
        tc._transfer_worker.shutdown.assert_called_once()
        assert tc._shutdown is True

    def test_shutdown_is_idempotent(self):
        tc = self._tc()
        tc.shutdown()
        tc.shutdown()  # second call short-circuits on the _shutdown guard.
        tc._transfer_worker.shutdown.assert_called_once()
