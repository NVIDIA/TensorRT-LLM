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
"""Tests for adapter transfer spans, anchor trims, and anchored alignment."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    _CacheReuseAdapterV1,
    _CacheReuseAdapterV2,
    split_packed_beam_block_ids,
)
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, LocalLayer
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

pytestmark = pytest.mark.cpu_only


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


class _FakeReq:
    def __init__(self, prompt_len: int, beam_width: int = 1, request_id: int = 0):
        self.prompt_len = prompt_len
        self.py_beam_width = beam_width
        self.py_request_id = request_id


class _FakeSamplingConfig:
    def __init__(self, beam_width: int):
        self.beam_width = beam_width


class _FakeV1Mgr:
    """C++-manager stand-in for _CacheReuseAdapterV1.get_transfer_span.

    Pool translation is identity but records what it was asked to translate,
    so tests can assert dangling (evicted) or scratch block IDs never reach
    the pool-pointer arithmetic.
    """

    enable_block_reuse = True

    def __init__(
        self,
        block_ids,
        *,
        tokens_per_block: int = 8,
        num_extra_kv_tokens: int = 0,
        front_blocks_removed: int = 0,
        cp_size: int = 1,
    ):
        self.tokens_per_block = tokens_per_block
        self.num_extra_kv_tokens = num_extra_kv_tokens
        self.mapping = SimpleNamespace(cp_size=cp_size)
        self._block_ids = list(block_ids)
        self._front_blocks_removed = front_blocks_removed
        self.requested_beam_width = None
        self.translated_ids = None
        self.translated_window = None

    def get_batch_cache_indices(self, request_ids, layer_idx=None, beam_width=1):
        self.requested_beam_width = beam_width
        return [list(self._block_ids)]

    def get_num_front_blocks_removed(self, request_id, window_size):
        return self._front_blocks_removed

    def get_memory_pool_block_indices(self, block_ids, window_size):
        self.translated_ids = list(block_ids)
        self.translated_window = window_size
        return block_ids


def _lg(window=None):
    return AttentionLayerGroup(
        pool_group_idx=0,
        sliding_window_size=window,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=0)],
    )


class TestPackedBeamBlockLayout:
    """Verify beam search block IDs stay 1-D with only final tail blocks appended."""

    def test_v1_adapter_uses_request_py_beam_width(self):
        req = _FakeReq(prompt_len=7, beam_width=4, request_id=1)
        req.sampling_config = _FakeSamplingConfig(beam_width=1)
        mgr = _FakeV1Mgr([10, 11, 12, 13], tokens_per_block=32)

        pages, anchor = _CacheReuseAdapterV1(mgr).get_transfer_span(req, 0, _lg(window=512))

        assert mgr.requested_beam_width == 4
        assert mgr.translated_window == 512
        assert anchor == 0
        np.testing.assert_array_equal(pages, [10, 11, 12, 13])

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

    def test_split_single_block_prompt_separates_beam_tails(self):
        beam0, tails = split_packed_beam_block_ids(
            np.array([10, 11, 12, 13], dtype=np.int64), beam_width=4, beam0_len=1
        )

        np.testing.assert_array_equal(beam0, [10])
        np.testing.assert_array_equal(tails, [11, 12, 13])

    def test_split_long_prompt_separates_beam_tails(self):
        beam0, tails = split_packed_beam_block_ids(
            np.array([10, 11, 12, 13, 14, 15], dtype=np.int64), beam_width=4, beam0_len=3
        )

        np.testing.assert_array_equal(beam0, [10, 11, 12])
        np.testing.assert_array_equal(tails, [13, 14, 15])

    def test_split_with_fewer_shared_tails_than_beam_width(self):
        # The packer appends only UNSHARED final blocks, so a beam_width=4 list
        # can carry fewer than 3 tails. The tail count is size - beam0_len,
        # never a beam_width guess (which would misclassify beam-0 blocks
        # as tails here).
        beam0, tails = split_packed_beam_block_ids(
            np.array([10, 11, 12, 14, 15], dtype=np.int64), beam_width=4, beam0_len=3
        )

        np.testing.assert_array_equal(beam0, [10, 11, 12])
        np.testing.assert_array_equal(tails, [14, 15])

    def test_split_anchored_span_beam0_len(self):
        # beam0_len is span_end - first_ordinal: a 4-entry list whose beam-0
        # span holds 1 block (prompt 3 blocks, anchored at 2) has 3 tails.
        beam0, tails = split_packed_beam_block_ids(
            np.array([12, 13, 14, 15], dtype=np.int64), beam_width=4, beam0_len=1
        )

        np.testing.assert_array_equal(beam0, [12])
        np.testing.assert_array_equal(tails, [13, 14, 15])

    def test_split_within_beam0_len_has_no_tails(self):
        beam0, tails = split_packed_beam_block_ids(
            np.array([10, 11], dtype=np.int64), beam_width=4, beam0_len=3
        )

        np.testing.assert_array_equal(beam0, [10, 11])
        assert tails.size == 0

    def test_split_single_beam_passthrough(self):
        beam0, tails = split_packed_beam_block_ids(
            np.array([10, 11, 12, 13], dtype=np.int64), beam_width=1, beam0_len=1
        )

        np.testing.assert_array_equal(beam0, [10, 11, 12, 13])
        assert tails.size == 0

    def test_align_packed_single_block_prompt_keeps_all_beam_blocks(self):
        # Both spans anchored at 0 with a single beam-0 block each; tails ride
        # behind the intersected beam-0 spans (see _build_kv_write_meta).
        src_block_ids = np.array([10, 10, 10, 10], dtype=np.int64)
        dst_block_ids = np.array([20, 21, 22, 23], dtype=np.int64)
        tpb = 32
        src_beam0, src_tail = split_packed_beam_block_ids(src_block_ids, beam_width=4, beam0_len=1)
        dst_beam0, dst_tail = split_packed_beam_block_ids(dst_block_ids, beam_width=4, beam0_len=1)

        src, dst = Sender._align_kv_blocks(
            src_beam0,
            dst_beam0,
            src_token_start=0,
            dst_token_start=0,
            tokens_per_block=tpb,
        )
        src = np.concatenate([src, src_tail])
        dst = np.concatenate([dst, dst_tail])

        np.testing.assert_array_equal(src, [10, 10, 10, 10])
        np.testing.assert_array_equal(dst, [20, 21, 22, 23])


# ---------------------------------------------------------------------------
# Windowed layer group where only the generation side runs speculative
# decoding: the receiver keeps a larger window, so its span starts earlier.
# Anchors resolve the asymmetry by plain interval intersection — the deleted
# _trim_receiver_window_head count-based head trim has no successor mechanism
# beyond this.
# ---------------------------------------------------------------------------


class TestReceiverLargerWindowViaAnchors:
    TPB = 128

    def _align(self, src, src_anchor, dst, dst_anchor):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_anchor * self.TPB,
            dst_token_start=dst_anchor * self.TPB,
            tokens_per_block=self.TPB,
        )

    def test_receiver_extra_head_block_gets_no_source(self):
        src, dst = self._align([10], src_anchor=1224, dst=[20, 21], dst_anchor=1223)

        np.testing.assert_array_equal(src, [10])
        np.testing.assert_array_equal(dst, [21])

    def test_last_prompt_block_maps_exactly(self):
        # Regression (formerly against _trim_receiver_window_head): dropping the
        # receiver's tail instead of its head paired dst block 20 with src block
        # 10 one position early, so the last prompt block was never written.
        # With anchors the pairing is by equal block ordinal by construction.
        src, dst = self._align([10], src_anchor=1224, dst=[20, 21], dst_anchor=1223)

        assert (dst.tolist(), src.tolist()) == ([21], [10])

    def test_equal_windows_are_untouched(self):
        src, dst = self._align([10, 11], src_anchor=1223, dst=[20, 21], dst_anchor=1223)

        np.testing.assert_array_equal(src, [10, 11])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_smaller_receiver_resolves_via_its_own_anchor(self):
        # Generation prefix-cache reuse: dst declares a later-starting span.
        src, dst = self._align([10, 11, 12], src_anchor=0, dst=[20], dst_anchor=2)

        np.testing.assert_array_equal(src, [12])
        np.testing.assert_array_equal(dst, [20])


# ---------------------------------------------------------------------------
# _CacheReuseAdapterV1.get_transfer_span: manager-truth span and anchor.
# ---------------------------------------------------------------------------


class TestV1AdapterTransferSpan:
    TPB = 8

    def _span(self, mgr, *, prompt_len, beam_width=1, window=None):
        req = _FakeReq(prompt_len=prompt_len, beam_width=beam_width)
        window = window if window is not None else 1 << 30  # full attention
        return _CacheReuseAdapterV1(mgr).get_transfer_span(req, 0, _lg(window=window))

    def test_span_covers_ceil_prompt_blocks(self):
        # prompt 17 + extra 7 = 24 tokens = 3 blocks allocated; prompt needs
        # ceil(17/8) = 3, so nothing is scratch.
        mgr = _FakeV1Mgr([0, 1, 2], num_extra_kv_tokens=7)

        pages, anchor = self._span(mgr, prompt_len=17)

        np.testing.assert_array_equal(pages, [0, 1, 2])
        assert anchor == 0

    def test_extra_tokens_crossing_block_boundary_are_stripped(self):
        # prompt 16 = 2 blocks; extra 7 pushes allocation to 3 blocks. The
        # scratch block holds no prompt KV and must not be transferred.
        mgr = _FakeV1Mgr([0, 1, 2], num_extra_kv_tokens=7)

        pages, anchor = self._span(mgr, prompt_len=16)

        np.testing.assert_array_equal(pages, [0, 1])
        assert anchor == 0
        assert mgr.translated_ids == [0, 1]

    def test_no_extra_defaults_to_prompt_len(self):
        mgr = _FakeV1Mgr([0, 1, 2])

        pages, anchor = self._span(mgr, prompt_len=17)

        np.testing.assert_array_equal(pages, [0, 1, 2])
        assert anchor == 0

    def test_draft_over_allocation_is_capped(self):
        # Draft-token allocation can extend the list past the allocated bound.
        mgr = _FakeV1Mgr([100, 101, 102, 103, 104])

        pages, anchor = self._span(mgr, prompt_len=32)

        np.testing.assert_array_equal(pages, [100, 101, 102, 103])
        assert anchor == 0

    def test_scratch_requires_single_beam(self):
        mgr = _FakeV1Mgr([100, 101, 102, 103, 104], num_extra_kv_tokens=2)

        with pytest.raises(ValueError, match="speculative scratch blocks require beam_width == 1"):
            self._span(mgr, prompt_len=32, beam_width=4)

    def test_beam_tails_split_exactly_at_the_allocated_bound(self):
        # 4 allocated beam-0 blocks + 3 packed tails: the tail count comes from
        # size - allocated, so all tails survive.
        mgr = _FakeV1Mgr([100, 101, 102, 103, 200, 201, 202])

        pages, anchor = self._span(mgr, prompt_len=32, beam_width=4)

        np.testing.assert_array_equal(pages, [100, 101, 102, 103, 200, 201, 202])
        assert anchor == 0

    def test_fewer_shared_tails_than_beam_width_split_correctly(self):
        # The exact bug the beam0_len rename fixed: with beam_width=4 but only
        # 2 UNSHARED tail blocks packed, the old beam_width-1 guess would have
        # misclassified beam-0 block 103 as a tail.
        mgr = _FakeV1Mgr([100, 101, 102, 103, 200, 201])

        pages, anchor = self._span(mgr, prompt_len=32, beam_width=4)

        np.testing.assert_array_equal(pages, [100, 101, 102, 103, 200, 201])
        assert anchor == 0

    def test_scratch_crossing_boundary_with_evicted_front_blocks(self):
        """PR-17619 regression: boundary-crossing scratch with evicted fronts.

        prompt 32 = 4 blocks; extra 8 tokens allocate a 5th (scratch) block
        holding garbage; the manager already detached the first 2 blocks, whose
        IDs are dangling. The span must be honestly anchored at 2, exclude both
        the dangling head and the garbage scratch tail, and never hand either
        to pool translation.
        """
        mgr = _FakeV1Mgr(
            [100, 101, 102, 103, 104],  # 104 = scratch garbage; 100, 101 = dangling
            num_extra_kv_tokens=8,
            front_blocks_removed=2,
        )

        pages, anchor = self._span(mgr, prompt_len=32, window=16)

        np.testing.assert_array_equal(pages, [102, 103])
        assert anchor == 2
        # Dangling and scratch IDs are stripped BEFORE pool translation: a
        # detached block may have been reused or offloaded by now.
        assert mgr.translated_ids == [102, 103]

    def test_eviction_counter_exceeding_the_stale_formula_is_honored(self):
        """The eviction counter is authoritative over the stale formula.

        Extra/draft tokens can push eviction one block past the
        (prompt_len + 1 - window) formula; the counter wins and the
        dangling IDs are stripped before pool translation.
        """
        tpb = 128
        prompt_len = 1150  # 9 blocks
        mgr = _FakeV1Mgr(
            list(range(200, 210)),  # 9 prompt blocks + 1 scratch block
            tokens_per_block=tpb,
            num_extra_kv_tokens=5,
            front_blocks_removed=8,  # formula would give (1150+1-133)//128 = 7
        )
        req = _FakeReq(prompt_len=prompt_len)

        pages, anchor = _CacheReuseAdapterV1(mgr).get_transfer_span(req, 0, _lg(window=133))

        np.testing.assert_array_equal(pages, [208])
        assert anchor == 8
        assert mgr.translated_ids == [208]

    @pytest.mark.parametrize("prompt_len", (1150, 1151))
    def test_dspark_disagg_boundary_keeps_only_initialized_swa(self, prompt_len):
        # 9 prompt blocks + 1 scratch block; 7 fronts already evicted.
        tpb = 128
        mgr = _FakeV1Mgr(
            list(range(200, 210)),
            tokens_per_block=tpb,
            num_extra_kv_tokens=5,
            front_blocks_removed=7,
        )
        req = _FakeReq(prompt_len=prompt_len)

        pages, anchor = _CacheReuseAdapterV1(mgr).get_transfer_span(req, 0, _lg(window=133))

        np.testing.assert_array_equal(pages, [207, 208])
        assert anchor == 7

    def test_reconciliation_mismatch_raises(self):
        """Anchor + len(beam0) != ceil(prompt_len / tpb) refuses to transfer."""
        mgr = _FakeV1Mgr([100, 101, 102])  # 3 blocks for a 4-block prompt

        with pytest.raises(RuntimeError, match="refusing to transfer misaligned KV blocks"):
            self._span(mgr, prompt_len=32)

    def test_reconciliation_mismatch_after_eviction_raises(self):
        # Counter says 1 evicted but the surviving list is still one short.
        mgr = _FakeV1Mgr([100, 101, 102], front_blocks_removed=1)

        with pytest.raises(RuntimeError, match="refusing to transfer misaligned KV blocks"):
            self._span(mgr, prompt_len=32)

    def test_everything_evicted_returns_empty_anchor_zero(self):
        mgr = _FakeV1Mgr([100, 101, 102, 103], front_blocks_removed=4)

        pages, anchor = self._span(mgr, prompt_len=32)

        assert pages.size == 0
        assert anchor == 0

    def test_empty_manager_list_returns_empty_anchor_zero(self):
        mgr = _FakeV1Mgr([])

        pages, anchor = self._span(mgr, prompt_len=32)

        assert pages.size == 0
        assert anchor == 0

    def test_helix_cp_returns_local_ordinals_anchor_zero(self):
        # Helix lists are strided local subsets; global eviction/scratch
        # bookkeeping does not apply and the anchor stays 0.
        mgr = _FakeV1Mgr([100, 102], num_extra_kv_tokens=8, front_blocks_removed=2, cp_size=2)

        pages, anchor = self._span(mgr, prompt_len=32)

        np.testing.assert_array_equal(pages, [100, 102])
        assert anchor == 0


# ---------------------------------------------------------------------------
# _CacheReuseAdapterV2.get_transfer_span: anchor derived from unbacked
# ordinals (valid_only=False keeps index == ordinal).
# ---------------------------------------------------------------------------


class _FakeKvCacheV2:
    def __init__(self, pages, history_length, scratch_range=None, num_sink_blocks=0):
        self._pages = list(pages)
        self.history_length = history_length
        self._scratch_range = scratch_range
        # The sink guard reads the manager-internal life cycle (no public API).
        self.manager = SimpleNamespace(
            _life_cycles=[SimpleNamespace(num_sink_blocks=num_sink_blocks)]
        )

    def get_aggregated_page_indices(self, group_idx, valid_only=True):
        # index == ordinal only holds for valid_only=False; the adapter must
        # never ask for the compacted view.
        assert valid_only is False
        return list(self._pages)

    def get_scratch_desc(self, group_idx):
        if self._scratch_range is None:
            return None
        return SimpleNamespace(range=self._scratch_range)


class _FakeV2Mgr:
    enable_block_reuse = False

    def __init__(self, kv_cache, tokens_per_block=8, cp_size=1):
        self.tokens_per_block = tokens_per_block
        self.kv_cache_map = {0: kv_cache}
        self.mapping = SimpleNamespace(cp_size=cp_size)


class TestV2AdapterTransferSpan:
    TPB = 8
    PROMPT = 32  # 4 blocks

    def _span(self, kv_cache, window=None, prompt_len=PROMPT, cp_size=1):
        adapter = _CacheReuseAdapterV2(
            _FakeV2Mgr(kv_cache, tokens_per_block=self.TPB, cp_size=cp_size)
        )
        req = _FakeReq(prompt_len=prompt_len)
        return adapter.get_transfer_span(req, 0, _lg(window=window))

    def test_fully_backed_prompt_anchor_zero(self):
        kv = _FakeKvCacheV2([10, 11, 12, 13], history_length=self.PROMPT)

        pages, anchor = self._span(kv)

        np.testing.assert_array_equal(pages, [10, 11, 12, 13])
        assert anchor == 0

    def test_swa_stale_holes_anchor_the_backed_run(self):
        # window 16 → stale_end = (32 + 1 - 16) // 8 = 2: ordinals 0-1 unbacked.
        kv = _FakeKvCacheV2([BAD_PAGE_INDEX, BAD_PAGE_INDEX, 12, 13], history_length=self.PROMPT)

        pages, anchor = self._span(kv, window=16)

        np.testing.assert_array_equal(pages, [12, 13])
        assert anchor == 2

    def test_scratch_hole_is_explained(self):
        # Ordinal 0 was written to a rotating shared scratch slot.
        kv = _FakeKvCacheV2(
            [BAD_PAGE_INDEX, 11, 12, 13], history_length=self.PROMPT, scratch_range=(0, 1)
        )

        pages, anchor = self._span(kv)

        np.testing.assert_array_equal(pages, [11, 12, 13])
        assert anchor == 1

    def test_unexplained_hole_raises(self):
        """An unexplained unbacked ordinal means the span cannot be trusted."""
        kv = _FakeKvCacheV2([10, BAD_PAGE_INDEX, 12, 13], history_length=self.PROMPT)

        with pytest.raises(RuntimeError, match="refusing to transfer misaligned KV blocks"):
            self._span(kv)

    def test_hole_past_swa_stale_end_raises(self):
        # window 16 explains ordinals < 2 only; a hole at ordinal 2 is a bug.
        kv = _FakeKvCacheV2(
            [BAD_PAGE_INDEX, BAD_PAGE_INDEX, BAD_PAGE_INDEX, 13], history_length=self.PROMPT
        )

        with pytest.raises(RuntimeError, match="unbacked"):
            self._span(kv, window=16)

    def test_scratch_tail_past_prompt_is_trimmed(self):
        # A 5th page past the prompt (speculative scratch) is never transferred.
        kv = _FakeKvCacheV2([10, 11, 12, 13, 99], history_length=self.PROMPT)

        pages, anchor = self._span(kv)

        np.testing.assert_array_equal(pages, [10, 11, 12, 13])
        assert 99 not in pages
        assert anchor == 0

    def test_all_unbacked_returns_empty_anchor_zero(self):
        kv = _FakeKvCacheV2([BAD_PAGE_INDEX] * 4, history_length=self.PROMPT)

        pages, anchor = self._span(kv, window=16)

        assert pages.size == 0
        assert anchor == 0

    def test_hole_at_last_prompt_block_returns_empty(self):
        # The contiguous backed run must end at the last prompt block.
        kv = _FakeKvCacheV2([10, 11, 12, BAD_PAGE_INDEX], history_length=self.PROMPT)

        pages, anchor = self._span(kv)

        assert pages.size == 0
        assert anchor == 0

    def test_sink_blocks_with_holes_raise(self):
        # A single anchor cannot represent a backed sink prefix followed by an
        # unbacked hole; a sink-configured life cycle must fail loud.
        kv = _FakeKvCacheV2(
            [BAD_PAGE_INDEX, BAD_PAGE_INDEX, 12, 13],
            history_length=self.PROMPT,
            num_sink_blocks=1,
        )

        with pytest.raises(RuntimeError, match="token sinks"):
            self._span(kv, window=16)

    def test_sink_blocks_without_holes_pass(self):
        # Sinks only conflict with the anchor when there are unbacked ordinals.
        kv = _FakeKvCacheV2([10, 11, 12, 13], history_length=self.PROMPT, num_sink_blocks=1)

        pages, anchor = self._span(kv)

        np.testing.assert_array_equal(pages, [10, 11, 12, 13])
        assert anchor == 0

    def test_helix_cp_returns_local_list_anchor_zero(self):
        # Helix lists are strided local subsets; the prompt cap and global
        # stale/scratch bookkeeping do not apply and the anchor stays 0.
        kv = _FakeKvCacheV2([10, 12, 14, 16, 18], history_length=self.PROMPT)

        pages, anchor = self._span(kv, window=16, cp_size=2)

        np.testing.assert_array_equal(pages, [10, 12, 14, 16, 18])
        assert anchor == 0

    def test_helix_cp_with_unbacked_ordinals_raises(self):
        # Local block lists carry no global stale/scratch bookkeeping, so an
        # unbacked ordinal cannot be anchored under helix.
        kv = _FakeKvCacheV2([BAD_PAGE_INDEX, 12, 14], history_length=self.PROMPT)

        with pytest.raises(RuntimeError, match="helix"):
            self._span(kv, window=16, cp_size=2)


# ---------------------------------------------------------------------------
# _create_kv_slice: transfer-policy head trims (SWA bandwidth skip, gen-side
# reuse skip) are explicit head-slices that advance the anchor.
# ---------------------------------------------------------------------------


def _build_transceiver_for_kv_slice(
    *,
    prompt_len: int,
    block_ids,
    span_anchor: int = 0,
    tokens_per_block: int = 8,
    sliding_window_size=None,
    cached_tokens: int = 0,
    is_generation_only: bool = False,
    beam_width: int = 1,
    cp_size: int = 1,
):
    """Stub a KvCacheTransceiverV2 so _create_kv_slice runs without dist setup.

    Wires only the attributes the method touches:
      - reuse adapter: tokens_per_block, per-layer-group cached count, and the
        anchored (pages, first_block_ordinal) span
      - page table:    layer groups
      - mapping:       cp_size (helix skips global-position trims)
    """
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=sliding_window_size,
    )
    block_ids = np.asarray(block_ids, dtype=np.int64)

    reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_cached_token_count_per_layer_group=lambda req, layer_groups: [cached_tokens]
        * len(layer_groups),
        get_transfer_span=lambda req, idx, lg: (block_ids, span_anchor),
    )

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._reuse_adapter = reuse_adapter
    transceiver._page_table = SimpleNamespace(layer_groups=[layer_group])
    transceiver._kv_cache_manager = SimpleNamespace()
    transceiver._mapping = SimpleNamespace(cp_size=cp_size)

    req = SimpleNamespace(
        prompt_len=prompt_len,
        py_request_id=0,
        py_beam_width=beam_width,
        is_generation_only_request=lambda: is_generation_only,
    )
    return transceiver, req


class TestCreateKvSliceAnchorTrims:
    """Anchor-trim tests: tpb=8, prompt_len=32 (4 blocks), window start 2."""

    def _slice(self, **kwargs):
        include = kwargs.pop("include_window_groups", True)
        transceiver, req = _build_transceiver_for_kv_slice(**kwargs)
        return transceiver._create_kv_slice(req, include_window_groups=include)

    def test_full_attention_span_passes_through(self):
        kv_slice = self._slice(prompt_len=32, block_ids=[100, 101, 102, 103])

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, 102, 103])
        assert kv_slice.first_ordinals == [0]
        assert kv_slice.is_last_slice is True

    def test_swa_policy_trim_advances_anchor(self):
        # Pre-window blocks are a bandwidth-only skip; the head-slice advances
        # the anchor so the remaining blocks stay honestly positioned.
        kv_slice = self._slice(
            prompt_len=32, block_ids=[100, 101, 102, 103], sliding_window_size=16
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [102, 103])
        assert kv_slice.first_ordinals == [2]

    def test_swa_trim_respects_adapter_anchor(self):
        # V2-style span: the adapter already excluded the stale prefix.
        kv_slice = self._slice(
            prompt_len=32, block_ids=[102, 103], span_anchor=2, sliding_window_size=16
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [102, 103])
        assert kv_slice.first_ordinals == [2]

    def test_adapter_anchor_beyond_policy_target_is_kept(self):
        # The manager evicted more than the policy would trim; the manager wins.
        kv_slice = self._slice(
            prompt_len=32, block_ids=[103], span_anchor=3, sliding_window_size=16
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [103])
        assert kv_slice.first_ordinals == [3]

    def test_gen_reuse_skip_advances_anchor(self):
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            cached_tokens=16,
            is_generation_only=True,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [102, 103])
        assert kv_slice.first_ordinals == [2]

    def test_gen_reuse_and_swa_take_the_max(self):
        # window start 2 vs cached 3 blocks → skip to 3.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            sliding_window_size=16,
            cached_tokens=24,
            is_generation_only=True,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [103])
        assert kv_slice.first_ordinals == [3]

    def test_gen_reuse_inside_stale_region_adds_no_skip(self):
        # Regression: reuse-hit (1 block) below the window start (2) must not
        # skip anything beyond the SWA trim itself.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[102, 103],
            span_anchor=2,
            sliding_window_size=16,
            cached_tokens=8,
            is_generation_only=True,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [102, 103])
        assert kv_slice.first_ordinals == [2]

    def test_ctx_side_ignores_cached_tokens(self):
        # get_cached_token_count_per_layer_group is a gen-side concern; the ctx
        # sender transfers its full window.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            sliding_window_size=16,
            cached_tokens=24,
            is_generation_only=False,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [102, 103])
        assert kv_slice.first_ordinals == [2]

    def test_window_offset_skip_subtracts_the_adapter_anchor(self):
        # window=24 → window start 1; cached 2 blocks → skip only 1 more block
        # from the anchored span, not cached//tpb blocks from its head.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[10, 11, 12],
            span_anchor=1,
            sliding_window_size=24,
            cached_tokens=16,
            is_generation_only=True,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [11, 12])
        assert kv_slice.first_ordinals == [2]

    def test_window_covering_prompt_behaves_like_full_attention(self):
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[10, 11, 12, 13],
            sliding_window_size=32,
            cached_tokens=8,
            is_generation_only=True,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [11, 12, 13])
        assert kv_slice.first_ordinals == [1]

    def test_skip_covering_the_span_yields_empty_anchor_zero(self):
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            cached_tokens=32,
            is_generation_only=True,
        )

        assert kv_slice.block_ids_per_layer_groups[0].size == 0
        assert kv_slice.first_ordinals == [0]

    def test_head_slice_preserves_packed_beam_tails(self):
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 200, 201, 202],
            sliding_window_size=16,
            beam_width=4,
        )

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0], [102, 103, 200, 201, 202]
        )
        assert kv_slice.first_ordinals == [2]

    def test_skip_covering_beam0_drops_tails_too(self):
        kv_slice = self._slice(
            prompt_len=8,  # 1 block
            block_ids=[10, 11, 12, 13],  # 1 beam-0 block + 3 tails
            beam_width=4,
            cached_tokens=8,
            is_generation_only=True,
        )

        assert kv_slice.block_ids_per_layer_groups[0].size == 0
        assert kv_slice.first_ordinals == [0]

    def test_helix_skips_global_position_trims(self):
        # Helix spans are strided local subsets; the SWA trim is a global-
        # ordinal concept and must not apply.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 102],
            sliding_window_size=16,
            cp_size=2,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 102])
        assert kv_slice.first_ordinals == [0]

    def test_deferred_window_group_is_empty_with_anchor_zero(self):
        # Pipelined senders defer SWA groups to the final chunk.
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            sliding_window_size=16,
            include_window_groups=False,
        )

        assert kv_slice.block_ids_per_layer_groups[0].size == 0
        assert kv_slice.first_ordinals == [0]

    def test_window_at_least_prompt_is_not_deferred(self):
        kv_slice = self._slice(
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            sliding_window_size=32,
            include_window_groups=False,
        )

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, 102, 103])
        assert kv_slice.first_ordinals == [0]

    def test_empty_span_stays_empty_with_anchor_zero(self):
        kv_slice = self._slice(prompt_len=32, block_ids=[])

        assert kv_slice.block_ids_per_layer_groups[0].size == 0
        assert kv_slice.first_ordinals == [0]


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

    def get_transfer_span(self, req, group_idx, lg):  # noqa: ARG002
        return np.array([], dtype=np.int64), 0

    def commit_blocks_for_reuse(self, req):  # noqa: ARG002
        pass


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
        # scalar — the call site reconciles with the window start via max().
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
# Sender anchored token starts: src/dst starts come straight from the
# per-group anchors (first_ordinal * tpb), replacing the deleted
# (total_blocks - len) * tpb suffix derivation and its SWA clamp.
# ---------------------------------------------------------------------------


class TestSenderAnchoredStarts:
    TPB = 8

    def _align(self, src, src_anchor, dst, dst_anchor):
        return Sender._align_kv_blocks(
            np.array(src, dtype=np.int64),
            np.array(dst, dtype=np.int64),
            src_token_start=src_anchor * self.TPB,
            dst_token_start=dst_anchor * self.TPB,
            tokens_per_block=self.TPB,
        )

    def test_full_prompt_no_cache(self):
        src, dst = self._align([10, 11, 12, 13], 0, [20, 21, 22, 23], 0)
        np.testing.assert_array_equal(src, [10, 11, 12, 13])
        np.testing.assert_array_equal(dst, [20, 21, 22, 23])

    def test_dst_cached_prefix(self):
        # dst reused 2 blocks → its span is anchored at 2; src head is trimmed.
        src, dst = self._align([10, 11, 12, 13], 0, [22, 23], 2)
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [22, 23])

    def test_src_cached_prefix(self):
        src, dst = self._align([12, 13], 2, [20, 21, 22, 23], 0)
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [22, 23])

    def test_swa_both_sides_anchored_at_window_start(self):
        src, dst = self._align([10, 11], 2, [20, 21], 2)
        np.testing.assert_array_equal(src, [10, 11])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_swa_asymmetric_anchors(self):
        # ctx trimmed window start + reuse skip → anchor 2; gen window only → 1.
        src, dst = self._align([10, 11], 2, [20, 21, 22], 1)
        np.testing.assert_array_equal(src, [10, 11])
        np.testing.assert_array_equal(dst, [21, 22])

    def test_chunk_anchored_at_its_start(self):
        # Non-final chunk covering blocks [0, 2) against an uncached receiver.
        src, dst = self._align([10, 11], 0, [20, 21, 22, 23], 0)
        np.testing.assert_array_equal(src, [10, 11])
        np.testing.assert_array_equal(dst, [20, 21])

    def test_chunk_entirely_within_receiver_cached_prefix(self):
        src, dst = self._align([10, 11], 0, [22, 23], 2)
        assert src.size == 0
        assert dst.size == 0


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
