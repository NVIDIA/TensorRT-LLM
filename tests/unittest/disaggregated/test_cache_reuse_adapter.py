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

from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    _CacheReuseAdapterV1,
    _CacheReuseAdapterV2,
)
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, LocalLayer
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

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


class TestPackedBeamBlockLayout:
    """Verify beam search block IDs stay 1-D with only final tail blocks appended."""

    def test_v1_adapter_uses_request_py_beam_width(self):
        class _FakeMgr:
            enable_block_reuse = True
            tokens_per_block = 32

            def __init__(self):
                self.beam_width = None
                self.pool_indices_window = None

            def get_batch_cache_indices(self, request_ids, layer_idx=None, beam_width=1):
                self.beam_width = beam_width
                return [[10, 11, 12, 13]]

            def get_memory_pool_block_indices(self, block_ids, window_size):
                # Identity translation: nothing offloaded, block_id == pool slot.
                self.pool_indices_window = window_size
                return block_ids

        req = _FakeReq(prompt_len=7)
        req.py_request_id = 1
        req.py_beam_width = 4
        req.sampling_config = _FakeSamplingConfig(beam_width=1)
        mgr = _FakeMgr()

        block_ids = _CacheReuseAdapterV1(mgr).get_block_ids(req, 0, _lg(window=512))

        assert mgr.beam_width == 4
        assert mgr.pool_indices_window == 512
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


# ---------------------------------------------------------------------------
# Windowed layer group where only the generation side runs speculative decoding.
# ---------------------------------------------------------------------------


class TestTrimReceiverWindowHead:
    """Sender._trim_receiver_window_head drops the receiver's extra head blocks.

    The receiver keeps a larger window when only it runs speculative decoding,
    so its suffix starts earlier. Both token starts are derived from list
    length, so the extra blocks must come off the head.
    """

    WINDOW = 128

    def test_extra_receiver_blocks_come_off_the_head(self):
        src_block_ids = np.array([10], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)

        trimmed = Sender._trim_receiver_window_head(
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW, beam_width=1
        )

        np.testing.assert_array_equal(trimmed, [21])

    def test_trimmed_receiver_maps_onto_the_last_prompt_block(self):
        # Regression: trimming the tail leaves [20], which _align_kv_blocks then
        # pairs with src block 10 -- one block early, so the last prompt block
        # is never written.
        src_block_ids = np.array([10], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)
        total_blocks = 1225
        tpb = 128

        dst_block_ids = Sender._trim_receiver_window_head(
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW, beam_width=1
        )
        src_start = (total_blocks - Sender._beam0_block_count(src_block_ids, total_blocks, 1)) * tpb
        dst_start = (total_blocks - Sender._beam0_block_count(dst_block_ids, total_blocks, 1)) * tpb

        src, dst = Sender._align_kv_blocks(
            src_block_ids,
            dst_block_ids,
            src_token_start=src_start,
            dst_token_start=dst_start,
            tokens_per_block=tpb,
        )

        np.testing.assert_array_equal(src, [10])
        np.testing.assert_array_equal(dst, [21])

    def test_equal_counts_are_untouched(self):
        src_block_ids = np.array([10, 11], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)

        trimmed = Sender._trim_receiver_window_head(
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW, beam_width=1
        )

        np.testing.assert_array_equal(trimmed, [20, 21])

    def test_smaller_receiver_is_untouched(self):
        # Generation prefix-cache reuse: handled downstream via dst_start.
        src_block_ids = np.array([10, 11, 12], dtype=np.int64)
        dst_block_ids = np.array([20], dtype=np.int64)

        trimmed = Sender._trim_receiver_window_head(
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW, beam_width=1
        )

        np.testing.assert_array_equal(trimmed, [20])

    def test_non_windowed_group_still_raises(self):
        src_block_ids = np.array([10], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)

        with pytest.raises(ValueError, match="block count mismatch"):
            Sender._trim_receiver_window_head(
                src_block_ids, dst_block_ids, peer_window_size=None, beam_width=1
            )

    def test_multi_beam_still_raises(self):
        src_block_ids = np.array([10], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)

        with pytest.raises(ValueError, match="block count mismatch"):
            Sender._trim_receiver_window_head(
                src_block_ids, dst_block_ids, peer_window_size=self.WINDOW, beam_width=4
            )


# ---------------------------------------------------------------------------
# _create_kv_slice: the block list spans prompt_len, excluding the extra KV
# slots speculative decoding reserves.
# ---------------------------------------------------------------------------


def _build_transceiver_for_kv_slice(
    num_extra_kv_tokens: int,
    prompt_len: int,
    *,
    tokens_per_block: int = 8,
    block_ids=None,
    sliding_window_size=None,
    cached_tokens: int = 0,
    is_generation_only: bool = False,
    beam_width: int = 1,
    beam_tails=None,
):
    """Stub a KvCacheTransceiverV2 so _create_kv_slice runs without dist setup.

    Wires only the attributes the method touches:
      - reuse adapter: tokens_per_block, per-layer-group cached count, block ids
      - page table:    layer groups
      - cache manager: num_extra_kv_tokens (read in this code path)

    For beam_width == 1, `block_ids` is beam-0's stale-stripped chain (the
    positional ordinals prepend -1 holes). For beam_width > 1, it is still
    beam-0's chain; `beam_tails` supplies the divergent per-beam final blocks
    carried after the window.
    """
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=sliding_window_size,
    )
    total_blocks = (prompt_len + num_extra_kv_tokens + tokens_per_block - 1) // tokens_per_block
    if block_ids is None:
        block_ids = np.arange(total_blocks, dtype=np.int64)
    else:
        block_ids = np.asarray(block_ids, dtype=np.int64)

    # The positional ordinal list the adapter hands the beam_width == 1 path:
    # the stale (out-of-window) front is reported as -1 holes, followed by the
    # valid window blocks, the speculative/scratch tail, and the ctx
    # first-token over-hang. `block_ids` here is the stale-stripped chain, so
    # prepend the -1 holes. (V1 masks its full chain to this shape; V2 reports
    # it directly via valid_only=False.)
    stale_end = 0
    if sliding_window_size is not None:
        stale_end = max(0, (prompt_len + 1 - sliding_window_size) // tokens_per_block)
    ordinals = np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids])
    tails = np.asarray([] if beam_tails is None else beam_tails, dtype=np.int64)

    reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_cached_token_count_per_layer_group=lambda req, layer_groups: [cached_tokens]
        * len(layer_groups),
        get_block_ids=lambda req, idx, lg: block_ids,
        get_block_ordinals=lambda req, idx, lg: ordinals,
        # beam_width > 1: beam-0's positional ordinals + the divergent tails.
        get_beam0_ordinals_and_tails=lambda req, idx, lg: (ordinals, tails),
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
        py_beam_width=beam_width,
        is_generation_only_request=is_generation_only,
        # num_extra_kv_tokens == max_draft_len - 1
        py_draft_tokens=[0 for _ in range(num_extra_kv_tokens + 1)],
    )
    return transceiver, req


class TestCreateKvSliceBlockSpan:
    """The block list must span prompt_len, not prompt_len + num_extra_kv_tokens.

    A monolithic slice carries no extent of its own: the sender's suffix
    arithmetic anchors on the session's prompt_len and assumes the list is the
    tail of ceil(prompt_len / tpb) blocks. An extra block would shift every
    per-layer token start.
    """

    def test_excludes_num_extra_kv_tokens(self):
        prompt_len = 17
        num_extra_kv_tokens = 7
        transceiver, req = _build_transceiver_for_kv_slice(num_extra_kv_tokens, prompt_len)
        tpb = transceiver._reuse_adapter.tokens_per_block

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == (prompt_len + tpb - 1) // tpb

    def test_extra_tokens_do_not_cross_block_boundary(self):
        prompt_len = 16
        num_extra_kv_tokens = 7
        transceiver, req = _build_transceiver_for_kv_slice(num_extra_kv_tokens, prompt_len)
        tpb = transceiver._reuse_adapter.tokens_per_block

        # Setup must actually exercise a boundary crossing: prompt_len ends on a
        # block boundary and the extra tokens would otherwise add a block.
        assert prompt_len % tpb == 0
        assert (prompt_len + num_extra_kv_tokens + tpb - 1) // tpb == prompt_len // tpb + 1

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == prompt_len // tpb

    def test_defaults_to_prompt_len_when_no_extra(self):
        prompt_len = 17
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0, prompt_len=prompt_len
        )
        tpb = transceiver._reuse_adapter.tokens_per_block

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == (prompt_len + tpb - 1) // tpb

    def test_swa_ordinal_drops_stale_front_and_ctx_first_token_overhang(self):
        # prompt_len=32, window=16 -> stale_end=2, so the adapter reports the
        # positional list [-1, -1, 102, 103, 104]: two out-of-window holes, the
        # two in-window prompt blocks, and the ctx first-token block (ordinal 4).
        # The prompt_blocks=4 upper bound drops 104; the -1 holes are filtered.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[102, 103, 104],
            sliding_window_size=16,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103], dtype=np.int64),
        )

    def test_swa_allocation_cap_preserves_packed_beam_tails(self):
        # beam_width > 1: beam-0's window is selected positionally (stale front
        # holes + the ctx first-token block 104 dropped by prompt_blocks=4), and
        # the divergent per-beam tails ride along verbatim.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[102, 103, 104],  # beam-0 chain (window + ctx first-token)
            beam_tails=[200, 201, 202],
            sliding_window_size=16,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103, 200, 201, 202], dtype=np.int64),
        )

    def test_beam_gt1_drops_tails_when_beam0_fully_cached(self):
        # Receiver already holds all of beam-0's prompt (cached=16 -> 2 blocks =
        # prompt_blocks), so the window is empty and the tails go with it.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=16,
            block_ids=[100, 101],  # beam-0 chain
            beam_tails=[200, 201, 202],
            cached_tokens=16,
            is_generation_only=True,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == 0

    def test_swa_trims_speculative_tail_before_stale_prompt_blocks(self):
        # MTP gen request: num_extra=2 adds a speculative tail block, window=16
        # evicts the front (stale_end=2), and the receiver already holds the
        # first 16 cached tokens. Positional list [-1, -1, 102, 103, 104]:
        # start = cached//tpb = 2 skips the holes, prompt_blocks=4 drops the
        # speculative block 104. V1 (masked full chain) and V2 (valid_only=False)
        # both produce this same list, so a single case covers both.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=2,
            prompt_len=32,
            block_ids=[102, 103, 104],
            sliding_window_size=16,
            cached_tokens=16,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103], dtype=np.int64),
        )

    def test_swa_ordinal_skips_reuse_hit_reaching_into_window(self):
        # Reuse hit reaches past the stale head: window=32 -> stale_end=2, but
        # the receiver already holds 32 cached tokens (4 blocks), so start=4 must
        # skip two in-window blocks the receiver already has. prompt_len=48 ->
        # prompt_blocks=6; positional list [-1, -1, 102, 103, 104, 105]. Only
        # ordinals [4:6] survive.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=48,
            block_ids=[102, 103, 104, 105],
            sliding_window_size=32,
            cached_tokens=32,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([104, 105], dtype=np.int64),
        )

    def test_full_attention_beam_gt1_preserves_tails(self):
        # Full attention (no window) with beam_width > 1: beam-0's window is the
        # prompt (ctx first-token block 104 dropped by prompt_blocks=4) and the
        # divergent per-beam tails survive.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 104],  # beam-0 chain (prompt + first-token)
            beam_tails=[200, 201, 202],
            sliding_window_size=None,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([100, 101, 102, 103, 200, 201, 202], dtype=np.int64),
        )

    @pytest.mark.parametrize("prompt_len", (1150, 1151))
    def test_dspark_disagg_boundary_keeps_only_initialized_swa(self, prompt_len):
        tokens_per_block = 128
        total_blocks = (prompt_len + tokens_per_block - 1) // tokens_per_block
        sliding_window_size = 128 + 5
        stale_end = max(
            0,
            (prompt_len + 1 - sliding_window_size) // tokens_per_block,
        )
        valid_prompt_blocks = total_blocks - stale_end
        block_ids = np.arange(
            200,
            200 + valid_prompt_blocks + 1,
            dtype=np.int64,
        )
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=5,
            prompt_len=prompt_len,
            tokens_per_block=tokens_per_block,
            block_ids=block_ids,
            sliding_window_size=sliding_window_size,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            block_ids[:-1],
        )


class TestKvSliceProperty:
    """Randomized differential test of _create_kv_slice.

    Over many (prompt_len, tpb, window, cached, beam, stale, spec, tails)
    combos, the real slice must equal an independent reference that selects the
    in-window prompt blocks the receiver still needs, purely from token
    positions -- a different derivation than the production code. The runtime
    self-check is enabled too, so both guards run. This is what would fail CI if
    a future edit reintroduced a count-infers-position (or any) selection bug,
    instead of the regression passing silently.
    """

    @staticmethod
    def _reference(prompt_len, tpb, window, cached_tokens, is_gen, beam, window_slots, tails):
        prompt_blocks = (prompt_len + tpb - 1) // tpb
        stale = 0 if window is None else max(0, (prompt_len + 1 - window) // tpb)
        start = (cached_tokens // tpb) if is_gen else 0
        out = []
        for p in range(prompt_blocks):
            if p < stale:  # out-of-window hole
                continue
            if p < start:  # receiver already holds it
                continue
            out.append(window_slots[p - stale])
        if beam > 1 and out and len(tails):
            out = out + list(tails)
        return np.array(out, dtype=np.int64)

    def test_matches_independent_reference_over_random_configs(self):
        rng = np.random.default_rng(20240917)
        next_slot = [1000]

        def fresh(n):
            s = list(range(next_slot[0], next_slot[0] + n))
            next_slot[0] += n
            return s

        for _ in range(500):
            tpb = int(rng.choice([8, 16, 128]))
            prompt_len = int(rng.integers(1, 25 * tpb))
            prompt_blocks = (prompt_len + tpb - 1) // tpb
            windowed = bool(rng.integers(0, 2))
            window = int(rng.integers(1, prompt_len + tpb + 1)) if windowed else None
            stale = 0 if window is None else max(0, (prompt_len + 1 - window) // tpb)
            stale = min(stale, prompt_blocks)  # cannot evict more than the prompt
            n_window = prompt_blocks - stale
            beam = int(rng.choice([1, 4]))
            is_gen = bool(rng.integers(0, 2))
            cached_tokens = int(rng.integers(0, prompt_len + 1)) if is_gen else 0

            window_slots = fresh(n_window)
            spec_slots = fresh(int(rng.integers(0, 3)))  # phantom / speculative tail
            tails = fresh(int(rng.integers(0, beam))) if beam > 1 else []

            transceiver, req = _build_transceiver_for_kv_slice(
                num_extra_kv_tokens=0,
                prompt_len=prompt_len,
                tokens_per_block=tpb,
                block_ids=window_slots + spec_slots,  # beam-0 chain (stale-stripped)
                beam_tails=tails,
                sliding_window_size=window,
                cached_tokens=cached_tokens,
                is_generation_only=is_gen,
                beam_width=beam,
            )

            got = transceiver._create_kv_slice(req).block_ids_per_layer_groups[0]
            expected = self._reference(
                prompt_len, tpb, window, cached_tokens, is_gen, beam, window_slots, tails
            )
            np.testing.assert_array_equal(
                got,
                expected,
                err_msg=(
                    f"prompt_len={prompt_len} tpb={tpb} window={window} stale={stale} "
                    f"cached={cached_tokens} is_gen={is_gen} beam={beam} "
                    f"window_slots={window_slots} spec={spec_slots} tails={tails}"
                ),
            )


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

    def get_block_ordinals(self, req, group_idx, lg):  # noqa: ARG002
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


class TestAdapterBlockOrdinals:
    """get_block_ordinals / get_beam0_ordinals_and_tails on the real V1/V2 adapters.

    The positional list is index==ordinal with -1 holes for stale/unbound
    blocks: V1 masks its pre-eviction chain using the manager's authoritative
    front-removed count; V2 reports the holes directly via valid_only=False.
    """

    class _V1Mgr:
        enable_block_reuse = True
        tokens_per_block = 32

        def __init__(self, chain, front_removed):
            self._chain = chain
            self._front_removed = front_removed
            self.asked = None

        def get_batch_cache_indices(self, request_ids, layer_idx=None, beam_width=1):
            return [list(self._chain)]

        def get_memory_pool_block_indices(self, block_ids, window_size):
            # Identity translation (nothing offloaded).
            return block_ids

        def get_num_front_blocks_removed(self, request_id, window_size=None):
            self.asked = (request_id, window_size)
            return self._front_removed

    def _v1_req(self):
        req = _FakeReq(prompt_len=7)
        req.py_request_id = 1
        req.py_beam_width = 1
        return req

    def test_v1_masks_stale_front_by_authoritative_count(self):
        mgr = self._V1Mgr([10, 11, 12, 13], front_removed=2)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [-1, -1, 12, 13])

    def test_v1_no_eviction_keeps_full_positional_chain(self):
        mgr = self._V1Mgr([10, 11, 12, 13], front_removed=0)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [10, 11, 12, 13])

    def test_v1_beam0_ordinals_and_tails(self):
        # Raw per-beam chains: beam-0 owns the shared prefix; the others differ
        # only in the final block. beam-3 matches beam-0's last, so it is not a
        # tail. Front eviction of 1 masks beam-0's leading block to -1.
        class _Impl:
            def get_batch_cache_block_ids(self, request_ids, window_size):
                return [[[10, 11, 12], [10, 11, 20], [10, 11, 21], [10, 11, 12]]]

        class _Mgr:
            enable_block_reuse = True
            tokens_per_block = 32
            impl = _Impl()

            def get_memory_pool_block_indices(self, block_ids, window_size):
                return block_ids

            def get_num_front_blocks_removed(self, request_id, window_size=None):
                return 1

        req = self._v1_req()
        req.py_beam_width = 4
        beam0, tails = _CacheReuseAdapterV1(_Mgr()).get_beam0_ordinals_and_tails(
            req, 0, _lg(window=512)
        )
        np.testing.assert_array_equal(beam0, [-1, 11, 12])
        np.testing.assert_array_equal(tails, [20, 21])

    def test_v2_reports_stale_as_holes_via_valid_only_false(self):
        class _KvCache:
            def __init__(self):
                self.valid_only = None

            def get_aggregated_page_indices(self, group_idx, valid_only=False):
                self.valid_only = valid_only
                return iter([-1, -1, 12, 13])

        class _V2Mgr:
            enable_block_reuse = True
            tokens_per_block = 32

            def __init__(self):
                self.kv_cache_map = {7: _KvCache()}

        mgr = _V2Mgr()
        req = _FakeReq(prompt_len=7)
        req.py_request_id = 7
        req.py_beam_width = 1
        ordinals = _CacheReuseAdapterV2(mgr).get_block_ordinals(req, 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [-1, -1, 12, 13])
        # Must request the positional view (holes kept), not the compact one.
        assert mgr.kv_cache_map[7].valid_only is False

    def test_v2_beam_search_unsupported(self):
        # V2 reads only the default beam, so packed beam search over disagg is
        # unsupported and must raise rather than silently drop tails.
        class _V2Mgr:
            enable_block_reuse = True
            tokens_per_block = 32
            kv_cache_map = {}

        req = _FakeReq(prompt_len=7)
        req.py_request_id = 7
        req.py_beam_width = 4
        with pytest.raises(NotImplementedError, match="beam_width > 1"):
            _CacheReuseAdapterV2(_V2Mgr()).get_beam0_ordinals_and_tails(req, 0, _lg(window=512))


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
    """Replicate the beam_width==1 ordinal selection of _create_kv_slice.

    Inputs:
      block_ids: the adapter's stale-stripped positional chain (valid window +
        speculative/over-hang tail). The SWA-evicted front is reported by the
        adapter as -1 holes -- V1 masks its pre-eviction chain to this shape,
        V2 reports it via valid_only=False -- modeled here by prepending
        stale_end of them.
      cached_tokens: reuse-hit prefix reported by the adapter (token-aligned).
      is_gen_only: True mirrors the gen-side path; False mirrors the ctx-side
        path where ``cached_per_lg`` is synthetically 0.
    """
    block_ids = np.array(block_ids, dtype=np.int64)
    prompt_blocks = (prompt_len + tpb - 1) // tpb
    stale_end = max(0, (prompt_len + 1 - window_size) // tpb)
    ordinals = np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids])
    # Ctx side bypasses adapter (cached=0); gen side uses adapter scalar.
    cached_lg = cached_tokens if is_gen_only else 0
    start = cached_lg // tpb
    # Position, not count: the prompt_blocks upper bound drops the speculative
    # and over-hang tail; -1 holes (stale front) are filtered after slicing.
    window = ordinals[start:prompt_blocks]
    return window[window >= 0]


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

    def test_masked_stale_front_gives_window(self):
        # The adapter masks the two out-of-window blocks to -1, so the ordinal
        # chain is [-1, -1, 12, 13]; the holes are filtered out.
        out = _swa_trim([12, 13], self.PROMPT_LEN, self.TPB, self.WINDOW, 0)
        np.testing.assert_array_equal(out, [12, 13])

    def test_ctx_side_no_adapter_no_skip(self):
        # Ctx-side path: adapter not invoked, cached_per_lg synthetically 0.
        # cache_skip = max(0, 0 - stale_end) = 0 — full valid window is sent.
        out = _swa_trim([20, 21], self.PROMPT_LEN, self.TPB, self.WINDOW, 0, is_gen_only=False)
        np.testing.assert_array_equal(out, [20, 21])

    def test_ctx_side_masked_stale_front(self):
        # Ctx-side path (cached synthetically 0): the masked stale front is
        # filtered and the full valid window is sent.
        out = _swa_trim([12, 13], self.PROMPT_LEN, self.TPB, self.WINDOW, 0, is_gen_only=False)
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
        tc._shutdown_complete = False
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
        assert tc._shutdown_complete is True

    def test_exit_calls_shutdown_on_exception(self):
        tc = self._tc()
        with pytest.raises(RuntimeError, match="boom"):
            with tc:
                raise RuntimeError("boom")
        # __exit__ still ran shutdown despite the in-block exception.
        tc._transfer_worker.shutdown.assert_called_once()
        assert tc._shutdown_complete is True

    def test_shutdown_is_idempotent(self):
        tc = self._tc()
        tc.shutdown()
        tc.shutdown()  # second call short-circuits after completed teardown.
        tc._transfer_worker.shutdown.assert_called_once()
