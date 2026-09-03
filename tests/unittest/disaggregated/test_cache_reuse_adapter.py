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
# Beam-0-only block layout.
# ---------------------------------------------------------------------------


class TestBeam0BlockLayout:
    """Verify disaggregated transfer requests only beam 0's block IDs."""

    def test_v1_adapter_requests_beam0_only(self):
        class _FakeMgr:
            enable_block_reuse = True
            tokens_per_block = 32

            def __init__(self):
                self.calls = []
                self.pool_indices_window = None

            def get_batch_cache_indices(self, request_ids, layer_idx=None):
                self.calls.append((request_ids, layer_idx))
                return [[10, 11, 12]]

            def get_memory_pool_block_indices(self, block_ids, window_size):
                # Identity translation: nothing offloaded, block_id == pool slot.
                self.pool_indices_window = window_size
                return block_ids

        req = _FakeReq(prompt_len=7)
        req.py_request_id = 1
        req.py_beam_width = 4
        mgr = _FakeMgr()

        block_ids = _CacheReuseAdapterV1(mgr).get_block_ids(req, 0, _lg(window=512))

        assert mgr.calls == [([1], 0)]
        assert mgr.pool_indices_window == 512
        np.testing.assert_array_equal(block_ids, [10, 11, 12])


class TestBeamTailCopy:
    def test_refreshes_v1_blocks_after_copy(self):
        manager = object.__new__(KVCacheManager)
        manager.impl = MagicMock()
        manager.impl.copy_last_attention_block_to_all_beams.return_value = True
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._kv_cache_manager = manager
        req = SimpleNamespace(py_beam_width=4)

        transceiver._copy_last_attention_block_to_all_beams(req)

        manager.impl.copy_last_attention_block_to_all_beams.assert_called_once_with(req)
        manager.impl.refresh_blocks.assert_called_once_with()

    def test_skips_refresh_when_blocks_are_shared(self):
        manager = object.__new__(KVCacheManager)
        manager.impl = MagicMock()
        manager.impl.copy_last_attention_block_to_all_beams.return_value = False
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._kv_cache_manager = manager
        req = SimpleNamespace(py_beam_width=4)

        transceiver._copy_last_attention_block_to_all_beams(req)

        manager.impl.refresh_blocks.assert_not_called()

    def test_single_beam_is_a_noop(self):
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._kv_cache_manager = MagicMock()

        transceiver._copy_last_attention_block_to_all_beams(SimpleNamespace(py_beam_width=1))

        transceiver._kv_cache_manager.assert_not_called()


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
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW
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
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW
        )
        src_start = (total_blocks - src_block_ids.size) * tpb
        dst_start = (total_blocks - dst_block_ids.size) * tpb

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
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW
        )

        np.testing.assert_array_equal(trimmed, [20, 21])

    def test_smaller_receiver_is_untouched(self):
        # Generation prefix-cache reuse: handled downstream via dst_start.
        src_block_ids = np.array([10, 11, 12], dtype=np.int64)
        dst_block_ids = np.array([20], dtype=np.int64)

        trimmed = Sender._trim_receiver_window_head(
            src_block_ids, dst_block_ids, peer_window_size=self.WINDOW
        )

        np.testing.assert_array_equal(trimmed, [20])

    def test_non_windowed_group_still_raises(self):
        src_block_ids = np.array([10], dtype=np.int64)
        dst_block_ids = np.array([20, 21], dtype=np.int64)

        with pytest.raises(ValueError, match="block count mismatch"):
            Sender._trim_receiver_window_head(src_block_ids, dst_block_ids, peer_window_size=None)


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
):
    """Stub a KvCacheTransceiverV2 so _create_kv_slice runs without dist setup.

    Wires only the attributes the method touches:
      - reuse adapter: tokens_per_block, per-layer-group cached count, block ids
      - page table:    layer groups
      - cache manager: num_extra_kv_tokens (read in this code path)
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

    reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_cached_token_count_per_layer_group=lambda req, layer_groups: [cached_tokens]
        * len(layer_groups),
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
        py_beam_width=beam_width,
        is_generation_only_request=lambda: is_generation_only,
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

    def test_swa_caps_oversized_non_speculative_v1_list_before_window_trim(self):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 104],
            sliding_window_size=16,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103], dtype=np.int64),
        )

    def test_swa_beam_request_transfers_beam0_only(self):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 104],
            sliding_window_size=16,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103], dtype=np.int64),
        )

    @pytest.mark.parametrize(
        "block_ids",
        (
            pytest.param([100, 101, 102, 103, 104], id="v1-pre-eviction"),
            pytest.param([102, 103, 104], id="v2-valid-only"),
        ),
    )
    def test_swa_trims_speculative_tail_before_stale_prompt_blocks(self, block_ids):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=2,
            prompt_len=32,
            block_ids=block_ids,
            sliding_window_size=16,
            cached_tokens=16,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.array([102, 103], dtype=np.int64),
        )

    def test_swa_speculative_tail_requires_single_beam(self):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=2,
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 104],
            sliding_window_size=16,
            beam_width=4,
        )

        with pytest.raises(ValueError, match="speculative scratch blocks require beam_width == 1"):
            transceiver._create_kv_slice(req)

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
