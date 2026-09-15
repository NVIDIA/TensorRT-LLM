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
"""Tests for CacheReuseAdapter, the positional _create_kv_slice, and Sender ordinal pairing."""

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


def _arr(values):
    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sender._pair_ordinals: the whole alignment protocol is one intersection.
# ---------------------------------------------------------------------------


class TestPairOrdinals:
    """Both tables are positional (index == block ordinal, -1 == not held).

    The sender moves a block only where both sides hold it. Every previously
    special-cased situation (SWA eviction, receiver prefix reuse, wider receiver
    window, pipelined chunk bounds) is just a different hole pattern.
    """

    def test_identity(self):
        src, dst = Sender._pair_ordinals(_arr([10, 11, 12]), _arr([20, 21, 22]))
        np.testing.assert_array_equal(src, [10, 11, 12])
        np.testing.assert_array_equal(dst, [20, 21, 22])

    def test_receiver_prefix_reuse_is_a_dst_hole(self):
        src, dst = Sender._pair_ordinals(_arr([10, 11, 12, 13]), _arr([-1, -1, 22, 23]))
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [22, 23])

    def test_sender_chunk_bounds_are_src_holes(self):
        src, dst = Sender._pair_ordinals(_arr([-1, 11, 12, -1]), _arr([20, 21, 22, 23]))
        np.testing.assert_array_equal(src, [11, 12])
        np.testing.assert_array_equal(dst, [21, 22])

    def test_holes_on_both_sides_take_the_intersection(self):
        src, dst = Sender._pair_ordinals(_arr([10, -1, 12, 13]), _arr([-1, 21, 22, -1]))
        np.testing.assert_array_equal(src, [12])
        np.testing.assert_array_equal(dst, [22])

    def test_no_overlap_is_empty(self):
        src, dst = Sender._pair_ordinals(_arr([10, 11, -1]), _arr([-1, -1, 22]))
        assert src.size == 0
        assert dst.size == 0

    def test_dst_start_block_masks_the_head(self):
        src, dst = Sender._pair_ordinals(
            _arr([10, 11, 12, 13]), _arr([20, 21, 22, 23]), dst_start_block=2
        )
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [22, 23])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="differ in length"):
            Sender._pair_ordinals(_arr([10, 11]), _arr([20, 21, 22]))

    def test_sender_not_yet_evicted_but_receiver_has(self):
        # Regression (DeepSeek-V4 SWA, prompt_len == 2 blocks, window == 1
        # block): ctx still holds ordinal 0 because its history_length has not
        # advanced, while gen pre-declared history_length=prompt_len and never
        # allocated ordinal 0. Pairing by ordinal copies X1 -> G1; the old
        # length-based suffix arithmetic copied X0 -> G1.
        src, dst = Sender._pair_ordinals(_arr([100, 101]), _arr([-1, 201]))
        np.testing.assert_array_equal(src, [101])
        np.testing.assert_array_equal(dst, [201])

    def test_receiver_with_wider_speculative_window(self):
        # Only gen runs spec decoding, so its SWA window keeps one more block.
        # That block has no source; it simply is not written.
        src, dst = Sender._pair_ordinals(_arr([-1, -1, 12, 13]), _arr([-1, 21, 22, 23]))
        np.testing.assert_array_equal(src, [12, 13])
        np.testing.assert_array_equal(dst, [22, 23])


class TestPairBeamTails:
    def test_equal_tails_pair_index_wise(self):
        src, dst = Sender._pair_beam_tails(_arr([1, 2, 3]), _arr([7, 8, 9]))
        np.testing.assert_array_equal(src, [1, 2, 3])
        np.testing.assert_array_equal(dst, [7, 8, 9])

    @pytest.mark.parametrize("src,dst", [([], [7, 8]), ([1, 2], []), ([], [])])
    def test_either_side_opting_out_yields_nothing(self, src, dst):
        s, d = Sender._pair_beam_tails(_arr(src), _arr(dst))
        assert s.size == 0
        assert d.size == 0

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="beam tail count mismatch"):
            Sender._pair_beam_tails(_arr([1, 2]), _arr([7]))

    def test_beam_tails_accessor(self):
        assert Sender._beam_tails(None, 0).size == 0
        assert Sender._beam_tails([_arr([1])], 3).size == 0
        np.testing.assert_array_equal(Sender._beam_tails([_arr([1, 2])], 0), [1, 2])


# ---------------------------------------------------------------------------
# Packed 1-D beam block layout (manager side).
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


# ---------------------------------------------------------------------------
# _create_kv_slice: positional table over ceil(prompt_len / tpb) ordinals.
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
      - reuse adapter: tokens_per_block, per-layer-group cached count, ordinals
      - page table:    layer groups

    `block_ids` is beam-0's stale-stripped chain (valid window followed by any
    speculative / ctx first-token over-hang). The adapter's positional table is
    modeled by prepending the manager-reported SWA holes: V1 masks its
    pre-eviction chain to this shape, V2 reports it via valid_only=False. For
    beam_width > 1, `beam_tails` supplies the divergent per-beam final blocks.
    """
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=sliding_window_size,
        local_layers=[LocalLayer(local_layer_id=0, global_layer_id=0)],
    )
    total_blocks = (prompt_len + num_extra_kv_tokens + tokens_per_block - 1) // tokens_per_block
    if block_ids is None:
        block_ids = np.arange(total_blocks, dtype=np.int64)
    else:
        block_ids = np.asarray(block_ids, dtype=np.int64)

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
    """The table spans exactly ceil(prompt_len / tpb) ordinals.

    Position is carried by the index, so the table must neither include the
    num_extra_kv_tokens over-hang nor shrink when blocks are evicted or cached;
    those become -1 entries instead.
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

    def test_incremental_allocation_is_padded_with_holes(self):
        # Ctx during chunked prefill has allocated only 2 of 4 prompt blocks.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0, prompt_len=32, block_ids=[100, 101]
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, -1, -1])
        assert kv_slice.beam_tails_per_layer_groups is None

    def test_swa_holes_and_first_token_overhang(self):
        # prompt_len=32, window=16 -> the adapter reports [-1, -1, 102, 103, 104]:
        # two out-of-window holes, the two in-window prompt blocks, and the ctx
        # first-token block (ordinal 4). prompt_blocks=4 drops 104; the holes
        # stay in place so the receiver can pair by ordinal.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[102, 103, 104],
            sliding_window_size=16,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [-1, -1, 102, 103])

    def test_beam_gt1_carries_tails_separately(self):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[102, 103, 104],
            beam_tails=[200, 201, 202],
            sliding_window_size=16,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [-1, -1, 102, 103])
        np.testing.assert_array_equal(kv_slice.beam_tails_per_layer_groups[0], [200, 201, 202])

    def test_beam_gt1_drops_tails_when_beam0_fully_cached(self):
        # Receiver already holds all of beam-0's prompt (cached=16 -> 2 blocks =
        # prompt_blocks), so the window is all holes and the tails go with it.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=16,
            block_ids=[100, 101],
            beam_tails=[200, 201, 202],
            cached_tokens=16,
            is_generation_only=True,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [-1, -1])
        assert kv_slice.beam_tails_per_layer_groups is None

    def test_receiver_cached_prefix_is_masked(self):
        # MTP gen request: num_extra=2 adds a speculative tail block, window=16
        # evicts the front, and the receiver already holds the first 16 tokens.
        # Adapter table [-1, -1, 102, 103, 104]; the cached prefix (2 blocks)
        # overlaps the holes; prompt_blocks=4 drops the speculative block 104.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=2,
            prompt_len=32,
            block_ids=[102, 103, 104],
            sliding_window_size=16,
            cached_tokens=16,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [-1, -1, 102, 103])

    def test_receiver_reuse_hit_reaching_into_window(self):
        # window=32 -> 2 holes; the receiver holds 32 cached tokens (4 blocks),
        # so two in-window blocks it already has are masked as well.
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
            kv_slice.block_ids_per_layer_groups[0], [-1, -1, -1, -1, 104, 105]
        )

    def test_ctx_side_ignores_cached_tokens(self):
        # The ctx request is not generation-only, so the adapter's cached count
        # is never consulted: ctx offers every block it holds.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[100, 101, 102, 103],
            cached_tokens=16,
            is_generation_only=False,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, 102, 103])

    def test_full_attention_beam_gt1_preserves_tails(self):
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=32,
            block_ids=[100, 101, 102, 103, 104],  # beam-0 chain (prompt + first-token)
            beam_tails=[200, 201, 202],
            sliding_window_size=None,
            beam_width=4,
        )

        kv_slice = transceiver._create_kv_slice(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, 102, 103])
        np.testing.assert_array_equal(kv_slice.beam_tails_per_layer_groups[0], [200, 201, 202])

    @pytest.mark.parametrize("prompt_len", (1150, 1151))
    def test_dspark_disagg_boundary_keeps_only_initialized_swa(self, prompt_len):
        tokens_per_block = 128
        total_blocks = (prompt_len + tokens_per_block - 1) // tokens_per_block
        sliding_window_size = 128 + 5
        stale_end = max(0, (prompt_len + 1 - sliding_window_size) // tokens_per_block)
        valid_prompt_blocks = total_blocks - stale_end
        block_ids = np.arange(200, 200 + valid_prompt_blocks + 1, dtype=np.int64)
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=5,
            prompt_len=prompt_len,
            tokens_per_block=tokens_per_block,
            block_ids=block_ids,
            sliding_window_size=sliding_window_size,
            is_generation_only=True,
        )

        kv_slice = transceiver._create_kv_slice(req)

        expected = np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids[:-1]])
        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], expected)


class TestKvSliceProperty:
    """Randomized differential test of _create_kv_slice.

    Over many (prompt_len, tpb, window, cached, beam, stale, spec, tails)
    combos, the real table must equal an independent reference built purely
    from token positions: -1 where the block is out of window or already held
    by the receiver, else the slot; the divergent beam tails ride separately.
    """

    @staticmethod
    def _reference(prompt_len, tpb, window, cached_tokens, is_gen, window_slots):
        prompt_blocks = (prompt_len + tpb - 1) // tpb
        stale = 0 if window is None else max(0, (prompt_len + 1 - window) // tpb)
        start = (cached_tokens // tpb) if is_gen else 0
        out = np.full(prompt_blocks, -1, dtype=np.int64)
        for p in range(prompt_blocks):
            if p < stale or p < start:
                continue
            out[p] = window_slots[p - stale]
        return out

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

            kv_slice = transceiver._create_kv_slice(req)
            got = kv_slice.block_ids_per_layer_groups[0]
            expected = self._reference(prompt_len, tpb, window, cached_tokens, is_gen, window_slots)
            ctx = (
                f"prompt_len={prompt_len} tpb={tpb} window={window} stale={stale} "
                f"cached={cached_tokens} is_gen={is_gen} beam={beam} "
                f"window_slots={window_slots} spec={spec_slots} tails={tails}"
            )
            np.testing.assert_array_equal(got, expected, err_msg=ctx)
            expect_tails = beam > 1 and len(tails) > 0 and (expected >= 0).any()
            if expect_tails:
                np.testing.assert_array_equal(
                    kv_slice.beam_tails_per_layer_groups[0], tails, err_msg=ctx
                )
            else:
                assert kv_slice.beam_tails_per_layer_groups is None, ctx


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

    The positional table is index==ordinal with -1 holes for stale/unbound
    blocks: V1 masks its pre-eviction chain using the manager's authoritative
    front-removed count; V2 reports the holes directly via valid_only=False.
    """

    class _V1Mgr:
        enable_block_reuse = True
        tokens_per_block = 32

        def __init__(self, beams, front_removed):
            self._beams = beams
            self._front_removed = front_removed
            self.asked = None
            self.translated = []
            self.impl = SimpleNamespace(
                get_batch_cache_block_ids=lambda request_ids, window_size: [
                    [list(b) for b in self._beams]
                ]
            )

        def get_memory_pool_block_indices(self, block_ids, window_size):
            # Identity translation (nothing offloaded); record what was asked.
            self.translated.append(list(block_ids))
            return list(block_ids)

        def get_num_front_blocks_removed(self, request_id, window_size=None):
            self.asked = (request_id, window_size)
            return self._front_removed

    def _v1_req(self):
        req = _FakeReq(prompt_len=7)
        req.py_request_id = 1
        req.py_beam_width = 1
        return req

    def test_v1_masks_stale_front_by_authoritative_count(self):
        mgr = self._V1Mgr([[10, 11, 12, 13]], front_removed=2)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [-1, -1, 12, 13])
        assert mgr.asked == (1, 512)

    def test_v1_evicted_ids_are_never_translated(self):
        # Evicted blocks were released to the free pool and may be offloaded by
        # now; the primary-pool translation would abort on them.
        mgr = self._V1Mgr([[10, 11, 12, 13]], front_removed=2)
        _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        assert mgr.translated == [[12, 13]]

    def test_v1_no_eviction_keeps_full_positional_chain(self):
        mgr = self._V1Mgr([[10, 11, 12, 13]], front_removed=0)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [10, 11, 12, 13])

    def test_v1_fully_evicted_chain_is_all_holes(self):
        mgr = self._V1Mgr([[10, 11]], front_removed=5)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        np.testing.assert_array_equal(ordinals, [-1, -1])
        assert mgr.translated == []

    def test_v1_empty_chain(self):
        mgr = self._V1Mgr([], front_removed=0)
        ordinals = _CacheReuseAdapterV1(mgr).get_block_ordinals(self._v1_req(), 0, _lg(window=512))
        assert ordinals.size == 0

    def test_v1_beam0_ordinals_and_tails(self):
        # Raw per-beam chains: beam-0 owns the shared prefix; the others differ
        # only in the final block. beam-3 matches beam-0's last, so it is not a
        # tail. Front eviction of 1 masks beam-0's leading block to -1.
        mgr = self._V1Mgr([[10, 11, 12], [10, 11, 20], [10, 11, 21], [10, 11, 12]], front_removed=1)
        req = self._v1_req()
        req.py_beam_width = 4
        beam0, tails = _CacheReuseAdapterV1(mgr).get_beam0_ordinals_and_tails(
            req, 0, _lg(window=512)
        )
        np.testing.assert_array_equal(beam0, [-1, 11, 12])
        np.testing.assert_array_equal(tails, [20, 21])
        assert mgr.translated == [[11, 12], [20, 21]]

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
        ad = _StubAdapter(scalar=0, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg(window=64)])
        assert out == [0, 0]

    def test_full_attn_passthrough(self):
        ad = _StubAdapter(scalar=64, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), [_lg(), _lg()])
        assert out == [64, 64]

    def test_swa_passthrough_above_stale(self):
        ad = _StubAdapter(scalar=24, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [24]

    def test_swa_passthrough_below_stale(self):
        # The adapter returns the raw scalar; masking it into the positional
        # table (where it may fall inside the SWA holes anyway) is
        # _create_kv_slice's job.
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), [_lg(window=16)])
        assert out == [8]

    def test_mixed_groups(self):
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(
            _FakeReq(32), [_lg(), _lg(window=16), _lg(window=32)]
        )
        assert out == [8, 8, 8]


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
        tc._transfer_worker.shutdown.assert_called_once()
        assert tc._shutdown_complete is True

    def test_shutdown_is_idempotent(self):
        tc = self._tc()
        tc.shutdown()
        tc.shutdown()  # second call short-circuits after completed teardown.
        tc._transfer_worker.shutdown.assert_called_once()
