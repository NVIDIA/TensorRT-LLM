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
"""Tests for CacheReuseAdapter, the positional _create_chunk, and Sender ordinal pairing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import Sender
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import (
    CacheReuseAdapter,
    _CacheReuseAdapterV2,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

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


# ---------------------------------------------------------------------------
# _create_chunk: positional table over ceil(prompt_len / tpb) ordinals.
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
):
    """Stub a KvCacheTransceiverV2 so _create_chunk runs without dist setup.

    Wires only the attributes the method touches:
      - reuse adapter: tokens_per_block, per-layer-group cached count, ordinals
      - page table:    layer groups

    `block_ids` is beam-0's stale-stripped chain (valid window followed by any
    speculative / ctx first-token over-hang). The adapter's positional table is
    modeled by prepending the manager-reported SWA holes, as V2 reports
    them via valid_only=False.
    """
    layer_group = SimpleNamespace(kind="attention", kept_window=sliding_window_size)
    total_blocks = (prompt_len + num_extra_kv_tokens + tokens_per_block - 1) // tokens_per_block
    if block_ids is None:
        block_ids = np.arange(total_blocks, dtype=np.int64)
    else:
        block_ids = np.asarray(block_ids, dtype=np.int64)

    # The positional ordinal list the adapter hands _create_chunk:
    # the stale (out-of-window) front is reported as -1 holes, followed by the
    # valid window blocks, the speculative/scratch tail, and the ctx
    # first-token over-hang. `block_ids` here is the stale-stripped chain, so
    # prepend the -1 holes, as V2 reports via valid_only=False.
    def get_block_ordinals(req, idx):
        transfer_len = getattr(req, "py_csa2_remote_tail_start", None)
        if transfer_len is None:
            transfer_len = prompt_len
        stale_end = 0
        if sliding_window_size is not None:
            stale_end = max(0, (transfer_len + 1 - sliding_window_size) // tokens_per_block)
        return np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids])

    reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_cached_token_count_per_layer_group=lambda req, num_layer_groups: (
            [cached_tokens] * num_layer_groups
        ),
        get_block_ids=lambda req, idx: block_ids,
        get_block_ordinals=get_block_ordinals,
    )
    page_table = SimpleNamespace(page_layout=SimpleNamespace(layer_groups=[layer_group]))

    def transfer_length(req):
        # A CSA2 manager under remote-tail replay moves the prompt before the remote tail.
        start = getattr(req, "py_csa2_remote_tail_start", None)
        return req.prompt_len if start is None else start

    cache_manager = SimpleNamespace(
        num_extra_kv_tokens=num_extra_kv_tokens,
        disagg_transfer_length=transfer_length,
        disagg_untransferred_buffers=lambda req: frozenset(),
        disagg_state_slot=lambda req, layer_group: None,
    )

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._reuse_adapter = reuse_adapter
    transceiver._page_table = page_table
    transceiver._kv_cache_manager = cache_manager

    req = SimpleNamespace(
        prompt_len=prompt_len,
        py_request_id=0,
        py_beam_width=1,
        is_generation_only_request=is_generation_only,
        # num_extra_kv_tokens == max_draft_len - 1
        py_draft_tokens=[0 for _ in range(num_extra_kv_tokens + 1)],
    )
    return transceiver, req


@pytest.mark.parametrize("encoder_hit", [False, True])
def test_v2_receive_uses_each_resumed_group(encoder_hit):
    transceiver, req = _build_transceiver_for_kv_slice(
        0, 257, tokens_per_block=128, is_generation_only=True
    )
    # Native lifecycle IDs, not physical pool IDs, index reuse_status. Two
    # distinct window groups can share a physical pool without sharing reuse.
    groups = [
        SimpleNamespace(kind="attention", kept_window=None),
        SimpleNamespace(kind="attention", kept_window=128),
        SimpleNamespace(kind="attention", kept_window=128),
    ]
    pages = [[10, 11, 12], [-1, 21, 22], [-1, 31, 32]]
    cache = SimpleNamespace(
        num_committed_tokens=256,
        reuse_status=[
            SimpleNamespace(endpoint=256, complete=True),
            SimpleNamespace(endpoint=256, complete=encoder_hit),
            SimpleNamespace(endpoint=256, complete=False),
        ],
        get_aggregated_page_indices=lambda group, valid_only: iter(pages[group]),
    )
    manager = SimpleNamespace(
        enable_block_reuse=True, tokens_per_block=128, kv_cache_map={0: cache}
    )
    adapter = _CacheReuseAdapterV2(manager)
    transceiver._reuse_adapter = adapter
    transceiver._page_table.page_layout.layer_groups = groups
    assert adapter.get_cached_token_count_per_layer_group(req, len(groups)) == [
        256,
        256 if encoder_hit else 0,
        0,
    ]
    received = transceiver._create_chunk(req).block_ids_per_layer_groups
    for actual, expected in zip(
        received, [[-1, -1, 12], [-1, -1, 22] if encoder_hit else [-1, 21, 22], [-1, 31, 32]]
    ):
        np.testing.assert_array_equal(actual, expected)

    # A later commit must not make newly allocated, unreceived pages look like
    # an admission-time prefix hit.
    cache.num_committed_tokens = 384
    assert adapter.get_cached_token_count_per_layer_group(req, len(groups))[0] == 256
    cache.reuse_status[1].complete = False  # manager declined an available OPTIONAL group
    assert adapter.get_cached_token_count_per_layer_group(req, len(groups)) == [256, 0, 0]
    manager.enable_block_reuse = False
    assert adapter.get_cached_token_count_per_layer_group(req, len(groups)) == [0, 0, 0]
    manager.enable_block_reuse = True
    manager.kv_cache_map.clear()
    assert adapter.get_cached_token_count_per_layer_group(req, len(groups)) == [0, 0, 0]


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

        kv_slice = transceiver._create_chunk(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == (prompt_len + tpb - 1) // tpb

    def test_extra_tokens_do_not_cross_block_boundary(self):
        prompt_len = 16
        num_extra_kv_tokens = 7
        transceiver, req = _build_transceiver_for_kv_slice(num_extra_kv_tokens, prompt_len)
        tpb = transceiver._reuse_adapter.tokens_per_block

        assert prompt_len % tpb == 0
        assert (prompt_len + num_extra_kv_tokens + tpb - 1) // tpb == prompt_len // tpb + 1

        kv_slice = transceiver._create_chunk(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == prompt_len // tpb

    def test_defaults_to_prompt_len_when_no_extra(self):
        prompt_len = 17
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0, prompt_len=prompt_len
        )
        tpb = transceiver._reuse_adapter.tokens_per_block

        kv_slice = transceiver._create_chunk(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == (prompt_len + tpb - 1) // tpb

    def test_incremental_allocation_is_padded_with_holes(self):
        # Ctx during chunked prefill has allocated only 2 of 4 prompt blocks.
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0, prompt_len=32, block_ids=[100, 101]
        )

        kv_slice = transceiver._create_chunk(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, -1, -1])

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

        kv_slice = transceiver._create_chunk(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [-1, -1, 102, 103])

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

        kv_slice = transceiver._create_chunk(req)

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

        kv_slice = transceiver._create_chunk(req)

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

        kv_slice = transceiver._create_chunk(req)

        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], [100, 101, 102, 103])

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

        kv_slice = transceiver._create_chunk(req)

        expected = np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids[:-1]])
        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], expected)

    def test_dspark_gen_init_draft_reserve_block_is_trimmed(self):
        # A DSpark gen-init request reserves room for its draft tokens past the
        # prompt. Here only that reserve spills into a new block: the prompt
        # plus num_extra_kv_tokens still ends inside the last prompt block. The
        # spilled block must be dropped without shifting the in-window SWA
        # blocks that are paired with the ctx side.
        prompt_len = 1144
        tokens_per_block = 128
        num_extra_kv_tokens = 4
        num_draft_tokens = num_extra_kv_tokens + 1  # max_draft_len, as the helper models it
        sliding_window_size = 128 + 5
        prompt_blocks = (prompt_len + tokens_per_block - 1) // tokens_per_block
        assert (prompt_len + num_extra_kv_tokens - 1) // tokens_per_block == prompt_blocks - 1
        assert (
            prompt_len + num_extra_kv_tokens + num_draft_tokens - 1
        ) // tokens_per_block == prompt_blocks

        stale_end = max(0, (prompt_len + 1 - sliding_window_size) // tokens_per_block)
        valid_prompt_blocks = prompt_blocks - stale_end
        # The in-window prompt blocks followed by the draft-reserve block.
        block_ids = np.arange(200, 200 + valid_prompt_blocks + 1, dtype=np.int64)
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=num_extra_kv_tokens,
            prompt_len=prompt_len,
            tokens_per_block=tokens_per_block,
            block_ids=block_ids,
            sliding_window_size=sliding_window_size,
            is_generation_only=True,
        )
        assert len(req.py_draft_tokens) == num_draft_tokens

        kv_slice = transceiver._create_chunk(req)

        expected = np.concatenate([np.full(stale_end, -1, dtype=np.int64), block_ids[:-1]])
        np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], expected)


class TestKvSliceProperty:
    """Randomized differential test of _create_chunk.

    Over many (prompt_len, tpb, window, cached, stale, spec) combos, the real
    table must equal an independent reference built purely from token
    positions: -1 where the block is out of window or already held by the
    receiver, else the slot.
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
            is_gen = bool(rng.integers(0, 2))
            cached_tokens = int(rng.integers(0, prompt_len + 1)) if is_gen else 0

            window_slots = fresh(n_window)
            spec_slots = fresh(int(rng.integers(0, 3)))  # phantom / speculative tail

            transceiver, req = _build_transceiver_for_kv_slice(
                num_extra_kv_tokens=0,
                prompt_len=prompt_len,
                tokens_per_block=tpb,
                block_ids=window_slots + spec_slots,  # beam-0 chain (stale-stripped)
                sliding_window_size=window,
                cached_tokens=cached_tokens,
                is_generation_only=is_gen,
            )

            kv_slice = transceiver._create_chunk(req)
            got = kv_slice.block_ids_per_layer_groups[0]
            expected = self._reference(prompt_len, tpb, window, cached_tokens, is_gen, window_slots)
            ctx = (
                f"prompt_len={prompt_len} tpb={tpb} window={window} stale={stale} "
                f"cached={cached_tokens} is_gen={is_gen} "
                f"window_slots={window_slots} spec={spec_slots}"
            )
            np.testing.assert_array_equal(got, expected, err_msg=ctx)

    @pytest.mark.parametrize(
        "prompt_len,remote_tail_start,source_pages,destination_pages",
        (
            pytest.param(129, 1, [100], [200, 201], id="partial-prefix"),
            pytest.param(1024, 896, [106], [206, 207], id="evicted-prefix"),
        ),
    )
    def test_remote_tail_preserves_source_swa_prefix(
        self,
        prompt_len: int,
        remote_tail_start: int,
        source_pages: list[int],
        destination_pages: list[int],
    ) -> None:
        slices = []
        for mode, block_ids in (("source", source_pages), ("destination", destination_pages)):
            transceiver, req = _build_transceiver_for_kv_slice(
                num_extra_kv_tokens=0,
                prompt_len=prompt_len,
                tokens_per_block=128,
                block_ids=block_ids,
                sliding_window_size=128,
                is_generation_only=mode == "destination",
            )
            req.py_csa2_remote_tail_mode = mode
            req.py_csa2_remote_tail_start = remote_tail_start
            req.py_draft_tokens = []
            slices.append(transceiver._create_chunk(req).block_ids_per_layer_groups[0])

        # CTX owns only the computed prefix. GEN also reserves the replay tail,
        # whose uninitialized page must not replace the prefix's receive target.
        prefix_blocks = (remote_tail_start + 127) // 128
        stale = max(0, (remote_tail_start + 1 - 128) // 128)
        np.testing.assert_array_equal(
            slices[0], [-1] * stale + source_pages[: prefix_blocks - stale]
        )
        np.testing.assert_array_equal(
            slices[1], [-1] * stale + destination_pages[: prefix_blocks - stale]
        )

    @pytest.mark.parametrize("mode", ("source", "destination"))
    @pytest.mark.parametrize("prompt_len", (64, 128))
    @pytest.mark.parametrize("window", (None, 128))
    def test_remote_tail_zero_prefix_transfers_no_pages(
        self, mode: str, prompt_len: int, window: int | None
    ) -> None:
        transceiver, req = _build_transceiver_for_kv_slice(
            num_extra_kv_tokens=0,
            prompt_len=prompt_len,
            tokens_per_block=128,
            block_ids=[100],
            sliding_window_size=window,
            is_generation_only=mode == "destination",
        )
        req.py_csa2_remote_tail_mode = mode
        req.py_csa2_remote_tail_start = 0
        req.py_draft_tokens = []

        kv_slice = transceiver._create_chunk(req)

        assert kv_slice.block_ids_per_layer_groups[0].size == 0


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

    def get_block_ids(self, req, group_idx):  # noqa: ARG002
        return np.array([], dtype=np.int64)

    def get_block_ordinals(self, req, group_idx):  # noqa: ARG002
        return np.array([], dtype=np.int64)

    def commit_blocks_for_reuse(self, req):  # noqa: ARG002
        pass


class _FakeReq:
    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len


class TestAdapterBlockOrdinals:
    """V2 positional block ordinals preserve stale and unbound slots as holes."""

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
        ordinals = _CacheReuseAdapterV2(mgr).get_block_ordinals(req, 0)
        np.testing.assert_array_equal(ordinals, [-1, -1, 12, 13])
        # Must request the positional view (holes kept), not the compact one.
        assert mgr.kv_cache_map[7].valid_only is False


class TestAdapterPerLayerGroup:
    """Per-layer cached prefix: adapter reports only the reuse-hit scalar."""

    TPB = 8

    def test_reuse_disabled(self):
        ad = _StubAdapter(scalar=128, tpb=self.TPB, enabled=False)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), 2)
        assert out == [0, 0]

    def test_zero_scalar(self):
        ad = _StubAdapter(scalar=0, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), 2)
        assert out == [0, 0]

    def test_full_attn_passthrough(self):
        ad = _StubAdapter(scalar=64, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(256), 2)
        assert out == [64, 64]

    def test_swa_passthrough_above_stale(self):
        ad = _StubAdapter(scalar=24, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), 1)
        assert out == [24]

    def test_swa_passthrough_below_stale(self):
        # The adapter returns the raw scalar; masking it into the positional
        # table (where it may fall inside the SWA holes anyway) is
        # _create_chunk's job.
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), 1)
        assert out == [8]

    def test_mixed_groups(self):
        ad = _StubAdapter(scalar=8, tpb=self.TPB)
        out = ad.get_cached_token_count_per_layer_group(_FakeReq(32), 3)
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


def test_a_state_layer_group_carries_its_slot_or_the_transfer_is_refused():
    """A recurrent-state group moves the one slot the cache manager names.

    Naming none would leave the state behind while both sides expect no bytes of it, so the
    transfer refuses it.
    """
    transceiver, req = _build_transceiver_for_kv_slice(0, 64, tokens_per_block=32)
    transceiver._page_table = SimpleNamespace(
        page_layout=SimpleNamespace(layer_groups=[SimpleNamespace(kind="state", kept_window=None)])
    )
    transceiver._kv_cache_manager.disagg_state_slot = lambda req, layer_group: 5 + layer_group
    assert transceiver._create_chunk(req).block_ids_per_layer_groups[0].tolist() == [5]
    transceiver._kv_cache_manager.disagg_state_slot = lambda req, layer_group: None
    with pytest.raises(RuntimeError, match="names no slot of layer group 0's recurrent state"):
        transceiver._create_chunk(req)
