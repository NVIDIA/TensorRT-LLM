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
"""Unit tests for chunked and pipelined KV cache transfer (sender-only chunking).

These tests validate the session state machine using the real
TxSession/RxSession classes with lightweight stub sender/receiver objects.
"""

import itertools
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVSendTask,
    Receiver,
    RecvReqInfo,
    RxSession,
    Sender,
    TaskStatus,
    TxSession,
)
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, LlmRequestType
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(rid: int = 42) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=rid)


def _stub_sender():
    """Create a stub sender with no-op methods needed by TxSession."""
    sender = MagicMock()
    sender.setup_session = MagicMock()
    sender._get_req_info = MagicMock(return_value=None)
    sender.dispatch_task = MagicMock()
    return sender


def _stub_receiver():
    """Create a stub receiver with no-op methods needed by RxSession."""
    receiver = MagicMock()
    receiver.setup_session = MagicMock()
    receiver.dispatch_task = MagicMock()
    return receiver


def _make_tx_session(num_slices: int, rid: int = 42, prompt_len: int = 8, **kwargs) -> TxSession:
    """Create a real TxSession and send num_slices slices into it."""
    params = _make_params(rid)
    session = TxSession(
        request_id=rid,
        params=params,
        sender=_stub_sender(),
        prompt_len=prompt_len,
        **kwargs,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.send(s)
    return session


def _make_rx_session(num_slices: int, rid: int = 42, prompt_len: int = 8) -> RxSession:
    """Create a real RxSession and receive num_slices slices into it."""
    params = _make_params(rid)
    session = RxSession(
        request_id=rid,
        params=params,
        receiver=_stub_receiver(),
        prompt_len=prompt_len,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.receive(s)
    return session


# ---------------------------------------------------------------------------
# Anchored span intersection tests
#
# The deleted project_blocks_to_global_chunk inferred a list's global position
# from its length (suffix convention). Anchors make the position explicit;
# _align_kv_blocks intersects two anchored spans directly.
# ---------------------------------------------------------------------------

_ANCHOR_TPB = 8


def _align_anchored(src_ids, src_anchor, dst_ids, dst_anchor, tpb=_ANCHOR_TPB):
    return Sender._align_kv_blocks(
        np.asarray(src_ids, dtype=np.int64),
        np.asarray(dst_ids, dtype=np.int64),
        src_token_start=src_anchor * tpb,
        dst_token_start=dst_anchor * tpb,
        tokens_per_block=tpb,
    )


def test_chunk_anchored_past_short_group_span_is_a_noop():
    """A chunk anchored past a short layer group's declared span transfers nothing."""
    src, dst = _align_anchored(
        src_ids=np.arange(4, 8, dtype=np.int64),  # chunk covering block ordinals [4, 8)
        src_anchor=4,
        dst_ids=np.array([10, 11, 12], dtype=np.int64),  # short group spans [0, 3)
        dst_anchor=0,
    )

    assert src.size == 0
    assert dst.size == 0


@pytest.mark.parametrize(
    "chunk_anchor,chunk_ids,expected_dst",
    [
        (0, np.arange(16, dtype=np.int64), np.arange(100, 116, dtype=np.int64)),
        (16, np.arange(16, 32, dtype=np.int64), np.arange(116, 132, dtype=np.int64)),
    ],
    ids=["first_chunk", "later_chunk"],
)
def test_anchored_chunks_map_incrementally_allocated_source(chunk_anchor, chunk_ids, expected_dst):
    """Each chunk's anchor addresses the receiver's whole-prompt list directly."""
    dst_ids = np.arange(100, 132, dtype=np.int64)  # 32 blocks anchored at 0

    src, dst = _align_anchored(chunk_ids, chunk_anchor, dst_ids, 0)

    assert np.array_equal(src, chunk_ids)
    assert np.array_equal(dst, expected_dst)


def test_anchored_chunks_map_prefix_reuse_suffix_by_overlap():
    """A receiver suffix anchored at its true ordinal overlaps only later chunks."""
    dst_ids = np.array([104, 105, 106, 107], dtype=np.int64)  # spans [4, 8)
    dst_anchor = 4

    first_src, first_dst = _align_anchored(np.arange(4), 0, dst_ids, dst_anchor)
    second_src, second_dst = _align_anchored(np.arange(4, 8), 4, dst_ids, dst_anchor)

    assert first_src.size == 0 and first_dst.size == 0
    assert np.array_equal(second_src, np.arange(4, 8))
    assert np.array_equal(second_dst, dst_ids)


_PROJECTION_TPB = 8
_PROJECTION_PROMPT_TOKENS = 8 * _PROJECTION_TPB


def _make_projection_sender(num_groups: int = 2, tpb: int = _PROJECTION_TPB) -> Sender:
    """Create a Sender wired to a stub registrar with full-attention layer groups.

    Full attention is spelled as a window the prompt never outgrows, which is
    how both extractor paths build it; they read max_attention_window_vec, so an
    attention group never carries None.
    """
    peer_ri = SimpleNamespace(
        dp_rank=0,
        device_id=0,
        instance_name="decode",
        instance_rank=0,
        self_endpoint="tcp://decode:0",
        cp_size=1,
        cp_rank=0,
    )

    extractor = MagicMock()
    extractor.page_table = SimpleNamespace(
        tokens_per_block=tpb,
        layer_groups=[
            SimpleNamespace(kind=CacheKind.PAGED, sliding_window_size=_PROJECTION_PROMPT_TOKENS)
            for _ in range(num_groups)
        ],
    )
    # extract(region_ids, layer_group_id, pool_idx) - the caller passes the
    # layer-group and pool indices positionally, so swallow them.
    extractor.extract.side_effect = lambda block_ids, *_, **__: SimpleNamespace(
        memory=SimpleNamespace(
            ptrs=np.asarray(block_ids, dtype=np.int64),
            bytes_per_region=1,
        )
    )

    mapper = MagicMock()
    mapper.map.side_effect = lambda src_region, dst_region: SimpleNamespace(
        src=src_region,
        dst=dst_region,
    )

    registrar = MagicMock()
    registrar.self_rank_info = SimpleNamespace(cp_size=1, cp_rank=0)
    registrar.self_extractor = extractor
    registrar.get_peer_rank_info.return_value = peer_ri
    registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    registrar.should_send_kv.return_value = True
    registrar.get_pool_mapping.return_value = {(lg, 0): (lg, 0) for lg in range(num_groups)}
    registrar.peer_extractor.return_value = extractor
    registrar.get_kv_map.return_value = mapper

    sender = Sender.__new__(Sender)
    sender._registrar = registrar
    return sender


def _make_projection_task(slice_id: int = 1, beam_width: int = 1) -> KVSendTask:
    # Group 0 carries the chunk covering block ordinals [4, 8); group 1 is a
    # shorter (asymmetric) group whose resident suffix spans [5, 8).
    return KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.array([4, 5, 6, 7], dtype=np.int64),
                np.array([10, 11, 12], dtype=np.int64),
            ],
            first_ordinals=[4, 5],
        ),
        _make_params(),
        slice_id=slice_id,
        prompt_len=_PROJECTION_PROMPT_TOKENS,
        beam_width=beam_width,
    )


def _make_projection_req_info(slice_id=None) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.array([104, 105, 106, 107], dtype=np.int64),
            np.array([200, 201, 202], dtype=np.int64),
        ],
        unique_rid=42,
        first_ordinals=[4, 5],
        slice_id=slice_id,
    )


def test_build_kv_write_meta_projects_asymmetric_layer_group_chunk():
    """A short layer group's suffix blocks transfer with the overlapping global chunk."""
    sender = _make_projection_sender()

    write_meta = sender._build_kv_write_meta(_make_projection_task(), _make_projection_req_info())

    assert np.array_equal(
        write_meta.src_ptrs,
        np.array([4, 5, 6, 7, 10, 11, 12], dtype=np.int64),
    )
    assert np.array_equal(
        write_meta.dst_ptrs,
        np.array([104, 105, 106, 107, 200, 201, 202], dtype=np.int64),
    )
    assert np.array_equal(write_meta.sizes, np.ones(7, dtype=np.int64))
    assert write_meta.slice_id == 1
    # A receiver that sends no slice_id is addressed as its single task 0.
    assert write_meta.receiver_slice_id == 0


def test_final_swa_slice_keeps_the_receivers_complete_active_window():
    """The final SWA transfer is not projected down to the final context chunk."""
    sender = _make_projection_sender()
    sender._registrar.self_extractor.page_table.layer_groups[0].sliding_window_size = (
        6 * _PROJECTION_TPB
    )
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.arange(2, 8, dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
            # The deferred SWA group is anchored at its true window start (2),
            # not at the final context chunk's start (6).
            first_ordinals=[2, 0],
        ),
        _make_params(),
        slice_id=1,
        prompt_len=8 * _PROJECTION_TPB,
    )
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.arange(102, 108, dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        unique_rid=42,
        first_ordinals=[2, 0],
    )

    write_meta = sender._build_kv_write_meta(task, req_info)

    assert np.array_equal(write_meta.src_ptrs, np.arange(2, 8, dtype=np.int64))
    assert np.array_equal(write_meta.dst_ptrs, np.arange(102, 108, dtype=np.int64))


def test_chunked_slices_reassemble_to_the_monolithic_transfer():
    """Anchored chunks concatenate to exactly the monolithic slice's pairs."""
    sender = _make_projection_sender()
    src_group0 = np.arange(8, dtype=np.int64)  # full-attention, anchored at 0
    src_group1 = np.array([10, 11, 12], dtype=np.int64)  # short group, spans [5, 8)

    monolithic_task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[src_group0, src_group1],
            first_ordinals=[0, 5],
        ),
        _make_params(),
        slice_id=0,
        prompt_len=64,
    )
    chunk_tasks = [
        KVSendTask(
            KVSlice(
                is_last_slice=False,
                block_ids_per_layer_groups=[src_group0[:4], src_group1[:0]],
                first_ordinals=[0, 0],
            ),
            _make_params(),
            slice_id=0,
            prompt_len=64,
        ),
        KVSendTask(
            KVSlice(
                is_last_slice=True,
                block_ids_per_layer_groups=[src_group0[4:], src_group1],
                first_ordinals=[4, 5],
            ),
            _make_params(),
            slice_id=1,
            prompt_len=64,
        ),
    ]

    monolithic = sender._build_kv_write_meta(monolithic_task, _make_projection_req_info())
    chunked = [sender._build_kv_write_meta(t, _make_projection_req_info()) for t in chunk_tasks]

    chunked_src = np.concatenate([m.src_ptrs for m in chunked])
    chunked_dst = np.concatenate([m.dst_ptrs for m in chunked])
    # Pair sets are equal: every (src, dst) block pair is transferred exactly once.
    assert sorted(zip(chunked_src.tolist(), chunked_dst.tolist())) == sorted(
        zip(monolithic.src_ptrs.tolist(), monolithic.dst_ptrs.tolist())
    )
    assert np.concatenate([m.sizes for m in chunked]).sum() == monolithic.sizes.sum()


def test_build_kv_write_meta_tracks_sender_and_receiver_slice_ids():
    """Write metadata retains both the local task and peer task indices."""
    sender = _make_projection_sender()

    write_meta = sender._build_kv_write_meta(
        _make_projection_task(slice_id=1), _make_projection_req_info(slice_id=3)
    )

    assert write_meta.slice_id == 1
    assert write_meta.receiver_slice_id == 3


def test_helix_receiver_requires_zero_block_anchors():
    """A helix peer rejects anchored (trimmed/reused) spans loudly."""
    sender = _make_projection_sender()
    sender._registrar.get_peer_rank_info.return_value.cp_size = 2
    sender._registrar.get_peer_rank_info.return_value.cp_rank = 0

    # slice_id=0 and is_last_slice=True: not a partial chunk, so the pipelined
    # CP guard passes and the anchor guard is what must fire.
    with pytest.raises(ValueError, match="helix CP requires zero block anchors"):
        sender._build_kv_write_meta(_make_projection_task(slice_id=0), _make_projection_req_info())


def test_beam_tails_pair_positionally_after_the_beam0_spans():
    """Packed beam-tail blocks ride behind the anchored beam-0 intersection."""
    sender = _make_projection_sender(num_groups=1)
    beam_width = 4
    # 8 prompt blocks + 3 per-beam tail blocks on each side, both anchored at 0.
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.concatenate([np.arange(8), np.array([70, 71, 72])]).astype(np.int64)
            ],
            first_ordinals=[0],
        ),
        _make_params(),
        slice_id=0,
        prompt_len=_PROJECTION_PROMPT_TOKENS,
        beam_width=beam_width,
    )
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.concatenate([np.arange(100, 108), np.array([170, 171, 172])]).astype(np.int64)
        ],
        unique_rid=42,
        first_ordinals=[0],
    )

    write_meta = sender._build_kv_write_meta(task, req_info)

    assert np.array_equal(
        write_meta.src_ptrs, np.concatenate([np.arange(8), np.array([70, 71, 72])])
    )
    assert np.array_equal(
        write_meta.dst_ptrs, np.concatenate([np.arange(100, 108), np.array([170, 171, 172])])
    )


def test_fewer_shared_tails_than_beam_width_pair_correctly():
    """Anchored beam-0 length splits partially-shared beam tails exactly.

    The packer appends only UNSHARED final blocks, so beam_width=4 lists can
    carry just 2 tails; the old beam_width-1 guess misclassified the last
    beam-0 block as a tail here.
    """
    sender = _make_projection_sender(num_groups=1)
    beam_width = 4
    # 8 prompt blocks + 2 unshared tail blocks on each side, both anchored at 0.
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.concatenate([np.arange(8), np.array([70, 71])]).astype(np.int64)
            ],
            first_ordinals=[0],
        ),
        _make_params(),
        slice_id=0,
        prompt_len=_PROJECTION_PROMPT_TOKENS,
        beam_width=beam_width,
    )
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.concatenate([np.arange(100, 108), np.array([170, 171])]).astype(np.int64)
        ],
        unique_rid=42,
        first_ordinals=[0],
    )

    write_meta = sender._build_kv_write_meta(task, req_info)

    # All 8 beam-0 blocks pair by ordinal; exactly the 2 tails ride behind.
    assert np.array_equal(write_meta.src_ptrs, np.concatenate([np.arange(8), np.array([70, 71])]))
    assert np.array_equal(
        write_meta.dst_ptrs, np.concatenate([np.arange(100, 108), np.array([170, 171])])
    )


def test_beam_tail_count_mismatch_raises():
    """Unequal packed beam-tail counts cannot pair positionally and must fail."""
    sender = _make_projection_sender(num_groups=1)
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.concatenate([np.arange(8), np.array([70, 71, 72])]).astype(np.int64)
            ],
            first_ordinals=[0],
        ),
        _make_params(),
        slice_id=0,
        prompt_len=_PROJECTION_PROMPT_TOKENS,
        beam_width=4,
    )
    # dst has no tail blocks (its final blocks are shared across beams).
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[np.arange(100, 108, dtype=np.int64)],
        unique_rid=42,
        first_ordinals=[0],
    )

    with pytest.raises(ValueError, match="packed beam-tail count mismatch"):
        sender._build_kv_write_meta(task, req_info)


def test_receiver_with_larger_window_resolves_via_anchors():
    """The receiver's earlier-starting window is aligned by anchors, not trimmed.

    Old behavior (_trim_receiver_window_head, deleted): a receiver keeping a
    larger SWA window (only it runs speculative decoding) held extra leading
    blocks, which the sender dropped by count so that suffix-derived starts
    lined up. With explicit anchors the sender simply intersects the two
    declared spans: the receiver's extra head block gets no source and the
    last prompt block maps exactly (the old off-by-one regression).
    """
    tpb = 128
    total_blocks = 1225
    sender = _make_projection_sender(num_groups=1, tpb=tpb)
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[np.array([10], dtype=np.int64)],
            first_ordinals=[total_blocks - 1],  # sender window: last block only
        ),
        _make_params(),
        slice_id=0,
        prompt_len=total_blocks * tpb,
    )
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[np.array([20, 21], dtype=np.int64)],
        unique_rid=42,
        first_ordinals=[total_blocks - 2],  # receiver window starts one block earlier
    )

    write_meta = sender._build_kv_write_meta(task, req_info)

    # src block 10 lands on dst block 21 (same ordinal); dst block 20 has no
    # source and is not written.
    assert np.array_equal(write_meta.src_ptrs, np.array([10], dtype=np.int64))
    assert np.array_equal(write_meta.dst_ptrs, np.array([21], dtype=np.int64))


# ---------------------------------------------------------------------------
# Anchored alignment property test
#
# For a grid over (tpb, prompt length, per-side window sizes, gen cached
# prefix, chunk boundaries, beam width): every (src, dst) pointer pair maps
# equal block ordinals, and the transferred set equals the intersection of the
# declared spans. Block IDs encode their ordinal so the mocked extractors
# (identity ptrs) expose the mapping directly.
# ---------------------------------------------------------------------------

_SRC_ID_BASE = 1_000
_DST_ID_BASE = 5_000
_SRC_TAIL_BASE = 3_000
_DST_TAIL_BASE = 7_000


def _swa_start_block(prompt_len, window, tpb):
    if window is None or window >= prompt_len:
        return 0
    return (prompt_len + 1 - window) // tpb


def _grid_req_info(dst_ids: np.ndarray, dst_anchor: int) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[dst_ids],
        unique_rid=42,
        first_ordinals=[dst_anchor],
    )


def test_anchored_alignment_property_grid():
    tpb_values = (4, 16)
    prompt_block_counts = (1, 3, 8)
    window_block_options = (None, 2, 5)  # window sizes in blocks (None = full attention)
    gen_cached_block_options = (0, 1, 4)
    beam_widths = (1, 2)

    for tpb, prompt_blocks, src_wblocks, dst_wblocks, gen_cached, beam_width in itertools.product(
        tpb_values,
        prompt_block_counts,
        window_block_options,
        window_block_options,
        gen_cached_block_options,
        beam_widths,
    ):
        prompt_len = prompt_blocks * tpb - 1  # exercise the ceil in prompt_blocks
        src_window = src_wblocks * tpb + 1 if src_wblocks is not None else None
        dst_window = dst_wblocks * tpb + 1 if dst_wblocks is not None else None

        src_anchor = _swa_start_block(prompt_len, src_window, tpb)
        dst_anchor = max(
            _swa_start_block(prompt_len, dst_window, tpb), min(gen_cached, prompt_blocks)
        )
        # Beam search never combines with eviction/reuse trims in production
        # (front-block eviction and chunked transfer assert beam_width == 1).
        if beam_width > 1 and (src_anchor > 0 or dst_anchor > 0):
            continue

        case = (
            f"tpb={tpb} prompt_blocks={prompt_blocks} src_w={src_wblocks} "
            f"dst_w={dst_wblocks} cached={gen_cached} beam={beam_width}"
        )
        src_span = np.arange(_SRC_ID_BASE + src_anchor, _SRC_ID_BASE + prompt_blocks)
        dst_span = np.arange(_DST_ID_BASE + dst_anchor, _DST_ID_BASE + prompt_blocks)
        n_tails = beam_width - 1
        src_ids = np.concatenate([src_span, np.arange(_SRC_TAIL_BASE, _SRC_TAIL_BASE + n_tails)])
        dst_ids = np.concatenate([dst_span, np.arange(_DST_TAIL_BASE, _DST_TAIL_BASE + n_tails)])

        sender = _make_projection_sender(num_groups=1, tpb=tpb)

        # --- monolithic slice ---
        task = KVSendTask(
            KVSlice(
                is_last_slice=True,
                block_ids_per_layer_groups=[src_ids.astype(np.int64)],
                first_ordinals=[src_anchor],
            ),
            _make_params(),
            slice_id=0,
            prompt_len=prompt_len,
            beam_width=beam_width,
        )
        meta = sender._build_kv_write_meta(
            task, _grid_req_info(dst_ids.astype(np.int64), dst_anchor)
        )

        expected_ordinals = list(range(max(src_anchor, dst_anchor), prompt_blocks))
        got_pairs = list(zip(meta.src_ptrs.tolist(), meta.dst_ptrs.tolist()))
        beam0_pairs = [p for p in got_pairs if p[0] < _SRC_TAIL_BASE]
        tail_pairs = [p for p in got_pairs if p[0] >= _SRC_TAIL_BASE]
        # Every pair maps equal ordinals on both sides.
        assert all(s - _SRC_ID_BASE == d - _DST_ID_BASE for s, d in beam0_pairs), case
        # The transferred range is exactly the intersection of the two spans.
        assert [s - _SRC_ID_BASE for s, _ in beam0_pairs] == expected_ordinals, case
        if expected_ordinals and n_tails:
            assert tail_pairs == [
                (_SRC_TAIL_BASE + i, _DST_TAIL_BASE + i) for i in range(n_tails)
            ], case
        else:
            assert tail_pairs == [], case

        # --- chunked slices (full-attention source only; SWA groups are
        # deferred whole to the final chunk by _build_prefill_chunk) ---
        if src_wblocks is not None or beam_width > 1:
            continue
        for chunk_blocks in (1, 2, prompt_blocks):
            seen_ordinals = []
            bounds = list(range(0, prompt_blocks, chunk_blocks)) + [prompt_blocks]
            for slice_id, (c0, c1) in enumerate(zip(bounds[:-1], bounds[1:])):
                if c0 >= c1:
                    continue
                chunk_ids = np.arange(_SRC_ID_BASE + c0, _SRC_ID_BASE + c1)
                chunk_task = KVSendTask(
                    KVSlice(
                        is_last_slice=(c1 == prompt_blocks),
                        block_ids_per_layer_groups=[chunk_ids.astype(np.int64)],
                        first_ordinals=[c0],
                    ),
                    _make_params(),
                    slice_id=slice_id,
                    prompt_len=prompt_len,
                )
                chunk_meta = sender._build_kv_write_meta(
                    chunk_task, _grid_req_info(dst_ids.astype(np.int64), dst_anchor)
                )
                for s, d in zip(chunk_meta.src_ptrs.tolist(), chunk_meta.dst_ptrs.tolist()):
                    assert s - _SRC_ID_BASE == d - _DST_ID_BASE, f"{case} chunk={chunk_blocks}"
                    seen_ordinals.append(s - _SRC_ID_BASE)
            # Chunks tile the intersection exactly once, in order.
            assert seen_ordinals == expected_ordinals, f"{case} chunk={chunk_blocks}"


# ---------------------------------------------------------------------------
# RecvReqInfo wire format
# ---------------------------------------------------------------------------


def test_recv_req_info_first_ordinals_roundtrip():
    """first_ordinals survive to_bytes/from_bytes unchanged."""
    info = RecvReqInfo(
        sender_req_id=7,
        instance_name="decode",
        instance_rank=3,
        block_ids_per_layer_groups=[
            np.array([104, 105], dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        unique_rid=42,
        first_ordinals=[6, 0],
        aux_slot=1,
        slice_id=2,
        bounce_dst_base=None,
    )

    restored = RecvReqInfo.from_bytes(info.to_bytes())

    assert restored.first_ordinals == [6, 0]
    assert restored.sender_req_id == 7
    assert restored.instance_name == "decode"
    assert restored.instance_rank == 3
    assert restored.unique_rid == 42
    assert restored.aux_slot == 1
    assert restored.slice_id == 2
    assert np.array_equal(restored.block_ids_per_layer_groups[0], [104, 105])
    assert restored.block_ids_per_layer_groups[1].size == 0


def test_recv_req_info_rejects_pre_anchor_wire_layout():
    """Bytes from a peer still sending dst_start_token fail loudly, never misalign."""
    old_layout = msgpack.packb(
        {
            "sender_req_id": 7,
            "instance_name": "decode",
            "instance_rank": 0,
            "block_ids_per_layer_groups": [np.array([104], dtype=np.int64).tobytes()],
            "unique_rid": 42,
            "dst_start_token": 128,  # pre-anchor field, no first_ordinals
            "aux_slot": None,
            "slice_id": 0,
            "bounce_dst_base": None,
        }
    )

    with pytest.raises(TypeError):
        RecvReqInfo.from_bytes(old_layout)


# ---------------------------------------------------------------------------
# Receiver-side anchor validation
# ---------------------------------------------------------------------------


def _make_recv_task(kv_slice: KVSlice) -> SimpleNamespace:
    return SimpleNamespace(
        _unique_rid=42,
        _params=SimpleNamespace(ctx_request_id=7, disagg_request_id=42),
        _kv_slice=kv_slice,
        _aux_slot=None,
        slice_id=0,
    )


def _make_bare_receiver() -> Receiver:
    receiver = Receiver.__new__(Receiver)
    receiver._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(instance_name="decode", instance_rank=0)
    )
    return receiver


def test_recv_req_info_requires_one_anchor_per_layer_group():
    """An unanchored slice never reaches the wire."""
    receiver = _make_bare_receiver()
    kv_slice = KVSlice(
        is_last_slice=True,
        block_ids_per_layer_groups=[
            np.array([1, 2], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ],
        first_ordinals=[0],  # one anchor missing
    )

    with pytest.raises(ValueError, match="first_ordinals"):
        receiver._build_recv_req_info(_make_recv_task(kv_slice))


def test_recv_req_info_carries_the_slice_anchors():
    receiver = _make_bare_receiver()
    kv_slice = KVSlice(
        is_last_slice=True,
        block_ids_per_layer_groups=[
            np.array([1, 2], dtype=np.int64),
            np.array([], dtype=np.int64),
        ],
        first_ordinals=[3, 0],
    )

    info = receiver._build_recv_req_info(_make_recv_task(kv_slice))

    assert info.first_ordinals == [3, 0]
    assert info.unique_rid == 42
    assert info.sender_req_id == 7


def test_process_kv_agent_result_resolves_task_by_receiver_slice_id():
    """The receiver resolves the task using its own slice id."""
    session = _make_rx_session(1)
    session._kv_tasks[0].expected_transfers = 1
    session._receiver._bounce.is_bounced.return_value = False

    session.process_kv_agent_result(
        peer_rank=0,
        receiver_slice_id=0,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
    )

    assert session._kv_tasks[0].status == TaskStatus.TRANSFERRED


def test_process_kv_agent_result_rejects_unknown_receiver_slice_id():
    """Indexing is bounded by the receiver's own task count."""
    session = _make_rx_session(1)

    with pytest.raises(AssertionError, match=r"receiver_slice_id=2"):
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=2,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )


# ---------------------------------------------------------------------------
# TxSession multi-slice status tests (real class)
# ---------------------------------------------------------------------------


def test_tx_session_status_init_until_all_transferred():
    """TxSession status is not KV_TRANSFERRED until ALL tasks complete."""
    session = _make_tx_session(3)
    session.receiver_ready = True
    assert session.status == SessionStatus.TRANSFERRING or session.status == SessionStatus.READY

    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    assert session.status != SessionStatus.KV_TRANSFERRED

    session.kv_tasks[1].status = TaskStatus.TRANSFERRED
    assert session.status != SessionStatus.KV_TRANSFERRED

    session.kv_tasks[2].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_tx_session_intermediate_slice_cannot_complete_session():
    """Transferred intermediate tasks do not make an open session complete."""
    session = TxSession(
        request_id=42,
        params=_make_params(),
        sender=_stub_sender(),
        prompt_len=8,
    )
    session.send(
        KVSlice(
            is_last_slice=False,
            block_ids_per_layer_groups=[[0]],
        )
    )
    session.kv_tasks[0].complete()

    assert session.status != SessionStatus.KV_TRANSFERRED
    assert not session.is_completed()


def test_tx_session_wait_complete_all_tasks():
    """TxSession.wait_complete blocks on all task futures."""
    session = _make_tx_session(3)
    for task in session.kv_tasks:
        task.complete()

    result = session.wait_complete()
    assert result == WaitResult.COMPLETED


# ---------------------------------------------------------------------------
# RxSession status tests (real class)
#
# Chunking is sender-side only: request_and_receive_async posts one slice, so a
# receive session has exactly one task no matter how many chunks arrive. These
# cases stay single-task for that reason, unlike the TxSession ones above.
# ---------------------------------------------------------------------------


def test_rx_session_status_follows_its_single_task():
    """RxSession status mirrors the state of the one task it posted."""
    session = _make_rx_session(1)
    assert session.status == SessionStatus.INIT

    session._kv_tasks[0].status = TaskStatus.TRANSFERRING
    assert session.status == SessionStatus.TRANSFERRING

    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_rx_session_process_aux_completes_at_expected_transfers():
    """Aux completes only once the expected transfer count is reached."""
    session = _make_rx_session(1)
    session._kv_tasks[0].expected_transfers = 2

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session._aux_status != TaskStatus.TRANSFERRED

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session._aux_status == TaskStatus.TRANSFERRED


def test_rx_session_wait_complete_reports_task_outcome():
    """RxSession.wait_complete blocks on its task future and reports its outcome."""
    session = _make_rx_session(1)
    session._kv_tasks[0].complete()

    assert session.wait_complete() == WaitResult.COMPLETED


# ---------------------------------------------------------------------------
# Pipelined transfer tests
# ---------------------------------------------------------------------------


def _make_respond_transceiver(session, kv_slice, *, pipelined=True):
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._enable_pipelined_transfer = pipelined
    transceiver._get_or_create_send_session = MagicMock(return_value=session)
    transceiver._build_prefill_chunk = MagicMock(return_value=kv_slice)
    transceiver._create_kv_slice = MagicMock(return_value=kv_slice)
    transceiver._finalize_send = MagicMock()
    return transceiver


def test_pipelined_transfer_disabled_by_default():
    """pipeline_transfer_enabled reflects the configured flag."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = False

    result = KvCacheTransceiverV2.pipeline_transfer_enabled.fget(transceiver)
    assert result is False


def test_pipelined_transfer_allows_pipeline_parallelism_at_initialization():
    """Generation workers may initialize pipelining with pipeline parallelism."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(pp_size=2)
    transceiver._kv_cache_manager = MagicMock()
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
    )

    assert KvCacheTransceiverV2._resolve_pipelined_transfer(transceiver, cache_transceiver_config)


def test_pipelined_transfer_rejects_bounce_buffer():
    """Bounce buffers stage whole requests rather than individual chunks."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(pp_size=1)
    transceiver._kv_cache_manager = MagicMock()
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
        kv_cache_bounce_size_mb=1,
    )

    with pytest.raises(
        ValueError,
        match="not supported with kv_cache_bounce_size_mb=1",
    ):
        KvCacheTransceiverV2._resolve_pipelined_transfer(transceiver, cache_transceiver_config)


def test_pipelined_transfer_rejects_mamba_cache_manager():
    """Pipelined transfer does not support recurrent-state cache managers."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._mapping = SimpleNamespace(pp_size=1)
    transceiver._kv_cache_manager = MagicMock(spec=MambaHybridCacheManager)
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
    )

    with pytest.raises(
        ValueError,
        match="not supported with a Mamba/hybrid cache manager",
    ):
        KvCacheTransceiverV2._resolve_pipelined_transfer(transceiver, cache_transceiver_config)


def test_python_transceiver_rejects_cpp_mamba_cache_manager():
    """Python transceiver requires separate Python-managed Mamba state."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import CppMambaHybridCacheManager

    kv_cache_manager = object.__new__(CppMambaHybridCacheManager)
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        transceiver_runtime="PYTHON",
    )

    # A hybrid manager arrives as both kv_cache_manager and mamba_cache_manager,
    # the way _util.py passes it.
    with pytest.raises(
        ValueError,
        match="cannot drive CppMambaHybridCacheManager",
    ):
        create_kv_cache_transceiver(
            MagicMock(),
            MagicMock(),
            kv_cache_manager,
            MagicMock(),
            cache_transceiver_config,
            kv_cache_manager,
        )


def test_pipelined_transfer_requires_gen_first_flow():
    """ValueError when a real request is not using gen-first flow."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine.attn_runtime_features.chunked_prefill = True
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.is_context_only_request = True
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST
    )

    with pytest.raises(
        ValueError,
        match="requires schedule_style='generation_first' on the request",
    ):
        PyExecutor._validate_request(executor, request)


def test_pipelined_transfer_requires_chunked_prefill_for_auto_deploy():
    """AutoDeploy reports disabled chunking with the intended validation error."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine = SimpleNamespace(_enable_chunked_prefill=False)

    request = MagicMock()
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )

    with pytest.raises(
        ValueError,
        match="enable_chunked_prefill is required when enable_pipelined_transfer is set",
    ):
        PyExecutor._validate_request(executor, request)


def test_pipelined_transfer_accepts_chunked_prefill_for_auto_deploy():
    """AutoDeploy's chunked-prefill flag satisfies context validation."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine = SimpleNamespace(_enable_chunked_prefill=True)
    executor.dist.pp_size = 1
    executor.dist.cp_size = 1
    executor.max_beam_width = 1
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )

    PyExecutor._validate_request(executor, request)

    executor.sampler.validate_request.assert_called_once_with(request)


def test_pipelined_transfer_rejects_pipeline_parallelism_for_context_request():
    """Context workers require all layers on one pipeline rank."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine.attn_runtime_features.chunked_prefill = True
    executor.dist.pp_size = 2

    request = MagicMock()
    request.py_beam_width = 1
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )

    with pytest.raises(
        ValueError,
        match="not supported with pipeline_parallel_size=2 on context workers",
    ):
        PyExecutor._validate_request(executor, request)


def test_pipelined_transfer_rejects_helix_receiver():
    sender = _make_projection_sender()
    sender._registrar.get_peer_rank_info.return_value.cp_size = 2

    with pytest.raises(
        ValueError,
        match=r"context parallelism \(sender cp_size=1, receiver cp_size=2\)",
    ):
        sender._build_kv_write_meta(_make_projection_task(), _make_projection_req_info())


def test_pipelined_transfer_requires_single_beam_for_context_request():
    """Context-side pipelining rejects beam search before scheduling."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine.attn_runtime_features.chunked_prefill = True

    request = MagicMock()
    request.py_beam_width = 2
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.is_context_only_request = True
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST
    )

    with pytest.raises(ValueError, match="requires beam_width == 1, got 2"):
        PyExecutor._validate_request(executor, request)


def test_pipelined_transfer_allows_non_disaggregated_request():
    """Requests without disaggregated parameters do not transfer KV cache."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.max_beam_width = 1
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY
    request.is_context_only_request = True
    request.py_disaggregated_params = None

    PyExecutor._validate_request(executor, request)

    executor.sampler.validate_request.assert_called_once_with(request)


def test_pipelined_transfer_allows_generation_only_request():
    """Generation workers do not build prefill chunks or enforce sender-only limits."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.max_beam_width = 2
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.model_engine.attn_runtime_features.chunked_prefill = False
    executor.dist.pp_size = 2
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 2
    request.llm_request_type = LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    request.is_context_only_request = False
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST
    )

    PyExecutor._validate_request(executor, request)

    executor.sampler.validate_request.assert_called_once_with(request)


def test_pipelined_last_chunk_sends_and_finalizes():
    """respond_and_send_async sends the built chunk and finalizes on the last chunk."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.kv_tasks = []

    last_slice = KVSlice(
        is_last_slice=True,
        block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
    )

    transceiver = _make_respond_transceiver(session, last_slice)

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._build_prefill_chunk.assert_called_once_with(request)
    transceiver._create_kv_slice.assert_not_called()
    session.send.assert_called_once_with(last_slice)
    transceiver._finalize_send.assert_called_once_with(request, session)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS


def test_non_pipelined_transfer_builds_whole_slice():
    """respond_and_send_async builds a monolithic slice when pipelining is disabled."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    whole_slice = KVSlice(
        is_last_slice=True,
        block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
    )
    transceiver = _make_respond_transceiver(session, whole_slice, pipelined=False)
    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._create_kv_slice.assert_called_once_with(request)
    transceiver._build_prefill_chunk.assert_not_called()
    session.send.assert_called_once_with(whole_slice)
    transceiver._finalize_send.assert_called_once_with(request, session)


def test_pipelined_non_last_chunk_does_not_finalize():
    """respond_and_send_async sends non-final chunks without finalizing."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.kv_tasks = []

    mid_slice = KVSlice(
        is_last_slice=False,
        block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
    )

    transceiver = _make_respond_transceiver(session, mid_slice)

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._build_prefill_chunk.assert_called_once_with(request)
    transceiver._create_kv_slice.assert_not_called()
    session.send.assert_called_once_with(mid_slice)
    transceiver._finalize_send.assert_not_called()


def test_pipelined_chunk_without_a_complete_block_is_not_sent():
    """A chunk that completes no block leaves the session untouched."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.kv_tasks = []

    transceiver = _make_respond_transceiver(session, None)

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._build_prefill_chunk.assert_called_once_with(request)
    transceiver._create_kv_slice.assert_not_called()
    session.send.assert_not_called()
    transceiver._finalize_send.assert_not_called()


def test_failed_pipelined_send_retires_without_mutating_request_state():
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    request = SimpleNamespace(
        state=LlmRequestState.CONTEXT_INIT,
        py_kv_send_session_retired=False,
    )
    session = MagicMock()

    KvCacheTransceiverV2._close_failed_sessions(
        MagicMock(), {42: session}, {42: request}, [42], mark_retired=True
    )

    assert request.state == LlmRequestState.CONTEXT_INIT
    assert request.py_kv_send_session_retired
    session.close.assert_called_once()


def test_pipelined_multiple_chunks_use_real_builder_and_tx_session():
    """Drive two chunks through respond_and_send_async and a real TxSession."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    rid = 42
    tokens_per_block = 4
    source_block_ids = np.arange(4, dtype=np.int64)
    session = TxSession(
        request_id=rid,
        params=_make_params(rid),
        sender=_stub_sender(),
        prompt_len=16,
    )

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._enable_pipelined_transfer = True
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._ever_had_send_session = False
    transceiver._transfer_worker = SimpleNamespace(create_tx_session=lambda _req: session)
    transceiver._reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_transfer_span=lambda _req, _idx, _lg: (source_block_ids, 0),
    )
    transceiver._page_table = SimpleNamespace(
        layer_groups=[SimpleNamespace(kind=CacheKind.PAGED, sliding_window_size=None)]
    )
    transceiver._kv_cache_manager = SimpleNamespace(tokens_per_block=tokens_per_block)
    transceiver._mapping = SimpleNamespace(cp_size=1)
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "ctx"

    request = SimpleNamespace(
        py_disaggregated_params=_make_params(rid),
        request_id=rid,
        py_request_id=rid,
        prompt_len=16,
        py_beam_width=1,
        py_kv_send_session_retired=False,
        prepopulated_prompt_len=0,
        is_generation_only_request=lambda: False,
        set_kv_cache_transfer_start=lambda _ts: None,
        state=LlmRequestState.CONTEXT_INIT,
    )

    request.py_last_context_chunk = (0, 8)
    request.context_remaining_length = 8
    transceiver.respond_and_send_async(request)

    request.py_last_context_chunk = (8, 16)
    request.context_remaining_length = 0
    transceiver.respond_and_send_async(request)

    assert [task._slice.first_ordinals for task in session.kv_tasks] == [[0], [2]]
    assert [task._slice.block_ids_per_layer_groups[0].tolist() for task in session.kv_tasks] == [
        [0, 1],
        [2, 3],
    ]
    assert [task._slice.is_last_slice for task in session.kv_tasks] == [False, True]
    assert transceiver._send_sessions == {rid: session}
    assert transceiver._send_reqs == {rid: request}
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    session.close()


# ---------------------------------------------------------------------------
# Context-side prefix reuse
# ---------------------------------------------------------------------------

_REUSE_TPB = 4
_REUSE_TOTAL_BLOCKS = 8


def _build_prefill_chunk_tokens_for(
    prepopulated_tokens,
    chunk_start_pos,
    chunk_end_pos,
    resident_blocks=None,
    sliding_window_size=_REUSE_TOTAL_BLOCKS * _REUSE_TPB,
    source_block_ids=None,
    span_anchor=0,
):
    """Drive the real _build_prefill_chunk for one chunk, in token coordinates.

    ``resident_blocks`` is how many blocks the mocked ``_create_kv_slice`` hands
    back, and defaults to the block holding ``chunk_end_pos``. ``span_anchor``
    is the block ordinal the mocked base slice is anchored at.

    ``sliding_window_size`` defaults to a full-attention layer as the V1
    extractor actually builds one: its groups come from max_attention_window_vec
    (resource_manager._get_window_size_to_layers), which has no None to hand
    out, so a full-attention group carries max_attention_window. V2 spells the
    same thing as None. Passing a window shorter than the prompt makes the group
    genuinely windowed.
    """
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    if resident_blocks is None:
        resident_blocks = (chunk_end_pos + _REUSE_TPB - 1) // _REUSE_TPB
    if source_block_ids is None:
        source_block_ids = np.arange(resident_blocks, dtype=np.int64)
    base_slice = KVSlice(
        block_ids_per_layer_groups=[np.asarray(source_block_ids, dtype=np.int64)],
        first_ordinals=[span_anchor],
    )

    transceiver = MagicMock()
    # Bind the real predicate: a bare MagicMock's truthy auto-mock would defer
    # every group (even full-attention) to the final slice.
    transceiver._defers_to_final_slice = KvCacheTransceiverV2._defers_to_final_slice
    transceiver._kv_cache_manager.tokens_per_block = _REUSE_TPB
    transceiver._create_kv_slice.return_value = base_slice
    transceiver._page_table = SimpleNamespace(
        layer_groups=[
            SimpleNamespace(kind=CacheKind.PAGED, sliding_window_size=sliding_window_size)
        ]
    )
    transceiver._send_reqs = {}

    req = MagicMock()
    req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=42)
    req.py_beam_width = 1
    req.prompt_len = _REUSE_TOTAL_BLOCKS * _REUSE_TPB
    req.prepopulated_prompt_len = prepopulated_tokens
    req.py_last_context_chunk = (chunk_start_pos, chunk_end_pos)
    req.context_remaining_length = req.prompt_len - chunk_end_pos

    return KvCacheTransceiverV2._build_prefill_chunk(transceiver, req)


def _build_prefill_chunk_for(
    prepopulated_blocks,
    chunk_start_block,
    chunk_end_block,
    resident_blocks=None,
):
    """Drive the real _build_prefill_chunk for one block-aligned chunk."""
    return _build_prefill_chunk_tokens_for(
        prepopulated_tokens=prepopulated_blocks * _REUSE_TPB,
        chunk_start_pos=chunk_start_block * _REUSE_TPB,
        chunk_end_pos=chunk_end_block * _REUSE_TPB,
        resident_blocks=chunk_end_block if resident_blocks is None else resident_blocks,
    )


def test_build_prefill_chunk_rounds_unaligned_non_final_end_down():
    """An unaligned non-final end stops at the last block it finished computing."""
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=0,
        chunk_end_pos=6,
    )

    assert kv_slice.is_last_slice is False
    assert kv_slice.first_ordinals == [0]
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(1, dtype=np.int64))


def test_unaligned_chunk_boundaries_tile_block_space_exactly():
    """Rounding both bounds down covers every block exactly once."""
    chunk_bounds = [(0, 6), (6, 12), (12, _REUSE_TOTAL_BLOCKS * _REUSE_TPB)]
    slices = [
        _build_prefill_chunk_tokens_for(
            prepopulated_tokens=0,
            chunk_start_pos=start,
            chunk_end_pos=end,
        )
        for start, end in chunk_bounds
    ]

    block_spans = [
        (s.first_ordinals[0], s.first_ordinals[0] + s.block_ids_per_layer_groups[0].size)
        for s in slices
    ]
    assert block_spans == [(0, 1), (1, 3), (3, _REUSE_TOTAL_BLOCKS)]

    covered = [set(range(start, end)) for start, end in block_spans]
    assert set().union(*covered) == set(range(_REUSE_TOTAL_BLOCKS))
    assert sum(len(c) for c in covered) == _REUSE_TOTAL_BLOCKS
    assert slices[-1].is_last_slice is True


@pytest.mark.parametrize(
    "chunk_start_pos,chunk_end_pos",
    [(0, 2), (5, 7)],
    ids=["first_chunk", "later_chunk"],
)
def test_chunk_completing_no_block_sends_nothing(chunk_start_pos, chunk_end_pos):
    """A chunk inside a single block has nothing to send; the next chunk covers it."""
    assert (
        _build_prefill_chunk_tokens_for(
            prepopulated_tokens=0,
            chunk_start_pos=chunk_start_pos,
            chunk_end_pos=chunk_end_pos,
        )
        is None
    )


def test_swa_blocks_are_deferred_until_the_complete_final_window():
    """Non-final slices omit SWA; the final slice carries its complete active suffix."""
    first_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=0,
        chunk_end_pos=16,
        resident_blocks=4,
        sliding_window_size=16,
    )
    final_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=16,
        chunk_end_pos=_REUSE_TOTAL_BLOCKS * _REUSE_TPB,
        resident_blocks=_REUSE_TOTAL_BLOCKS,
        sliding_window_size=16,
        source_block_ids=np.arange(4, _REUSE_TOTAL_BLOCKS),
        span_anchor=4,
    )

    assert first_slice is None
    assert final_slice.is_last_slice is True
    assert np.array_equal(
        final_slice.block_ids_per_layer_groups[0],
        np.arange(4, _REUSE_TOTAL_BLOCKS, dtype=np.int64),
    )
    # The deferred window rides anchored at its true window-start ordinal, not
    # at the final chunk's start.
    assert final_slice.first_ordinals == [4]


@pytest.mark.parametrize(
    "window_tokens",
    [None, _REUSE_TOTAL_BLOCKS * _REUSE_TPB, _REUSE_TOTAL_BLOCKS * _REUSE_TPB + _REUSE_TPB],
    ids=["no_window_v2", "window_equals_prompt", "window_exceeds_prompt"],
)
def test_window_covering_the_whole_prompt_streams_like_full_attention(window_tokens):
    """A window the prompt never outgrows evicts nothing, so its chunks stream.

    The two managers describe a full-attention layer differently: V2 uses None
    while V1 hands over max_attention_window, so a group is only genuinely
    windowed when its window cuts the prompt short. Deferring on a non-None
    window alone would hold every V1 group back to the last chunk and silently
    disable pipelining. All three spellings must behave identically.
    """
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=0,
        chunk_end_pos=16,
        resident_blocks=4,
        sliding_window_size=window_tokens,
    )

    assert kv_slice is not None
    assert kv_slice.is_last_slice is False
    assert kv_slice.first_ordinals == [0]
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(4, dtype=np.int64))


def test_unaligned_reuse_prefix_still_extends_first_chunk_to_block_zero():
    """A partial-block reuse hit leaves the first chunk unaligned at both ends."""
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=6,
        chunk_start_pos=6,
        chunk_end_pos=14,
    )

    assert kv_slice.first_ordinals == [0]
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(3, dtype=np.int64))


def test_first_chunk_covers_ctx_prefix_reuse():
    """The reused prefix is resident but no chunk spans it, so slice 0 extends to block 0."""
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=3,
        chunk_start_block=3,
        chunk_end_block=6,
    )

    assert kv_slice.first_ordinals == [0]
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(6, dtype=np.int64))
    assert kv_slice.is_last_slice is False


@pytest.mark.parametrize(
    "prepopulated_blocks,chunk_start_block,chunk_end_block,expected_start_block",
    [
        (3, 6, 8, 6),
        (0, 4, 8, 4),
        (0, 0, 4, 0),
    ],
    ids=["after_reuse_hit", "no_reuse_later_chunk", "no_reuse_first_chunk"],
)
def test_only_the_first_chunk_extends_to_block_zero(
    prepopulated_blocks, chunk_start_block, chunk_end_block, expected_start_block
):
    """Chunks past the first keep their own start; without reuse nothing changes."""
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=prepopulated_blocks,
        chunk_start_block=chunk_start_block,
        chunk_end_block=chunk_end_block,
        resident_blocks=_REUSE_TOTAL_BLOCKS,
    )

    assert kv_slice.first_ordinals == [expected_start_block]
    assert np.array_equal(
        kv_slice.block_ids_per_layer_groups[0],
        np.arange(expected_start_block, chunk_end_block, dtype=np.int64),
    )


def test_single_chunk_with_reuse_degenerates_to_monolithic_slice():
    """One chunk plus a reuse hit yields the same slice shape a monolithic send would.

    The chunk still spans [0, total_blocks) anchored at 0, which
    _build_kv_write_meta addresses exactly as an unpipelined write — see
    test_chunked_slices_reassemble_to_the_monolithic_transfer.
    """
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=3,
        chunk_start_block=3,
        chunk_end_block=_REUSE_TOTAL_BLOCKS,
        resident_blocks=_REUSE_TOTAL_BLOCKS,
    )

    assert kv_slice.is_last_slice is True
    assert kv_slice.first_ordinals == [0]
    assert np.array_equal(
        kv_slice.block_ids_per_layer_groups[0],
        np.arange(_REUSE_TOTAL_BLOCKS, dtype=np.int64),
    )


def test_chunk_before_an_anchored_group_span_contributes_nothing():
    """A chunk entirely before a group's anchored span sends no blocks for it."""
    # The group's span covers block ordinals [4, 8); the chunk covers [0, 4).
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=0,
        chunk_end_pos=4 * _REUSE_TPB,
        resident_blocks=4,
        source_block_ids=np.array([104, 105, 106, 107], dtype=np.int64),
        span_anchor=4,
    )

    # Nothing to send yet: the only group contributes no blocks.
    assert kv_slice is None


def test_chunk_overlapping_an_anchored_group_span_starts_at_the_anchor():
    """The chunk's sub-slice starts at max(chunk_start, span anchor)."""
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=3 * _REUSE_TPB,
        chunk_end_pos=6 * _REUSE_TPB,
        resident_blocks=4,
        source_block_ids=np.array([104, 105, 106, 107], dtype=np.int64),
        span_anchor=4,
    )

    # Span [4, 8) ∩ chunk [3, 6) = [4, 6): blocks 104, 105 anchored at 4.
    assert kv_slice.first_ordinals == [4]
    assert np.array_equal(
        kv_slice.block_ids_per_layer_groups[0], np.array([104, 105], dtype=np.int64)
    )


def test_span_not_covering_the_chunk_end_raises():
    """A short streaming-group span cannot address the chunk by ordinal.

    A ValueError (not a bare assert) so the guard survives python -O: a short
    span silently transferring misaligned chunk blocks is the failure mode the
    anchors exist to prevent.
    """
    with pytest.raises(ValueError, match="does not cover the chunk end"):
        _build_prefill_chunk_tokens_for(
            prepopulated_tokens=0,
            chunk_start_pos=0,
            chunk_end_pos=_REUSE_TOTAL_BLOCKS * _REUSE_TPB,  # final chunk: end block 8
            resident_blocks=2,
            source_block_ids=np.array([104, 105], dtype=np.int64),  # span [4, 6)
            span_anchor=4,
        )


# ---------------------------------------------------------------------------
# Transfer activity as a dimension owned by the transceiver
# ---------------------------------------------------------------------------


def _make_transfer_state_transceiver(session=None, rid: int = 42):
    """Transceiver stub whose session maps are the ownership record."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    sessions = {rid: session} if session is not None else {}
    transceiver = SimpleNamespace(
        _wait_reqs={},
        _send_sessions=dict(sessions),
        _send_reqs={rid: MagicMock()} if session is not None else {},
        _recv_sessions={},
        _recv_reqs={},
    )
    # Real teardown, so the predicate is checked against the actual bookkeeping.
    transceiver._retire_send_session = MethodType(
        KvCacheTransceiverV2._retire_send_session, transceiver
    )
    return transceiver


def _make_transfer_state_request(rid=42, request_id: int = 42):
    return SimpleNamespace(
        py_disaggregated_params=(
            DisaggregatedParams(disagg_request_id=rid) if rid is not None else None
        ),
        request_id=request_id,
    )


def test_has_inflight_transfer_tracks_send_session_lifetime():
    """Session membership answers the predicate, before and after teardown."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.has_transferring_tasks.return_value = False
    transceiver = _make_transfer_state_transceiver(session)
    request = _make_transfer_state_request()

    assert KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)

    assert KvCacheTransceiverV2.cancel_request(transceiver, request)

    assert not KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)


def test_has_inflight_transfer_false_without_disagg_params():
    """A request that never registered a session owns no transfer resources."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = _make_transfer_state_transceiver()
    request = _make_transfer_state_request(rid=None, request_id=7)

    assert not KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)


def test_is_request_in_transmission_uses_transceiver_predicate():
    """A mid-prefill request still counts as transmitting despite CONTEXT_INIT."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.has_inflight_transfer.return_value = True

    request = SimpleNamespace(state=LlmRequestState.CONTEXT_INIT)

    assert PyExecutor._is_request_in_transmission(executor, request)
    executor.kv_cache_transceiver.has_inflight_transfer.assert_called_once_with(request)


def test_is_request_in_transmission_false_when_nothing_in_flight():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.has_inflight_transfer.return_value = False

    request = SimpleNamespace(state=LlmRequestState.CONTEXT_INIT)

    assert not PyExecutor._is_request_in_transmission(executor, request)


def _make_send_kv_executor(canceled_req_ids):
    executor = MagicMock()
    executor.kv_connector_manager = None
    executor.canceled_req_ids = list(canceled_req_ids)
    executor._pending_ctx_transfer_failures = set()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = None
    executor.kv_cache_transceiver.has_retired_send_session.return_value = False
    return executor


def _make_send_kv_request(is_last_chunk: bool, request_id: int = 7):
    return SimpleNamespace(
        is_context_only_request=True,
        is_finished_due_to_cancellation=False,
        is_context_finished=False,
        is_finished_due_to_length=is_last_chunk,
        is_child=False,
        parent_request_id=None,
        py_disaggregated_params=None,
        request_id=request_id,
        py_request_id=request_id,
        py_kv_transfer_start_time=None,
        state=(
            LlmRequestState.GENERATION_COMPLETE if is_last_chunk else LlmRequestState.CONTEXT_INIT
        ),
    )


def test_send_disagg_ctx_kv_sends_intermediate_chunk_when_not_cancelled():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    request = _make_send_kv_request(is_last_chunk=False)

    PyExecutor._send_disagg_ctx_kv_async(executor, [request])

    executor.kv_cache_transceiver.respond_and_send_async.assert_called_once_with(request)


def test_send_disagg_ctx_kv_sends_final_chunk_in_generation_complete_state():
    """The final context slice is not suppressed by the intermediate-slice state gate."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    request = _make_send_kv_request(is_last_chunk=True)

    PyExecutor._send_disagg_ctx_kv_async(executor, [request])

    executor.async_transfer_manager.start_transfer.assert_called_once_with(request)
    executor.kv_cache_transceiver.respond_and_send_async.assert_called_once_with(request)


@pytest.mark.parametrize(
    ("is_last_chunk", "expected_state"),
    [
        (False, LlmRequestState.CONTEXT_INIT),
        (True, LlmRequestState.GENERATION_COMPLETE),
    ],
)
def test_send_disagg_ctx_kv_skips_retired_session_without_mutating_state(
    is_last_chunk, expected_state
):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    executor.kv_cache_transceiver.has_retired_send_session.return_value = True
    request = _make_send_kv_request(is_last_chunk=is_last_chunk)

    PyExecutor._send_disagg_ctx_kv_async(executor, [request])

    assert request.state == expected_state
    executor.async_transfer_manager.start_transfer.assert_not_called()
    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()


def test_context_send_failure_is_applied_at_next_loop_boundary():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    request = _make_send_kv_request(is_last_chunk=False)
    request.py_kv_transfer_timed_out = False
    executor.active_requests = [request]
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = SimpleNamespace(
        completed_request_ids=[], error_request_ids=[request.py_request_id]
    )
    executor.async_transfer_manager.requests_in_transfer.return_value = {}

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor)

    assert request.state == LlmRequestState.CONTEXT_INIT
    assert executor._pending_ctx_transfer_failures == {request.py_request_id}
    executor.async_transfer_manager.end_transfer.assert_not_called()

    executor.enable_attention_dp = False
    executor.dist.world_size = 1
    executor._is_disagg_inflight_cancel_active.return_value = False

    PyExecutor._handle_disagg_cache_errors_synced(executor)

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not executor._pending_ctx_transfer_failures
    executor._check_cache_transfer_errors.assert_called_with("context requests")


@pytest.mark.parametrize("is_last_chunk", [False, True])
def test_send_disagg_ctx_kv_skips_pending_cancellation(is_last_chunk):
    """A pending cancellation suppresses both intermediate and final slices."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    request = _make_send_kv_request(is_last_chunk=is_last_chunk)
    executor = _make_send_kv_executor([request.py_request_id])

    PyExecutor._send_disagg_ctx_kv_async(executor, [request])

    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()
    executor.async_transfer_manager.start_transfer.assert_not_called()
    executor.kv_cache_manager.release_index_slot.assert_not_called()
