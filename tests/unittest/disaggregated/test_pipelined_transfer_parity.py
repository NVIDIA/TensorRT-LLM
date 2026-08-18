# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Behavioural parity checks for pipelined KV transfer.

These assertions are ported from an alternative implementation of the same
feature so the two can be compared on behaviour rather than on internals. The
chunk position is carried here by ``KVSlice.token_range`` (which main already
defines for multi-slice transfers) instead of a dedicated coordinate object, so
the setup differs while the expected outcomes do not.

Two of that implementation's assertions are deliberately NOT reproduced, because
this implementation makes the opposite choice on purpose:

* it rounds an unaligned non-final chunk end *up*, so the block holding the cut
  is sent twice -- once with an uncomputed tail -- and correctness then depends
  on the two writes landing in order. This implementation rounds *down*, so that
  block is deferred to the chunk that completes it and is sent exactly once.
  See ``test_unaligned_boundary_block_is_sent_once`` in test_pipelined_transfer.
* it widens the KV_AGENT_RESULT wire struct to carry the sender's chunk index
  alongside the receiver's. That breaks ctx/gen version compatibility for a field
  used only in logs, so here the sender index stays process-local and only the
  receiver's index goes on the wire (asserted below).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, TokenRange
from tensorrt_llm._torch.disaggregation.native.transfer import (
    KVSendTask,
    RecvReqInfo,
    Sender,
    project_blocks_to_chunk,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams

_TPB = 8
_PROMPT_LEN = 64  # 8 blocks


def _make_params(rid: int = 42) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=rid)


def _make_projection_sender() -> Sender:
    """A Sender wired to a stub registrar with two non-windowed layer groups."""
    peer_ri = SimpleNamespace(
        dp_rank=0,
        device_id=0,
        instance_name="decode",
        instance_rank=0,
        self_endpoint="tcp://decode:0",
    )
    extractor = MagicMock()
    extractor.page_table = SimpleNamespace(
        tokens_per_block=_TPB,
        layer_groups=[
            SimpleNamespace(sliding_window_size=None),
            SimpleNamespace(sliding_window_size=None),
        ],
    )
    extractor.extract.side_effect = lambda block_ids, **_: SimpleNamespace(
        memory=SimpleNamespace(ptrs=np.asarray(block_ids, dtype=np.int64), bytes_per_region=1)
    )
    mapper = MagicMock()
    mapper.map.side_effect = lambda src_region, dst_region: SimpleNamespace(
        src=src_region, dst=dst_region
    )
    registrar = MagicMock()
    registrar.self_rank_info = SimpleNamespace()
    registrar.self_extractor = extractor
    registrar.get_peer_rank_info.return_value = peer_ri
    registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    registrar.should_send_pool.return_value = True
    registrar.get_pool_mapping.return_value = {(0, 0): (0, 0), (1, 0): (1, 0)}
    registrar.peer_extractor.return_value = extractor
    registrar.get_kv_map.return_value = mapper

    sender = Sender.__new__(Sender)
    sender._registrar = registrar
    return sender


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
        slice_id=slice_id,
    )


def _make_task(block_start, block_end, slice_id=1, src_per_group=None):
    """A send task for the chunk covering global blocks [block_start, block_end)."""
    if src_per_group is None:
        src_per_group = [
            np.arange(8, dtype=np.int64),
            np.array([10, 11, 12], dtype=np.int64),
        ]
    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=src_per_group,
            token_range=TokenRange(start=block_start * _TPB, end=block_end * _TPB),
        ),
        _make_params(),
        slice_id=slice_id,
        prompt_len=_PROMPT_LEN,
    )
    task._unique_rid = 42
    return task


class TestChunkProjection:
    def test_noops_when_chunk_is_outside_short_layer_group(self):
        """A chunk past a short layer group's resident range is a no-op."""
        block_ids = np.array([10, 11, 12], dtype=np.int64)
        assert project_blocks_to_chunk(block_ids, 4, 8, resident_block_end=3).size == 0

    @pytest.mark.parametrize(
        "resident_block_end,chunk_block_offset,expected",
        [(4, 0, [0, 1, 2, 3]), (8, 4, [4, 5, 6, 7]), (8, 0, [0, 1, 2, 3, 4, 5, 6, 7])],
    )
    def test_maps_incrementally_allocated_source(
        self, resident_block_end, chunk_block_offset, expected
    ):
        """Source blocks end at the current chunk, not at the full prompt."""
        block_ids = np.arange(resident_block_end, dtype=np.int64)
        got = project_blocks_to_chunk(
            block_ids, chunk_block_offset, chunk_block_offset + 16, resident_block_end
        )
        assert np.array_equal(got, np.array(expected, dtype=np.int64))

    def test_maps_prefix_reuse_suffix_by_overlap(self):
        """Suffixes are matched by overlap, not by raw chunk-offset indexing."""
        block_ids = np.array([104, 105, 106, 107], dtype=np.int64)
        first = project_blocks_to_chunk(block_ids, 0, 4, resident_block_end=8)
        second = project_blocks_to_chunk(block_ids, 4, 8, resident_block_end=8)
        assert first.size == 0
        assert np.array_equal(second, block_ids)


class TestWriteMetaAddressing:
    def test_projects_asymmetric_layer_group_chunk(self):
        """A short layer group's suffix transfers with the overlapping global chunk."""
        sender = _make_projection_sender()
        write_meta = sender._build_kv_write_meta(_make_task(4, 8), _make_projection_req_info())

        assert np.array_equal(
            write_meta.src_ptrs, np.array([4, 5, 6, 7, 10, 11, 12], dtype=np.int64)
        )
        assert np.array_equal(
            write_meta.dst_ptrs, np.array([104, 105, 106, 107, 200, 201, 202], dtype=np.int64)
        )
        assert np.array_equal(write_meta.sizes, np.ones(7, dtype=np.int64))

    def test_whole_prompt_chunk_addresses_like_a_monolithic_slice(self):
        """A chunk spanning [0, total_blocks) writes exactly what an unpipelined send does.

        This is the degenerate slice produced when the whole prompt fits in one
        chunk, so the chunked branch must not perturb its addressing.
        """
        sender = _make_projection_sender()
        chunked = sender._build_kv_write_meta(_make_task(0, 8), _make_projection_req_info())
        monolithic = sender._build_kv_write_meta(
            _make_task(0, _PROMPT_LEN // _TPB), _make_projection_req_info()
        )
        assert np.array_equal(chunked.src_ptrs, monolithic.src_ptrs)
        assert np.array_equal(chunked.dst_ptrs, monolithic.dst_ptrs)
        assert np.array_equal(chunked.sizes, monolithic.sizes)

    def test_receiver_slice_id_comes_from_the_peer(self):
        """The result must address the receiver by its task index, not the sender's chunk.

        The peer's index is echoed from RecvReqInfo; the sender's own chunk index
        stays local (WriteMeta.slice_id) and never reaches the wire, so the
        KV_AGENT_RESULT layout is unchanged.
        """
        sender = _make_projection_sender()
        write_meta = sender._build_kv_write_meta(
            _make_task(4, 8, slice_id=1), _make_projection_req_info(slice_id=3)
        )
        assert write_meta.slice_id == 1, "sender index stays local"
        assert write_meta.receiver_slice_id == 3, "peer index is what gets sent"

    def test_receiver_without_slice_id_is_addressed_as_task_zero(self):
        """A sender chunk index past the receiver's task count still resolves task 0."""
        sender = _make_projection_sender()
        write_meta = sender._build_kv_write_meta(
            _make_task(4, 8, slice_id=4), _make_projection_req_info(slice_id=None)
        )
        assert write_meta.slice_id == 4
        assert write_meta.receiver_slice_id == 0
