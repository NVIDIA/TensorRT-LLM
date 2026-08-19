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
"""CPU-only unit tests for pipelined (chunked) KV cache transfer.

Covers the block-space projection that lets a chunking sender address a
monolithic receiver, plus the chunk geometry built by
``KvCacheTransceiverV2._build_prefill_chunk``.
"""

import itertools
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

# Nothing here needs a GPU; without this the file lands in no CI stage.
pytestmark = pytest.mark.cpu_only


class TestProjectBlocksToChunk:
    """``block_ids`` is the resident suffix of ``[0, resident_block_end)``."""

    @pytest.mark.parametrize("prompt_blocks", range(0, 8))
    def test_whole_request_is_identity(self, prompt_blocks):
        """The monolithic path must stay bit-identical.

        A whole-request slice projects [0, P) onto a list resident in [0, P),
        so every resident suffix has to come back unchanged.
        """
        for resident in range(0, prompt_blocks + 1):
            blocks = np.arange(100, 100 + resident, dtype=np.int64)
            out = project_blocks_to_chunk(blocks, 0, prompt_blocks, prompt_blocks)
            assert np.array_equal(out, blocks)

    @pytest.mark.parametrize("prompt_blocks", range(1, 8))
    def test_split_reassembles_the_request(self, prompt_blocks):
        """Two abutting chunks rebuild the list with no gap and no overlap."""
        full = np.arange(500, 500 + prompt_blocks, dtype=np.int64)
        for cut in range(0, prompt_blocks + 1):
            head = project_blocks_to_chunk(full, 0, cut, prompt_blocks)
            tail = project_blocks_to_chunk(full, cut, prompt_blocks, prompt_blocks)
            assert np.array_equal(np.concatenate([head, tail]), full)

    def test_suffix_resident_list_projects_by_global_id(self):
        """A prefix-reuse hit shortens the list; projection stays global.

        Each block's value equals its own global index, so a wrong offset shows
        up as wrong values rather than only a wrong length.
        """
        for resident_end in range(0, 9):
            for n in range(0, resident_end + 1):
                resident_start = resident_end - n
                blocks = np.arange(resident_start, resident_end, dtype=np.int64)
                for start, end in itertools.combinations_with_replacement(
                    range(0, resident_end + 4), 2
                ):
                    out = project_blocks_to_chunk(blocks, start, end, resident_end)
                    want = np.array(
                        [g for g in range(resident_start, resident_end) if start <= g < end],
                        dtype=np.int64,
                    )
                    assert np.array_equal(out, want), (
                        f"resident=[{resident_start},{resident_end}) chunk=[{start},{end})"
                    )

    def test_degenerate_ranges_are_empty_not_errors(self):
        blocks = np.arange(4, dtype=np.int64)
        assert project_blocks_to_chunk(np.array([], dtype=np.int64), 0, 4, 4).size == 0
        assert project_blocks_to_chunk(blocks, 2, 2, 4).size == 0  # zero width
        assert project_blocks_to_chunk(blocks, 3, 1, 4).size == 0  # inverted
        assert project_blocks_to_chunk(blocks, 9, 12, 4).size == 0  # past resident

    def test_dtype_is_preserved(self):
        """extract() indexes KV pools with these, so they must stay int64."""
        out = project_blocks_to_chunk(np.arange(6, dtype=np.int64), 1, 4, 6)
        assert out.dtype == np.int64


_TPB = 4
_TOTAL_BLOCKS = 8
_PROMPT_LEN = _TOTAL_BLOCKS * _TPB


def _drive_build_prefill_chunk(
    prepopulated_tokens, chunk_start_pos, chunk_end_pos, resident_blocks=None
):
    """Run the real ``_build_prefill_chunk`` for one chunk, in token coordinates.

    Only the collaborators it reads are stubbed, so the chunk geometry, the
    projection and the token_range it emits are the shipped ones.
    """
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    if resident_blocks is None:
        resident_blocks = min((chunk_end_pos + _TPB - 1) // _TPB, _TOTAL_BLOCKS)
    transceiver = MagicMock()
    transceiver._reuse_adapter.tokens_per_block = _TPB
    transceiver._create_kv_slice.return_value = KVSlice(
        block_ids_per_layer_groups=[np.arange(resident_blocks, dtype=np.int64)]
    )
    transceiver._need_aux_transfer.return_value = True

    req = MagicMock()
    req.py_beam_width = 1
    req.prompt_len = _PROMPT_LEN
    req.prepopulated_prompt_len = prepopulated_tokens
    req.py_last_context_chunk = (chunk_start_pos, chunk_end_pos)
    req.context_remaining_length = req.prompt_len - chunk_end_pos
    return KvCacheTransceiverV2._build_prefill_chunk(transceiver, req)


def _blocks_of(kv_slice):
    """(start_block, end_block) a slice addresses, read back from token_range."""
    return kv_slice.token_range.start // _TPB, kv_slice.token_range.end // _TPB


class TestPrefillChunkRealFunction:
    """Drives the shipped ``_build_prefill_chunk`` rather than restating its math."""

    def test_aligned_chunks_meet_exactly(self):
        first = _drive_build_prefill_chunk(0, 0, 4 * _TPB)
        second = _drive_build_prefill_chunk(0, 4 * _TPB, _PROMPT_LEN)
        assert _blocks_of(first) == (0, 4)
        assert _blocks_of(second) == (4, 8)
        assert first.is_last_slice is False
        assert second.is_last_slice is True

    def test_unaligned_boundary_block_is_sent_once(self):
        """Token 10 sits inside block 2, only half computed when the cut lands."""
        first = _drive_build_prefill_chunk(0, 0, 10)
        second = _drive_build_prefill_chunk(0, 10, _PROMPT_LEN)
        assert _blocks_of(first) == (0, 2), "end rounds down: block 2 is incomplete"
        assert _blocks_of(second) == (2, 8)
        assert not (set(range(*_blocks_of(first))) & set(range(*_blocks_of(second))))

    def test_first_chunk_reaches_back_over_a_reused_prefix(self):
        """A ctx-side reuse hit would otherwise leave [0, prepopulated) unsent."""
        kv_slice = _drive_build_prefill_chunk(3 * _TPB, 3 * _TPB, 6 * _TPB)
        assert _blocks_of(kv_slice) == (0, 6)
        assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(6, dtype=np.int64))

    @pytest.mark.parametrize(
        "prepopulated_blocks,start_block,end_block,expected_start",
        [(0, 2, 4, 2), (0, 4, 6, 4), (3, 4, 6, 4), (2, 2, 5, 0)],
    )
    def test_only_the_first_chunk_extends_to_block_zero(
        self, prepopulated_blocks, start_block, end_block, expected_start
    ):
        kv_slice = _drive_build_prefill_chunk(
            prepopulated_blocks * _TPB,
            start_block * _TPB,
            end_block * _TPB,
            resident_blocks=_TOTAL_BLOCKS,
        )
        assert _blocks_of(kv_slice) == (expected_start, end_block)
        assert np.array_equal(
            kv_slice.block_ids_per_layer_groups[0],
            np.arange(expected_start, end_block, dtype=np.int64),
        )

    def test_single_chunk_degenerates_to_a_whole_request_slice(self):
        """One chunk covering the prompt must address exactly like a monolithic send."""
        kv_slice = _drive_build_prefill_chunk(0, 0, _PROMPT_LEN, resident_blocks=_TOTAL_BLOCKS)
        assert kv_slice.is_last_slice is True
        assert _blocks_of(kv_slice) == (0, _TOTAL_BLOCKS)

    def test_chunk_inside_one_block_sends_nothing(self):
        """No block completed, so there is nothing to ship yet."""
        assert _drive_build_prefill_chunk(0, 1, 3) is None

    def test_last_chunk_always_reaches_the_prompt_end(self):
        """Even an unaligned tail is picked up, since prefill has finished."""
        kv_slice = _drive_build_prefill_chunk(0, 6 * _TPB, _PROMPT_LEN)
        assert kv_slice.is_last_slice is True
        assert _blocks_of(kv_slice)[1] == _TOTAL_BLOCKS

    def test_rejects_beam_search(self):
        from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

        transceiver = MagicMock()
        req = MagicMock()
        req.py_beam_width = 2
        with pytest.raises(ValueError, match="beam_width == 1"):
            KvCacheTransceiverV2._build_prefill_chunk(transceiver, req)

    def test_rejects_context_first_schedule_style(self):
        """generation_first is what keeps a session from completing mid-prefill."""
        from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

        transceiver = MagicMock()
        transceiver._need_aux_transfer.return_value = False
        req = MagicMock()
        req.py_beam_width = 1
        with pytest.raises(ValueError, match="generation_first"):
            KvCacheTransceiverV2._build_prefill_chunk(transceiver, req)

    def test_chunks_exactly_partition_the_prompt(self):
        """Replay a full scheduler sequence: every block sent once, none twice."""
        for chunk_tokens in (4, 6, 7, 12, 40):
            seen, pos = [], 0
            while pos < _PROMPT_LEN:
                end = min(pos + chunk_tokens, _PROMPT_LEN)
                kv_slice = _drive_build_prefill_chunk(0, pos, end)
                if kv_slice is not None:
                    seen.extend(range(*_blocks_of(kv_slice)))
                pos = end
            assert sorted(seen) == list(range(_TOTAL_BLOCKS)), f"chunk_tokens={chunk_tokens}"
            assert len(seen) == len(set(seen)), f"duplicate block, chunk_tokens={chunk_tokens}"


_WM_TPB = 8
_WM_PROMPT_LEN = 64  # 8 blocks


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
        tokens_per_block=_WM_TPB,
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
            token_range=TokenRange(start=block_start * _WM_TPB, end=block_end * _WM_TPB),
        ),
        _make_params(),
        slice_id=slice_id,
        prompt_len=_WM_PROMPT_LEN,
    )
    task._unique_rid = 42
    return task


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
            _make_task(0, _WM_PROMPT_LEN // _WM_TPB), _make_projection_req_info()
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
