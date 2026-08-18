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
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice
from tensorrt_llm._torch.disaggregation.native.transfer import project_blocks_to_chunk


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
        for resident_end in range(0, 7):
            for n in range(0, resident_end + 1):
                resident_start = resident_end - n
                blocks = np.arange(resident_start, resident_end, dtype=np.int64)
                for start, end in itertools.combinations_with_replacement(
                    range(0, resident_end + 2), 2
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
