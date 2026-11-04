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

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.runs import RunsMapper
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    BufferRuns,
    PositionRule,
    TransferWindow,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc, ring

pytestmark = pytest.mark.cpu_only

_BUFFER = BufferDesc(0, "k", 0, 0, 0, 64)


def _rings(src_slots: int, dst_slots: int) -> PositionRule:
    """Rings of two-byte slots on both sides, in pages of 8 tokens."""
    return PositionRule(ring(src_slots, 2)[0].axes[0], ring(dst_slots, 2)[0].axes[0], 8)


def _runs(src, dst, sizes, ring=None) -> BufferRuns:
    return BufferRuns(
        _BUFFER,
        _BUFFER,
        np.array(src, dtype=np.int64),
        np.array(dst, dtype=np.int64),
        np.array(sizes, dtype=np.int64),
        by_position=ring,
    )


def _fragments(mapper, *args, **kwargs):
    """(src pointer, dst pointer, size) of every fragment the mapper gives, sorted."""
    return sorted(
        (int(s), int(d), size)
        for src, dst, size in mapper.map(*args, **kwargs)
        for s, d in zip(src, dst)
    )


def test_runs_mapper_whole_slot_copy():
    """One run over the whole slot is a pass-through copy of each slot."""
    mapper = RunsMapper([_runs([0], [0], [32])])
    ((src, dst, size),) = mapper.map(np.array([100, 200]), np.array([300, 400]))
    np.testing.assert_array_equal(src, [100, 200])
    np.testing.assert_array_equal(dst, [300, 400])
    assert size == 32


def test_runs_mapper_shifts_each_side_by_its_own_offset():
    """Each side's slots shift by that side's own offset of the run.

    The run lies at different offsets of the two slots, so applying one side's offset to the
    other would fail here.
    """
    mapper = RunsMapper([_runs([16], [32], [32])])
    ((src, dst, size),) = mapper.map(np.array([10, 20]), np.array([30, 40]))
    np.testing.assert_array_equal(src, [26, 36])
    np.testing.assert_array_equal(dst, [62, 72])
    assert size == 32


def test_runs_mapper_moves_runs_alike_on_every_page_whatever_their_positions():
    """Runs alike on every page ignore the pages and window a transfer passes."""
    mapper = RunsMapper([_runs([0], [8], [16])])
    bases = (np.array([100, 200]), np.array([300, 400]))
    assert _fragments(mapper, *bases) == _fragments(
        mapper, *bases, pages=np.array([0, 5]), window=TransferWindow(0, 3)
    )


def test_runs_mapper_moves_ring_positions_by_page():
    """Each kept position goes from slot ``p % 4`` to slot ``p % 6``, split where a ring wraps.

    The source has 4 two-byte slots and the destination 6, with 8 tokens a page. At end position
    11, page 1 holds positions 8..10 and page 0 its last four, 4..7. A fixed run beside them
    moves alike on every page.
    """
    ring = _rings(4, 6)
    mapper = RunsMapper([_runs([100], [200], [2], ring=ring), _runs([0], [0], [16])])
    pages = np.array([0, 1])
    got = _fragments(
        mapper,
        np.array([1000, 2000]),
        np.array([5000, 6000]),
        pages=pages,
        window=TransferWindow(0, 11),
    )
    expected = [(1000, 5000, 16), (2000, 6000, 16)]
    # Page 0: positions 4..7 sit in source slots 0..3 and destination slots 4, 5, 0, 1.
    expected += [(1100, 5208, 4), (1104, 5200, 4)]
    # Page 1: positions 8..10 sit in source slots 0..2 and destination slots 2..4.
    expected += [(2100, 6204, 6)]
    assert got == sorted(expected)


def test_runs_mapper_shares_one_page_computation_between_buffers_of_one_shape():
    """Two layers' rings, alike but for where they lie, move the same positions."""
    ring = _rings(4, 4)
    mapper = RunsMapper([_runs([0], [0], [2], ring=ring), _runs([64], [32], [2], ring=ring)])
    got = _fragments(
        mapper, np.array([1000]), np.array([5000]), pages=np.array([0]), window=TransferWindow(0, 6)
    )
    # Positions 2..5 of page 0 fill slots 2, 3, 0, 1 on both sides: the whole ring of each layer,
    # one run at that layer's offsets.
    assert got == [(1000, 5000, 8), (1064, 5032, 8)]


def test_runs_mapper_refuses_ring_runs_without_positions():
    ring = _rings(4, 4)
    mapper = RunsMapper([_runs([0], [0], [2], ring=ring)])
    with pytest.raises(ValueError, match="transfer's window"):
        mapper.map(np.array([1000]), np.array([5000]))
