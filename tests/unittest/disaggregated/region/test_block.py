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
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import BufferRuns
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc

pytestmark = pytest.mark.cpu_only

_BUFFER = BufferDesc(0, "k", 0, 0, 0, 64)


def _runs(src, dst, sizes) -> BufferRuns:
    return BufferRuns(
        _BUFFER,
        _BUFFER,
        np.array(src, dtype=np.int64),
        np.array(dst, dtype=np.int64),
        np.array(sizes, dtype=np.int64),
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
    """Runs alike on every page ignore the pages and end position a transfer passes."""
    mapper = RunsMapper([_runs([0], [8], [16])])
    bases = (np.array([100, 200]), np.array([300, 400]))
    assert _fragments(mapper, *bases) == _fragments(mapper, *bases, pages=np.array([0, 5]), end=3)
