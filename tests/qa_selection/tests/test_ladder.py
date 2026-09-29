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
"""AC-3: the ladder routes each selected test to the smallest allocation that holds it.

Each criterion collects one module from `cases/` for one machine:

    selection.run(LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8",
                  "--selection-out-dir={out}")

    run.ids(4)    the 4-GPU allocation's published list

    B200    8 GPUs/node, which can run every test here
"""

from mock_suite import BothGpuMarkers, LadderDemands, MixedGpuMarkers


def test_smallest_rung_that_fits(selection):
    """1, 2 and 8 GPUs land on the 1-, 4- and 8-GPU allocations."""
    run = selection.run(
        LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )

    assert run.ids(1) == [LadderDemands.UNMARKED]
    assert run.ids(4) == [LadderDemands.NEEDS_TWO]
    assert run.ids(8) == [LadderDemands.NEEDS_EIGHT]


def test_rungs_partition_the_selected_set(selection):
    """One list per rung, pairwise disjoint, and together the whole selected set.

    A four-rung ladder over the same three demands, so that rung 4 has no
    members and an empty rung's list is observable.
    """
    ladder = (1, 2, 4, 8)
    run = selection.run(
        LadderDemands.MODULE,
        "--machine=B200",
        "--ladder=1,2,4,8",
        "--selection-out-dir={out}",
    )

    assert run.written == [
        "B200-1gpu.ids",
        "B200-2gpu.ids",
        "B200-4gpu.ids",
        "B200-8gpu.ids",
        "B200.json",
    ]
    assert run.ids(4) == []

    published = [nodeid for rung in ladder for nodeid in run.ids(rung)]
    assert len(published) == len(set(published))
    assert sorted(published) == sorted(run.selected)


def test_both_gpu_markers_are_credited(selection):
    """Both GPU markers are read, and equal bounds credit both.

    The shape at `tests/integration/defs/examples/test_deepseek_v4_pro.py:138,140`.
    Ranks are measured one per GPU; see `ref/mpi-world-size.md`.
    """
    run = selection.run(
        BothGpuMarkers.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    assert run.ids(8) == [BothGpuMarkers.NEEDS_EIGHT_OF_BOTH]

    outcome = run.outcome(BothGpuMarkers.NEEDS_EIGHT_OF_BOTH)
    assert outcome["required_gpus"] == 8
    assert outcome["required_gpus_from"] == [
        "skip_less_device(8)",
        "skip_less_mpi_world_size(8)",
    ]


def test_unmarked_lands_on_the_smallest_rung(selection):
    """A test stating no bound is assumed to want one GPU, and the record says so."""
    run = selection.run(
        LadderDemands.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    outcome = run.outcome(LadderDemands.UNMARKED)
    assert outcome["rung"] == 1
    assert outcome["required_gpus"] == 1
    assert outcome["required_gpus_from"] == []


def test_larger_bound_decides(selection):
    """The larger of two disagreeing bounds decides, and is credited alone.

    A rule probe, not an observed shape: no collected item in the repository
    carries the two markers at different values.
    """
    run = selection.run(
        MixedGpuMarkers.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    assert run.ids(8) == [MixedGpuMarkers.TWO_DEVICES_EIGHT_RANKS]

    outcome = run.outcome(MixedGpuMarkers.TWO_DEVICES_EIGHT_RANKS)
    assert outcome["required_gpus"] == 8
    assert outcome["required_gpus_from"] == ["skip_less_mpi_world_size(8)"]
