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
"""AC-3: each selected test lands on the smallest rung that holds it.

Each criterion collects one module from `cases/` for one machine:

    selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8",
                  "--selection-out-dir={out}")

    run.ids(4)    the 4-GPU rung's published list

    B200    8 GPUs/node, which can hold every GPU demand here
"""

from mocks import DeviceCases, MarkerCases


def test_one_demand_per_rung(selection):
    """1, 2 and 8 GPUs land on the 1-, 4- and 8-GPU rungs, in collection order."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )

    assert run.ids(1) == [DeviceCases.UNMARKED]
    assert run.ids(4) == [DeviceCases.TWO_GPUS]
    assert run.ids(8) == [DeviceCases.EIGHT_GPUS, DeviceCases.EIGHT_RANKS]


def test_lists_partition_the_selection(selection):
    """One list per rung, empty rungs included, pairwise disjoint, and together the selection.

    A four-rung ladder over the same demands, so that rung 4 has no members.
    """
    ladder = (1, 2, 4, 8)
    run = selection.run(
        DeviceCases.MODULE,
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


def test_both_markers_credited(selection):
    """Equal `skip_less_device` and `skip_less_mpi_world_size` bounds are both credited.

    The shape at `tests/integration/defs/examples/test_deepseek_v4_pro.py:138,140`.
    """
    run = selection.run(
        MarkerCases.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    assert MarkerCases.AGREEING_BOUNDS in run.ids(8)

    outcome = run.outcome(MarkerCases.AGREEING_BOUNDS)
    assert outcome["required_gpus"] == 8
    assert outcome["required_gpus_from"] == [
        "skip_less_device(8)",
        "skip_less_mpi_world_size(8)",
    ]


def test_larger_bound_wins(selection):
    """The larger of two disagreeing bounds decides the rung, and is credited alone.

    A rule probe: no collected item in the repository carries the two markers
    at different values.
    """
    run = selection.run(
        MarkerCases.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    assert MarkerCases.DISAGREEING_BOUNDS in run.ids(8)

    outcome = run.outcome(MarkerCases.DISAGREEING_BOUNDS)
    assert outcome["required_gpus"] == 8
    assert outcome["required_gpus_from"] == ["skip_less_mpi_world_size(8)"]


def test_unmarked_means_one_gpu(selection):
    """A test stating no bound lands on the smallest rung, and the record names no marker."""
    run = selection.run(
        DeviceCases.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--selection-out-dir={out}",
    )

    outcome = run.outcome(DeviceCases.UNMARKED)
    assert outcome["rung"] == 1
    assert outcome["required_gpus"] == 1
    assert outcome["required_gpus_from"] == []


def test_leftover_machine_list_is_an_orphan(selection):
    """A leftover `<machine>.ids` is an orphan: every list a run writes names its rung."""
    selection.out_dir.mkdir()
    (selection.out_dir / "B200.ids").write_text("")

    run = selection.refuse(DeviceCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    run.result.stderr.fnmatch_lines(
        ["*already holds B200.ids for B200*writes B200.json, B200-8gpu.ids*"]
    )
