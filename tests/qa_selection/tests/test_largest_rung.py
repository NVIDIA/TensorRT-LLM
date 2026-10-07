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
"""AC-5: a test above the largest rung is deselected with a reason.

Each criterion collects a module from `cases/` into an output directory:

    selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4",
                  "--selection-out-dir={out}")

    B200    8 GPUs/node
    GB200   4 GPUs/node

The blocker names the demand, the markers that stated it, and the largest
rung, never the machine. The record holds no other population for such a test.
"""

from mocks import DeviceCases, MarkerCases


class Blocker:
    """The GPU-count blockers these criteria expect, spelled out."""

    EIGHT_GPUS_ABOVE_FOUR = "skip_less_device(8): needs 8 GPUs, largest rung is 4"
    EIGHT_RANKS_ABOVE_FOUR = "skip_less_mpi_world_size(8): needs 8 GPUs, largest rung is 4"


def test_short_ladder(selection):
    """A ladder below the node deselects the 8-GPU test with one reason, and warns nothing."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert DeviceCases.EIGHT_GPUS not in run.ids(1) + run.ids(4)
    assert run.outcome(DeviceCases.EIGHT_GPUS)["blockers"] == [Blocker.EIGHT_GPUS_ABOVE_FOUR]
    assert run.record["deselected_by_reason"][Blocker.EIGHT_GPUS_ABOVE_FOUR] == [
        DeviceCases.EIGHT_GPUS
    ]
    assert "warnings summary" not in run.result.stdout.str()


def test_same_reason_either_way(selection):
    """A 4-GPU node and a 4-GPU ladder on an 8-GPU node give the same blocker."""
    gb200 = selection.run(DeviceCases.MODULE, "--machine=GB200", "--selection-out-dir={out}")
    b200 = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert (
        gb200.outcome(DeviceCases.EIGHT_RANKS)["blockers"]
        == b200.outcome(DeviceCases.EIGHT_RANKS)["blockers"]
        == [Blocker.EIGHT_RANKS_ABOVE_FOUR]
    )


def test_one_reason_for_two_markers(selection):
    """Agreeing GPU-count markers give one blocker naming both."""
    run = selection.run(MarkerCases.MODULE, "--machine=GB200", "--selection-out-dir={out}")

    assert run.outcome(MarkerCases.AGREEING_BOUNDS)["blockers"] == [
        "skip_less_device(8), skip_less_mpi_world_size(8): needs 8 GPUs, largest rung is 4"
    ]


def test_no_stranded_population(selection):
    """The record has no `unassignable` key, count or per-test field, and no `live` count."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert "unassignable" not in run.record
    assert "unassignable" not in run.record["counts"]
    assert "live" not in run.record["counts"]
    assert all("unassignable" not in outcome for outcome in run.record["tests"])
