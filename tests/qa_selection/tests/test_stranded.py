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
"""AC-5: a ladder shorter than the machine strands feasible tests audibly.

`--ladder=1,4` on a machine that can run an 8-GPU test leaves it above every
rung. Two machines separate the two ways a test ends up in no `.ids` file:

    B200    8 GPUs/node   can run it   -> stranded
    GB200   4 GPUs/node   cannot       -> infeasible

Both are recorded in `<machine>.json`; `selected` and `unassignable` tell them
apart, and only the first warns.
"""

from mock_suite import DeviceCount


def test_stranded_test_is_named_counted_and_warned(selection):
    """A test above every rung is kept, published nowhere, listed, counted and warned."""
    run = selection.run(
        DeviceCount.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    # Still collected -- so its absence from every list is the only sign.
    assert run.selected == [DeviceCount.UNMARKED, DeviceCount.NEEDS_EIGHT]
    assert run.ids(1) == [DeviceCount.UNMARKED]
    assert run.ids(4) == []

    # The per-test flag and the record's two summaries are separate fields and
    # are asserted separately: the count is derived by a second feasibility
    # filter, so it survives a broken flag.
    assert run.outcome(DeviceCount.NEEDS_EIGHT)["unassignable"] is True
    assert run.record["unassignable"] == {
        "largest_rung": 4,
        "nodeids": [DeviceCount.NEEDS_EIGHT],
    }
    assert run.record["counts"]["unassignable"] == 1

    run.result.stdout.fnmatch_lines(
        [
            "*UnassignableWarning: 1 feasible test(s) need more than 4 GPUs "
            "and fit no rung of --ladder=1,4*"
        ]
    )


def test_an_infeasible_test_is_never_stranded(selection):
    """A test the machine cannot run is deselected, not counted as waiting for an allocation."""
    run = selection.run(
        DeviceCount.MODULE, "--machine=GB200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert run.selected == [DeviceCount.UNMARKED]
    assert run.outcome(DeviceCount.NEEDS_EIGHT)["unassignable"] is False
    assert run.record["counts"]["unassignable"] == 0
    assert run.record["unassignable"]["nodeids"] == []
    assert run.record["deselected_by_reason"] == {
        "skip_less_device: needs 8 GPUs, target has 4": [DeviceCount.NEEDS_EIGHT]
    }

    assert "UnassignableWarning" not in run.result.stdout.str()


def test_the_record_tells_stranded_from_infeasible(selection):
    """Both end up in no `.ids` file and in the record; `rung` cannot tell them apart."""
    stranded = selection.run(
        DeviceCount.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )
    infeasible = selection.run(
        DeviceCount.MODULE, "--machine=GB200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    for run in (stranded, infeasible):
        assert DeviceCount.NEEDS_EIGHT in [outcome["nodeid"] for outcome in run.record["tests"]]
        assert DeviceCount.NEEDS_EIGHT not in run.ids(1) + run.ids(4)
        assert run.outcome(DeviceCount.NEEDS_EIGHT)["rung"] is None

    assert stranded.outcome(DeviceCount.NEEDS_EIGHT)["selected"] is True
    assert stranded.outcome(DeviceCount.NEEDS_EIGHT)["blockers"] == []
    assert stranded.outcome(DeviceCount.NEEDS_EIGHT)["unassignable"] is True

    assert infeasible.outcome(DeviceCount.NEEDS_EIGHT)["selected"] is False
    assert infeasible.outcome(DeviceCount.NEEDS_EIGHT)["blockers"] == [
        "skip_less_device: needs 8 GPUs, target has 4"
    ]
    assert infeasible.outcome(DeviceCount.NEEDS_EIGHT)["unassignable"] is False
