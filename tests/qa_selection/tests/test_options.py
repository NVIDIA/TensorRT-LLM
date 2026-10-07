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
"""AC-4: each option answers one question, and a rung run matches what was published.

    --ladder=R,..  allocation sizes that exist      partition
    --rung=N       the allocation this run holds    neither

    B200    8 GPUs/node

`--rung` and `--selection-out-dir` are refused together, so a rung run is read
through `run.selected` and `run.reported(...)`; a ladder alone publishes every
rung's list.
"""

from mock_suite import LadderDemands


def test_rung_run_selects_its_published_list(selection):
    """The tests a rung run keeps are the tests that rung published, in order."""
    plan = selection.run(
        LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )
    rung = selection.run(LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--rung=4")

    assert plan.ids(4) == [LadderDemands.NEEDS_TWO]
    assert rung.selected == plan.ids(4)


def test_rung_narrows_what_executes_not_the_machine(selection):
    """A rung miss is reported feasible with no blocker, never as infeasible."""
    rung = selection.run(LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--rung=4")
    plan = selection.run(
        LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )

    assert LadderDemands.NEEDS_EIGHT not in rung.selected

    # Feasibility was decided against the whole node.
    assert rung.reported("target") == "B200, ladder 1,4,8"
    assert rung.reported("feasible") == "3  (0 deselected)"
    assert rung.reported("live") == "1  (rung 4 only)"

    # The record, which only the plan run writes, says the same.
    assert plan.outcome(LadderDemands.NEEDS_EIGHT)["selected"] is True
    assert plan.outcome(LadderDemands.NEEDS_EIGHT)["blockers"] == []
    assert plan.outcome(LadderDemands.NEEDS_EIGHT)["rung"] == 8


def test_a_rung_run_differs_from_the_plan_only_in_what_runs(selection):
    """Plan and rung runs agree on the machine and on feasibility; `live` differs.

    The plan run keeps every feasible test and publishes one list per rung; the
    rung run keeps that rung's share of them.
    """
    plan = selection.run(
        LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )
    rung = selection.run(LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--rung=4")

    assert plan.selected == [
        LadderDemands.UNMARKED,
        LadderDemands.NEEDS_TWO,
        LadderDemands.NEEDS_EIGHT,
    ]
    assert plan.written == [
        "B200-1gpu.ids",
        "B200-4gpu.ids",
        "B200-8gpu.ids",
        "B200.json",
    ]
    assert rung.selected == [LadderDemands.NEEDS_TWO]

    assert plan.reported("target") == rung.reported("target") == "B200, ladder 1,4,8"
    assert plan.reported("feasible") == rung.reported("feasible") == "3  (0 deselected)"
    assert plan.reported("live") == "3"
    assert rung.reported("live") == "1  (rung 4 only)"


def test_rung_outside_the_ladder(selection):
    """A rung the ladder does not hold is refused, not silently empty."""
    run = selection.refuse(LadderDemands.MODULE, "--machine=B200", "--ladder=1,4,8", "--rung=2")

    run.result.stderr.fnmatch_lines(["*--rung: 2 is not a rung of --ladder=1,4,8*"])


def test_rung_with_an_output_directory(selection):
    """`--rung` and an output directory are refused, naming the command that writes them."""
    run = selection.refuse(
        LadderDemands.MODULE,
        "--machine=B200",
        "--ladder=1,4,8",
        "--rung=4",
        "--selection-out-dir={out}",
    )

    run.result.stderr.fnmatch_lines(
        ["*--selection-out-dir: cannot be combined with --rung*without --rung*every rung*"]
    )
