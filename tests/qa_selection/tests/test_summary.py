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
"""AC-7: a run explains itself in the terminal.

Each criterion collects `DeviceCases.MODULE` and reads what the plugin printed:

    run.summary    the block after the run, line by line
    run.result     the whole output, for the session header

    B200    sm 100, 8 GPUs/node

The block's layout is pinned exactly: it is what a reader of a job log sees.
"""

from mocks import DeviceCases


class Reason:
    """The two GPU-count reasons a `--ladder=1,4` run on B200 gives, spelled out."""

    EIGHT_GPUS = "skip_less_device(8): needs 8 GPUs, largest rung is 4"
    EIGHT_RANKS = "skip_less_mpi_world_size(8): needs 8 GPUs, largest rung is 4"


def test_header_names_the_target(selection):
    """The session header names the machine, its card and node, and the ladder."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8")

    assert "qa selection: B200 (sm 100, 8 GPUs/node), ladder 1,4,8" in run.result.stdout.lines


def test_quiet_run_has_no_header(selection):
    """`-q` drops the header, as it drops pytest's own, and keeps the block."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8", "-q")

    assert not any(line.startswith("qa selection:") for line in run.result.stdout.lines)
    assert run.summary[0] == "target        B200, ladder 1,4,8"


def test_counts_and_their_parts(selection):
    """Selected tests by rung, deselected tests by reason, and the files on one line."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert run.summary == [
        "target        B200, ladder 1,4",
        "candidates    4",
        "selected      2",
        "  1gpu        1",
        "  4gpu        1",
        "deselected    2",
        f"  1  {Reason.EIGHT_GPUS}",
        f"  1  {Reason.EIGHT_RANKS}",
        f"written       {run.out_dir}/: B200.json B200-1gpu.ids B200-4gpu.ids",
    ]


def test_one_rung_needs_no_breakdown(selection):
    """A one-rung ladder is already in the target line; nothing written, no `written` line."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200")

    assert run.summary == [
        "target        B200, ladder 8",
        "candidates    4",
        "selected      4",
        "deselected    0",
    ]


def test_verbose_lists_the_deselected(selection):
    """`-v` puts each reason's node ids beneath it."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "-v")

    deselected = run.summary[run.summary.index("deselected    2") :]
    assert deselected == [
        "deselected    2",
        f"  1  {Reason.EIGHT_GPUS}",
        f"       {DeviceCases.EIGHT_GPUS}",
        f"  1  {Reason.EIGHT_RANKS}",
        f"       {DeviceCases.EIGHT_RANKS}",
    ]
