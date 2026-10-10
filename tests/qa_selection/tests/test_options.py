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
"""AC-4: a machine and a ladder are the whole command.

Each criterion collects `DeviceCases.MODULE` for B200 (8 GPUs/node) into an
output directory. The ladder chooses the question:

    what can this machine run?                 --machine=B200                 B200-8gpu.ids
    what can one N-GPU allocation of it run?   --machine=B200 --ladder=N      one list
    how does an allocation policy divide it?   --machine=B200 --ladder=1,4,8  one list per rung

Every kept test is in exactly one published list, and nothing else is kept.
"""

from mocks import DeviceCases


def test_whole_node_by_default(selection):
    """Without `--ladder`, the ladder is one rung of the machine's GPUs per node."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    assert run.selected == [
        DeviceCases.UNMARKED,
        DeviceCases.TWO_GPUS,
        DeviceCases.EIGHT_GPUS,
        DeviceCases.EIGHT_RANKS,
    ]
    assert run.written == ["B200-8gpu.ids", "B200.json"]
    assert run.record["ladder"] == [8]


def test_smaller_allocation(selection):
    """`--ladder=4` asks what one 4-GPU allocation can run, and publishes that one list."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=4", "--selection-out-dir={out}"
    )

    assert run.selected == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]
    assert run.written == ["B200-4gpu.ids", "B200.json"]
    assert run.ids(4) == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]


def test_allocation_policy(selection):
    """`--ladder=1,4,8` divides the machine: one list per rung."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )

    assert run.written == ["B200-1gpu.ids", "B200-4gpu.ids", "B200-8gpu.ids", "B200.json"]
    assert run.record["ladder"] == [1, 4, 8]


def test_kept_equals_published(selection):
    """The kept node ids are exactly the published ones; a test above every rung is in neither."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    published = run.ids(1) + run.ids(4)
    assert sorted(run.selected) == sorted(published)
    assert DeviceCases.EIGHT_GPUS not in published
    assert DeviceCases.EIGHT_RANKS not in published


def test_no_rung_option(selection):
    """`--rung` is not an option: a run keeps every rung's tests and publishes each list."""
    run = selection.refuse(DeviceCases.MODULE, "--machine=B200", "--rung=4")

    run.result.stderr.fnmatch_lines(["*unrecognized arguments: --rung=4*"])
