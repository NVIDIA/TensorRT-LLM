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
"""Each selected test lands on the smallest rung that holds it.

The QA pipeline reads each test's `selected` and `rung` from the record, and
splits the list by rung. Each criterion collects one module from `cases/`:

    selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8",
                  "--selection-out-dir={out}")

    run.ids(4)    the 4-GPU rung's published list

    B200    8 GPUs/node, which can hold every GPU demand here
"""

from mocks import DeviceCases


def test_one_demand_per_rung(selection):
    """1, 2 and 8 GPUs land on the 1-, 4- and 8-GPU rungs, in the lists and in the record."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4,8", "--selection-out-dir={out}"
    )

    assert run.ids(1) == [DeviceCases.UNMARKED]
    assert run.ids(4) == [DeviceCases.TWO_GPUS]
    assert run.ids(8) == [DeviceCases.EIGHT_GPUS, DeviceCases.EIGHT_RANKS]

    assert {t["nodeid"]: (t["selected"], t["rung"]) for t in run.record["tests"]} == {
        DeviceCases.UNMARKED: (True, 1),
        DeviceCases.TWO_GPUS: (True, 4),
        DeviceCases.EIGHT_GPUS: (True, 8),
        DeviceCases.EIGHT_RANKS: (True, 8),
    }


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
