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

from mocks import DeviceCases


def test_an_infeasible_test_is_never_stranded(selection):
    """A test the machine cannot run is deselected, not counted as waiting for an allocation."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=GB200", "--ladder=1,4", "--selection-out-dir={out}"
    )

    assert run.selected == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]
    assert run.outcome(DeviceCases.EIGHT_GPUS)["unassignable"] is False
    assert run.record["counts"]["unassignable"] == 0
    assert run.record["unassignable"]["nodeids"] == []
    assert run.record["deselected_by_reason"] == {
        "skip_less_device(8): needs 8 GPUs, largest rung is 4": [DeviceCases.EIGHT_GPUS],
        "skip_less_mpi_world_size(8): needs 8 GPUs, largest rung is 4": [DeviceCases.EIGHT_RANKS],
    }

    assert "UnassignableWarning" not in run.result.stdout.str()
