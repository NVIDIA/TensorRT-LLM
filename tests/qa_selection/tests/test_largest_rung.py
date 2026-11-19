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
"""A test above the largest rung is deselected with a reason.

    selection.run(DeviceCases.MODULE, "--machine=B200", "--ladder=1,4",
                  "--selection-out-dir={out}")

    B200    8 GPUs/node

The blocker names the demand, the markers that stated it, and the largest
rung, never the machine.
"""

from mocks import DeviceCases


def test_short_ladder(selection):
    """A ladder below the node deselects the 8-GPU test with one reason, and warns nothing."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=1,4", "--selection-out-dir={out}"
    )
    blocker = "skip_less_device(8): needs 8 GPUs, largest rung is 4"

    assert DeviceCases.EIGHT_GPUS not in run.ids(1) + run.ids(4)
    assert run.outcome(DeviceCases.EIGHT_GPUS)["selected"] is False
    assert run.outcome(DeviceCases.EIGHT_GPUS)["blockers"] == [blocker]
    assert run.record["deselected_by_reason"][blocker] == [DeviceCases.EIGHT_GPUS]
    assert "warnings summary" not in run.result.stdout.str()
