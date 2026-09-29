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
"""AC-2: the available GPU count decides what is selected.

Each criterion collects one module from `cases/` for one or more machines:

    selection.run(DeviceCount.MODULE, "--machine=GB200")
    -> pytest --collect-only -p qa_selection.plugin --machine=GB200 test_device_count.py

    B200    8 GPUs/node
    GB200   4 GPUs/node

`--gpus=N` lowers the count below the node; above it, the run is refused.
"""

from mock_suite import ClosestMarker, DeviceCount, MpiSize


def test_device_count(selection):
    """`skip_less_device(8)` fits B200's node and not GB200's."""
    big = selection.run(DeviceCount.MODULE, "--machine=B200")
    small = selection.run(DeviceCount.MODULE, "--machine=GB200", "--selection-out-dir={out}")

    assert big.selected == [
        DeviceCount.UNMARKED,
        DeviceCount.NEEDS_EIGHT,
    ]
    assert small.selected == [DeviceCount.UNMARKED]

    # The module was imported on both runs -- the control for test_gpus_above_node.
    assert (small.path / DeviceCount.IMPORT_SENTINEL).exists()

    (blocker,) = small.outcome(DeviceCount.NEEDS_EIGHT)["blockers"]
    assert blocker == "skip_less_device: needs 8 GPUs, target has 4"


def test_mpi_world_size(selection):
    """`skip_less_mpi_world_size` is measured in GPUs: one rank per GPU."""
    big = selection.run(MpiSize.MODULE, "--machine=B200")
    small = selection.run(MpiSize.MODULE, "--machine=GB200", "--selection-out-dir={out}")

    assert big.selected == [
        MpiSize.UNMARKED,
        MpiSize.NEEDS_EIGHT_RANKS,
    ]
    assert small.selected == [MpiSize.UNMARKED]

    (blocker,) = small.outcome(MpiSize.NEEDS_EIGHT_RANKS)["blockers"]
    assert blocker == "skip_less_mpi_world_size: needs 8 ranks, target has 4"


def test_gpus_lowers_ceiling(selection):
    """`--gpus` resizes the profile, so B200's eight GPUs per node are not what is asked of."""
    run = selection.run(
        DeviceCount.MODULE, "--machine=B200", "--gpus=4", "--selection-out-dir={out}"
    )

    assert run.selected == [DeviceCount.UNMARKED]

    (blocker,) = run.outcome(DeviceCount.NEEDS_EIGHT)["blockers"]
    assert blocker == "skip_less_device: needs 8 GPUs, target has 4"


def test_closest_marker(selection):
    """`skip_less_device` reads the closest marker only: a method's bound replaces its class's.

    The deliberate opposite of `skip_less_device_memory`, which is demanded at
    every level (AC-1, `test_arch.py::test_memory_every_level`). Both reproduce
    the fixtures that consume each marker; neither is a defect to tidy away.
    """
    run = selection.run(ClosestMarker.MODULE, "--machine=GB200", "--selection-out-dir={out}")

    assert run.selected == [
        ClosestMarker.UNMARKED,
        ClosestMarker.NEEDS_TWO,
    ]
    assert run.outcome(ClosestMarker.NEEDS_TWO)["blockers"] == []


def test_gpus_above_node(selection):
    """A GPU count above the node is a usage error, raised before collection."""
    run = selection.refuse(DeviceCount.MODULE, "--machine=GB200", "--gpus=8")

    run.result.stderr.fnmatch_lines(["*--gpus: 8 exceeds GB200, which has 4 GPUs per node*"])
    assert not (run.path / DeviceCount.IMPORT_SENTINEL).exists()
