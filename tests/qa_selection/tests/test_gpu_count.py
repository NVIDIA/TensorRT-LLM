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
"""A test's GPU count comes from its markers, read as the integration conftest reads them.

Each criterion collects one module from `cases/` for one or more machines:

    selection.run(DeviceCases.MODULE, "--machine=GB200")
    -> pytest --collect-only -p qa_selection.plugin --machine=GB200 test_device_cases.py

    B200    8 GPUs/node
    GB200   4 GPUs/node

The count a machine offers is the largest rung of the run's ladder, which
defaults to the machine's profile.
"""

from mocks import DeviceCases, MarkerCases


def test_node_sized_test(selection):
    """`skip_less_device(8)` fits B200's node and not GB200's."""
    b200 = selection.run(DeviceCases.MODULE, "--machine=B200")
    gb200 = selection.run(DeviceCases.MODULE, "--machine=GB200")

    assert DeviceCases.EIGHT_GPUS in b200.selected
    assert DeviceCases.EIGHT_GPUS not in gb200.selected

    # The module was imported: the control for the usage errors that must not import it.
    assert (gb200.path / DeviceCases.IMPORT_SENTINEL).exists()


def test_mpi_ranks_count_as_gpus(selection):
    """`skip_less_mpi_world_size` is measured in GPUs: one rank per GPU."""
    b200 = selection.run(DeviceCases.MODULE, "--machine=B200")
    gb200 = selection.run(DeviceCases.MODULE, "--machine=GB200")

    assert DeviceCases.EIGHT_RANKS in b200.selected
    assert DeviceCases.EIGHT_RANKS not in gb200.selected


def test_closest_marker_wins(selection):
    """`skip_less_device` reads the closest marker only: a method's bound replaces its class's.

    The deliberate opposite of `skip_less_device_memory`, which is demanded at
    every level (`test_arch.py::test_memory_every_level`). Both reproduce the
    fixtures that consume each marker.
    """
    run = selection.run(MarkerCases.MODULE, "--machine=GB200")

    assert MarkerCases.CLOSEST_GPUS in run.selected


def test_parameter_level_gpu_bound(selection):
    """A GPU bound on one `pytest.param` is that case's demand alone."""
    b200 = selection.run(MarkerCases.MODULE, "--machine=B200")
    gb200 = selection.run(MarkerCases.MODULE, "--machine=GB200")

    assert MarkerCases.PARAM_EIGHT_GPUS in b200.selected
    assert MarkerCases.PARAM_EIGHT_GPUS not in gb200.selected
    assert MarkerCases.PARAM_SKIP in gb200.selected
