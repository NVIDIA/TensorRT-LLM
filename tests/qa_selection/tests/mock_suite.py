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
"""What `cases/` holds: one class per mock module, naming its file and node ids.

    selection.run(DeviceCount.MODULE, "--machine=GB200")
    assert run.selected == [DeviceCount.UNMARKED]

One definition per node id, so two criteria over one mock cannot disagree about
what it contains. Every mock carries an unmarked control that states no
requirement and therefore survives on every machine.

These are facts about this suite's own fixtures, never read from `qa_selection`
-- unlike the reason strings in `cases/conftest.py`, which are duplicated from
the real decorators on purpose.
"""


class PreBlackwell:
    """`cases/test_pre_blackwell.py`: a floor gate, `sm < 100`."""

    MODULE = "test_pre_blackwell.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    BLACKWELL_AND_NEWER = f"{MODULE}::test_blackwell_and_newer"


class PostBlackwell:
    """`cases/test_post_blackwell.py`: a ceiling gate, `sm >= 100`."""

    MODULE = "test_post_blackwell.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    HOPPER_AND_OLDER = f"{MODULE}::test_hopper_and_older"


class CpuArch:
    """`cases/test_cpu_arch.py`: a host-CPU gate, mentioning no GPU."""

    MODULE = "test_cpu_arch.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    X86_ONLY = f"{MODULE}::test_x86_only"


class DeviceMemory:
    """`cases/test_device_memory.py`: a memory demand at class and method level."""

    MODULE = "test_device_memory.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    MODEST_METHOD = f"{MODULE}::TestHungry::test_modest_method"


class UncuratedSkip:
    """`cases/test_uncurated_skip.py`: one skip the rule table encodes, one it declines to."""

    MODULE = "test_uncurated_skip.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    UNCURATED_REASON = f"{MODULE}::test_skipped_for_an_uncurated_reason"
    BY_THE_TABLE = f"{MODULE}::test_skipped_by_the_table"


class DeviceCount:
    """`cases/test_device_count.py`: a GPU lower bound, and an import sentinel."""

    MODULE = "test_device_count.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    NEEDS_EIGHT = f"{MODULE}::test_needs_eight"

    #: Touched at import time, so a criterion can require that no test module
    #: was imported. Relative to the run's own directory, `run.path`.
    IMPORT_SENTINEL = "imported"


class MpiSize:
    """`cases/test_mpi_size.py`: a rank lower bound and no other resource marker."""

    MODULE = "test_mpi_size.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    NEEDS_EIGHT_RANKS = f"{MODULE}::test_needs_eight_ranks"


class ClosestMarker:
    """`cases/test_closest_marker.py`: a GPU bound at class and method level."""

    MODULE = "test_closest_marker.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    NEEDS_TWO = f"{MODULE}::TestEightGpus::test_needs_two"


class LadderDemands:
    """`cases/test_ladder_demands.py`: three GPU demands, one per rung of `1,4,8`."""

    MODULE = "test_ladder_demands.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    NEEDS_TWO = f"{MODULE}::test_needs_two"
    NEEDS_EIGHT = f"{MODULE}::test_needs_eight"


class BothGpuMarkers:
    """`cases/test_both_gpu_markers.py`: both GPU markers on one test, agreeing at 8."""

    MODULE = "test_both_gpu_markers.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    NEEDS_EIGHT_OF_BOTH = f"{MODULE}::test_needs_eight_of_both"


class MixedGpuMarkers:
    """`cases/test_mixed_gpu_markers.py`: both GPU markers on one test, disagreeing."""

    MODULE = "test_mixed_gpu_markers.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    TWO_DEVICES_EIGHT_RANKS = f"{MODULE}::test_two_devices_eight_ranks"
