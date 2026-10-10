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

    selection.run(DeviceCases.MODULE, "--machine=GB200")
    assert run.selected == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]

One module per kind of fact selection reads:

    ArchCases     the card's and host's architecture    AC-1
    DeviceCases   how many GPUs a test needs            AC-2 .. AC-6
    MarkerCases   how marks combine across levels       AC-1 .. AC-3

`cases/test_device_cases.py` is `DeviceCases`. Each id is named after its
test: `test_eight_gpus` is `EIGHT_GPUS`, `TestClosestGpus::test_method_bound`
is `CLOSEST_GPUS`, and `test_param[skip]` is `PARAM_SKIP`.

One definition per node id, so two criteria over one mock cannot disagree about
what it contains. Every mock carries an unmarked control that states no
requirement and therefore survives on every machine.

These are facts about this suite's own fixtures, never read from `qa_selection`
-- unlike the reason strings in `cases/conftest.py`, which are duplicated from
the real decorators on purpose.
"""


class ArchCases:
    """`cases/test_arch_cases.py`: one test per architecture gate."""

    MODULE = "test_arch_cases.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    BLACKWELL_AND_NEWER = f"{MODULE}::test_blackwell_and_newer"
    HOPPER_AND_OLDER = f"{MODULE}::test_hopper_and_older"
    X86_ONLY = f"{MODULE}::test_x86_only"


class DeviceCases:
    """`cases/test_device_cases.py`: one GPU demand per rung of `1,4,8`, and 8 in MPI ranks."""

    MODULE = "test_device_cases.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    TWO_GPUS = f"{MODULE}::test_two_gpus"
    EIGHT_GPUS = f"{MODULE}::test_eight_gpus"
    EIGHT_RANKS = f"{MODULE}::test_eight_ranks"

    #: Touched at import time, so a criterion can require that no test module
    #: was imported. Relative to the run's own directory, `run.path`.
    IMPORT_SENTINEL = "imported"


class MarkerCases:
    """`cases/test_marker_cases.py`: where marks sit, and how two of them combine."""

    MODULE = "test_marker_cases.py"
    UNMARKED = f"{MODULE}::test_unmarked"
    CLOSEST_GPUS = f"{MODULE}::TestClosestGpus::test_method_bound"
    EVERY_LEVEL_MEMORY = f"{MODULE}::TestEveryLevelMemory::test_method_bound"
    AGREEING_BOUNDS = f"{MODULE}::test_agreeing_bounds"
    DISAGREEING_BOUNDS = f"{MODULE}::test_disagreeing_bounds"
    CURATED_SKIP = f"{MODULE}::test_curated_skip"
    UNCURATED_SKIP = f"{MODULE}::test_uncurated_skip"
    CLASS_SKIP = f"{MODULE}::TestClassSkip::test_method"
    PARAM_SKIP = f"{MODULE}::test_param[skip]"
    PARAM_EIGHT_GPUS = f"{MODULE}::test_param[eight_gpus]"
