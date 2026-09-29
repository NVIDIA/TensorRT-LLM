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
"""AC-1: the target machine's architecture decides what is selected.

Each criterion collects one module from `cases/` for one or more machines:

    selection.run(CpuArch.MODULE, "--machine=GB200")
    -> pytest --collect-only -p qa_selection.plugin --machine=GB200 test_cpu_arch.py

    H100    sm  90    x86_64
    B200    sm 100    x86_64
    GB200   sm 100    aarch64
"""

from mock_suite import CpuArch, DeviceMemory, PostBlackwell, PreBlackwell, UncuratedSkip


def test_pre_blackwell_gate(selection):
    """`skip_pre_blackwell` is `sm < 100`: fires on H100, not on B200."""
    hopper = selection.run(PreBlackwell.MODULE, "--machine=H100")
    blackwell = selection.run(PreBlackwell.MODULE, "--machine=B200")

    assert hopper.selected == [PreBlackwell.UNMARKED]
    assert blackwell.selected == [
        PreBlackwell.UNMARKED,
        PreBlackwell.BLACKWELL_AND_NEWER,
    ]


def test_post_blackwell_ceiling(selection):
    """`skip_post_blackwell` is `sm >= 100` -- inclusive, so it fires *on* B200."""
    blackwell = selection.run(PostBlackwell.MODULE, "--machine=B200")
    hopper = selection.run(PostBlackwell.MODULE, "--machine=H100")

    assert blackwell.selected == [PostBlackwell.UNMARKED]
    assert hopper.selected == [
        PostBlackwell.UNMARKED,
        PostBlackwell.HOPPER_AND_OLDER,
    ]


def test_cpu_arch(selection):
    """`skip_arm` is `cpu_arch == "aarch64"`; B200 and GB200 differ only there."""
    x86 = selection.run(CpuArch.MODULE, "--machine=B200")
    arm = selection.run(CpuArch.MODULE, "--machine=GB200")

    assert x86.selected == [
        CpuArch.UNMARKED,
        CpuArch.X86_ONLY,
    ]
    assert arm.selected == [CpuArch.UNMARKED]


def test_memory_every_level(selection):
    """Every level's memory demand applies; a method's does not replace its class's."""
    run = selection.run(DeviceMemory.MODULE, "--machine=B200", "--selection-out-dir={out}")

    assert run.selected == [DeviceMemory.UNMARKED]

    outcome = run.outcome(DeviceMemory.MODEST_METHOD)
    assert outcome["selected"] is False

    # The class's 1000000 blocked it; the method's 1000 neither blocked nor
    # displaced it. `target has N MiB` is the datum and is not asserted.
    (blocker,) = outcome["blockers"]
    assert blocker.startswith("skip_less_device_memory:")
    assert "needs 1000000 MiB" in blocker
    assert "needs 1000 MiB" not in blocker


def test_uncurated_skipif(selection):
    """A reason with no rule keeps its test and produces no blocker."""
    run = selection.run(UncuratedSkip.MODULE, "--machine=H100", "--selection-out-dir={out}")

    assert run.selected == [
        UncuratedSkip.UNMARKED,
        UncuratedSkip.UNCURATED_REASON,
    ]
    assert run.record["deselected_by_reason"] == {
        "This test is not supported in pre-Blackwell architecture": [UncuratedSkip.BY_THE_TABLE]
    }
