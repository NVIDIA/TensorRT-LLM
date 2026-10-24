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
"""AC-1: the target's architecture decides what is selected, wherever the mark sits.

Each criterion collects one module from `cases/` for two machines, and checks
the one test whose gate it is about:

    selection.run(ArchCases.MODULE, "--machine=GB200")
    -> pytest --collect-only -p qa_selection.plugin --machine=GB200 test_arch_cases.py

    H100    sm  90    x86_64
    B200    sm 100    x86_64
    GB200   sm 100    aarch64
"""

from mocks import ArchCases, MarkerCases


def test_pre_blackwell_gate(selection):
    """`skip_pre_blackwell` is `sm < 100`: fires on H100, not on B200.

    Its condition was frozen True on the GPU-less collecting host, so B200
    keeping the test shows the condition is never read.
    """
    h100 = selection.run(ArchCases.MODULE, "--machine=H100")
    b200 = selection.run(ArchCases.MODULE, "--machine=B200")

    assert ArchCases.BLACKWELL_AND_NEWER not in h100.selected
    assert ArchCases.BLACKWELL_AND_NEWER in b200.selected


def test_post_blackwell_ceiling(selection):
    """`skip_post_blackwell` is `sm >= 100` -- inclusive, so it fires *on* B200."""
    b200 = selection.run(ArchCases.MODULE, "--machine=B200")
    h100 = selection.run(ArchCases.MODULE, "--machine=H100")

    assert ArchCases.HOPPER_AND_OLDER not in b200.selected
    assert ArchCases.HOPPER_AND_OLDER in h100.selected


def test_cpu_arch(selection):
    """`skip_arm` is `cpu_arch == "aarch64"`; B200 and GB200 differ only there."""
    b200 = selection.run(ArchCases.MODULE, "--machine=B200")
    gb200 = selection.run(ArchCases.MODULE, "--machine=GB200")

    assert ArchCases.X86_ONLY in b200.selected
    assert ArchCases.X86_ONLY not in gb200.selected


def test_memory_every_level(selection):
    """Every level's memory demand applies; a method's does not replace its class's."""
    run = selection.run(MarkerCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    outcome = run.outcome(MarkerCases.EVERY_LEVEL_MEMORY)
    assert outcome["selected"] is False

    # The class's 1000000 blocked it; the method's 1000 neither blocked nor
    # displaced it. `target has N MiB` is the datum and is not asserted.
    (blocker,) = outcome["blockers"]
    assert blocker.startswith("skip_less_device_memory:")
    assert "needs 1000000 MiB" in blocker
    assert "needs 1000 MiB" not in blocker


def test_uncurated_skipif(selection):
    """A reason with no rule keeps its test and produces no blocker."""
    run = selection.run(MarkerCases.MODULE, "--machine=H100", "--selection-out-dir={out}")

    assert MarkerCases.UNCURATED_SKIP in run.selected
    assert run.outcome(MarkerCases.UNCURATED_SKIP)["blockers"] == []

    # The curated skip beside it is dropped, so the table was consulted.
    reason = "This test is not supported in pre-Blackwell architecture"
    assert MarkerCases.CURATED_SKIP in run.record["deselected_by_reason"][reason]


def test_class_level_skip(selection):
    """A skip on a class reaches its methods, decided by the target."""
    h100 = selection.run(MarkerCases.MODULE, "--machine=H100")
    b200 = selection.run(MarkerCases.MODULE, "--machine=B200")

    assert MarkerCases.CLASS_SKIP not in h100.selected
    assert MarkerCases.CLASS_SKIP in b200.selected


def test_parameter_level_skip(selection):
    """A skip on one `pytest.param` drops that case only, decided by the target."""
    h100 = selection.run(MarkerCases.MODULE, "--machine=H100")
    b200 = selection.run(MarkerCases.MODULE, "--machine=B200")

    assert MarkerCases.PARAM_SKIP not in h100.selected
    assert MarkerCases.PARAM_EIGHT_GPUS in h100.selected
    assert MarkerCases.PARAM_SKIP in b200.selected
