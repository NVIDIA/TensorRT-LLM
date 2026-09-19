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
"""Verdict logic for the MegaMoE DeepGEMM SymmBuffer preflight.

Both directions are covered on purpose. A gate tested only on the cases it
should trip cannot tell you whether it also trips on healthy runs, and a
preflight that warns on every healthy configuration gets turned off long
before it ever catches anything.
"""

import pytest

from tensorrt_llm._torch.moe.fused_moe.mega_moe._symm_preflight import (
    _fabric_identity,
    preflight_failed,
)


@pytest.mark.parametrize(
    "fields,ep_size,reason",
    [
        # A single-rank group never reaches symmetric memory: DeepGEMM
        # allocates with torch.empty and fabricates the handle, so nothing
        # reported for it can be a symmetric-memory fault.
        ({"supports_mnnvl": False, "probe": "FAILED:RuntimeError"}, 1, "single-rank"),
        ({"supports_mnnvl": True, "probe": "ok", "probe_mc_ptr": "0x9360000000"}, 32, "healthy"),
        ({"supports_mnnvl": True, "probe": "single-rank"}, 32, "probe deliberately skipped"),
        ({"supports_mnnvl": True, "probe": "no-group-name"}, 32, "cannot probe"),
        # An unusable capability check is not evidence that the capability is
        # absent. Scoring a broken probe as a negative result is how a probe
        # starts manufacturing conclusions.
        ({"supports_mnnvl": "check-failed:NVMLError", "probe": "ok"}, 32, "check unusable"),
    ],
)
def test_healthy_configurations_do_not_trip(fields, ep_size, reason):
    assert preflight_failed(fields, ep_size=ep_size) is False, reason


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"supports_mnnvl": False, "probe": "ok"}, "capability genuinely absent"),
        (
            {"supports_mnnvl": True, "probe": "MULTICAST_UNAVAILABLE", "probe_mc_ptr": "0x0"},
            "multicast claimed locally but the pointer is null",
        ),
        ({"supports_mnnvl": True, "probe": "FAILED:AcceleratorError"}, "probe raised"),
        # Missing key rather than a failure string: stage 2 never ran at all,
        # which must not read as success.
        ({"supports_mnnvl": True}, "stage 2 never ran"),
    ],
)
def test_real_problems_trip(fields, reason):
    assert preflight_failed(fields, ep_size=32) is True, reason


def test_fabric_identity_is_always_reported_never_silent():
    """An unusable fabric probe must say so rather than return nothing.

    The caller compares these dicts across ranks. A silently empty result
    would compare equal everywhere and look like agreement.
    """
    out = _fabric_identity(0)
    assert "fabric" in out
    assert isinstance(out["fabric"], str) and out["fabric"]
