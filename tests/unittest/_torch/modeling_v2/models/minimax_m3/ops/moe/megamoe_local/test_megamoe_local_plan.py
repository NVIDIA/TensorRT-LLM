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
"""CPU tests for the MegaMoE local backend's switches and deployment checks (``local_plan``)."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS = 128


def _raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def _admission(**overrides):
    kwargs = dict(
        ep_size=8,
        num_experts=NUM_EXPERTS,
        num_slots=NUM_EXPERTS,
        moe_tp_size=1,
        eplb_enabled=False,
        world_size=8,
    )
    kwargs.update(overrides)
    return plan.vd_static_rejection(**kwargs)


def test_default_on_and_master_kill_switch():
    assert plan.megamoe_local_requested({})
    assert plan.megamoe_local_requested({plan.DISABLE_ENV: "0"})
    assert not plan.megamoe_local_requested({plan.DISABLE_ENV: "1"})


def test_lever_switches():
    env = {}
    assert all(plan.lever_enabled(lever, env) for lever in plan.LEVERS)
    env[f"{plan.DISABLE_ENV}_TAIL"] = "1"
    assert not plan.lever_enabled("TAIL", env)
    assert plan.lever_enabled("ROUTEQ", env)
    assert not plan.lever_enabled("ROUTEQ", {plan.DISABLE_ENV: "1"})
    assert _raises_value_error(lambda: plan.lever_enabled("NOPE", env))


def test_vd_refuses_any_eplb():
    assert _admission() is None
    assert _admission(ep_size=16, world_size=16) is None
    assert "load balancer" in _admission(eplb_enabled=True)
    assert "redundant" in _admission(num_slots=NUM_EXPERTS + 16)


def test_vd_static_preconditions():
    assert "moe_tp_size" in _admission(moe_tp_size=2)
    assert "multi-rank" in _admission(ep_size=1, world_size=1)
    assert "divisible" in _admission(ep_size=3, world_size=3)
    assert "WORLD" in _admission(world_size=16)
    assert _admission(world_size=None) is None
