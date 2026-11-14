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
"""CPU tests for the decline rules of ``MegaMoECuteDslLocal.try_fused_route_quant``.

Each scenario breaks exactly one condition of an otherwise admissible M3.1 decode
chunk. The routing conditions are checked before any device condition, so on CPU
an admissible chunk declines only for its device.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.routing import (
    DeepSeekV3MoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    MiniMaxM3MoeRoutingMethod,
)

pytestmark = pytest.mark.cpu_only

EXPERTS = 128
TOP_K = 4
HIDDEN = 6144
ROWS = 37
DEVICE_REASON = "a tensor is not on the activation's CUDA device"


def _m3_routing(top_k=TOP_K, bias_dtype=torch.float32, experts=EXPERTS):
    bias = torch.zeros(experts, dtype=bias_dtype)
    return MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=2.0,
    )


def _v3_routing(n_group, topk_group):
    bias = torch.zeros(EXPERTS, dtype=torch.float32)
    return DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=2.0,
        callable_e_score_correction_bias=lambda: bias,
    )


class _M3Subclass(MiniMaxM3MoeRoutingMethod):
    pass


def _seat(routing_method, enabled=True):
    return SimpleNamespace(
        _route_quant_enabled=enabled,
        routing_method=routing_method,
        hidden_size=HIDDEN,
        tactic_autotune=False,
        fc31_input_scale=torch.ones(1),
    )


def _reason(seat, logits_dtype=torch.float32, experts=EXPERTS, x_dtype=torch.bfloat16):
    x = torch.zeros(ROWS, HIDDEN, dtype=x_dtype)
    logits = torch.zeros(ROWS, experts, dtype=logits_dtype)
    return TrtllmCutedslMegaMoeLocalNvfp4Impl._route_quant_decline_reason(seat, x, logits)


def test_admissible_chunk_declines_only_for_its_device():
    assert _reason(_seat(_m3_routing())) == DEVICE_REASON
    assert _reason(_seat(_v3_routing(1, 1))) == DEVICE_REASON


@pytest.mark.parametrize(
    "routing",
    [
        pytest.param(
            lambda: MiniMaxM2MoeRoutingMethod(TOP_K, EXPERTS, lambda: torch.zeros(EXPERTS)), id="m2"
        ),
        pytest.param(
            lambda: _M3Subclass(TOP_K, EXPERTS, lambda: torch.zeros(EXPERTS), 2.0), id="subclass"
        ),
    ],
)
def test_declines_routing_methods_other_than_deepseekv3_impl(routing):
    assert _reason(_seat(routing())).startswith("routing method ")


def test_declines_unfused_deepseekv3_impl():
    routing = _m3_routing()
    routing.routing_impl.is_fused = False
    assert _reason(_seat(routing)) == "routing is not the fused Deepseekv3RoutingImpl"


def test_declines_n_group_above_1():
    assert _reason(_seat(_v3_routing(2, 1))) == "n_group 2, topk_group 1"


def test_declines_topk_group_above_1():
    assert _reason(_seat(_v3_routing(1, 2))) == "n_group 1, topk_group 2"


def test_declines_non_fp32_logits():
    assert (
        _reason(_seat(_m3_routing()), logits_dtype=torch.bfloat16)
        == "router logits are not float32"
    )


def test_declines_non_fp32_bias():
    reason = _reason(_seat(_m3_routing(bias_dtype=torch.bfloat16)))
    assert reason == "e_score_correction_bias is not float32"


def test_declines_more_than_256_experts():
    seat = _seat(_m3_routing(experts=257))
    assert _reason(seat, experts=257) == "router logits shape (37, 257)"


def test_declines_top_k_above_32():
    assert _reason(_seat(_m3_routing(top_k=33))) == "top_k 33"


def test_declines_non_bf16_activation():
    assert _reason(_seat(_m3_routing()), x_dtype=torch.float16) == "the activation is not bf16"


def test_declines_with_the_routeq_lever_off():
    assert _reason(_seat(_m3_routing(), enabled=False)) == "the ROUTEQ lever is off"


@pytest.mark.parametrize(
    "environ, enabled",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ": "0"}, True),
    ],
)
def test_routeq_lever_reads_its_disable_switch(monkeypatch, environ, enabled):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
        local_impl as local,
    )

    monkeypatch.setattr(local.os, "environ", environ)
    assert local.lever_enabled("ROUTEQ", local.os.environ) is enabled


# --- the seat staged beside a serial shared expert ---


def _inputs():
    return torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16), torch.zeros(ROWS, EXPERTS)


def _prestage_seat(result=("staged",)):
    """A seat whose own launch returns ``result``, recording the scheduler hint each launch ran under."""
    seat = SimpleNamespace(_prestaged_route_quant=None, _active_launch_max_tokens=None, hints=[])

    def launch(x, router_logits):
        seat.hints.append(seat._active_launch_max_tokens)
        return result

    seat.try_fused_route_quant = launch
    return seat


def _prestage(seat, x, router_logits, chunk_max_tokens=88):
    TrtllmCutedslMegaMoeLocalNvfp4Impl.prestage_route_quant(
        seat, x, router_logits, chunk_max_tokens
    )


def _take(seat, x, router_logits):
    return TrtllmCutedslMegaMoeLocalNvfp4Impl._take_prestaged_route_quant(seat, x, router_logits)


def test_the_prestaged_seat_launches_at_the_steps_bucket_and_restores_the_scheduler_hint():
    seat = _prestage_seat()
    seat._active_launch_max_tokens = 7
    _prestage(seat, *_inputs(), chunk_max_tokens=88)
    assert seat.hints == [88] and seat._active_launch_max_tokens == 7


def test_the_prestaged_seat_is_taken_once_by_the_same_tensors():
    seat, (x, logits) = _prestage_seat(), _inputs()
    _prestage(seat, x, logits)
    assert _take(seat, x, logits) == ("staged",)
    assert _take(seat, x, logits) is None


def test_a_prestaged_seat_for_other_tensors_is_dropped():
    seat, (x, logits) = _prestage_seat(), _inputs()
    _prestage(seat, x, logits)
    assert _take(seat, x.clone(), logits) is None
    assert seat._prestaged_route_quant is None


def test_a_declined_prestage_stages_nothing():
    seat, (x, logits) = _prestage_seat(result=None), _inputs()
    _prestage(seat, x, logits)
    assert seat._prestaged_route_quant is None and _take(seat, x, logits) is None


def test_the_fused_route_quant_takes_a_prestaged_seat_before_any_check():
    def relaunch(x, router_logits):
        raise AssertionError("the staged seat launched again")

    seat = SimpleNamespace(
        _take_prestaged_route_quant=lambda x, router_logits: ("staged",),
        _route_quant_decline_reason=relaunch,
    )
    assert TrtllmCutedslMegaMoeLocalNvfp4Impl.try_fused_route_quant(seat, *_inputs()) == ("staged",)


@pytest.mark.parametrize(
    "enabled, serial, beside", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_route_beside_shared_needs_its_lever_and_a_serial_step(enabled, serial, beside):
    seat = SimpleNamespace(
        _route_beside_shared_enabled=enabled, shared_expert_serial=lambda max_rows: serial
    )
    assert TrtllmCutedslMegaMoeLocalNvfp4Impl.route_beside_shared(seat, 88) is beside


@pytest.mark.parametrize(
    "environ, enabled",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTE_BESIDE_SHARED": "1"}, False),
        ({"TRTLLM_M31_DISABLE_MEGAMOE_LOCAL": "1"}, False),
    ],
)
def test_the_route_beside_lever_reads_its_disable_switch(environ, enabled):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
        local_plan as plan,
    )

    assert plan.lever_enabled("ROUTE_BESIDE_SHARED", environ) is enabled
