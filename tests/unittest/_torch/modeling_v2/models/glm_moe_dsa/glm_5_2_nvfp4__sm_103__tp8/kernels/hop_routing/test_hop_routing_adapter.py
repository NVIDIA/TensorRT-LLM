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
"""Control flow of the hop_routing call sites in the target's decoder layer and MoE.

With fake collaborators (no device work), on a decode step, checks that

* the plan runs hop_routing with its closure, and not without it;
* ``GlmDecoderLayer.forward_MoE`` runs the hop_routing hop instead of the into-hop entry's
  exactly when the tail published and the hop_routing transport can run, hands the hop's
  routing and logits to the MoE, and otherwise runs the into-hop entry's hop with no routing;
* ``GlmMoE.compute_routed_output`` tries routed_moe's hop_routing fork first only when routing
  is present (routed_moe_into_hop entry with the hop, the plain entry without), and falls back
  to routed_moe on the same logits when the fork returns None.
"""

import os
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
    dispatch as _dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
    GlmDecoderLayer,
    GlmMoE,
)

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "shared", "kernels"
    ),
)

import plan_testlib  # noqa: E402

FEATURES = (
    "attention_output_tail,post_moe_hop,post_attention_hop,attention_output_into_hop,"
    "routed_moe,routed_moe_into_hop,hop_routing"
)


@pytest.fixture(autouse=True)
def decode_step():
    """The owned call sites run on decode steps."""
    with _dispatch.step(_dispatch.Phase.DECODE):
        yield


@pytest.fixture
def selector(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)
    plan_testlib.select(dispatch, FEATURES, monkeypatch)
    return dispatch


def test_selector_accepts_hop_routing(selector):
    dispatch = selector
    assert dispatch.is_enabled("hop_routing")
    for base in dispatch.REQUIRED_FEATURES["hop_routing"]:
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: base})
        assert not plan.enabled("hop_routing"), base


class _Calls:
    def __init__(self):
        self.log = []

    def fn(self, name, result):
        def call(*args, **kwargs):
            self.log.append((name, args, kwargs))
            return result

        return call


def _decoder(hop_routing_can_run, calls):
    dec = GlmDecoderLayer.__new__(GlmDecoderLayer)
    torch.nn.Module.__init__(dec)
    rows = 6
    normed = torch.zeros(rows, 4)
    res = torch.ones(rows, 4)
    logits = torch.full((rows, 256), 2.0)
    routing = torch.zeros(2, rows, 8, dtype=torch.int32)
    bias = torch.zeros(256, dtype=torch.bfloat16)
    weight = torch.zeros(256, 4)

    class _HOP_ROUTING:
        def can_run(self, *args, **kwargs):
            calls.log.append(("hop_routing.can_run", args, kwargs))
            return hop_routing_can_run

        def __call__(self, *args, **kwargs):
            calls.log.append(("hop_routing", args, kwargs))
            return normed, res, logits, routing

    class _INTO_HOP:
        def __call__(self, *args, **kwargs):
            calls.log.append(("into_hop", args, kwargs))
            return normed, res, logits

    class _MoE:
        gate = types.SimpleNamespace(e_score_correction_bias=bias)
        experts = types.SimpleNamespace(has_nvfp4=True)

        def __call__(self, hidden_states, hidden_states_fp4, **kwargs):
            calls.log.append(("mlp", (hidden_states,), kwargs))
            return hidden_states

    dec.__dict__.update(
        _hop_routing=_HOP_ROUTING(),
        _into_hop=_INTO_HOP(),
        mlp=_MoE(),
        _post_attn_router_weight=lambda: weight,
        post_attention_layernorm=types.SimpleNamespace(
            weight=torch.ones(4), variance_epsilon=1e-5, nvfp4_scale=None
        ),
        fusion_config=types.SimpleNamespace(PRE_MOE_FUSION=True, POST_MOE_FUSION=False),
        mapping=types.SimpleNamespace(tp_size=8, is_multi_node=lambda: True),
        post_moe_hop=None,
        next_layer_layernorm=None,
        layer_idx=3,
        _apply_next_layer_layernorm=lambda h, r: (h, r),
    )
    return dec, (normed, res, logits, routing, bias, weight)


@pytest.mark.parametrize("hop_routing_can_run", [True, False])
def test_forward_moe_selects_the_hop(hop_routing_can_run):
    calls = _Calls()
    dec, (normed, res, logits, routing, bias, weight) = _decoder(hop_routing_can_run, calls)
    meta = types.SimpleNamespace(all_rank_num_tokens=[6])
    GlmDecoderLayer.forward_MoE(dec, torch.zeros(6, 4), meta, res, into_hop_published=True)
    names = [c[0] for c in calls.log]
    assert names[0] == "hop_routing.can_run"
    can = calls.log[0]
    assert can[1][2] is weight and can[1][3] is bias
    mlp = [c for c in calls.log if c[0] == "mlp"][0]
    assert mlp[2]["router_logits"] is logits
    if hop_routing_can_run:
        assert "hop_routing" in names and "into_hop" not in names
        assert mlp[2]["hop_routing_out"] is routing
    else:
        assert "into_hop" in names and "hop_routing" not in names
        assert mlp[2]["hop_routing_out"] is None


def _moe(selector, calls, hop_routing_into, hop_routing_plain, hop):
    dispatch = selector
    for feature, name, result in (
        ("hop_routing", "routed_output_into_hop", hop_routing_into),
        ("hop_routing", "routed_output", hop_routing_plain),
        ("routed_moe", "routed_output_into_hop", ["routed_moe_into"]),
        ("routed_moe", "routed_output", ["routed_moe_plain"]),
    ):
        dispatch.register(feature, name, calls.fn(f"{feature}.{name}", result))
    moe = GlmMoE.__new__(GlmMoE)
    torch.nn.Module.__init__(moe)
    moe.__dict__.update(
        use_dp=False,
        mapping=types.SimpleNamespace(tp_size=8),
        router_gemm=None,
        gate=types.SimpleNamespace(e_score_correction_bias=torch.zeros(256)),
        experts=object(),
        _routed_into_hop=hop,
    )
    return moe


@pytest.fixture
def restore_registry():
    import importlib

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.modeling import (  # noqa: E501
        TARGET_PATHS,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    # Import both packages (which registers their real entries) before the snapshot: a package registers only on
    # its first import, so entries dropped by a restore would never come back.
    for feature in ("hop_routing", "routed_moe"):
        importlib.import_module(TARGET_PATHS.kernel_modules[feature])
        dispatch.get_impl(feature, "routed_output")
    saved = {k: dict(v) for k, v in dispatch._REGISTRY.items()}
    yield
    dispatch._REGISTRY.clear()
    dispatch._REGISTRY.update(saved)


@pytest.mark.parametrize(
    "hop, routing_given, fork_result, want",
    [
        ("hop", True, ["fork_into"], ["hop_routing.routed_output_into_hop"]),
        (
            "hop",
            True,
            None,
            ["hop_routing.routed_output_into_hop", "routed_moe.routed_output_into_hop"],
        ),
        ("hop", False, ["fork_into"], ["routed_moe.routed_output_into_hop"]),
        (None, True, ["fork_plain"], ["hop_routing.routed_output"]),
        (None, True, None, ["hop_routing.routed_output", "routed_moe.routed_output"]),
        (None, False, ["fork_plain"], ["routed_moe.routed_output"]),
    ],
)
def test_compute_routed_output_order(
    selector, restore_registry, hop, routing_given, fork_result, want
):
    calls = _Calls()
    moe = _moe(selector, calls, fork_result, fork_result, object() if hop else None)
    x = torch.zeros(6, 4)
    logits = torch.zeros(6, 256)
    routing = torch.zeros(2, 6, 8, dtype=torch.int32) if routing_given else None
    out = moe.compute_routed_output(
        x, None, None, False, router_logits=logits, hop_routing_out=routing
    )
    assert [c[0] for c in calls.log] == want
    for name, args, _ in calls.log:
        assert args[3] is logits  # every entry receives the hop's logits
        if name.startswith("hop_routing."):
            assert args[4] is routing
    results = {
        "hop_routing.routed_output_into_hop": fork_result,
        "hop_routing.routed_output": fork_result,
        "routed_moe.routed_output_into_hop": ["routed_moe_into"],
        "routed_moe.routed_output": ["routed_moe_plain"],
    }
    assert out == results[want[-1]]
