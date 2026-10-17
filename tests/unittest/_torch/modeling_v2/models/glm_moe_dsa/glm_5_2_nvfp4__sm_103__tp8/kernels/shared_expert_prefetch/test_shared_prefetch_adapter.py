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
"""Control flow of the shared_expert_prefetch call site in the target's MoE.

With fake collaborators (no device work), on a decode step, checks that

* the plan runs shared_expert_prefetch with its closure (hop_routing's fork into
  routed_moe_into_hop's FP32 accumulator), not without it, and records no numerical variant
  for it;
* deferred_input_prep's tables cover the shared_expert_prefetch directory (with hop_routing's and
  routed_moe's) and its side selectors;
* ``GlmMoE.compute_routed_output`` tries ``shared_expert_prefetch.routed_output_into_hop`` (with the
  module's shared expert) before hop_routing's fork, only with the hop's routing and
  only when shared_expert_prefetch is selected; ``None`` falls through to hop_routing's fork; the
  shared branch stays the native shared expert.
"""

import os
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
    dispatch as _dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import layers

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "shared", "kernels"
    ),
)

import plan_testlib  # noqa: E402

F_FEATURES = (
    "absorbed_query,indexer_topk,attention_output_tail,post_moe_hop,draft_front,"
    "router_gemm,post_attention_hop,draft_tail,sparse_mla_decode,draft_absorbed_query,"
    "rope_kv_append,draft_loop,shared_indexer_absorbed_query,routed_moe,cross_rank_topk,"
    "attention_output_tail_prefetch,attention_output_into_hop,routed_moe_into_hop,"
    "deferred_norm,dense_mlp,deferred_input_prep,hop_routing,oproj_weight_staging,"
    "fused_kv_indexer_projection,host_page_retention"
)


@pytest.fixture(autouse=True)
def decode_step():
    """The owned call site runs on decode steps."""
    with _dispatch.step(_dispatch.Phase.DECODE):
        yield


def _selector(monkeypatch, features):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)
    plan_testlib.select(dispatch, features, monkeypatch)
    return dispatch


@pytest.fixture
def shared_prefetch_on(monkeypatch):
    return _selector(monkeypatch, F_FEATURES + ",shared_expert_prefetch")


@pytest.fixture
def shared_prefetch_off(monkeypatch):
    return _selector(monkeypatch, F_FEATURES)


def test_selector(shared_prefetch_on, monkeypatch):
    dispatch = shared_prefetch_on
    assert dispatch.is_enabled("shared_expert_prefetch")
    assert "shared_expert_prefetch" not in dispatch.NUMERICAL_VARIANTS
    paths = dispatch.feature_record(dispatch.enabled_features()).split()[0]
    assert paths.startswith("paths=") and "shared_expert_prefetch" in paths[6:].split(",")
    closure = (
        "attention_output_tail",
        "post_moe_hop",
        "routed_moe",
        "post_attention_hop",
        "attention_output_into_hop",
        "routed_moe_into_hop",
        "hop_routing",
    )
    assert set(dispatch.REQUIRED_FEATURES["shared_expert_prefetch"]) == set(closure)
    _selector(monkeypatch, closure + ("shared_expert_prefetch",))
    assert dispatch.installed() == frozenset(closure + ("shared_expert_prefetch",))
    for missing in closure:
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: missing})
        assert not plan.enabled("shared_expert_prefetch"), missing


@pytest.fixture
def restore_registry():
    import importlib

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.modeling import (  # noqa: E501
        TARGET_PATHS,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    for feature in ("hop_routing", "routed_moe", "shared_expert_prefetch"):
        importlib.import_module(TARGET_PATHS.kernel_modules[feature])
        dispatch.get_impl(feature, "routed_output_into_hop")
    saved = {k: dict(v) for k, v in dispatch._REGISTRY.items()}
    yield
    dispatch._REGISTRY.clear()
    dispatch._REGISTRY.update(saved)


class _Log:
    def __init__(self):
        self.calls = []

    def fn(self, name, result):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return result

        return call

    def names(self):
        return [c[0] for c in self.calls]


def _moe(dispatch, log, r7_result, monkeypatch):
    from tensorrt_llm._torch.utils import EventType

    for feature, result in (
        ("shared_expert_prefetch", r7_result),
        ("hop_routing", ["hop_routing_into"]),
        ("routed_moe", ["routed_moe_into"]),
    ):
        dispatch.register(
            feature, "routed_output_into_hop", log.fn(f"{feature}.routed_output_into_hop", result)
        )

    def sequential(fn0, fn1, *args, **kwargs):
        log.calls.append(("parallel", (), {}))
        return fn0(), fn1()

    monkeypatch.setattr(layers, "maybe_execute_in_parallel", sequential)
    moe = layers.GlmMoE.__new__(layers.GlmMoE)
    torch.nn.Module.__init__(moe)

    class _Shared:
        def __call__(self, x):
            log.calls.append(("native_shared", (x,), {}))
            return torch.full((1,), 7.0)

    moe.__dict__.update(
        use_dp=False,
        mapping=types.SimpleNamespace(tp_size=8),
        router_gemm=None,
        gate=types.SimpleNamespace(e_score_correction_bias=torch.zeros(256)),
        experts=object(),
        _routed_into_hop=object(),
        shared_experts=_Shared(),
        shared_output_scale=None,
        shared_experts_use_fp4=False,
        event_dict={k: None for k in EventType},
        aux_stream=None,
    )
    return moe


@pytest.mark.parametrize(
    "routing_given, r7_result, want_routed, want_out",
    [
        (
            True,
            ["shared_prefetch_into"],
            ["shared_expert_prefetch.routed_output_into_hop"],
            ["shared_prefetch_into"],
        ),
        (
            True,
            None,
            ["shared_expert_prefetch.routed_output_into_hop", "hop_routing.routed_output_into_hop"],
            ["hop_routing_into"],
        ),
        (
            False,
            ["shared_prefetch_into"],
            ["routed_moe.routed_output_into_hop"],
            ["routed_moe_into"],
        ),
    ],
)
def test_routed_order(
    shared_prefetch_on,
    restore_registry,
    monkeypatch,
    routing_given,
    r7_result,
    want_routed,
    want_out,
):
    log = _Log()
    moe = _moe(shared_prefetch_on, log, r7_result, monkeypatch)
    x = torch.zeros(6, 4)
    logits = torch.zeros(6, 256)
    routing = torch.zeros(2, 6, 8, dtype=torch.int32) if routing_given else None
    out = moe.forward(x, do_finalize=False, router_logits=logits, hop_routing_out=routing)
    names = log.names()
    assert [n for n in names if n.endswith("routed_output_into_hop")] == want_routed
    assert names.count("native_shared") == 1
    assert float(out[0][0]) == 7.0 and out[1:] == want_out
    for name, args, kwargs in log.calls:
        if name == "shared_expert_prefetch.routed_output_into_hop":
            want = (
                moe.experts,
                moe.gate.e_score_correction_bias,
                x,
                logits,
                routing,
                moe._routed_into_hop,
                moe.shared_experts,
            )
            assert not kwargs and len(args) == len(want) and all(a is b for a, b in zip(args, want))


def test_not_selected_never_calls_shared_prefetch(
    shared_prefetch_off, restore_registry, monkeypatch
):
    log = _Log()
    moe = _moe(shared_prefetch_off, log, ["shared_prefetch_into"], monkeypatch)
    out = moe.forward(
        torch.zeros(6, 4),
        do_finalize=False,
        router_logits=torch.zeros(6, 256),
        hop_routing_out=torch.zeros(2, 6, 8, dtype=torch.int32),
    )
    assert not [n for n in log.names() if n.startswith("shared_expert_prefetch.")]
    assert float(out[0][0]) == 7.0 and out[1:] == ["hop_routing_into"]
