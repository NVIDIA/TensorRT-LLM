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
"""Router-producer invariants of the target's MoE and layers (router_gemm, post_attention_hop).

* one ``is_enabled("router_gemm")`` gate, at ``GlmMoE`` adoption, and the router GEMM is
  None (the native gate) unless adoption attaches it;
* ``GlmMoE.compute_routed_output`` has exactly one router_gemm router call and one native
  gate call; router_gemm runs only when no logits were passed in, the gate only when neither
  the caller (post_attention_hop) nor router_gemm produced logits, in that order;
* the post_attention_hop's logits reach ``compute_routed_output`` only through
  ``router_logits`` and only on the tensor-parallel path.
"""

import ast
import inspect
import textwrap

import pytest

modeling = pytest.importorskip(
    "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers"
)


def _function(cls, name):
    source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    return ast.parse(source).body[0]


def _self_calls(node, attr):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]


def _enclosing_ifs(func, target):
    """Tests of the ``if`` statements that contain ``target`` (outermost first)."""
    tests = []

    def visit(node, stack):
        if node is target:
            tests.extend(stack)
            return True
        children = ast.iter_child_nodes(node)
        if isinstance(node, ast.If):
            for child in node.body:
                if visit(child, stack + [node.test]):
                    return True
            for child in node.orelse:
                if visit(child, stack):
                    return True
            return False
        return any(visit(child, stack) for child in children)

    visit(func, [])
    return [ast.unparse(t) for t in tests]


@pytest.fixture(scope="module")
def integrated():
    return inspect.getsource(modeling)


def test_single_router_gemm_gate(integrated):
    assert integrated.count('is_enabled("router_gemm")') == 1
    adopt = inspect.getsource(modeling.GlmMoE.adopt)
    assert 'is_enabled("router_gemm")' in adopt
    assert modeling.GlmMoE.router_gemm is None


def test_router_producer_order(integrated):
    func = _function(modeling.GlmMoE, "compute_routed_output")
    args = [a.arg for a in func.args.args]
    assert "router_logits" in args
    defaults = dict(zip(args[-len(func.args.defaults) :], func.args.defaults))
    assert ast.unparse(defaults["router_logits"]) == "None"

    router_gemm_calls = _self_calls(func, "router_gemm")
    gate_calls = _self_calls(func, "gate")
    assert len(router_gemm_calls) == 1 and len(gate_calls) == 1
    assert router_gemm_calls[0].lineno < gate_calls[0].lineno

    router_gemm_guard = " and ".join(_enclosing_ifs(func, router_gemm_calls[0]))
    assert "router_logits is None" in router_gemm_guard
    assert "self.router_gemm is not None" in router_gemm_guard
    gate_guard = _enclosing_ifs(func, gate_calls[0])
    assert gate_guard == ["router_logits is None"], gate_guard

    # router_logits is assigned only by the two producers.
    assigned = [
        n.value
        for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "router_logits" for t in n.targets)
    ]
    assert sorted(ast.unparse(v) for v in assigned) == sorted(
        [
            ast.unparse(router_gemm_calls[0]),
            ast.unparse(gate_calls[0]),
        ]
    )


def test_attn_hop_logits_only_through_the_argument(integrated):
    forward = inspect.getsource(modeling.GlmMoE.forward)
    assert "assert router_logits is None or not self.use_dp" in forward
    for cls, name in (
        (modeling.GlmDecoderLayer, "forward_MoE"),
        (modeling.GlmMtpLayer, "forward"),
    ):
        func = _function(cls, name)
        hop_calls = _self_calls(func, "_post_attn_hop")
        assert len(hop_calls) == 1, (cls.__name__, name)
        guard = " and ".join(_enclosing_ifs(func, hop_calls[0]))
        assert "PRE_MOE_FUSION" in guard and "_can_use_post_attn_hop" in guard
        source = ast.unparse(func)
        assert "router_logits=router_logits" in source
