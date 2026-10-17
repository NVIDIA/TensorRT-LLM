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
"""Call-site invariants of the post_attention_hop in the target's layers.

The post_attention_hop keeps one workspace and one device-side buffer state per TP group
for every layer, CUDA graph and eager call, as the native MNNVL all-reduce
does. That is correct only while every rank issues the same sequence of hop
calls on one stream, deciding each call from rank-uniform facts. These checks
pin the preconditions on the target's layer module:

* the hop and the into-hop transport (attention_output_into_hop) are created at one place each, at
  layer adoption, through the dispatch entry points;
* every hop call (post_attention_hop, into-hop entry) is issued directly on the layer's stream:
  never inside a ``with`` block (e.g. ``torch.cuda.stream``) and never inside
  a nested function or lambda (e.g. a closure handed to
  ``maybe_execute_in_parallel``, which runs one branch on an auxiliary
  stream);
* the per-call decision reads only shapes, dtypes, contiguity and the
  rank-agreed SM count: no device-to-host transfer or tensor value;
* dense layers keep the native post-attention path: ``forward_mlp`` never
  reaches the hop (scope of the served post_attention_hop: MoE and MTP layers).
"""

import ast
import inspect
import textwrap

import pytest

modeling = pytest.importorskip(
    "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers"
)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_attention_hop import (  # noqa: E402, E501
    hop,
)

_HOP_CALLS = ("_post_attn_hop", "_into_hop", "post_attn_hop")


@pytest.fixture(scope="module")
def tree():
    return ast.parse(inspect.getsource(modeling))


def _self_attr_calls(node, attrs):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in attrs
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]


def _parents(tree):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _methods(tree, cls_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"{cls_name} not found")


def test_hop_created_once_through_dispatch(tree):
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Call)
        and isinstance(n.func.func, ast.Attribute)
        and n.func.func.attr == "get_impl"
        and len(n.func.args) == 2
        and all(isinstance(a, ast.Constant) for a in n.func.args)
    ]
    entries = [(c.func.args[0].value, c.func.args[1].value) for c in calls]
    assert entries.count(("post_attention_hop", "make_post_attn_hop")) == 1, entries
    assert entries.count(("post_attention_hop", "make_into_hop_transport")) <= 1, entries
    methods = _methods(tree, "_AdoptedLayerMixin")
    init = methods["_adopt_pre_moe"]
    in_init = [
        c for c in ast.walk(init) if c in calls and c.func.args[0].value == "post_attention_hop"
    ]
    assert len(in_init) == len(
        [e for e in entries if e[0] == "post_attention_hop" and e[1].startswith("make_")]
    )


def test_hop_calls_are_issued_on_the_layer_stream(tree):
    parent = _parents(tree)
    sites = []
    for call in _self_attr_calls(tree, _HOP_CALLS):
        node = call
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.With, ast.AsyncWith)):
                raise AssertionError(
                    f"hop call at line {call.lineno} inside a with block: "
                    f"{ast.unparse(node.items[0])}"
                )
            if isinstance(node, ast.Lambda):
                raise AssertionError(f"hop call at line {call.lineno} inside a lambda")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outer = parent.get(node)
                assert isinstance(outer, ast.ClassDef), (
                    f"hop call at line {call.lineno} inside nested function {node.name}"
                )
                sites.append((outer.name, node.name))
                break
    # The served call sites: target MoE layers (forward_MoE) and the MTP
    # layer; _post_attn_hop itself wraps the hop object.
    assert ("GlmDecoderLayer", "forward_MoE") in sites, sites
    assert ("GlmMtpLayer", "forward") in sites, sites


def test_dense_layers_never_reach_the_hop(tree):
    methods = _methods(tree, "GlmDecoderLayer")
    forward_mlp = methods["forward_mlp"]
    names = {n.attr for n in ast.walk(forward_mlp) if isinstance(n, ast.Attribute)}
    assert not names & {"post_attn_hop", "_post_attn_hop", "_can_use_post_attn_hop", "_into_hop"}, (
        names
    )


def test_call_decision_reads_no_tensor_values():
    for fn in (hop.PostAttnHop.call_ineligibility, hop.PostAttnHop.can_run, hop.required_sms):
        node = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        assert not attrs & {
            "item",
            "tolist",
            "cpu",
            "numpy",
            "any",
            "all",
            "sum",
            "max",
            "min",
            "nonzero",
            "synchronize",
        }, (fn.__name__, attrs)
