# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CPU-only tests for sharding-related FX node predicates in ``node_utils``."""

import operator

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register custom ops
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ShardableNode
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_any_split_op, is_any_view_op


def _call_function_nodes(gm: GraphModule):
    return [n for n in gm.graph.nodes if n.op == "call_function"]


def test_is_any_view_op_aten_view():
    class ViewModel(nn.Module):
        def forward(self, x):
            return x.view(2, 4)

    # ``symbolic_trace`` records ``Tensor.view`` as ``call_method``; ``torch.export`` lowers to
    # ``torch.ops.aten.view.default``, which ``is_any_view_op`` matches.
    exported = torch.export.export(ViewModel(), (torch.randn(8),))
    gm = exported.module()
    assert any(n.target == torch.ops.aten.view.default for n in _call_function_nodes(gm))
    assert any(is_any_view_op(n) for n in _call_function_nodes(gm)), (
        f"Expected aten view in graph, got targets: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_is_any_view_op_auto_deploy():
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(x, [2, 4]),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    graph.output(out)
    gm = GraphModule(nn.Module(), graph)
    view_nodes = [n for n in _call_function_nodes(gm) if is_any_view_op(n)]
    assert len(view_nodes) == 1
    assert is_any_view_op(view_nodes[0])


def test_is_any_view_op_negative():
    class AtenLinearOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.ops.aten.linear.default(x, self.w, self.b)

    gm = torch.fx.symbolic_trace(AtenLinearOnly())
    assert not any(is_any_view_op(n) for n in _call_function_nodes(gm)), (
        f"Unexpected view op in linear-only graph: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_is_any_split_op_aten():
    class SplitModel(nn.Module):
        def forward(self, x):
            a, b = torch.split(x, [2, 2], dim=-1)
            return a + b

    exported = torch.export.export(SplitModel(), (torch.randn(2, 4),))
    gm = exported.module()
    assert any(
        n.target == torch.ops.aten.split_with_sizes.default for n in _call_function_nodes(gm)
    )
    assert any(is_any_split_op(n) for n in _call_function_nodes(gm)), (
        f"Expected split op in graph, got: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_is_any_split_op_auto_deploy():
    graph = fx.Graph()
    x = graph.placeholder("x")
    splits = graph.call_function(
        torch.ops.auto_deploy.split_with_sizes.default,
        args=(x, [2, 2], -1),
        kwargs={"enable_sharding": False, "layer_type": "unknown"},
    )
    first = graph.call_function(operator.getitem, args=(splits, 0))
    graph.output(first)
    gm = GraphModule(nn.Module(), graph)
    split_nodes = [n for n in _call_function_nodes(gm) if is_any_split_op(n)]
    assert len(split_nodes) == 1
    assert is_any_split_op(split_nodes[0])


def _minimal_graph_module_for_enable_sharding_linear():
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

    root = Shell()
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("w")
    b = graph.get_attr("b")
    lin = graph.call_function(
        torch.ops.auto_deploy.torch_linear_simple.default,
        args=(x, w, b, "none", None, 1, "unknown"),
        kwargs={},
    )
    graph.output(lin)
    return GraphModule(root, graph)


def test_enable_sharding_node_linear():
    gm = _minimal_graph_module_for_enable_sharding_linear()
    lin_nodes = [n for n in _call_function_nodes(gm) if ShardableNode.from_node(n) is not None]
    assert len(lin_nodes) == 1
    assert ShardableNode.from_node(lin_nodes[0]) is not None


def test_enable_sharding_node_view():
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(x, [2, 4]),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    graph.output(out)
    gm = GraphModule(nn.Module(), graph)
    view_nodes = [n for n in _call_function_nodes(gm) if ShardableNode.from_node(n) is not None]
    assert len(view_nodes) == 1
    assert ShardableNode.from_node(view_nodes[0]) is not None


def test_enable_sharding_node_none_for_aten():
    class AtenLinearOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.ops.aten.linear.default(x, self.w, self.b)

    gm = torch.fx.symbolic_trace(AtenLinearOnly())
    aten_linear = [n for n in _call_function_nodes(gm) if n.target == torch.ops.aten.linear.default]
    assert len(aten_linear) == 1
    assert ShardableNode.from_node(aten_linear[0]) is None
