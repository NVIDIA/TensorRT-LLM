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

"""CPU-only regression tests for ``_determine_fused_weight_dims``.

These guard the fix for the Phi-4 TP=2 regression (issue #11220). A fused QKV
linear is column-sharded, but the downstream ``split_with_sizes`` sizes were
never rescaled by ``world_size`` because ``_determine_fused_weight_dims``
computed the fused dims into a local variable and then fell off the end of the
function with no ``return`` statement, so the caller always received ``None``
and the ``if fused_weight_dims is not None`` rescaling branch never ran.
"""

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register custom ops
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import _determine_fused_weight_dims
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op


def _linear_node(gm: GraphModule):
    for n in gm.graph.nodes:
        if is_linear_op(n):
            return n
    raise AssertionError(f"no linear node in graph: {[n.target for n in gm.graph.nodes]}")


def test_fused_qkv_split_returns_split_sizes():
    """A linear feeding a ``split_with_sizes`` must report the fused split sizes.

    This is the exact Phi-4 fused-QKV shape: out_features = 5120 + 1280 + 1280.
    The pre-fix bug made this return ``None``.
    """
    q, k, v = 5120, 1280, 1280

    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    lin = graph.call_function(torch.ops.aten.linear, args=(x, w))
    split = graph.call_function(
        torch.ops.aten.split_with_sizes, args=(lin, [q, k, v]), kwargs={"dim": -1}
    )
    graph.output(split)
    gm = GraphModule(nn.Module(), graph)

    fused_weight_dims = _determine_fused_weight_dims([_linear_node(gm)])

    assert fused_weight_dims is not None, (
        "fused_weight_dims must not be None for a fused QKV split "
        "(regression: missing return in _determine_fused_weight_dims)"
    )
    assert list(fused_weight_dims) == [q, k, v]


def test_unfused_linear_returns_none():
    """A linear with no split/slice/chunk user is not fused and must return None.

    This guards against the fix over-reporting fused dims (which would make the
    caller wrongly rescale unrelated split nodes).
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    lin = graph.call_function(torch.ops.aten.linear, args=(x, w))
    relu = graph.call_function(torch.ops.aten.relu, args=(lin,))
    graph.output(relu)
    gm = GraphModule(nn.Module(), graph)

    assert _determine_fused_weight_dims([_linear_node(gm)]) is None


def test_more_than_one_linear_returns_none():
    """The helper only handles a single fused linear; multiple inputs return None."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    w0 = graph.placeholder("w0")
    w1 = graph.placeholder("w1")
    lin0 = graph.call_function(torch.ops.aten.linear, args=(x, w0))
    lin1 = graph.call_function(torch.ops.aten.linear, args=(x, w1))
    graph.output((lin0, lin1))
    gm = GraphModule(nn.Module(), graph)

    linears = [n for n in gm.graph.nodes if is_linear_op(n)]
    assert len(linears) == 2
    assert _determine_fused_weight_dims(linears) is None
