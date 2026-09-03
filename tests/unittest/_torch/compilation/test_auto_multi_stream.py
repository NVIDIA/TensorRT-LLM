# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scheduling of in-place side effects that the FX output does not reach.

Eagle3 captures decoder hidden states into a preallocated buffer with
``inplace_slice_copy``; the drafter reads that buffer outside the compiled
graph.  The multi-stream scheduler must emit every such mutation before ``output``
(a node emitted after ``output`` is dead code once the module is recompiled)
and make the exit wait on the mutating stream.
"""

import pytest
import torch
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.compilation.multi_stream.auto_multi_stream import (
    MultiStreamDAG,
    multi_stream_schedule,
)

COPY = torch.ops.trtllm.inplace_slice_copy.default


def _capture(graph: Graph, dest, src, layer: int):
    return graph.call_function(
        COPY, kwargs={"dest": dest, "src": src, "dim1_start": layer, "dim1_end": layer + 1}
    )


def _decoder_stack(n_layers: int, capture_layers: tuple[int, ...]):
    """A chain of layers; selected layers copy their output into ``dest``."""
    graph = Graph()
    dest = graph.placeholder("dest")
    x = graph.placeholder("x")
    hidden = x
    captures = []
    for layer in range(n_layers):
        mm = graph.call_function(torch.ops.aten.mm.default, args=(hidden, hidden))
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(mm, hidden))
        hidden = graph.call_function(torch.ops.aten.mul.Tensor, args=(add, 2.0))
        if layer in capture_layers:
            captures.append(_capture(graph, dest, hidden, layer))
    out = graph.call_function(torch.ops.aten.neg.default, args=(hidden,))
    graph.output((out,))
    return GraphModule({}, graph), dest, captures


def _index_of(nodes, predicate):
    return next(i for i, node in enumerate(nodes) if predicate(node))


def test_graph_exit_depends_on_unreturned_inplace_side_effect() -> None:
    graph = Graph()
    dest = graph.placeholder("dest")
    src = graph.placeholder("src")
    returned = graph.call_function(torch.ops.aten.neg.default, args=(src,))
    mutation = _capture(graph, dest, src, 0)
    output = graph.output(returned)
    graph_module = GraphModule({}, graph)

    dag = MultiStreamDAG(graph_module)
    assert dag.nodes[output].in_edges[dest] is dag.nodes[mutation]

    dag.assign_streams(max_num_streams=2)
    scheduled = dag.create_new_graph()
    scheduled.lint()
    nodes = list(scheduled.nodes)
    output_index = _index_of(nodes, lambda n: n.op == "output")
    mutation_index = _index_of(nodes, lambda n: n.target is COPY)
    assert mutation_index < output_index

    if dag.nodes[mutation].stream is not dag.nodes[output].stream:
        event = dag.nodes[mutation].event
        assert event is not None
        assert any(
            node.target is torch.ops.trtllm.wait_event and node.args == (event,)
            for node in nodes[mutation_index:output_index]
        )


@pytest.mark.parametrize("max_num_streams", [2, 3])
@pytest.mark.parametrize(
    "n_layers,capture_layers",
    [(6, (1, 3, 5)), (6, (1, 3, 4)), (12, (1, 5, 11)), (12, (1, 5, 8))],
)
def test_every_capture_precedes_output(n_layers, capture_layers, max_num_streams) -> None:
    graph_module, _, captures = _decoder_stack(n_layers, capture_layers)
    multi_stream_schedule(graph_module, max_num_streams)
    graph_module.graph.lint()
    nodes = list(graph_module.graph.nodes)
    output_index = _index_of(nodes, lambda n: n.op == "output")
    copy_indices = [i for i, node in enumerate(nodes) if node.target is COPY]
    assert len(copy_indices) == len(captures)
    assert all(i < output_index for i in copy_indices), (copy_indices, output_index)

    # Every capture must survive into the generated code ahead of `return`.
    graph_module.recompile()
    lines = [line.strip() for line in graph_module.code.splitlines() if line.strip()]
    return_index = _index_of(lines, lambda line: line.startswith("return"))
    copy_lines = [i for i, line in enumerate(lines) if "inplace_slice_copy" in line]
    assert len(copy_lines) == len(captures)
    assert all(i < return_index for i in copy_lines), (copy_lines, return_index)
