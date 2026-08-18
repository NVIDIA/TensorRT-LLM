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

import torch
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.compilation.multi_stream.auto_multi_stream import MultiStreamDAG


def test_graph_exit_waits_for_unreturned_inplace_side_effect() -> None:
    graph = Graph()
    dest = graph.placeholder("dest")
    src = graph.placeholder("src")

    # Put the returned computation first so that it occupies the primary
    # stream and the independent mutation is assigned to an auxiliary stream.
    returned = graph.call_function(torch.ops.aten.neg.default, args=(src,))
    mutation = graph.call_function(
        torch.ops.trtllm.inplace_slice_copy.default,
        kwargs={
            "dest": dest,
            "src": src,
            "dim1_start": 0,
            "dim1_end": 1,
        },
    )
    output = graph.output(returned)
    graph_module = GraphModule({}, graph)

    dag = MultiStreamDAG(graph_module)

    assert dag.nodes[output].in_edges[dest] is dag.nodes[mutation]

    num_events = dag.assign_streams(max_num_streams=2)
    assert num_events > 0
    assert dag.nodes[returned].stream.id == 0
    assert dag.nodes[mutation].stream.id == 1
    assert dag.nodes[output].stream.id == 0
    assert dag.nodes[mutation].event is not None
    assert dag.nodes[output].wait_on == [(dag.nodes[mutation], dest)]

    scheduled_graph = dag.create_new_graph()
    scheduled_graph.lint()
    scheduled_nodes = list(scheduled_graph.nodes)
    output_index = next(i for i, node in enumerate(scheduled_nodes) if node.op == "output")
    assert any(
        node.target == torch.ops.trtllm.wait_event and node.args == (dag.nodes[mutation].event,)
        for node in scheduled_nodes[:output_index]
    )
