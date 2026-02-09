from typing import Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
    aux_stream_wrapper,
    cuda_stream_manager,
    record_event_passthrough,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import canonicalize_graph
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


@torch.library.custom_op("auto_deploy::multi_stream_linear", mutates_args=())
def multi_stream_linear(
    input: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor
) -> torch.Tensor:
    output = torch.ops.aten.linear(input, weight0)
    output = torch.ops.aten.linear(output, weight1)
    return output


@multi_stream_linear.register_fake
def multi_stream_linear_fake(input, weight0, weight1):
    """Fake implementation of multi_stream_linear."""
    output = torch.ops.aten.linear(input, weight0)
    return torch.ops.aten.linear(output, weight1)


def replace_multi_stream_linear_with_aux_stream_wrapper(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Traverse ``gm`` and replace all ``auto_deploy::multi_stream_linear`` ops with ``aux_stream_wrapper``.

    For each target op we:
      1. Find the shared input (a node with >1 users, e.g. relu output used
         by both a main-stream consumer and the target op).
      2. Insert ``record_event_passthrough`` right after it to mark a CUDA
         synchronisation point — the aux stream waits for this event before
         starting.
      3. Wire the event node into the target op (other users of the shared
         input are unaffected) and replace the op with ``aux_stream_wrapper``.

    Returns:
        A tuple of (gm, num_replaced)
    """
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating
    target_nodes: list[Node] = [
        n for n in graph.nodes if is_op(n, torch.ops.auto_deploy.multi_stream_linear)
    ]

    for n in target_nodes:
        # Find the shared input — a node consumed by multiple ops.
        # Analogous to the routing input in the MoE transform.
        shared_input = None
        for input_node in n.all_input_nodes:
            if len(input_node.users) > 1:
                shared_input = input_node
                break
        if shared_input is None:
            raise ValueError(f"Shared input node not found for node {n}")

        # Record CUDA event right after the shared input so the aux stream
        # can start as soon as this dependency is ready, while the main
        # stream continues with other consumers (e.g. fc2).
        with graph.inserting_after(shared_input):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(shared_input,),
                kwargs={"device": torch.cuda.current_device()},
            )

        # Wire the event node into the target op only.
        n.args = tuple(rec_node if arg is shared_input else arg for arg in n.args)

        with graph.inserting_after(n):
            new_node = graph.call_function(
                aux_stream_wrapper, args=(n.target, *n.args), kwargs=n.kwargs
            )
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)
        num_replaced += 1

    if num_replaced:
        canonicalize_graph(gm)

    return gm, num_replaced


class ParallelTwoLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc10 = nn.Linear(in_dim, in_dim)
        self.fc11 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(x)
        y0 = self.fc2(x)
        y1 = torch.ops.auto_deploy.multi_stream_linear(x, self.fc10.weight, self.fc11.weight)
        return y0 + y1


def test_multi_stream_linear():
    in_dim, out_dim = 128, 256
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = (
        nn.Sequential(ParallelTwoLinear(in_dim, out_dim), ParallelTwoLinear(out_dim, out_dim))
        .eval()
        .to("cuda")
    )

    # Example input used for export
    example_input = torch.randn(4, in_dim).to("cuda")

    # Export the graph
    egm = torch.export.export(model, (example_input,))
    gm = egm.module()

    test_x = torch.randn(4, in_dim).to("cuda")
    ref_output = model(test_x)

    # pattern matching and replace
    gm, num_replaced = replace_multi_stream_linear_with_aux_stream_wrapper(gm)

    assert num_replaced == 2
    y = gm(test_x)
    assert torch.allclose(y, ref_output)

    static_x = torch.randn(4, in_dim).to("cuda")
    static_output = torch.randn(4, out_dim).to("cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_output, ref_output)
