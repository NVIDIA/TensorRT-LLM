"""Transform for multi-stream execution of MoE layers that have shared experts and routed experts."""

from typing import Callable, Dict, Tuple

import torch
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.auto_deploy.custom_ops.multi_stream import record_event_wrapper

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _execute_op_in_aux_stream(
    gm: GraphModule, op_dict: Dict[Callable, Callable]
) -> Tuple[GraphModule, int]:
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating
    target_nodes: list[Node] = []
    for n in graph.nodes:
        if is_op(n, op_dict.keys()):
            target_nodes.append(n)

    for n in target_nodes:
        target_input_node = None
        for input_node in n.all_input_nodes:
            if input_node.target == torch.ops.aten.view.default:
                target_input_node = input_node
                break

        if target_input_node is None:
            raise ValueError(f"Target input node not found for node {n}")
        with graph.inserting_before(target_input_node):
            new_node = graph.call_function(
                record_event_wrapper,
                args=(target_input_node.target, *target_input_node.args),
                kwargs=target_input_node.kwargs,
            )
        target_input_node.replace_all_uses_with(new_node)
        graph.erase_node(target_input_node)
        with graph.inserting_after(n):
            new_node = graph.call_function(op_dict[n.target], args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)
        num_replaced += 1
    if num_replaced:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return gm, num_replaced


@TransformRegistry.register("multi_stream_moe")
class MultiStreamMOE(BaseTransform):
    """Multi-stream execution of MoE layers that have shared experts and routed experts."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with open("graph-before-moe.txt", "w") as f:
            f.write(str(gm.graph))
        print("wrote graph to graph-before-moe.txt")

        op_dict = {
            torch.ops.auto_deploy.trtllm_moe_fused: torch.ops.auto_deploy.trtllm_moe_fused_aux,
            torch.ops.auto_deploy.triton_moe_fused: torch.ops.auto_deploy.triton_moe_fused_aux,
            torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused: torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused_aux,
        }

        gm, num_matches = _execute_op_in_aux_stream(gm, op_dict)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )
        with open("graph-after-moe.txt", "w") as f:
            f.write(str(gm.graph))
        print("wrote graph to graph-after-moe.txt")
        return gm, info
