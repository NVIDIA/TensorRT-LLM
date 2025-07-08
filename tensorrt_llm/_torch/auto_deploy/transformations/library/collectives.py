import operator

import torch
from torch.fx import GraphModule

from ...distributed.trtllm import is_trtllm_op_available
from ...utils.logger import ad_logger
from ...utils.node_utils import get_op_overload_packet, get_user_if_pattern_match, is_op
from .._graph import canonicalize_graph


# TODO: This is an overly simplified model that works well for vanilla Llama models.
# However, we eventually want to consider more sophisticated patterns such as
# * all_reduce(lin1(x) + lin2(x))
# * version above with fused GEMMs (i.e. with a split node)
# * all_reduce(pointwise_op(linear(x)))
# * ...
def fuse_collectives(gm: GraphModule) -> None:
    num_gemm_collective_fusions = 0
    ad_logger.debug("Before GEMM+Collective fusion: " + str(gm))

    # lookup for fused ops
    # TODO: avoid this hardcoded lookup, e.g., by generating fused ops on the fly.
    lookup = {
        torch.ops.auto_deploy.torch_linear_simple: torch.ops.auto_deploy.trtllm_dist_fused_linear_all_reduce,
        torch.ops.aten.linear: torch.ops.auto_deploy.trtllm_dist_fused_linear_all_reduce,
        torch.ops.auto_deploy.torch_quant_fp8_linear: torch.ops.auto_deploy.torch_quant_fused_fp8_linear_all_reduce,
    }

    # go through all nodes and find all_reduce nodes
    for node in gm.graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_dist_all_reduce):
            continue

        # check if args are as expected
        assert len(node.args) == 1 and not len(node.kwargs), "Unexpected args/kwargs for all_reduce"

        # retrieve parent and check a few conditions on the parent node
        parent_node = node.args[0]
        if not is_op(parent_node, lookup.keys()):
            continue
        if len(parent_node.users) > 1:
            continue

        with gm.graph.inserting_before(node):
            # insert fused node
            fused_linear_collective_node = gm.graph.call_function(
                lookup[get_op_overload_packet(parent_node.target)],
                args=parent_node.args,
                kwargs=parent_node.kwargs,
            )
        node.replace_all_uses_with(fused_linear_collective_node)
        gm.graph.erase_node(node)
        gm.graph.erase_node(parent_node)
        num_gemm_collective_fusions += 1

    canonicalize_graph(gm)
    ad_logger.info(f"Found {num_gemm_collective_fusions} GEMM+Collective fusions")
    ad_logger.debug("After GEMM+Collective fusion: " + str(gm))


def fuse_allreduce_residual_rmsnorm(gm: GraphModule) -> None:
    """Essentially, this function fuses the following operators into one allreduce trtllm implementation.

    * target pattern:
        x = all_reduce(x)
        y = x + residual
        return rmsnorm(y), y
    * replacement:
        fused_allreduce_residual_rmsnorm(x, residual, rmsnorm_weight, rmsnorm_eps)

    """
    if not is_trtllm_op_available():
        return

    num_ar_r_rms_fusions = 0
    ad_logger.debug("Before allreduce+residual+rmsnorm fusion: " + str(gm))

    def trace_and_fuse(allreduce_node, graph):
        # Check if all_reduce is followed by addition
        users = list(allreduce_node.users.keys())
        if len(users) != 1:
            return  # Skip if all_reduce has more than one consumer
        add_node = users[0]

        # Traverse nodes for RMSNorm pattern which is composed of to_copy, pow, mean, add, refer
        # the Huggingface LlamaRMSNorm implementation as example for more details
        to_copy_1 = get_user_if_pattern_match(add_node, [torch.ops.aten.add, operator.add], 2)
        # operand of pow and mul
        pow_node = get_user_if_pattern_match(
            to_copy_1, [torch.ops.aten._to_copy, torch.ops.aten.to], 2
        )
        mean_node = get_user_if_pattern_match(pow_node, torch.ops.aten.pow, 1)
        add_eps_node = get_user_if_pattern_match(mean_node, torch.ops.aten.mean, 1)
        rsqrt_node = get_user_if_pattern_match(add_eps_node, [torch.ops.aten.add, operator.add], 1)
        mul_node_1 = get_user_if_pattern_match(rsqrt_node, torch.ops.aten.rsqrt, 1)
        to_copy_2 = get_user_if_pattern_match(mul_node_1, torch.ops.aten.mul, 1)
        mul_node_2 = get_user_if_pattern_match(
            to_copy_2, [torch.ops.aten._to_copy, torch.ops.aten.to], 1
        )
        # check args of ops: pow(2) and mean(-1)
        ARGS_MATCH = pow_node is not None and pow_node.args[1] == 2  # exponent
        ARGS_MATCH &= mean_node is not None and mean_node.args[1] == [-1]  # dimensions

        # Match found: Replace with fused operation
        if (
            to_copy_1
            and pow_node
            and mean_node
            and add_eps_node
            and rsqrt_node
            and mul_node_1
            and to_copy_2
            and mul_node_2
            and ARGS_MATCH
        ):
            # Gather the inputs for the custom operation
            tensor = allreduce_node.args[0]
            # Identify the residual argument in the add operation
            # One of the args in add_node.args is the output of all_reduce
            # The same idea also applies to norm_weight
            residual = add_node.args[0] if add_node.args[1] is allreduce_node else add_node.args[1]
            norm_weight = (
                mul_node_2.args[0] if mul_node_2.args[1] is to_copy_2 else mul_node_2.args[1]
            )
            eps = add_eps_node.args[1]

            # Insert nodes
            with graph.inserting_before(allreduce_node):
                fused_node = graph.call_function(
                    torch.ops.dist.fused_allreduce_residual_rmsnorm,
                    args=(
                        tensor,
                        residual,
                        norm_weight,
                        eps,
                    ),
                )
                # Extract outputs from the tuple returned by `fused_node`
                final_output_node = gm.graph.create_node(
                    "call_function",
                    target=operator.getitem,
                    args=(fused_node, 0),
                )
                add_output_node = gm.graph.create_node(
                    "call_function",
                    target=operator.getitem,
                    args=(fused_node, 1),
                )

                # Replace all uses of rmsnorm_node with final_output_node
                mul_node_2.replace_all_uses_with(final_output_node)

                # Replace all uses of add_node with add_output_node
                add_node.replace_all_uses_with(add_output_node)

            nonlocal num_ar_r_rms_fusions
            num_ar_r_rms_fusions += 1

    # Traverse all nodes
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.auto_deploy.torch_dist_all_reduce):
            trace_and_fuse(allreduce_node=node, graph=gm.graph)

    canonicalize_graph(gm)
    ad_logger.info(f"Found {num_ar_r_rms_fusions} allreduce+residual+rmsnorm fusions")
    ad_logger.debug("After allreduce+residual+rmsnorm fusion: " + str(gm))
