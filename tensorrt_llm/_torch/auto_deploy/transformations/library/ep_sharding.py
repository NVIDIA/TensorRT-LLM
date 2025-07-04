"""
Expert Parallel Sharding for Mixture-of-Experts (MoE) Graphs.

This module implements graph transformations to enable expert sharding
for Mixture-of-Experts (MoE) models in a multi-GPU setting. The sharding
algorithm partitions the expert weights, as well as updates the routing
components (`selected_experts` and `final_scales`), so that each GPU only
processes a subset of experts.

The sharding process consists of:

1. Identify MoE nodes in the FX graph
2. Compute local sharding parameters (`selected_experts` and `final_scales`) to update the routing tensors.
3. Partition expert weight lists according to the current rank and world size,
    and replace the MoE node’s arguments with these sharded versions.
4. Append an all_reduce node after each MoE node to aggregate outputs across devices,
    then canonicalize the modified graph.

"""

import operator

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from .._graph import canonicalize_graph


def ep_shard(gm: GraphModule, rank: int, world_size: int) -> GraphModule:
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size < 2:
        ad_logger.info("Skipping sharding for single device")
        return gm

    assert isinstance(gm, GraphModule), "Expecting GraphModule"
    num_moe_patterns = 0
    for node in list(gm.graph.nodes):
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            continue
        _insert_sharded_moe(gm, node, rank, world_size)
        num_moe_patterns += 1
    # canonicalize and return
    gm = canonicalize_graph(gm)

    ad_logger.debug("After sharding: " + str(gm))
    ad_logger.info(f"Found {num_moe_patterns} MoE patterns")
    return gm


def _insert_sharded_moe(
    gm: GraphModule,
    node: Node,
    rank: int,
    world_size: int,
):
    """Update the torch_moe node with sharded weight lists,
    sharded `selected_experts` and `final_scales(router_logics)`.
    Add an all_reduce node after the moe node.
    """
    num_experts = len(node.args[3])
    args = list(node.args)

    # -- Handle selected_experts and final_scales sharding --
    selected_experts = args[1]
    final_scales = args[2]

    experts_per_rank = num_experts // world_size

    with gm.graph.inserting_before(node):
        lower = experts_per_rank * rank
        # selected_experts_local = selected_experts - low
        selected_experts_local = gm.graph.create_node(
            "call_function", operator.sub, args=(selected_experts, lower), kwargs={}
        )

        # For num_experts % world_size != 0 case,
        # assign the last (num_experts % world_size) experts to the last rank
        # if rank == world_size -1:
        #     rank_mask = (selected_experts // experts_per_rank) >= rank
        # else:
        #     rank_mask = (selected_experts // experts_per_rank) == rank
        div_node = gm.graph.create_node(
            "call_function", operator.floordiv, args=(selected_experts, experts_per_rank), kwargs={}
        )
        comp_op = torch.ge if rank == world_size - 1 else torch.eq
        rank_mask = gm.graph.create_node("call_function", comp_op, args=(div_node, rank), kwargs={})

        # final_scales_local = final_scales * rank_mask
        final_scales_local = gm.graph.create_node(
            "call_function", operator.mul, args=(final_scales, rank_mask), kwargs={}
        )

    # -- Shard expert weights --
    def get_partition(lst, world_size, rank):
        num_experts = len(lst)
        expert_size_per_partition = num_experts // world_size
        expert_start = rank * expert_size_per_partition
        # For num_experts % world_size != 0 case,
        # assign the last (num_experts % world_size) experts to the last rank
        expert_end = (
            num_experts if (rank == world_size - 1) else expert_start + expert_size_per_partition
        )
        return lst[expert_start:expert_end]

    w1_list_sharded = get_partition(args[3], world_size, rank)
    w2_list_sharded = get_partition(args[4], world_size, rank)
    w3_list_sharded = get_partition(args[5], world_size, rank)

    # -- Update args --
    args[1] = selected_experts_local
    args[2] = final_scales_local
    args[3] = w1_list_sharded
    args[4] = w2_list_sharded
    args[5] = w3_list_sharded

    ad_logger.debug(
        f"Updated node {node}: replaced original arguments {node.args} with sharded arguments {args}."
    )
    node.args = tuple(args)

    # -- add an all_reduce node --
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_reduce, args=(node,)
        )
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)
