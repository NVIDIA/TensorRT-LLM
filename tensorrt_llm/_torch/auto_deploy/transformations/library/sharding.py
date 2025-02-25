"""Transformations to support graph sharding.

Our sharding algorithm for tensor parallelism (TP) is based on the following steps:

    1. Initialize/construct unsharded model. Ideally, this should be done on device="meta" to avoid
       unnecessary memory allocation. In some cases, this is necessary if the model is too large to
       fit on a single device.
    2. Shard the graph IR of the model:
        a. Identify linear nodes that correspond to TP tuples.
        b. Reduce/Shard shape of weights in the corresponding linear nodes accordingly (either in
           row or column dimension). Add all_reduce nodes where necessary (--> only needed for
           fusing results in final linear node of the TP tuple).
        c. Add a checkpoint loading hook to the sharded linear nodes so that only the correct shard
           of the weight from the checkpoint gets loaded.
    3. Load the checkpoint and allocate the tensor. Loading the correct shard from the checkpoint
       happens automatically via the checkpoint loading hook added in step 2c.
"""

from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import extract_param_names_from_lin_node, is_linear_op, is_op
from ...utils.quantization_utils import QuantizationImpl
from .._graph import canonicalize_graph


def _load_hook(
    state_dict,
    prefix,
    *args,
    f_split: Callable[[torch.Tensor, int], torch.Tensor],
    param_key: str,
    param_shape: torch.Size,
):
    # TODO: we need to support loading either a sharded or unsharded checkpoint.
    # Otherwise, basic workflows like
    # model.load_state_dict(model.state_dict()) will fail.
    # This is quite a hacky solution. A better solution would be to store extra_state in
    # the state_dict to identify whether the state_dict is sharded or not.
    key = prefix + param_key
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    if key not in state_dict:
        return
    p_to_load = state_dict[key]
    p_to_load = p_to_load if param_shape == p_to_load.shape else f_split(p_to_load)
    state_dict[key] = p_to_load


def _load_hook_remove(
    state_dict: Dict,
    prefix: str,
    *args,
    param_key: str,
):
    key = prefix + param_key
    ad_logger.debug(f"Sharder LOAD hook is called for '{key}'")
    state_dict.pop(key, None)


def _insert_sharded_matmul(
    gm: GraphModule, node: Node, dim: int, rank: int, world_size: int, add_dist: bool = False
):
    """Replaces the matmul node with a new matmul node that accepts sharded weights.

    The state_dict is also updated to contain the sharded weights.
    """
    assert dim in [0, 1], "Only dim 0 and 1 are supported for sharding"
    assert add_dist or dim == 0, "For dim=1 sharding, dist_op is required."

    quantization_impl = QuantizationImpl.create(node)

    def split_tensor(
        t: torch.Tensor, d: int = dim, r: int = rank, ws: int = world_size
    ) -> torch.Tensor:
        return torch.tensor_split(t, ws, dim=d)[r]

    # get weight and bias key
    weight_key, bias_key = extract_param_names_from_lin_node(node)

    modname = weight_key.rpartition(".")[0]
    submod = gm.get_submodule(modname)

    def set_new_param(submod: nn.Module, param_key: str, remove: bool = False) -> torch.Size:
        # split or remove it
        param_new = (
            None
            if remove
            else nn.Parameter(
                split_tensor(gm.get_parameter(param_key)).detach().clone(),
                requires_grad=quantization_impl is None,
            )
        )

        # update the parameter
        param_name = param_key.rpartition(".")[-1]
        setattr(submod, param_name, param_new)
        return torch.Size() if param_new is None else param_new.shape

    # update weight
    weight_new_shape = set_new_param(submod, weight_key)
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook, f_split=split_tensor, param_key=weight_key, param_shape=weight_new_shape
        )
    )

    if bias_key is not None and dim == 0:
        # update bias for dim 0 --> we can handle it like the weight
        bias_new_shape = set_new_param(submod, bias_key)
        gm._register_load_state_dict_pre_hook(
            partial(
                _load_hook, f_split=split_tensor, param_key=bias_key, param_shape=bias_new_shape
            )
        )
    elif bias_key is not None and rank != world_size - 1:
        # update the bias for dim 1 --> in this case only the last rank gets the bias to avoid
        # double counting it. For all other we will delete the bias.
        args = list(node.args)
        node_bias = args[2]
        args[2] = None
        node.args = tuple(args)
        gm.graph.erase_node(node_bias)
        set_new_param(submod, bias_key, remove=True)
        gm._register_load_state_dict_pre_hook(partial(_load_hook_remove, param_key=bias_key))

    if quantization_impl:
        scales = {}
        for scale_name in quantization_impl.scale_names():
            scales[scale_name] = submod.get_buffer(scale_name)
        scales["weight_shape"] = weight_new_shape
        sharded_scales = quantization_impl.shard_scales(dim, rank, world_size, **scales)
        for k, v in sharded_scales.items():
            submod.register_buffer(k, v)

        gm._register_load_state_dict_pre_hook(
            partial(
                quantization_impl.shard_load_hook,
                weight_name=weight_key,
                weight_shape=weight_new_shape,
                dim=dim,
                rank=rank,
                world_size=world_size,
            )
        )

    # no comm node needed for single device
    if not add_dist:
        return

    # figure out the right dist op
    dist_lookup = {
        0: (torch.ops.dist.all_gather, -1),
        1: (torch.ops.dist.all_reduce,),
    }
    fn_dist, *dist_args = dist_lookup[dim]

    # add reduction node
    with gm.graph.inserting_after(node):
        dist_node = gm.graph.call_function(fn_dist, args=(node, *dist_args))
        node.replace_all_uses_with(dist_node)
        dist_node.replace_input_with(dist_node, node)


def _identify_sharding_regions(gm: GraphModule) -> List[Node]:
    """Identify regions of the graph that we can investigate further for sharding.

    Right now, we split the regions according to the following structure:
        1. Input node
        2. Embedding node
        3. Residual nodes from the embedding node onwards (no other nodes in-between0)
        4. Output node

    The list will contain the boundary nodes between the regions.
    """
    assert gm.graph.nodes, "Graph is empty"

    # get first input node and last output node
    input_id_node = None
    output_node = None
    for node in gm.graph.nodes:
        if input_id_node is None and node.op == "placeholder":
            input_id_node = node
        output_node = node
    assert input_id_node, "Could not find input node"
    assert output_node.op == "output", "Could not find output node"

    # start list of boundary nodes
    boundary_nodes = [input_id_node]

    # find embedding node which we assume to be the first node in a sequence of residual nodes
    for n_user in input_id_node.users:
        if is_op(n_user, torch.ops.aten.embedding):
            break
    else:
        # we could not identify any boundary regions via embedding nodes
        boundary_nodes.append(output_node)
        return boundary_nodes

    # add embedding node to boundary nodes
    boundary_nodes.append(n_user)

    # find residual nodes from here on
    # NOTE: for now, we assume that the residual nodes do not go through point-wise operations like
    # activations. We are just looking for a "straight" path to the output.
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.aten.add) and any(n == node for n in boundary_nodes[-1].users):
            boundary_nodes.append(node)

    # sanity check: we expect at most two users for any residual node
    res_nodes_more_users = [n for n in boundary_nodes[1:] if len(n.users) > 2]
    assert not res_nodes_more_users, f"Unexpected # of users for residuals: {res_nodes_more_users}"

    # add output node to boundary nodes
    boundary_nodes.append(output_node)

    return boundary_nodes


def _simple_shard(
    gm: GraphModule, nodes_linear: Dict[Node, List[Node]], rank: int, world_size: int
):
    # for every linear node:
    # --> row_split (dim 0 of weight) + all_gather (dim -1 of output)
    for node_group in nodes_linear.values():
        for n in node_group:
            _insert_sharded_matmul(gm, n, 0, rank, world_size, add_dist=True)


# TODO: eventually I think the transformation here needs to be more rigorous. However, for now it's
# good enough. Ideally, we want the following structure:
# 1. Identify boundary nodes as we do right now.
# 2. Identify the GEMM nodes that can be sharded
# 3. Trace through the subgraph using DFS/BFS between each pair of boundary nodes
# 4. Account for each node in the trace to ensure the op is correct even after sharding (e.g. view,
#    reshape, expand, math ops, etc.). This is necessary to ensure that the sharding is correct and
#    we need to be able to account for **all** nodes in the subgraph.
# 5. Shard the GEMM nodes or skip accordingly.
def column_row_shard_matmul_v3(gm: GraphModule, rank: int, world_size: int) -> GraphModule:
    ad_logger.info("Sharding graph")
    ad_logger.debug("Before sharding graph: " + str(gm))

    if world_size == 1:
        ad_logger.info("Skipping sharding for single device")
        return gm

    assert isinstance(gm, GraphModule), "Expecting GraphModule"

    # find boundary nodes of regions we want to shard
    boundary_nodes = _identify_sharding_regions(gm)

    # view ops
    view_ops = {
        torch.ops.aten.view,
        torch.ops.aten._unsafe_view,
        torch.ops.aten.expand,
        torch.ops.aten.reshape,
    }

    # let's look at linear nodes we can identify between pairs of boundary nodes
    # There is three potential cases we can handle:
    # 1. No linear nodes:
    #       --> just continue
    # 2. Two groups of linear nodes and we can account for all to the view nodes:
    #       --> row_split (dim 0) 1st group + adjust all view nodes according to heuristic +
    #           col_split (dim 1) 2nd group + all_reduce output of 2nd group
    # 3. Linear nodes that are not in two groups or we cannot account for all view nodes:
    #       --> row_split (dim 0 of weight) + all_gather (dim -1 of output) output
    for n_start, n_end in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # we iterate through all nodes between the two boundary nodes and store linear nodes
        # sorted by their input activation node. We also store all view nodes.
        nodes_linear = defaultdict(list)
        nodes_view = []
        current_node = n_start
        while current_node != n_end:
            if is_linear_op(current_node, include_quantization=True):
                nodes_linear[current_node.args[0]].append(current_node)
            elif is_op(current_node, view_ops):
                nodes_view.append(current_node)
            current_node = current_node.next
            assert current_node, "Could not identify next node"

        # nothing to shard
        if len(nodes_linear) == 0:
            continue

        # simple shard when we have != 2 groups of linear nodes
        if len(nodes_linear) != 2:  # remove this
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # let's handle view nodes
        # NOTE: we currently make the following assumption for updating the view nodes:
        #       1. There is one symbolic size in the view node corresponding to the sequence length.
        #       2. The sharding dimension is either before or after the sequence dimension and
        #          corresponds to the number of heads or the total hidden size.
        view_args_updated: Dict[Node, Optional[Tuple]] = {}
        for n_view in nodes_view:
            # We first look for the input node corresponding to the symbolic sequence length size
            # (assumed to be dim 1).
            for n_in in n_view.all_input_nodes:
                if is_op(n_in, torch.ops.aten.sym_size) and n_in.args[1] == 1:
                    node_sym = n_in
                    break
            else:
                # check if there is a "-1" in the view node args --> if yes, we assume it's
                # flexible and we don't need to update it
                if -1 in n_view.args[1]:
                    ad_logger.debug(f"Assuming flexible view node: {n_view}")
                    view_args_updated[n_view] = tuple(list(n_view.args))
                else:
                    ad_logger.debug(f"Could not find input node with symbolic size for {n_view}")
                    view_args_updated[n_view] = None
                continue

            # let's see if we can use our heuristic to identify the sharding dimension
            view_dims = n_view.args[1]
            seq_dim = view_dims.index(node_sym)

            # we are assuming the sharding dim is before or after the sequence dimension
            if seq_dim == 1 and len(view_dims) in [3, 4]:
                shard_dim = 2
            elif (seq_dim == 2 and len(view_dims) == 4) or (seq_dim == 3 and len(view_dims) == 5):
                shard_dim = 1
            else:
                ad_logger.debug(f"Could not correctly identify the sequence dimension for {n_view}")
                view_args_updated[n_view] = None
                continue

            # update args
            if view_dims[shard_dim] == -1:
                # no need to update dimension
                view_args_updated[n_view] = tuple(list(n_view.args))
            elif n_view.args[1][shard_dim] % world_size != 0:
                ad_logger.debug(f"Sharding dimension must be divisible by world size, {n_view}")
                view_args_updated[n_view] = None
            else:
                view_dims_new = list(view_dims)
                view_dims_new[shard_dim] = view_dims_new[shard_dim] // world_size
                updated_args = list(n_view.args)
                updated_args[1] = tuple(view_dims_new)
                view_args_updated[n_view] = tuple(updated_args)

        # let's see if we can account for all view nodes; otherwise let's do a simple shard
        if any(v is None for v in view_args_updated.values()):
            _simple_shard(gm, nodes_linear, rank, world_size)
            continue

        # If we can account for all view nodes, we can do a two-way shard
        # --> row_split (dim 0) + col_split (dim 1) + all_reduce
        for i, group in enumerate(nodes_linear.values()):
            for n in group:
                _insert_sharded_matmul(gm, n, i, rank, world_size, add_dist=i > 0)

        # also update all the view nodes accordingly
        for n_view, args_updated in view_args_updated.items():
            n_view.args = args_updated

    # canonicalize and return
    gm = canonicalize_graph(gm)
    ad_logger.debug("After sharding: " + str(gm))
    return gm
