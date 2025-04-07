"""Transformations to unfuse weights and standardize graph representation.

The algorithm works as follows:

1. Pattern match the fused weight:
    For each linear node, check if all users are chunk or slice nodes
    If so, grab the subgraph and replace old param with new param and delete subgraph in the graph
2. Update load_hook so that during weight loading, old weight param is mapped to new weight param
    with the subgraph in 1

TODO: Add support for quantized graph
"""

from collections import defaultdict
from functools import partial
from typing import DefaultDict, Dict, List, Tuple

import torch
from torch import nn
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import (
    add_new_attribute_to_submodule,
    extract_param_names_from_lin_node,
    is_chunk_or_slice_op,
    is_linear_op,
    is_op,
)
from .._graph import canonicalize_graph


def _create_and_register_weight(
    gm: GraphModule,
    user: Node,
    start_idx: int,
    end_idx: int,
    input_tensor: Node,
    weight_key: str,
    weight_split_info: DefaultDict[str, List[Tuple[str, int, int]]],
):
    orig_weight = gm.get_parameter(weight_key)

    # Register each unfused_weight in a uniquely-named submodule so unused ones
    # can be cleaned up by `gm.delete_all_unused_submodules()`. Direct attributes
    # are not deleted automatically.
    weight_module_path, sub_weight_key = weight_key.rsplit(".", 1)
    new_module_name = f"{weight_module_path}_unfused_{user.name}"
    new_param_name = f"{sub_weight_key}"

    split_weight = orig_weight[start_idx:end_idx]  # spilt weights on dim=0
    new_weight = nn.Parameter(split_weight.detach().clone())
    full_new_param_name = add_new_attribute_to_submodule(
        gm, new_module_name, new_param_name, new_weight
    )

    weight_split_info[weight_key].append((full_new_param_name, start_idx, end_idx))

    graph = gm.graph
    with graph.inserting_before(user):
        weight_node = graph.create_node("get_attr", full_new_param_name)
        new_linear = graph.call_function(
            torch.ops.linear.simple.default, args=(input_tensor, weight_node, None)
        )
        user.replace_all_uses_with(new_linear)


def unfuse_weights(gm: GraphModule) -> GraphModule:
    """Unfuse weights that were previously accessed through chunk/slice operations.

    Args:
        gm: The GraphModule to transform

    Returns:
        The transformed GraphModule with unfused weights
    """
    ad_logger.info("Unfusing GEMM")
    ad_logger.debug("Before Unfusing GEMM: " + str(gm))
    graph = gm.graph

    # track weight split info for load hook
    weight_split_info: DefaultDict[str, List[Tuple[str, int, int]]] = defaultdict(list)

    for node in graph.nodes:
        # Check if node is a linear op and all users are chunk or slice nodes
        if not is_linear_op(node):
            continue
        if not all(is_chunk_or_slice_op(user) for user in node.users):
            continue

        weight_key, _ = extract_param_names_from_lin_node(node)
        if not weight_key:
            continue

        input_tensor = node.args[0]

        # Create new unfused weight parameters for each chunk/slice user
        for user in list(node.users):
            if is_op(user, torch.ops.aten.chunk):
                num_chunks = user.args[1]
                dim = user.args[2] if len(user.args) > 2 else 0
                chunk_size = user.args[0].meta["val"].size(dim) // num_chunks

                for getitem_user in list(user.users):
                    chunk_idx = getitem_user.args[1]
                    start_idx = chunk_idx * chunk_size
                    end_idx = start_idx + chunk_size

                    _create_and_register_weight(
                        gm,
                        getitem_user,
                        start_idx,
                        end_idx,
                        input_tensor,
                        weight_key,
                        weight_split_info,
                    )
            else:
                dim = user.args[1]
                start_idx = user.args[2]
                end_idx = min(user.args[3], user.args[0].meta["val"].size(dim))

                _create_and_register_weight(
                    gm, user, start_idx, end_idx, input_tensor, weight_key, weight_split_info
                )
        # Clean up deleted modules to save GPU memory
        gm.graph.eliminate_dead_code()
        gm.delete_all_unused_submodules()

    def _load_hook(state_dict: Dict, prefix: str, *args, param_key: str):
        key = prefix + param_key
        if key not in state_dict:
            return

        orig_weight = state_dict[key]

        for full_new_param_name, start_idx, end_idx in weight_split_info[param_key]:
            new_key = f"{prefix}{full_new_param_name}"
            sliced_weight = orig_weight[start_idx:end_idx]  # spilt weights on dim=0
            state_dict[new_key] = sliced_weight
        state_dict.pop(key, None)  # remove old key to avoid unexpected key mismatch

    # Register the load hook for each unfused weight
    for weight_key in weight_split_info:
        gm._register_load_state_dict_pre_hook(partial(_load_hook, param_key=weight_key))

    gm = canonicalize_graph(gm)
    ad_logger.debug("After Unfusing GEMM: " + str(gm))
    return gm
