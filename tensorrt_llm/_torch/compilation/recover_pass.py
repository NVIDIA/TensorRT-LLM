from operator import add

import torch
from torch.fx import GraphModule

from .utils import get_arg, is_call_function


def recover_pass(gm: GraphModule):
    # Unfuse specific op to make the fusion pass can work properly
    graph = gm.graph
    nodes_to_remove = []
    node2idx = {}
    for idx, node in enumerate(graph.nodes):
        node2idx[node] = idx
    for idx, node in enumerate(graph.nodes):
        if is_call_function(
                node, torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default):
            input = get_arg(node, 0, 'input')
            residual = get_arg(node, 1, 'residual')
            weight = get_arg(node, 2, 'weight')
            eps = get_arg(node, 3, 'eps')
            with graph.inserting_before(node):
                new_add = graph.call_function(add, (input, residual))
                new_norm = graph.call_function(
                    torch.ops.trtllm.flashinfer_rmsnorm.default,
                    (new_add, weight, eps))
                input.replace_all_uses_with(
                    new_norm, lambda user: user != new_add and
                    (user in node2idx and idx < node2idx[user]))
                residual.replace_all_uses_with(
                    new_add, lambda user: user != new_add and
                    (user in node2idx and idx < node2idx[user]))
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.erase_node(node)

    gm.recompile()
    return gm
