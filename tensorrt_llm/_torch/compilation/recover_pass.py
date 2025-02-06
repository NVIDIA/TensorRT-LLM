from operator import add

import torch
from torch.fx import GraphModule


def recover_pass(gm: GraphModule):
    # Unfuse specific op to make the fusion pass can work properly
    graph = gm.graph
    nodes_to_remove = []
    node2idx = {}
    for idx, node in enumerate(graph.nodes):
        node2idx[node] = idx
    for idx, node in enumerate(graph.nodes):
        if (node.op == "call_function" and node.target
                == torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default):
            input = node.args[0]
            residual = node.args[1]
            weight = node.args[2]
            eps = node.args[3]
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
