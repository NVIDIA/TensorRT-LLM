from operator import getitem

import torch
from torch._higher_order_ops.auto_functionalize import (auto_functionalized,
                                                        auto_functionalized_v2)
from torch.fx import Graph, Node

from .utils import inplace_info, is_call_function

aten = torch.ops.aten


def remove_copy_for_mutates_args(graph: Graph):
    '''
    This pass is to remove the copy nodes for the mutates args introduced by the auto_functionalize.
    '''

    nodes_to_remove: list[Node] = []

    def remove_functionalize_inner(node: Node, mutates_args: dict, is_v2=False):
        getitem_nodes = [
            user for user in node.users if is_call_function(user, getitem)
        ]

        kwargs = {k: v for k, v in node.kwargs.items() if not k.startswith("_")}
        if is_v2:
            for k, v in mutates_args.items():
                kwargs[v] = node.kwargs["_all_bases"][k - 1]

        for getitem_node in getitem_nodes:
            idx = getitem_node.args[1]
            getitem_node.replace_all_uses_with(kwargs[mutates_args[idx]])
            nodes_to_remove.append(getitem_node)

        with graph.inserting_before(node):
            graph.call_function(inplace_func, kwargs=kwargs)

        nodes_to_remove.append(node)

    for node in graph.nodes:
        if not is_call_function(node, [
                auto_functionalized,
                auto_functionalized_v2,
        ]):
            continue

        inplace_func = node.args[0]

        inplace_map = inplace_info()
        if inplace_func not in inplace_map:
            # We do not know the inplace op
            continue

        remove_functionalize_inner(node, inplace_map[inplace_func])

    for node in nodes_to_remove:
        graph.erase_node(node)
