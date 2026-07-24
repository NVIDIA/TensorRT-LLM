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

    def remove_functionalize_inner(node: Node, mutates_args: dict):
        getitem_nodes = [
            user for user in node.users if is_call_function(user, getitem)
        ]

        kwargs = {k: v for k, v in node.kwargs.items() if not k.startswith("_")}
        if is_call_function(node, auto_functionalized_v2):
            all_bases = node.kwargs["_all_bases"]
            for arg_name in mutates_args.values():
                base_index = node.kwargs.get(f"_{arg_name}_base_index")
                kwargs[arg_name] = (None if base_index is None else
                                    all_bases[base_index])

        for getitem_node in getitem_nodes:
            idx = getitem_node.args[1]
            mutated_arg = mutates_args[idx]
            replacement = kwargs[mutated_arg]
            assert replacement is not None, (
                f"getitem user for optional output '{mutated_arg}' "
                "has no base tensor -- graph is malformed")
            getitem_node.replace_all_uses_with(replacement)
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

        remove_functionalize_inner(
            node,
            inplace_map[inplace_func],
            is_v2=node.target == auto_functionalized_v2,
        )

    for node in nodes_to_remove:
        graph.erase_node(node)
