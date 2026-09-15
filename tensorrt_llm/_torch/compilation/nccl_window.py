# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from operator import getitem

import torch
from torch.fx import GraphModule, Node

from ..modules.decoder_layer import DecoderLayer

_SCOPE_OPS = {
    torch.ops.trtllm.begin_nccl_window_tensor_scope.default,
    torch.ops.trtllm.end_nccl_window_tensor_scope.default,
}


def _decoder_layer(node: Node) -> Hashable | None:
    module_stack = node.meta.get("nn_module_stack")
    if node.op in ("placeholder", "output"):
        return None
    if not isinstance(module_stack, dict):
        return None

    for path, module in reversed(module_stack.items()):
        if not isinstance(module, tuple) or len(module) != 2:
            continue
        module_type = module[1]
        if isinstance(module_type, type) and issubclass(module_type, DecoderLayer):
            return path
    return None


def _is_tensor(node: Node) -> bool:
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    return isinstance(value, torch.Tensor)


def _insert_scope(graph, nodes: list[Node]) -> None:
    node_set = set(nodes)
    inputs = []
    outputs = []

    for node in nodes:
        for input_node in node.all_input_nodes:
            if input_node not in node_set and _is_tensor(input_node) and input_node not in inputs:
                inputs.append(input_node)
        if _is_tensor(node) and any(user not in node_set for user in node.users):
            outputs.append(node)

    # A layer can return one of its inputs unchanged. Preserve any input that
    # remains live after the layer just as the Python scope would.
    for input_node in inputs:
        if any(user not in node_set for user in input_node.users):
            outputs.append(input_node)

    if not inputs:
        return

    with graph.inserting_before(nodes[0]):
        begin = graph.call_function(
            torch.ops.trtllm.begin_nccl_window_tensor_scope.default,
            kwargs={"inputs": inputs},
        )
    with graph.inserting_after(nodes[-1]):
        end = graph.call_function(
            torch.ops.trtllm.end_nccl_window_tensor_scope.default,
            kwargs={"inputs": inputs, "outputs": outputs, "failed": False},
        )
    begin.meta["val"] = None
    end.meta["val"] = None


def insert_nccl_window_tensor_scopes(gm: GraphModule) -> GraphModule:
    """Insert runtime lease boundaries around Dynamo-inlined decoder layers."""
    regions: list[list[Node]] = []
    current_layer = None
    current_nodes: list[Node] = []

    for node in gm.graph.nodes:
        layer = _decoder_layer(node)
        if layer == current_layer:
            current_nodes.append(node)
            continue
        if current_layer is not None:
            regions.append(current_nodes)
        current_layer = layer
        current_nodes = [node] if layer is not None else []
    if current_layer is not None:
        regions.append(current_nodes)

    for nodes in regions:
        _insert_scope(gm.graph, nodes)
    if regions:
        gm.graph.lint()
        gm.recompile()
    return gm


def lower_nccl_window_tensor_scope_effects(gm: GraphModule) -> GraphModule:
    """Lower AOT effect wrappers after they have enforced scope ordering."""
    graph = gm.graph
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not torch.ops.higher_order.with_effects
            or len(node.args) < 2
            or node.args[1] not in _SCOPE_OPS
        ):
            continue

        token, op, *args = node.args
        if op is torch.ops.trtllm.begin_nccl_window_tensor_scope.default:
            kwargs = {"inputs": args[0]}
        else:
            kwargs = {"inputs": args[0], "outputs": args[1], "failed": args[2]}

        with graph.inserting_before(node):
            direct = graph.call_function(op, kwargs=kwargs)
        direct.meta = {key: value for key, value in node.meta.items() if key != "val"}
        direct.meta["val"] = None

        for user in list(node.users):
            if user.op != "call_function" or user.target is not getitem or user.args[1] != 0:
                raise RuntimeError("Unexpected NCCL window effect output")
            user.replace_all_uses_with(token)
            graph.erase_node(user)
        graph.erase_node(node)

    graph.lint()
    return gm
