import torch
from torch.fx import Node

_get_range_fn = []


def register_get_range(fn=None, aux_stream_num=1):
    if fn is None:

        def wrapper(fn):
            return register_get_range(fn, aux_stream_num)

        return wrapper

    _get_range_fn.append((fn, aux_stream_num))
    return fn


def get_range(node: Node, father: dict):
    for fn, aux_stream_num in _get_range_fn:
        start_node, end_node = fn(node, father)
        if start_node is not None and end_node is not None:
            return start_node, end_node, aux_stream_num
    return None, None, None


def find_first_gpu_node(node: Node, father: dict):
    if father[node] == node:
        return node
    father[node] = find_first_gpu_node(father[node], father)
    return father[node]


@register_get_range
def get_moe_range(node: Node, father: dict):
    if (node.op == "call_function"
            and node.target == torch.ops.trtllm.moe_allreduce.default):
        start_node = find_first_gpu_node(node.args[0], father)
        path_top_nodes = set(
            find_first_gpu_node(i, father) for i in node.args
            if isinstance(i, Node) and find_first_gpu_node(i, father).op !=
            "placeholder" and find_first_gpu_node(i, father) != start_node)
        return start_node, path_top_nodes

    return None, None


@register_get_range
def get_attention_norm_range(node: Node, father: dict):
    if node.op == "call_function" and node.target == torch.ops.trtllm.attention.default:
        start_node = node
        while start_node.target != torch.ops.trtllm.flashinfer_rmsnorm.default:
            start_node = start_node.prev
        prev = start_node.prev
        if prev.op == "call_function" and prev.target == torch.ops.trtllm.flashinfer_rmsnorm.default and find_first_gpu_node(
                prev.args[0], father) == find_first_gpu_node(
                    start_node.args[0], father):
            path_top_nodes = {start_node, prev}
            start_node = find_first_gpu_node(prev.args[0], father)
            return start_node, path_top_nodes
    return None, None
