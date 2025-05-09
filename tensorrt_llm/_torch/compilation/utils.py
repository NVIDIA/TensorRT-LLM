from typing import Callable, List, Union

import torch
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def get_symint_val(i: Union[torch.SymInt | int]):
    if isinstance(i, int):
        return i
    elif isinstance(i, torch.SymInt):
        node = i.node
        expr = node.expr
        shape_env: ShapeEnv = node.shape_env
        var_val = shape_env.var_to_val.get(expr, None) or expr.xreplace(
            shape_env.var_to_val)
        return var_val
    else:
        raise Exception("Only support int or torch.SymInt")


def get_arg(node, idx, arg_name):
    return node.args[idx] if len(node.args) > idx else node.kwargs[arg_name]


def is_call_function(node: Node, target: Union[List[Callable], Callable]):
    if isinstance(target, list):
        return node.op == "call_function" and node.target in target
    else:
        return node.op == "call_function" and node.target == target


_enable_piecewise_cuda_graph_capture = True


def set_enable_piecewise_cuda_graph_capture_flag(enable: bool):
    global _enable_piecewise_cuda_graph_capture
    _enable_piecewise_cuda_graph_capture = enable


def get_enable_piecewise_cuda_graph_capture_flag() -> bool:
    global _enable_piecewise_cuda_graph_capture
    return _enable_piecewise_cuda_graph_capture
