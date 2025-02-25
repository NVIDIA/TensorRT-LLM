"""Common utils for torch fx graph transformation."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import Graph, Node

from ..custom_ops.quant import QUANT_OPS

try:
    # import modelopt to get quantize_op
    from modelopt.torch.quantization import tensor_quant  # noqa: F401

    if hasattr(torch.ops, "tensorrt"):
        modelopt_quantize_op = torch.ops.tensorrt.quantize_op
        modelopt_dynamic_block_quantize_op = torch.ops.tensorrt.dynamic_block_quantize_op
    else:
        modelopt_quantize_op = None
        modelopt_dynamic_block_quantize_op = None
except ImportError:
    modelopt_quantize_op = None
    modelopt_dynamic_block_quantize_op = None


@dataclass
class modelopt_quant_params:
    input_node: torch.fx.node.Node = None
    amax: torch.fx.node.Node = None
    num_bits: int = 0
    exp_bits: int = 0
    is_unsigned: bool = False
    narrow_range: bool = False
    is_dynamic_block_quant: bool = False
    block_size: int = 0
    scale_num_bits: int = 0
    scale_exponent_bits: int = 0

    def is_fp8_e4m3(self):
        return self.num_bits == 8 and self.exp_bits == 4

    def is_fp4_e2m1(self):
        return self.num_bits == 4 and self.exp_bits == 2

    def get_quant_format_str(self):
        return (
            None
            if self.num_bits == 0 and self.exp_bits == 0
            else f"num_bits: {self.num_bits}, exp_bits: {self.exp_bits}"
        )

    @staticmethod
    def get_quant_params_from_quantize_node(input_node):
        params = None
        if is_op(input_node, modelopt_quantize_op):
            params = modelopt_quant_params(
                *input_node.args,
                is_dynamic_block_quant=False,
                block_size=0,
                scale_num_bits=0,
                scale_exponent_bits=0,
            )
        elif is_op(input_node, modelopt_dynamic_block_quantize_op):
            params = modelopt_quant_params(
                input_node=input_node.args[0],
                block_size=input_node.args[1],
                amax=input_node.args[2],
                num_bits=input_node.args[3],
                exp_bits=input_node.args[4],
                is_dynamic_block_quant=True,
                scale_num_bits=input_node.args[5],
                scale_exponent_bits=input_node.args[6],
            )

        def get_amax_from_detach(detach_node):
            if (
                not is_op(detach_node, torch.ops.aten.detach)
                or len(detach_node.all_input_nodes) != 1
            ):
                return detach_node
            return detach_node.all_input_nodes[0]

        if params:
            params.amax = get_amax_from_detach(params.amax)
        return params


def get_quantization_params_from_linear_node(linear_op: torch.fx.node.Node):
    """Return quantization parameters of the linear node."""
    input_params, weight_params, output_params = None, None, None

    if modelopt_quantize_op is not None and is_linear_op(linear_op):
        input_node, weight_node = linear_op.all_input_nodes[:2]
        # check if activation, weight, and output are quantized
        input_params = modelopt_quant_params.get_quant_params_from_quantize_node(input_node)
        weight_params = modelopt_quant_params.get_quant_params_from_quantize_node(weight_node)
        output_params = modelopt_quant_params.get_quant_params_from_quantize_node(
            list(linear_op.users.keys())[0]
        )

    return input_params, weight_params, output_params


def is_match(node: Node, names_to_skip: List[str]):
    if names_to_skip is None:
        return False
    for n in names_to_skip:
        module_stack = node.meta.get("nn_module_stack", None)
        if module_stack is None:
            return False
        module_stack = list(module_stack.keys())
        if n in module_stack[-1]:
            return True
    return False


def extract_param_names_from_lin_node(mm_node: Node) -> Tuple[str, Optional[str]]:
    """Extracts the name of the parameter associated with the given matmul node.

    Args:
        mm_node: Matmul node in the graph.
    """
    assert is_linear_op(mm_node, include_quantization=True), (
        f"Expecting linear node, Found: {mm_node}"
    )
    # second arg is the weight
    weight_node = mm_node.args[1]
    # for modelopt quantized graph, there will be a quantize_op
    _, weight_params, _ = get_quantization_params_from_linear_node(mm_node)
    weight_node = weight_params.input_node if weight_params else weight_node

    assert weight_node.op == "get_attr"
    # Map arg to named parameter
    weight_name = weight_node.target

    # check for bias
    bias_node = mm_node.args[2] if len(mm_node.args) > 2 else None
    assert bias_node is None or bias_node.op == "get_attr"
    bias_name = bias_node.target if bias_node is not None else None

    return weight_name, bias_name


def get_op_overload_packet(node: Union[OpOverloadPacket, OpOverload]) -> OpOverloadPacket:
    """Get the overload packet from the op overload."""
    if isinstance(node, OpOverloadPacket):
        return node
    elif isinstance(node, OpOverload):
        return node.overloadpacket
    else:
        raise ValueError(f"Expected OpOverloadPacket or OpOverload, got {type(node)}")


def is_op(node: Node, ops: Union[OpOverloadPacket, Iterable[OpOverloadPacket]]) -> bool:
    """Check if the node is a call to one of the ops."""
    if node.op != "call_function":
        return False

    # check if it's a single op that's provided
    if isinstance(ops, OpOverloadPacket):
        ops = [ops]

    # check if it's the op itself instead of an overload
    if any(node.target == op for op in ops):
        return True

    # check the overloads
    return any(node.target == getattr(op, overload) for op in ops for overload in op)


def is_linear_op(node: Node, include_quantization: bool = False) -> bool:
    """Check if the node is a linear op.

    Using this function is preferred over `is_op` for linear ops to ensure all variants are covered.
    """
    lin_ops = {
        torch.ops.aten.linear,
        torch.ops.linear.simple,
    }

    if include_quantization:
        lin_ops.update(QUANT_OPS)
    return is_op(node, lin_ops)


def is_dist_op(node: Node) -> bool:
    """Check if the node is a distributed op."""
    dist_ops = {
        torch.ops.dist.all_gather,
        torch.ops.dist.all_reduce,
    }
    return is_op(node, dist_ops)


def get_all_input_output_nodes(graph: Graph) -> Tuple[List[Node], List[Node]]:
    input_nodes: List[Node] = graph.find_nodes(op="placeholder")
    output_nodes: List[Node] = graph.find_nodes(op="output")
    return (input_nodes, output_nodes)


def get_user_if_pattern_match(node, ops, numusers, user_idx: int = 0):
    """Get a user from a node if the node matches a given op set and num of users."""
    if node is None:
        return None
    assert len(node.users) > user_idx
    return (
        list(node.users.keys())[user_idx]
        if node and len(list(node.users.keys())) == numusers and is_op(node, ops)
        else None
    )
