"""Common utils for torch fx graph transformation."""

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import GraphModule, Node

from .logger import ad_logger

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

OpOrOverload = Union[OpOverloadPacket, OpOverload]
OperatorLike = Union[OpOrOverload, Callable]


class LayerType(Enum):
    """Enum for layer type."""

    ATTENTION = "attention"
    SSM = "ssm"
    MLP = "mlp"
    MOE = "moe"
    MLA = "mla"
    UNKNOWN = "unknown"


class LayerSubgraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    opening_nodes: List[Node]
    subgraph_nodes: List[Node]
    terminating_node: Union[Node, None]
    layer_type: LayerType
    min_local_shape: int = 1


class WeightNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node: Node
    tensor: torch.Tensor
    node_key: str
    submod: nn.Module


class WeightNodes(BaseModel):
    weights: list[WeightNode]
    biases: list[WeightNode]


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


def get_all_weights_in_subgraph(
    sources: list[Node],
    sinks: list[Node],
):
    """Get all weight nodes (get_attr nodes) in the subgraph between sources and sinks."""
    weight_nodes = subgraph(sources, sinks, include=is_weight_node)
    return weight_nodes


def extract_weight_name(node: Node) -> Union[str, bool]:
    """
    Extract the weight parameter name for a compute node.

    Args:
        node: Compute node (linear, MoE, SSM, etc.)

    Returns:
        Weight parameter name (str), or False if no weight exists.
    """
    weight_node = get_weight_node(node)
    if weight_node is None:
        return False
    return weight_node.target


def get_param_or_buffer(tensor_name: str, gm: GraphModule) -> torch.Tensor:
    if tensor_name in dict(gm.named_parameters()):
        return gm.get_parameter(tensor_name)
    elif tensor_name in dict(gm.named_buffers()):
        return gm.get_buffer(tensor_name)
    else:
        raise KeyError(f"Tensor {tensor_name} not found in the graph")


def extract_weight_nodes(node: Node) -> WeightNodes:
    """Extracts the list of weight node and optional bias node from the given parametrized node"""
    gm = node.graph.owning_module
    param_names = {name for name, _ in gm.named_parameters()}.union(
        {name for name, _ in gm.named_buffers()}
    )

    def find_get_attr_node(weight_node: Node) -> Node:
        """Recursively traverse inputs of allowed nodes to find a node with 'get_attr' op."""
        # If node is a get_attr node return node
        # List of nodes allowed in between a get_attr node and the matmul node
        allowed_ops = {
            torch.ops.aten.to.dtype,
            torch.ops.aten.view.default,
        }

        if (
            weight_node.op == "get_attr"
            and weight_node.target in param_names
            and has_shape(weight_node)
            and len(shape(weight_node)) > 0
        ):
            return weight_node

        # If node is not in the list of allowable ops then return None
        if weight_node.target not in allowed_ops:
            return None

        for input_node in weight_node.all_input_nodes:
            result = find_get_attr_node(input_node)
            if result:
                return result
        return None

    if is_op(node, torch.ops.aten.bmm):
        # no bias for bmm
        weight_node = find_get_attr_node(node.args[1])
        return WeightNodes(
            weights=[
                WeightNode(
                    node=node.args[1],
                    node_key=weight_node.target,
                    tensor=get_param_or_buffer(weight_node.target, gm),
                    submod=gm.get_submodule(weight_node.target.rpartition(".")[0]),
                )
            ],
            biases=[],
        )
    # for other parametrized nodes, we need to find the weight node
    else:
        all_weight_nodes = [
            attr_node
            for n in node.all_input_nodes
            if (attr_node := find_get_attr_node(n)) is not None
        ]
        # separate weight nodes and bias nodes
        bias_nodes = [n for n in all_weight_nodes if n.target.endswith("bias")]
        weight_nodes = [n for n in all_weight_nodes if n not in bias_nodes]
        weight_nodes = [
            WeightNode(
                node=n,
                node_key=n.target,
                submod=gm.get_submodule(n.target.rpartition(".")[0]),
                tensor=get_param_or_buffer(n.target, gm),
            )
            for n in weight_nodes
        ]
        bias_nodes = [
            WeightNode(
                node=n,
                node_key=n.target,
                submod=gm.get_submodule(n.target.rpartition(".")[0]),
                tensor=get_param_or_buffer(n.target, gm),
            )
            for n in bias_nodes
        ]
    return WeightNodes(weights=weight_nodes, biases=bias_nodes)


def num_users_of_weight_node(node: Node) -> int:
    """
    Get the number of users of the weight node.

    Args:
        node: Compute node (linear, MoE, SSM, etc.)

    Returns:
        Number of users of the primary weight node, or 0 if no weight exists.
    """
    weight_node = get_weight_node(node)
    return len(weight_node.users) if weight_node is not None else 0


def get_op_overload_packet(node: Union[OpOverloadPacket, OpOverload]) -> OpOverloadPacket:
    """Get the overload packet from the op overload."""
    if isinstance(node, OpOverloadPacket):
        return node
    elif isinstance(node, OpOverload):
        return node.overloadpacket
    else:
        raise ValueError(f"Expected OpOverloadPacket or OpOverload, got {type(node)}")


def is_op(node: Node, ops: Union[OperatorLike, Iterable[OperatorLike]]) -> bool:
    """Check if the node is a call to one of the ops."""
    if not isinstance(node, Node):
        return False

    if node.op != "call_function":
        return False

    # check if it's a single op that's provided by checking if it's iterable
    if isinstance(ops, OpOverloadPacket) or not isinstance(ops, Iterable):
        ops = [ops]

    # now iterate through the operator list and see if there is a match
    is_match = True
    for op in ops:
        if node.target == op:
            break
        if isinstance(op, OpOverloadPacket):
            if any(node.target == getattr(op, overload) for overload in op):
                break
    else:
        is_match = False

    return is_match


def filtered_nodes(
    nodes: Iterable[Node],
    target: Union[Callable[[Node], bool], Union[OperatorLike, Iterable[OperatorLike]]] = None,
    ops: Union[OperatorLike, Iterable[OperatorLike]] = None,
    prune_dangling: bool = False,
) -> Iterable[Node]:
    """Iterate over nodes that are filtered by the given operations or target function.

    This utility function simplifies the common pattern of iterating through nodes
    and filtering by operation type or custom function.

    Args:
        nodes: Iterable of nodes to filter (e.g., gm.graph.nodes)
        target: Either a callable function that takes a Node and returns bool,
               or operation(s) to match against (deprecated, use ops parameter)
        ops: Operation(s) to match against (preferred over target for operations)

    Yields:
        Node: Nodes that match the given operations or target function

    Example:
        # Using callable function:
        for node in filtered_nodes(gm.graph.nodes, is_linear_op):
            # process node

        # Using operations:
        for node in filtered_nodes(gm.graph.nodes, ops=torch.ops.aten.linear):
            # process node

        # Using multiple operations:
        for node in filtered_nodes(gm.graph.nodes, ops=[torch.ops.aten.linear, torch.ops.aten.bmm]):
            # process node
    """
    # Handle the case where target is a callable function
    if callable(target) and not isinstance(target, (OpOverloadPacket, OpOverload)):
        for node in nodes:
            if target(node):
                if prune_dangling and len(successors(node, depth=3)) < 3:
                    continue
                yield node
    elif isinstance(target, Iterable) and all(isinstance(t, Callable) for t in target):
        for node in nodes:
            for t in target:
                if t(node):
                    if prune_dangling and len(successors(node, depth=3)) < 3:
                        continue
                    yield node
                    break
    else:
        # Handle the case where target or ops contains operations
        operations = ops if ops is not None else target
        for node in nodes:
            if is_op(node, operations):
                if prune_dangling and len(successors(node, depth=3)) < 3:
                    continue
                yield node


def is_any_lin_op(node: Node) -> bool:
    return is_linear_op(node) or is_fake_quantized_linear_op(node)


def is_fp4_op(node: Node) -> bool:
    return is_op(
        node,
        [
            torch.ops.auto_deploy.torch_quant_nvfp4_linear,
            torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear,
        ],
    )


def is_any_moe_op(node: Node) -> bool:
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_moe,
            torch.ops.auto_deploy.torch_quant_fp8_moe,
            torch.ops.auto_deploy.torch_quant_nvfp4_moe,
            torch.ops.auto_deploy.triton_mxfp4_moe,
        ],
    )


def is_residual_add(node: Node) -> bool:
    if is_op(node, torch.ops.aten.add):
        if len(list(filtered_nodes(node.args, is_any_lin_op))) == 1:
            return True
    return False


def is_any_ssm_op(node: Node) -> bool:
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_ssm,
        ],
    )


def is_any_conv_op(node: Node) -> bool:
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_causal_conv1d,
        ],
    )


def is_any_attention_op(node: Node) -> bool:
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_attention_sdpa,
            torch.ops.auto_deploy.torch_attention,
        ],
    )


def is_linear_op(node: Node) -> bool:
    """Check if the node is a linear op.

    Using this function is preferred over `is_op` for linear ops to ensure all variants are covered.
    """
    lin_ops = {
        torch.ops.aten.linear,
        torch.ops.auto_deploy.torch_linear_simple,
    }

    return is_op(node, lin_ops)


def is_fake_quantized_linear_op(node: Node) -> bool:
    quantized_linear_op = {
        torch.ops.auto_deploy.torch_fake_quant_fp8_linear,
        torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear,
    }

    return is_op(node, quantized_linear_op)


def is_bmm_op(node: Node) -> bool:
    bmm_ops = {torch.ops.aten.bmm}

    return is_op(node, bmm_ops)


def is_dist_op(node: Node) -> bool:
    """Check if the node is a distributed op (torch or trtllm backend)."""
    dist_ops = {
        # PyTorch backend ops
        torch.ops.auto_deploy.torch_dist_all_gather,
        torch.ops.auto_deploy.torch_dist_all_reduce,
        # TRT-LLM backend ops
        torch.ops.auto_deploy.trtllm_dist_all_gather,
        torch.ops.auto_deploy.trtllm_dist_all_reduce,
    }
    return is_op(node, dist_ops)


def is_weight_node(node: Node) -> bool:
    return node.op == "get_attr" and node.target and has_shape(node) and len(shape(node)) > 0


# Auxiliary ops that may appear between a weight node and its consumer compute node
_WEIGHT_AUX_OPS = frozenset(
    {
        torch.ops.aten.to.dtype,
        torch.ops.aten.view.default,
    }
)


def precompute_weight_node_mapping(gm: GraphModule) -> None:
    """
    Pre-compute weight-to-consumer mapping for all weight nodes in the graph.

    For each weight node (get_attr), finds the consumer compute node by traversing
    through auxiliary ops (to.dtype, view.default). Stores the mapping in consumer
    node's metadata:
      - node.meta["weight_nodes"]: list of weight nodes (non-bias)
      - node.meta["bias_nodes"]: list of bias nodes

    This enables O(1) weight node lookup instead of O(depth) backward traversal.
    Called automatically on first weight lookup via lazy initialization.

    GUARANTEES (verified by assertions for debugging):
      - Called exactly once per GraphModule
      - No duplicate weight/bias nodes in any consumer's lists
      - Each weight node mapped to exactly one consumer
    """
    # Early return if already computed
    if "_weight_mapping_computed" in gm.meta:
        return
    gm.meta["_weight_mapping_computed"] = True

    for node in gm.graph.nodes:
        if not is_weight_node(node):
            continue

        is_bias = node.target.endswith("bias")

        # Find the consumer compute node by traversing through auxiliary ops
        current = node
        visited = {current}

        while True:
            # Get users of current node
            users = list(current.users.keys())
            if not users:
                break

            # Check if any user is a compute node (not an auxiliary op)
            consumer_found = None
            aux_node = None

            for user in users:
                if is_bias:
                    if "bias_nodes" not in user.meta:
                        user.meta["bias_nodes"] = []
                    # ASSERTION: Each weight node should be mapped exactly once
                    assert node not in user.meta["bias_nodes"], (
                        f"Duplicate bias node {node.name} found for consumer {user.name}"
                    )
                    user.meta["bias_nodes"].append(node)
                else:
                    if "weight_nodes" not in user.meta:
                        user.meta["weight_nodes"] = []
                    # ASSERTION: Each weight node should be mapped exactly once
                    assert node not in user.meta["weight_nodes"], (
                        f"Duplicate weight node {node.name} found for consumer {user.name}"
                    )
                    user.meta["weight_nodes"].append(node)
                if user.target in _WEIGHT_AUX_OPS:
                    # This is an auxiliary op, continue traversing
                    aux_node = user
                else:
                    # This is a potential consumer compute node
                    consumer_found = user
                    break

            if consumer_found is not None:
                # Found the consumer, return
                break
            elif aux_node is not None and aux_node not in visited:
                # Continue through auxiliary op
                current = aux_node
                visited.add(current)
            else:
                # No more nodes to traverse
                break


def _ensure_weight_mapping(node: Node) -> None:
    """Ensure weight node mapping is computed. Lazily calls precompute if needed."""
    gm = node.graph.owning_module
    if "_weight_mapping_computed" not in gm.meta or not gm.meta["_weight_mapping_computed"]:
        precompute_weight_node_mapping(gm)


def get_weight_node(node: Node) -> Optional[Node]:
    """Get the primary weight node for a compute node"""
    _ensure_weight_mapping(node)
    weight_nodes = node.meta.get("weight_nodes", [])
    return weight_nodes[0] if weight_nodes else None


def get_weight_nodes(node: Node) -> List[Node]:
    """Get all weight nodes for a compute node"""
    _ensure_weight_mapping(node)
    return node.meta.get("weight_nodes", [])


def get_bias_nodes(node: Node) -> List[Node]:
    """Get all bias nodes for a compute node"""
    _ensure_weight_mapping(node)
    return node.meta.get("bias_nodes", [])


@dataclass
class WeightInfo:
    """Lightweight weight info extracted from a weight node."""

    node: Node
    node_key: str
    tensor: torch.Tensor
    submod: nn.Module


def _weight_node_to_info(weight_node: Node, gm: GraphModule) -> WeightInfo:
    """Convert a weight node to WeightInfo."""
    node_key = weight_node.target
    tensor = get_param_or_buffer(node_key, gm)
    submod = gm.get_submodule(node_key.rpartition(".")[0])
    return WeightInfo(node=weight_node, node_key=node_key, tensor=tensor, submod=submod)


def get_weight_info(node: Node) -> Optional[WeightInfo]:
    """Extract weight info for the primary weight of a compute node."""
    weight_node = get_weight_node(node)
    if weight_node is None:
        return None
    return _weight_node_to_info(weight_node, node.graph.owning_module)


@dataclass
class AllWeightInfos:
    """Container for all weight and bias infos of a compute node."""

    weights: List[WeightInfo]
    biases: List[WeightInfo]


def get_all_weight_infos(node: Node) -> AllWeightInfos:
    """Extract all weight and bias infos for a compute node."""
    gm = node.graph.owning_module
    weight_nodes = get_weight_nodes(node)
    bias_nodes = get_bias_nodes(node)

    return AllWeightInfos(
        weights=[_weight_node_to_info(wn, gm) for wn in weight_nodes],
        biases=[_weight_node_to_info(bn, gm) for bn in bias_nodes],
    )


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


def identify_regions_between_residuals(gm: GraphModule) -> List[Node]:
    """Identify regions of the graph that we can investigate further for patterning matching.

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
        if node.op == "output":
            output_node = node
    assert input_id_node, "Could not find input node"
    assert output_node, "Could not find output node"

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
    res_nodes_more_users = [n for n in boundary_nodes[2:] if len(n.users) > 2]
    if res_nodes_more_users:
        ad_logger.debug(f"Unexpected # of users for residuals: {res_nodes_more_users}")

    # add output node to boundary nodes
    boundary_nodes.append(output_node)

    return boundary_nodes


def get_all_layer_subgraphs(gm: GraphModule) -> tuple[List[LayerSubgraph], set[Node]]:
    """
    Get subgraphs for all consecutive layers (attention, MLP, SSM, MoE) in the graph.

    Pre-computes weight mappings and caches weight shapes for all linear nodes.
    Each layer is contained between opening linear layers and a single closing linear layer.

    Assumptions:
        1. each layer (each subgraph) is contained between a list of opening
        linear layers (e.g., q/k/v_proj, gate/up_proj, in_proj, etc.) and a single closing linear layer
        (e.g., out_proj, down_proj, etc.)
        2. all layers are connected in sequence, there are no "parallel" layers.

    Consequence:
        1. We can linearize both all linear nodes in the graph and the layers itself, having a single
        linear history and define layers by the indices of corresponding opening and closing linear nodes,
        with no overlap between layers.
        E.g., if layer i is defined between linear nodes start_i and end_i, then all linear nodes between
        start_i and end_i necessarily belong to layer i.
        2. It does not mean that every linear node in the graph belongs to a layer, that is. Then, if:
            end_i < start_{{i+1}}, then all linear nodes in[end_i + 1, start_{{i+1}} - 1] do not belong
            to any layer, and are marked as "unprocessed".
    Note:
        The interesting case is MoE with shared experts. In this case, there are two parallel paths:
        1. Routed experts
        2. Shared experts
        In this case, "shared experts" path will be marked as an MLP layer, and "routed experts" path will
        be "unprocessed". This is desired, since routed experts should not be sharded by the TP transform,
        but a corresponding EP/BMM transforms.
    """

    assert gm.graph.nodes, "Graph is empty"
    layer_subgraphs = []
    linear_nodes = list(filtered_nodes(gm.graph.nodes, is_any_lin_op))

    # Pre-compute weight-to-consumer mapping for O(1) weight node lookup
    precompute_weight_node_mapping(gm)

    # Cache weight shapes for all linear nodes
    for lin_node in linear_nodes:
        if "lin_node_shape" not in lin_node.meta:
            shape = get_weight_shape(lin_node)
            if shape is not None:
                lin_node.meta["lin_node_shape"] = shape

    # Find the embedding size from the first linear node
    embd = get_weight_shape(linear_nodes[0], dim=-1)
    if embd is None:
        raise ValueError("Failed to extract embedding size from first linear node")

    unprocessed_linear_nodes = set(linear_nodes)
    assert len(linear_nodes) > 0, "Could not find any linear nodes in the graph"

    terminating_indices = [-1]
    last_lin_index = terminating_indices[-1] + 1

    # For each linear node, find its layer subgraph defined as regions between consecutive linear nodes.
    while last_lin_index < len(linear_nodes):
        layer_subgraph = get_layer_after_linear_node(linear_nodes, terminating_indices, embd=embd)

        if layer_subgraph.opening_nodes is not None and len(layer_subgraph.opening_nodes) > 0:
            unprocessed_linear_nodes -= (
                set(layer_subgraph.opening_nodes)
                | set([layer_subgraph.terminating_node])
                | set(layer_subgraph.subgraph_nodes)
            )
            layer_subgraphs.append(layer_subgraph)
        last_lin_index = terminating_indices[-1] + 1

    # Unprocessed linear nodes can be "simple sharded".
    return layer_subgraphs, unprocessed_linear_nodes


def bfs(
    node: Node,
    target: Callable,
    attr_next: str = "users",
    boundary: Optional[Node] = None,
    include_root: bool = True,
) -> Tuple[Node, int]:
    """
    Breadth-first search of the graph.
    Returns the found node and the depth of the node.
    """
    depth = 0
    queue_at_depth = [node]
    queue_at_depth_next = []
    visited = set()
    while queue_at_depth or queue_at_depth_next:
        cur_node = queue_at_depth.pop(0)
        if boundary is not None and cur_node == boundary:
            continue  # Skip the boundary node.
        if target(cur_node) and (include_root or depth > 0):
            return cur_node, depth
        if hasattr(cur_node, attr_next):
            for next_node in getattr(cur_node, attr_next):
                if boundary is not None and next_node == boundary:
                    continue  # Do not expand past the boundary.
                if next_node not in visited:
                    visited.add(next_node)
                    queue_at_depth_next.append(next_node)
        if not queue_at_depth:
            queue_at_depth = queue_at_depth_next
            queue_at_depth_next = []
            depth += 1

    return None, -1


def extract_output_tuple(node: Node, count: int = 2):
    """
    Extract up to `count` outputs from a tuple-producing node.
    Returns a list of length `count`, with None if an output isn't found.
    """
    results = []
    for idx in range(count):
        user_node = next(
            (
                u
                for u in node.users
                if u.op == "call_function" and u.target == operator.getitem and u.args[1] == idx
            ),
            None,
        )
        results.append(user_node)
    return results


def extract_op_args(node: Node, *arg_names):
    """
    Given a call_function node for torch custom op,
    returns a tuple of values for each name in arg_names, trying in order:
    1. node.kwargs[name]
    2. node.args[position_in_schema]
    3. the schema default
    """
    if node.op != "call_function":
        raise ValueError(f"extract_op_args only supports call_function nodes, got {node.op}")

    op = node.target
    if hasattr(op, "_schemas"):
        schema = next(iter(op._schemas.values()))
    elif hasattr(op, "_schema"):
        schema = op._schema
    else:
        raise RuntimeError(f"No schema found on op {op}")
    args_meta = schema.arguments

    # name→index in signature, and name→default_value
    pos = {a.name: i for i, a in enumerate(args_meta)}
    defs = {a.name: a.default_value for a in args_meta if a.has_default_value}

    args = list(node.args)
    kwargs = node.kwargs or {}

    def _get(name):
        if name in kwargs:
            return kwargs[name]
        i = pos.get(name)
        if i is not None and i < len(args):
            return args[i]
        if name in defs:
            return defs[name]
        raise RuntimeError(f"Could not find a value for '{name}' on op {op}")

    return [_get(n) for n in arg_names]


def predecessors(
    node: Node,
    depth: int = 1,
    include: Optional[Callable[[Node], bool]] = None,
    exclude: Optional[Callable[[Node], bool]] = None,
) -> List[Node]:
    """
    Build predecessor tree of node by recursively traversing node.args up to depth depth.
    If include is provided, only include nodes that satisfy the condition.
    If exclude is provided, exclude nodes that satisfy the condition.
    """
    preds = []
    seen = set()
    for arg in node.all_input_nodes:
        if ((not include) or (include and include(arg))) and (not exclude or not exclude(arg)):
            if arg not in seen:
                preds.append(arg)
                seen.add(arg)
        if depth > 1:
            for p in predecessors(arg, depth - 1, include, exclude):
                if p not in seen:
                    preds.append(p)
                    seen.add(p)
    return preds


def successors(
    node: Node,
    depth: int = 1,
    include: Optional[Callable[[Node], bool]] = None,
    exclude: Optional[Callable[[Node], bool]] = None,
) -> List[Node]:
    """
    Build successor tree of node by recursively traversing node.users up to depth depth.
    If include is provided, only include nodes that satisfy the condition.
    If exclude is provided, exclude nodes that satisfy the condition.
    """
    succs = []
    seen = set()
    for user in node.users:
        if ((not include) or (include and include(user))) and (not exclude or not exclude(user)):
            if user not in seen:
                succs.append(user)
                seen.add(user)
        if depth > 1:
            for s in successors(user, depth - 1, include, exclude):
                if s not in seen:
                    succs.append(s)
                    seen.add(s)
    return succs


def subgraph(
    sources: Optional[list[Node]] = None,
    sinks: Optional[list[Node]] = None,
    include: Optional[Callable[[Node], bool]] = None,
    exclude: Optional[Callable[[Node], bool]] = None,
    boundary_condition: Optional[Callable[[Node], bool]] = None,
) -> List[Node]:
    """
    Returns a list of nodes in a subgraph in computation DAG defined as either:
    1. all nodes succeeding any of the node in sources and preceding any of the
      nodes in sinks. In this case, it is built by a BFS traversal from sinks,
      where the sources list acts as a boundary. We do it in this order (and not
      from sources to sinks) to include nodes like weights or other inputs (they
      are not successors of sinks, so otherwise they wouldn't be included).
    2. all nodes succeeding any of the node in sources, bounded by (BFS search
      not extending further than) boundary_condition.
    3. all nodes preceding any of the nodes in sinks, bounded by (BFS search
      not extending further than) boundary_condition.

    Optionally, include or exclude conditions may be specified to include [exclude]
      only nodes that meet [don't meet] certain condition.

    """
    subgraph_nodes = []
    seen = set()

    # differentiate between cases 1, 2, and 3 by checking if sinks and sources are provided
    if sinks is not None and sources is not None:
        # case 1
        queue = list(sinks)
        start_nodes = set(sinks)
        sources_set = set(sources)
        # Initialize queue with sinks and mark them as seen
        for node in sinks:
            if node not in seen:
                seen.add(node)
        if boundary_condition is None:

            def boundary_condition(n):
                return n in sources_set

        attr_next = "all_input_nodes"
    elif sources is not None:
        # case 2
        assert boundary_condition is not None, "boundary_condition must be provided for case 2"
        # Initialize queue with sinks and mark them as seen
        queue = list(sources)
        start_nodes = set(sources)
        attr_next = "users"
    elif sinks is not None:
        # case 3
        assert boundary_condition is not None, "boundary_condition must be provided for case 3"
        # Initialize queue with sinks and mark them as seen
        queue = list(sinks)
        start_nodes = set(sinks)
        attr_next = "all_input_nodes"
    else:
        raise ValueError("Either sinks or sources must be provided")

    # BFS traversal from sinks backwards through predecessors
    while queue:
        node = queue.pop(0)

        # Check if node should be included based on filters
        should_include = True
        if include is not None and not include(node):
            should_include = False
        if exclude is not None and exclude(node):
            should_include = False

        if should_include and node not in start_nodes:
            subgraph_nodes.append(node)

        # Stop traversal at boundary - don't explore their predecessors
        if boundary_condition(node) and node not in start_nodes:
            continue

        # Traverse to predecessor nodes (all inputs to this node)
        for arg in getattr(node, attr_next):
            if isinstance(arg, Node) and arg not in seen:
                seen.add(arg)
                queue.append(arg)

    return subgraph_nodes


def get_weight_shape(node: Node, dim: Optional[int] = None) -> Optional[Union[int, List[int]]]:
    """Get weight shape for a linear operation node. Returns None if no weight."""
    if not is_any_lin_op(node):
        return None

    weight_node = get_weight_node(node)
    if weight_node is None:
        return None

    s = list(shape(weight_node))

    if is_fp4_op(node):
        # FP4 weights are packed as uint8 type with 2 FP4 values per element
        s[-1] *= 2
    if dim is None:
        return s
    else:
        return s[dim]


def get_layer_after_linear_node(
    linear_nodes: List[Node],
    terminating_indices: List[int],
    embd: int,
    match_on_shapes: bool = True,
    enforce_strict_linear_history: bool = True,
) -> LayerSubgraph:
    """
    Get the next model layer.
    The previous layer was closed by the terminating linear node with index terminating_indices[-1].

    Since we assume a layer is always terminated by a single linear node, we iteratively query subgraph
    and check for the condition len(lin_nodes_in_subgraph) == 1. If a given linear node
    linear_nodes[start_lin_index] does not have a corresponding single sink linear node, it will
    be classified as "unprocessed", and the next linear node is picked as a candidate to open
    a new layer.

    match_on_shapes explanation: We assume that the opening linear weights have shape [hidden, embedding],
       where, while hidden may vary from layer to layer (e.g., MLP, MoE, latent projections) may have
       different hidden sizes, the embedding size is the property of the model across all layers.
       Similarly, the unique closing linear weight should have shape [embedding, hidden], to map back
       from the hidden space to the embedding space.
       If match_on_shapes is True, we require that the opening_layer.shape[-1] == closing_layer.shape[0]
       If match_on_shapes is False, we only require the topological connectivity and uniqueness of
       closing_layer being the only sink.
    Why it matters: For MLA, activation X goes through the latent space projection, following:
        Q = norm(X @ W_q_a) @ W_q_b   # <- two linear projections
        KV = norm(X @ W_kv_a) @ W_kv_b  # <- two linear projections
        Without match_on_shapes, we would treat norm(X @ W_q_a) @ W_q_b as entire MLP layer and apply
        column-row sharding to it. That would result with Q not being sharded, but replicated (after MLP all-reduce),
        and the entire attention computation being replicated.
        With match_on_shapes, the entire MLA will be treated as a single layer, with the o_proj as the
        unique closing linear node.

    Args:
        linear_nodes: List of linear nodes in the graph.
        terminating_indices: List of indices of terminating linear nodes.
        embd: Embedding size for shape matching.
        match_on_shapes: If True, match layers on embedding shapes.
        enforce_strict_linear_history: If True, enforce strict ordering constraints.

    Returns:
        LayerSubgraph containing opening nodes, subgraph nodes, and terminating node.
    """

    def boundary_condition(node: Node, dim: int) -> bool:
        if match_on_shapes:
            if is_any_lin_op(node):
                return node.meta["lin_node_shape"][dim] == embd
            return (
                is_any_moe_op(node)
                or is_op(node, ops=[torch.ops.aten.sym_size, torch.ops.aten.bmm])
                or is_residual_add(node)
            )
        else:
            return (
                is_any_lin_op(node)
                or is_any_moe_op(node)
                or is_op(node, ops=[torch.ops.aten.sym_size, torch.ops.aten.bmm])
                or is_residual_add(node)
            )

    def filter_condition(node: Node, dim: int) -> bool:
        if match_on_shapes:
            if is_any_lin_op(node):
                return node.meta["lin_node_shape"][dim] == embd
            return False
        else:
            return is_any_lin_op(node)

    lin_nodes_in_subgraph = []
    start_lin_index = terminating_indices[-1] + 1

    while len(lin_nodes_in_subgraph) != 1:
        if start_lin_index >= len(linear_nodes):
            terminating_indices.append(len(linear_nodes))
            return LayerSubgraph(
                opening_nodes=[],
                subgraph_nodes=[],
                terminating_node=None,
                layer_type=LayerType.UNKNOWN,
            )

        forward_subgraph = subgraph(
            sources=[linear_nodes[start_lin_index]],
            boundary_condition=lambda n: boundary_condition(n, dim=0),
        )
        lin_nodes_in_subgraph = list(
            filtered_nodes(forward_subgraph, lambda n: filter_condition(n, dim=0))
        )
        if len(lin_nodes_in_subgraph) > 1:
            # it means that probably we went over the boundary of the layer.
            # It may happen e.g., with MoLE (latent MoE), with the closing latent fc2 projection,
            # when the subgraph spanned over fc2 "spills" over consecutive layers.
            # Then, wrap this single linear node in  LayerType.UNKNOWN and return.
            terminating_indices.append(start_lin_index)
            return LayerSubgraph(
                opening_nodes=[linear_nodes[start_lin_index]],
                subgraph_nodes=[],
                terminating_node=linear_nodes[start_lin_index],
                layer_type=LayerType.UNKNOWN,
            )
        start_lin_index += 1
    start_lin_index -= 1
    terminating_linear_node = lin_nodes_in_subgraph[0]

    # For backward pass, match embedding on dim=-1
    backward_subgraph = subgraph(
        sinks=[terminating_linear_node], boundary_condition=lambda n: boundary_condition(n, dim=-1)
    )

    # Get all opening linear nodes
    opening_linear_nodes = list(
        filtered_nodes(backward_subgraph, lambda n: filter_condition(n, dim=-1))
    )

    if enforce_strict_linear_history:
        # opening nodes must succeed last terminating node
        last_terminating_index = terminating_indices[-1]
        opening_linear_nodes = [
            n for n in opening_linear_nodes if linear_nodes.index(n) > last_terminating_index
        ]

    # subgraph_nodes should not include opening nodes.
    # the entire layer =  opening_nodes + subgraph_nodes + terminating_node,
    # with these three sets being disjoint.
    interior_nodes = [
        n
        for n in set(backward_subgraph).union(forward_subgraph)
        if n not in set(opening_linear_nodes).union([terminating_linear_node])
    ]
    ssm_nodes = list(filtered_nodes(interior_nodes, is_any_ssm_op))
    attention_nodes = list(filtered_nodes(interior_nodes, is_any_attention_op))
    intermediate_lin_nodes = list(filtered_nodes(interior_nodes, is_any_lin_op))
    intermediate_weight_nodes = list(
        filtered_nodes(
            interior_nodes, lambda n: is_weight_node(n) and not is_any_lin_op(list(n.users)[0])
        )
    )

    ####################################################
    ########## LAYER TYPE CLASSIFICATION ###############
    ####################################################

    def classify_layer_type() -> [LayerType, int]:
        if len(ssm_nodes) + len(attention_nodes) > 1:
            return LayerType.UNKNOWN, 1

        if len(attention_nodes) == 1:
            head_size = shape(attention_nodes[0])[-1]
            # check if this is MLA:
            # these two intermediate linear nodes are the latent q and kv projections.
            if len(intermediate_lin_nodes) == 2:
                # MLA has a RMS norm inside, so it should have one (or two, couning biaas)
                # intermediate weight nodes
                if len(intermediate_weight_nodes) not in [1, 2]:
                    return LayerType.UNKNOWN, 1
                return LayerType.MLA, head_size
            else:
                if len(intermediate_lin_nodes) != 0:
                    return LayerType.UNKNOWN, 1
                return LayerType.ATTENTION, head_size

        if len(ssm_nodes) == 1:
            head_size = shape(ssm_nodes[0])[-1]
            # Mamba layers should not have any intermediate linear nodes.
            if len(intermediate_lin_nodes) > 0:
                return LayerType.UNKNOWN, 1
            # Mamba layer should have 3 to 6 intermediate weight nodes:
            # - conv1d weight
            # - A (A_log)
            # - D
            # - conv1d bias [optional]
            # - dt_bias [optional]
            # - RMS norm [optional]
            if len(intermediate_weight_nodes) not in list(range(3, 7)):
                return LayerType.UNKNOWN, 1
            return LayerType.SSM, head_size

        # if we reach here, it means the layer is a MLP.
        # MLP should not have any intermediate linear or weight nodes.
        if len(intermediate_lin_nodes) > 0 or len(intermediate_weight_nodes) > 0:
            return LayerType.UNKNOWN, 1
        return LayerType.MLP, 1

    layer_type, head_size = classify_layer_type()

    layer_subgraph = LayerSubgraph(
        opening_nodes=opening_linear_nodes,
        subgraph_nodes=interior_nodes,
        terminating_node=terminating_linear_node,
        layer_type=layer_type,
        min_local_shape=head_size,
    )
    assert linear_nodes[start_lin_index] in opening_linear_nodes, (
        f"Linear node not found in opening linear nodes - "
        f"terminating_linear_node:{terminating_linear_node.name}, "
        f"opening_linear_nodes: {[n.name for n in opening_linear_nodes]}"
    )

    # return the index of the terminating linear node
    if terminating_linear_node == linear_nodes[-1]:
        terminating_index = len(linear_nodes)
    else:
        terminating_index = (
            start_lin_index + len(opening_linear_nodes) + len(intermediate_lin_nodes)
        )

    if enforce_strict_linear_history:
        if terminating_index < len(linear_nodes):
            assert linear_nodes[terminating_index] == terminating_linear_node, (
                "ill-formed layer subgraph"
            )
        terminating_indices.append(terminating_index)

    return layer_subgraph


def has_shape(node: Node) -> bool:
    return hasattr(node, "meta") and "val" in node.meta and hasattr(node.meta["val"], "shape")


def shape(node: Node) -> Tuple[int, ...]:
    if not has_shape(node):
        return None
    return node.meta["val"].shape


def get_weight_tensor(node: Node) -> torch.Tensor:
    """Extract the weight tensor from a compute node."""
    weight_node = get_weight_node(node)
    if weight_node is None:
        raise ValueError(f"Node {node.name} has no weight")

    gm = node.graph.owning_module
    return get_param_or_buffer(weight_node.target, gm)


def draw_graph(gm: GraphModule, filename: str):
    """
    Dump graphmodule to SVG file using PyTorch's built-in drawer.
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    drawer = FxGraphDrawer(gm, filename)
    with open(f"{filename}.svg", "wb") as f:
        f.write(drawer.get_dot_graph().create_svg())
