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
    DELTA = "delta"
    UNKNOWN = "unknown"


class LayerSubgraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    layer_type: LayerType
    opening_nodes: List[Node]
    terminating_node: Union[Node, None]
    min_local_shape: int = 1
    subgraph_nodes: List[Node]


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
    try:
        weight_nodes = extract_weight_nodes(node)
    except Exception:
        return False
    if len(weight_nodes.weights) == 0:
        return False
    return weight_nodes.weights[0].node_key


def get_param_or_buffer(tensor_name: str, gm: GraphModule) -> torch.Tensor:
    param_dict = WeightBiasInfoCache.get_param_dict(gm)
    if tensor_name in param_dict:
        return param_dict[tensor_name]
    buffer_dict = WeightBiasInfoCache.get_buffer_dict(gm)
    if tensor_name in buffer_dict:
        return buffer_dict[tensor_name]
    raise KeyError(f"Tensor {tensor_name} not found in the graph")


class WeightBiasInfoCache:
    """Cache for weight and bias information to avoid repeated expensive operations.

    This class manages caches for parameter names and weight shapes that are used
    during graph transformation operations. Use it as a context manager to scope
    the cache lifetime.

    Example:
        with WeightBiasInfoCache() as cache:
            # All calls to get_weight_shape and extract_weight_nodes
            # within this block use caching
            layer_subgraphs, _ = get_all_layer_subgraphs(gm)
        # Caches are cleared here
    """

    # Class-level reference to the currently active cache instance
    _active_instance: "WeightBiasInfoCache" = None

    def __init__(self):
        # Cache for param/buffer dicts to avoid repeated expensive named_parameters/named_buffers calls
        self._param_dict_cache = {}
        self._buffer_dict_cache = {}
        # Cache for get_weight_shape to avoid repeated expensive extract_weight_nodes calls
        self._weight_shape_cache = {}
        # Activate this cache instance
        WeightBiasInfoCache._active_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly deactivate and clear the cache."""
        if WeightBiasInfoCache._active_instance is self:
            WeightBiasInfoCache._active_instance = None
        self._param_dict_cache.clear()
        self._buffer_dict_cache.clear()
        self._weight_shape_cache.clear()

    def __del__(self):
        """Cleanup when the cache is garbage collected."""
        self.close()

    @classmethod
    def is_active(cls) -> bool:
        """Check if caching is currently enabled."""
        return cls._active_instance is not None

    @classmethod
    def get_param_dict(cls, gm: GraphModule) -> dict:
        """Get cached parameters dict for a GraphModule, or compute and cache it."""
        if cls._active_instance is None:
            return dict(gm.named_parameters())

        cache = cls._active_instance._param_dict_cache
        if gm not in cache:
            cache[gm] = dict(gm.named_parameters())
        return cache[gm]

    @classmethod
    def get_buffer_dict(cls, gm: GraphModule) -> dict:
        """Get cached buffers dict for a GraphModule, or compute and cache it."""
        if cls._active_instance is None:
            return dict(gm.named_buffers())

        cache = cls._active_instance._buffer_dict_cache
        if gm not in cache:
            cache[gm] = dict(gm.named_buffers())
        return cache[gm]

    @classmethod
    def get_param_names(cls, gm: GraphModule) -> set:
        """Get cached parameter and buffer names for a GraphModule."""
        param_dict = cls.get_param_dict(gm)
        buffer_dict = cls.get_buffer_dict(gm)
        return set(param_dict.keys()).union(buffer_dict.keys())

    @classmethod
    def get_weight_shape(cls, node: Node) -> Tuple[bool, Optional[List[int]]]:
        """Get cached weight shape for a node.

        Returns:
            Tuple of (found, value). If found is False, value should be ignored.
        """
        if cls._active_instance is None:
            return False, None

        cache = cls._active_instance._weight_shape_cache
        if node in cache:
            return True, cache[node]
        return False, None

    @classmethod
    def set_weight_shape(cls, node: Node, shape: Optional[List[int]]):
        """Store weight shape in cache."""
        if cls._active_instance is not None:
            cls._active_instance._weight_shape_cache[node] = shape


def extract_weight_nodes(node: Node) -> WeightNodes:
    """Extracts the list of weight node and optional bias node from the given parametrized node"""
    gm = node.graph.owning_module

    # Use cached param_names to avoid repeated expensive named_parameters/named_buffers calls
    param_names = WeightBiasInfoCache.get_param_names(gm)

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
    elif is_fake_quantized_linear_op(node):
        # For quantized linear ops (FP8, FP4, etc.), only args[1] is the actual shardable
        # weight. Scale buffers (input_scale, weight_scale, alpha, ...) are also registered
        # as get_attr nodes in the graph and would otherwise be picked up by the generic
        # all_input_nodes scan below -- causing shard_weight_tensor to overwrite them as
        # nn.Parameters, which then breaks quantization_cb's get_buffer() call.
        # The quantization_cb (QuantizationShardingMixin) is responsible for sharding scales.
        weight_node = find_get_attr_node(node.args[1])
        if weight_node is None:
            return WeightNodes(weights=[], biases=[])
        biases = []
        if len(node.args) > 2 and isinstance(node.args[2], Node):
            b = find_get_attr_node(node.args[2])
            if b is not None and b.target.rsplit(".", 1)[-1] == "bias":
                biases = [
                    WeightNode(
                        node=node.args[2],
                        node_key=b.target,
                        submod=gm.get_submodule(b.target.rpartition(".")[0]),
                        tensor=get_param_or_buffer(b.target, gm),
                    )
                ]
        return WeightNodes(
            weights=[
                WeightNode(
                    node=node.args[1],
                    node_key=weight_node.target,
                    submod=gm.get_submodule(weight_node.target.rpartition(".")[0]),
                    tensor=get_param_or_buffer(weight_node.target, gm),
                )
            ],
            biases=biases,
        )
    elif is_weight_node(node):
        weights = []
        biases = []

        if node.target.rsplit(".", 1)[-1] == "bias":
            biases = [
                WeightNode(
                    node=node,
                    node_key=node.target,
                    tensor=get_param_or_buffer(node.target, gm),
                    submod=gm.get_submodule(node.target.rpartition(".")[0]),
                )
            ]
        else:
            weights = [
                WeightNode(
                    node=node,
                    node_key=node.target,
                    tensor=get_param_or_buffer(node.target, gm),
                    submod=gm.get_submodule(node.target.rpartition(".")[0]),
                )
            ]
        return WeightNodes(
            weights=weights,
            biases=biases,
        )
    # for other parametrized nodes, we need to find the weight node
    else:
        all_weight_nodes = [
            attr_node
            for n in node.all_input_nodes
            if (attr_node := find_get_attr_node(n)) is not None
        ]
        # separate weight nodes and bias nodes
        bias_nodes = [n for n in all_weight_nodes if n.target.rsplit(".", 1)[-1] == "bias"]
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


def get_weight_node(node: Node) -> Node:
    """Get the primary weight node for a compute node.

    When the node itself is a bias get_attr node (i.e. extract_weight_nodes
    puts it into .biases rather than .weights), return the bias node so that
    num_users_of_weight_node gives the correct user count instead of 0.
    """
    weight_nodes = extract_weight_nodes(node)
    if len(weight_nodes.weights) > 0:
        return weight_nodes.weights[0].node
    if len(weight_nodes.biases) > 0:
        return weight_nodes.biases[0].node
    raise ValueError(f"Node {node.name} has no weight or bias")


def num_users_of_weight_node(node: Node) -> int:
    """Returns the number of users of the weight node of the given parametrized node."""
    try:
        weight_node = get_weight_node(node)
    except ValueError:
        return 0
    return len(weight_node.users)


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


def is_any_delta_op(node: Node) -> bool:
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_gated_delta_rule,
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
            torch.ops.aten.conv1d,  # Support regular conv1d for tests
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


def is_any_mla_op(node: Node) -> bool:
    """Check if the node is a mla op."""
    return is_op(
        node,
        ops=[
            torch.ops.auto_deploy.torch_mla,
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
    if "_weight_mapping_computed" in gm.meta and gm.meta["_weight_mapping_computed"]:
        return
    gm.meta["_weight_mapping_computed"] = True

    for node in gm.graph.nodes:
        if not is_weight_node(node):
            continue

        is_bias = node.target.rsplit(".", 1)[-1] == "bias"

        # the weight to user mapping is reflective - the weight node "owns" itself
        node.meta["weight_nodes"] = [node]

        # Find the consumer compute node by traversing through auxiliary ops
        current = node
        visited = {current}

        while True:
            # Get users of current node
            users = list(current.users.keys())
            if not users:
                break

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

            if aux_node is not None and aux_node not in visited:
                # Continue through auxiliary op
                current = aux_node
                visited.add(current)
            else:
                # No more nodes to traverse
                break


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
        3. Residual nodes from the embedding node onwards (no other nodes in-between)
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
    while True:
        next_res_add, _ = bfs(
            boundary_nodes[-1], lambda n: is_op(n, torch.ops.aten.add), include_root=False
        )
        if next_res_add is None:
            break
        else:
            boundary_nodes.append(next_res_add)

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

    Caches weight shapes for all linear nodes using WeightBiasInfoCache.
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

    # get residual add nodes to correctly identify layer boundaries
    residuals = identify_regions_between_residuals(gm)

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
        layer_subgraph = get_layer_after_linear_node(
            linear_nodes, terminating_indices, embd=embd, residuals=residuals
        )

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


def _get_op_schema(node: Node):
    """Return the op schema for a call_function node."""
    if node.op != "call_function":
        raise ValueError(f"_get_op_schema only supports call_function nodes, got {node.op}")
    op = node.target
    if hasattr(op, "_schemas"):
        return next(iter(op._schemas.values()))
    elif hasattr(op, "_schema"):
        return op._schema
    raise RuntimeError(f"No schema found on op {op}")


def extract_op_args(node: Node, *arg_names):
    """
    Given a call_function node for torch custom op,
    returns a tuple of values for each name in arg_names, trying in order:
    1. node.kwargs[name]
    2. node.args[position_in_schema]
    3. the schema default
    """
    schema = _get_op_schema(node)
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
        raise RuntimeError(f"Could not find a value for '{name}' on op {node.target}")

    return [_get(n) for n in arg_names]


def set_op_args(node: Node, **name_value_pairs) -> None:
    """Set argument values on a call_function node by name, using the op schema.

    For each name=value pair, the value is placed according to where the argument
    currently lives (or would naturally live):

    1. If the name is already present in ``node.kwargs``, update it there.
    2. If the name corresponds to a positional slot that exists in ``node.args``,
       update that slot.
    3. Otherwise, add it to ``node.kwargs`` (safest default — downstream
       consumers using ``extract_op_args`` or ``node.kwargs`` will find it).

    This is the write-side complement to :func:`extract_op_args` and avoids
    manual index arithmetic when injecting new arguments into a node.
    """
    schema = _get_op_schema(node)
    pos = {a.name: i for i, a in enumerate(schema.arguments)}

    args = list(node.args)
    kwargs = dict(node.kwargs) if node.kwargs else {}

    for name, value in name_value_pairs.items():
        if name not in pos:
            raise RuntimeError(f"'{name}' is not a valid argument for op {node.target}")
        if name in kwargs:
            kwargs[name] = value
        elif pos[name] < len(args):
            args[pos[name]] = value
        else:
            kwargs[name] = value

    node.args = tuple(args)
    node.kwargs = kwargs


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
    """Get the shape of the weight node."""
    if not is_any_lin_op(node):
        return None

    # Try to get from cache first
    found, s = WeightBiasInfoCache.get_weight_shape(node)
    if not found:
        # Not in cache or caching not enabled - compute the shape
        s = list(shape(extract_weight_nodes(node).weights[0].node))
        if len(s) == 0:
            s = None
        elif is_fp4_op(node):
            # FP4 weights are packed as uint8 type with 2 FP4 values per element
            s[-1] *= 2
        # Store in cache if caching is enabled
        WeightBiasInfoCache.set_weight_shape(node, s)

    if s is None:
        return None
    if dim is None:
        return s
    else:
        return s[dim]


def get_layer_after_linear_node(
    linear_nodes: List[Node],
    terminating_indices: List[int],
    embd: int,
    residuals: List[Node],
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
                or node in residuals
            )
        else:
            return (
                is_any_lin_op(node)
                or is_any_moe_op(node)
                or is_op(node, ops=[torch.ops.aten.sym_size, torch.ops.aten.bmm])
                or node in residuals
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
    delta_nodes = list(filtered_nodes(interior_nodes, is_any_delta_op))
    attention_nodes = list(filtered_nodes(interior_nodes, is_any_attention_op))
    mla_nodes = list(filtered_nodes(interior_nodes, is_any_mla_op))
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
        if len(ssm_nodes) + len(attention_nodes) + len(mla_nodes) + len(delta_nodes) > 1:
            # ambiguous layer type
            return LayerType.UNKNOWN, 1

        if len(delta_nodes) == 1:
            head_size = shape(delta_nodes[0])[-1]
            # Gated DeltaNet layers should have 2 opening linear nodes (fused qkvz + ba,
            # e.g. Qwen3Next) or 4 opening linear nodes (unfused qkv + z + b + a,
            # e.g. Qwen3.5 MoE) and one terminating node.
            if len(intermediate_lin_nodes) > 0 or len(opening_linear_nodes) not in (2, 4):
                return LayerType.UNKNOWN, 1
            # Gated DeltaNet layer should have 4 to 6 intermediate weight nodes:
            # - conv1d weight
            # - attn_A (attn_a_log))
            # - attn_norm_weight
            # - layernorm_weight
            # - attn_dt_bias [optional]
            # - conv1d bias [optional]

            if len(intermediate_weight_nodes) not in list(range(4, 7)):
                return LayerType.UNKNOWN, 1
            return LayerType.DELTA, head_size

        if len(attention_nodes) == 1:
            head_size = shape(attention_nodes[0])[-1]
            if len(intermediate_lin_nodes) > 0:
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

        if len(mla_nodes) == 1:
            head_size = shape(mla_nodes[0])[-1]
            # MLA should have two intermediate linear nodes:
            # kv_b_proj and q_b_proj, but:
            # - kv_b_proj may be absorbed by the MLA op
            # - q_b_proj is skipped if q_lora_rank is None
            if len(intermediate_lin_nodes) > 2:
                return LayerType.UNKNOWN, 1
            return LayerType.MLA, head_size

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
    """Extract the weight tensor from a node within a GraphModule."""
    weight_nodes = extract_weight_nodes(node)
    if len(weight_nodes.weights) == 0:
        raise ValueError(f"Node {node.name} has no weight")
    return weight_nodes.weights[0].tensor


def draw_graph(gm: GraphModule, filename: str):
    """
    Dump graphmodule to SVG file using PyTorch's built-in drawer.
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    drawer = FxGraphDrawer(gm, filename)
    with open(f"{filename}.svg", "wb") as f:
        f.write(drawer.get_dot_graph().create_svg())


def sync_weight_meta_dtype(gm: GraphModule) -> int:
    """Sync .meta['val'] dtype with actual state_dict dtype for weight nodes.

    During graph tracing, .meta['val'] is set with the dtype at tracing time.
    However, quantization transforms may change the actual weight dtype (e.g., FP16 -> FP8)
    without updating .meta['val']. This causes ONNX export to see incorrect dtypes.

    This function iterates through all get_attr nodes and updates their .meta['val']
    to match the actual tensor dtype from the GraphModule's state.

    Args:
        gm: The GraphModule containing weights to sync.

    Returns:
        Number of weight meta dtypes that were synced.
    """
    num_synced = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            try:
                # Traverse the attribute path to get the actual tensor
                target_path = str(node.target)
                obj = gm
                for attr in target_path.split("."):
                    obj = getattr(obj, attr)

                # Update .meta["val"] if dtype differs
                if isinstance(obj, torch.Tensor) and "val" in node.meta:
                    old_val = node.meta["val"]
                    if hasattr(old_val, "dtype") and old_val.dtype != obj.dtype:
                        # Create new meta tensor with correct dtype
                        node.meta["val"] = torch.empty(
                            obj.shape, dtype=obj.dtype, device=old_val.device
                        )
                        ad_logger.debug(
                            f"Synced {node.target} meta dtype: {old_val.dtype} -> {obj.dtype}"
                        )
                        num_synced += 1
            except (AttributeError, RuntimeError):
                pass  # Skip if attribute not found

    return num_synced
