"""Common utils for torch fx graph transformation."""

import operator
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx import Graph, GraphModule, Node

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


def extract_weight_node(node: Node) -> int:
    """Extracts the weight node from the given parametrized node"""

    def find_get_attr_node(weight_node: Node) -> Node:
        """Recursively traverse inputs of allowed nodes to find a node with 'get_attr' op."""
        # If node is a get_attr node return node
        # List of nodes allowed in between a get_attr node and the matmul node
        allowed_ops = {
            torch.ops.aten.to.dtype,
            torch.ops.aten.view.default,
        }

        if weight_node.op == "get_attr":
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
        weight_node = node.args[1]
    # for other parametrized nodes, we need to find the weight node
    else:
        weight_nodes = [
            n for n in node.args if isinstance(n, Node) and find_get_attr_node(n) is not None
        ]
        # can be two weights (if bias weight is present)
        weight_node = None
        if weight_nodes:
            weight_node = weight_nodes[0]
        # for modelopt quantized graph, there will be a quantize_op
        _, weight_params, _ = get_quantization_params_from_linear_node(node)
        weight_node = weight_params.input_node if weight_params else weight_node
        assert weight_node is not None, (
            "Expected exactly at least one weight node in the parametrized node"
        )
    return find_get_attr_node(weight_node)


def num_users_of_weight_node(node: Node) -> int:
    """Returns the number of users of the weight node of the given parametrized node."""
    weight_node = extract_weight_node(node)
    return len(weight_node.users) if weight_node is not None else 0


def extract_param_names_from_node(node: Node) -> Tuple[str, Optional[str]]:
    """Extracts the name of the parameter associated with the given parametrized node.

    Args:
        node: node with weight parameters in the graph.
    """
    weight_node = extract_weight_node(node)

    assert weight_node, "Cannot identify weight parameter of linear node."

    # Map arg to named parameter
    weight_name = weight_node.target

    # check for bias
    if is_op(node, torch.ops.aten.bmm):
        bias_node = node.args[2] if len(node.args) > 2 else None
    else:
        weight_nodes = [n for n in node.args if isinstance(n, Node) and n.op == "get_attr"]
        if len(weight_nodes) > 1:
            bias_node = weight_nodes[1]
        else:
            bias_node = None
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
                yield node
    elif isinstance(target, Iterable) and all(isinstance(t, Callable) for t in target):
        for node in nodes:
            for t in target:
                if t(node):
                    yield node
                    break
    else:
        # Handle the case where target or ops contains operations
        operations = ops if ops is not None else target
        for node in nodes:
            if is_op(node, operations):
                yield node


def is_any_lin_op(node: Node) -> bool:
    return is_linear_op(node) or is_fake_quantized_linear_op(node)


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
    """Check if the node is a distributed op."""
    dist_ops = {
        torch.ops.auto_deploy.torch_dist_all_gather,
        torch.ops.auto_deploy.torch_dist_all_reduce,
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


def get_all_layer_subgraphs(gm: GraphModule) -> List[List[Node]]:
    """Get subgraphs corresponding to all consecutive layers (attention, MLP, SSM, MoE) in the graph."""

    assert gm.graph.nodes, "Graph is empty"
    layer_subgraphs = []
    linear_nodes = list(filtered_nodes(gm.graph.nodes, is_linear_op))
    unprocessed_linear_nodes = set(linear_nodes)
    assert len(linear_nodes) > 0, "Could not find any linear nodes in the graph"

    terminating_indices = [-1]

    # for each linear node, find its layer subgraph defined as regions between consecutive linear nodes
    while True:
        layer_subgraph = get_layer_after_linear_node(linear_nodes, terminating_indices)
        unprocessed_linear_nodes -= set(layer_subgraph)
        layer_subgraphs.append(layer_subgraph)
        last_lin_index = terminating_indices[-1] + 1
        if last_lin_index >= len(linear_nodes):
            break

    # unprocessed linear nodes can be "simple sharded".
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

    raise RuntimeError(f"Could not find node with target condition {target}.")


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
    for arg in node.args:
        if isinstance(arg, Node):
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

        attr_next = "args"
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
        attr_next = "args"
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


def get_layer_after_linear_node(
    linear_nodes: List[Node], terminating_indices: List[int]
) -> List[Node]:
    """Get the layer after a linear node."""
    lin_nodes_in_subgraph = []
    start_lin_index = terminating_indices[-1] + 1
    while len(lin_nodes_in_subgraph) != 1:
        forward_subgraph = subgraph(
            sources=[linear_nodes[start_lin_index]], boundary_condition=is_linear_op
        )
        lin_nodes_in_subgraph = list(
            filtered_nodes(forward_subgraph, ops=torch.ops.auto_deploy.torch_linear_simple)
        )
        start_lin_index += 1
    terminating_linear_node = lin_nodes_in_subgraph[0]
    # get all opening linear nodes
    opening_linear_nodes = subgraph(
        sinks=[terminating_linear_node], include=is_linear_op, boundary_condition=is_linear_op
    )
    # opening nodes must succeed last terminating node
    last_terminating_index = terminating_indices[-1]
    opening_linear_nodes = [
        n for n in opening_linear_nodes if linear_nodes.index(n) > last_terminating_index
    ]
    assert linear_nodes[start_lin_index - 1] in opening_linear_nodes, (
        "Linear node not found in opening linear nodes"
    )
    layer_subgraph = opening_linear_nodes + forward_subgraph

    # return the index of the terminating linear node
    if terminating_linear_node == linear_nodes[-1]:
        terminating_index = len(linear_nodes)
    else:
        terminating_index = start_lin_index + len(opening_linear_nodes) - 1
    terminating_indices.append(terminating_index)
    if terminating_index < len(linear_nodes):
        assert linear_nodes[terminating_index] == terminating_linear_node, (
            "Terminating linear node not found"
        )
    # otherwise, we are done. We processed the last linear node.
    return layer_subgraph


def draw_graph(gm: GraphModule, filename: str):
    """
    Dump graphmodule to SVG file using PyTorch's built-in drawer.
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    drawer = FxGraphDrawer(gm, filename)
    with open(f"{filename}.svg", "wb") as f:
        f.write(drawer.get_dot_graph().create_svg())
