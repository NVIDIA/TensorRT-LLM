from collections import defaultdict
from typing import Optional

import torch
from torch.fx import GraphModule, Node

from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import bfs, identify_regions_between_residuals, is_linear_op, is_op
from .._graph import canonicalize_graph


def match_moe_pattern(gm: GraphModule) -> GraphModule:
    graph = gm.graph

    ad_logger.debug("Before MoE Pattern Matching: " + str(gm))
    # Preprocessing: Identify boundary nodes (e.g. residual connections) in the graph.
    boundary_nodes = identify_regions_between_residuals(gm)

    num_moe_patterns = 0

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        # Step 1: Identify Expert Compute pattern
        pattern_input_nodes, pattern_output_nodes, expert_weights = _match_expert_compute_pattern(
            start_boundary, end_boundary
        )
        if not expert_weights:
            continue
        # TODO: naming convention to verify the order of the weight nodes

        # Step 2: Trace upwards to locate normalize_routing_weight and selected_experts:
        arg1_list, arg2_list = _extract_index_branches_from_expert_outputs(pattern_output_nodes)
        normalized_routing_weights = _find_lowest_common_ancessor(arg1_list)
        if not normalized_routing_weights:
            continue

        common_ancessor2 = _find_lowest_common_ancessor(arg2_list)
        if not common_ancessor2:
            continue
        selected_experts = bfs(
            common_ancessor2,
            lambda node: is_op(node, torch.ops.aten.one_hot),
            attr_next="all_input_nodes",
            boundary=start_boundary,
        ).args[0]
        if not selected_experts:
            continue

        # Step 3: Trace upwards to find input node:
        hidden_states = _find_lowest_common_ancessor(pattern_input_nodes)
        if not hidden_states:
            continue

        # Step 4: Find output node with the combine pattern
        final_hidden_state_node = _find_final_hidden_state_node(pattern_output_nodes, end_boundary)
        if final_hidden_state_node is None:
            continue

        # Step 5: Insert the moe op into the graph.
        ad_logger.debug(
            f"""Found MoE Pattern: between boundary {start_boundary} and {end_boundary}.\n
            Capturing input hidden states node: {hidden_states},
            selected_experts node: {selected_experts}, routing_weights node: {normalized_routing_weights},
            expert weights : {expert_weights} """
        )
        with graph.inserting_before(final_hidden_state_node):
            w1_list = expert_weights["w1"]
            w2_list = expert_weights["w2"]
            w3_list = expert_weights["w3"]

            fused_moe_node = graph.call_function(
                torch.ops.auto_deploy.torch_moe,
                args=(
                    hidden_states,
                    selected_experts,
                    normalized_routing_weights,
                    w1_list,
                    w2_list,
                    w3_list,
                ),
            )

        final_hidden_state_node.replace_all_uses_with(fused_moe_node)
        graph.erase_node(final_hidden_state_node)

        while _remove_dead_inplace_nodes_in_region(gm.graph, start_boundary, end_boundary):
            gm.graph.eliminate_dead_code()

        num_moe_patterns += 1

    gm = canonicalize_graph(gm)

    ad_logger.info(f"Found {num_moe_patterns} MoE Patterns")
    ad_logger.debug("After MoE Pattern Matching: " + str(gm))

    return gm


def fuse_moe(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Scan the FX graph and replace all calls to torch.ops.moe.torch_moe with
    torch.ops.auto_deploy.trtllm_moe_fused.
    """
    ad_logger.debug("Before MoE fusion: " + str(gm))

    with cuda_memory_tracker():
        fused_key_counter = _insert_fused_moe_ops(gm)
        if fused_key_counter:
            gm = canonicalize_graph(gm)

    ad_logger.info(f"Found {fused_key_counter} MoE fusions")
    ad_logger.debug("After MoE fusion: " + str(gm))
    return gm


def _insert_fused_moe_ops(gm: GraphModule) -> int:
    fused_key_counter = 0
    graph = gm.graph

    for node in list(graph.nodes):
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            continue

        ad_logger.debug(f"Found MoE op to fuse: {node} with args: {node.args}")
        hidden_states, selected_experts, routing_weights, w1_list, w2_list, w3_list = node.args

        fused_w3_w1_experts = torch.stack(
            [
                torch.cat(
                    [gm.get_parameter(w3_node.target), gm.get_parameter(w1_node.target)], dim=-2
                )
                for w1_node, w3_node in zip(w1_list, w3_list)
            ],
            dim=0,
        )

        fused_w2_experts = torch.stack([gm.get_parameter(n.target) for n in w2_list], dim=0)

        new_key_w3_w1 = f"fused_moe_w3_w1_stacked_{fused_key_counter}"
        new_key_w2 = f"fused_moe_w2_stacked_{fused_key_counter}"
        fused_key_counter += 1
        param_w3_w1 = torch.nn.Parameter(fused_w3_w1_experts)
        param_w2 = torch.nn.Parameter(fused_w2_experts)
        gm.register_parameter(new_key_w3_w1, param_w3_w1)
        gm.register_parameter(new_key_w2, param_w2)

        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.auto_deploy.trtllm_moe_fused,
                args=(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    graph.get_attr(new_key_w3_w1),
                    graph.get_attr(new_key_w2),
                ),
            )

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    return fused_key_counter


def _find_lowest_common_ancessor(nodes: list[Node]) -> Optional[Node]:
    """
    Find the lowest common ancestor for a list of nodes in a torch.fx Graph by following
    each node's primary branch (recursively following the first Node argument).

    It first finds the LCA of the first two nodes and then
    iteratively computes the LCA of the result with the next node, and so on.

    Returns:
        The common ancestor Node if found, otherwise None.
    """
    if not nodes:
        return None

    def get_parent(node: Node) -> Optional[Node]:
        """Return the first Node-valued argument for a given node, or None if not found."""
        for arg in node.args:
            if isinstance(arg, Node):
                return arg
        return None

    def get_depth(node: Node) -> int:
        """
        Recursively compute the depth of the node by following its primary branch.
        Depth is defined as the number of steps to reach a node with no parent.
        """
        parent = get_parent(node)
        if parent is None:
            return 0
        return 1 + get_depth(parent)

    def lca_two(a: Node, b: Node) -> Optional[Node]:
        """
        Find the lowest common ancestor of two nodes by first equalizing their depth
        and then moving upward until a common node is found.
        """
        depth_a = get_depth(a)
        depth_b = get_depth(b)

        # Equalize depths
        while depth_a > depth_b:
            a = get_parent(a)
            depth_a -= 1
        while depth_b > depth_a:
            b = get_parent(b)
            depth_b -= 1

        # Walk upward in lockstep
        while a is not None and b is not None:
            if a is b:
                return a
            a = get_parent(a)
            b = get_parent(b)
        return None

    # Iteratively compute the LCA across all nodes.
    common = nodes[0]
    for node in nodes[1:]:
        common = lca_two(common, node)
        if common is None:
            return None

    return common


def _match_expert_compute_pattern(start_boundary: Node, end_boundary: Node):
    """
    Match the expert compute pattern between the given boundaries.

    The expert compute pattern corresponds to:

        (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()

    For each expert, the function returns:
      - pattern_input_nodes: a list of input nodes (x) used for the expert compute.
      - pattern_output_nodes: a list of final expert output nodes (the linear op with weight w2).
      - expert_weights: a dict with keys "w1", "w2", and "w3" mapping to lists of
        corresponding weight nodes from the w1, w2, and w3 branches.
    """
    pattern_input_nodes, pattern_output_nodes = [], []
    expert_weights = defaultdict(list)

    nodes = list(start_boundary.graph.nodes)
    region_nodes = nodes[nodes.index(start_boundary) + 1 : nodes.index(end_boundary)]

    for node in region_nodes:
        if not is_linear_op(node):
            continue

        final_linear = node
        # Must have at least one argument, and that first argument must be a Node.
        if not final_linear.args or not isinstance(final_linear.args[0], Node):
            continue

        mul_node = final_linear.args[0]
        if not is_op(mul_node, torch.ops.aten.mul) or len(mul_node.args) < 2:
            continue

        arg_a, arg_b = mul_node.args[:2]
        # Pick the silu op from either arg_a or arg_b.
        silu_node = (
            arg_a
            if (isinstance(arg_a, Node) and is_op(arg_a, torch.ops.aten.silu))
            else arg_b
            if (isinstance(arg_b, Node) and is_op(arg_b, torch.ops.aten.silu))
            else None
        )
        if silu_node is None:
            continue

        if not (
            silu_node.args
            and isinstance(silu_node.args[0], Node)
            and is_linear_op(silu_node.args[0])
        ):
            continue
        linear_w1_node = silu_node.args[0]

        # The other branch should be a linear op (w3 branch).
        linear_w3_node = arg_b if arg_a is silu_node else arg_a
        if not (isinstance(linear_w3_node, Node) and is_linear_op(linear_w3_node)):
            continue
        if not (linear_w1_node.args and linear_w3_node.args):
            continue

        input_node_w1 = linear_w1_node.args[0]
        weight_w1 = linear_w1_node.args[1] if len(linear_w1_node.args) > 1 else None
        weight_w3 = linear_w3_node.args[1] if len(linear_w3_node.args) > 1 else None
        weight_w2 = final_linear.args[1] if len(final_linear.args) > 1 else None

        if None in (weight_w1, weight_w3, weight_w2):
            continue

        pattern_input_nodes.append(input_node_w1)
        pattern_output_nodes.append(final_linear)
        expert_weights["w1"].append(weight_w1)
        expert_weights["w3"].append(weight_w3)
        expert_weights["w2"].append(weight_w2)

    return pattern_input_nodes, pattern_output_nodes, expert_weights


def _find_final_hidden_state_node(
    pattern_output_nodes: list[Node], end_boundary: Node
) -> Optional[Node]:
    """
    Identify the final hidden state node corresponding to the combine pattern:

        (expert_output * routing_weight) → index_add_

    For each expert output node (from the expert compute pattern), this function:
      1. Retrieves a multiplication node from its users.
      2. Extracts the second argument from the multiplication node (assumed to be the index node).
      3. Uses a BFS to locate the subsequent index_add_ node (guarded by the end_boundary).

    After collecting all such index_add_ nodes, the final hidden state node is determined
    as the one that is not used by any of the other index_add_ nodes.

    If any required attribute (users or args) is missing during the process or if no valid
    final node is found, the function returns None.
    """

    if not pattern_output_nodes:
        return None

    index_add_nodes = []
    for node in pattern_output_nodes:
        if not node.users:
            return None
        mul_node = next(iter(node.users))
        if not (hasattr(mul_node, "args") and len(mul_node.args) >= 2):
            return None
        index_node = mul_node.args[1]
        index_add_node = bfs(
            index_node, lambda n: is_op(n, torch.ops.aten.index_add_), boundary=end_boundary
        )
        if not index_add_node:
            return None
        index_add_nodes.append(index_add_node)

    # The final node is defined as the index_add_node that is not used by any other index_add_nodes
    return next(
        (
            candidate
            for candidate in index_add_nodes
            if not any(
                candidate in other.args for other in index_add_nodes if candidate is not other
            )
        ),
        None,
    )


def _extract_index_branches_from_expert_outputs(
    pattern_output_nodes: list[Node],
) -> tuple[list[Node], list[Node]]:
    """
    Extract routing and experts branches from expert outputs.

    For each expert output, find its multiplication user. From the
    multiplication node's second argument (an index node),
    extract:
      - The first argument as the routing branch.
      - The second argument (flattened if a list/tuple) as the experts branch.

    Returns:
        A tuple (routing_branches, experts_branches).
    """
    routing_branches, experts_branches = [], []
    for out in pattern_output_nodes:
        mul = next((u for u in out.users if is_op(u, torch.ops.aten.mul)), None)
        if not mul or len(mul.args) < 2:
            continue
        idx_node = mul.args[1]
        if not (isinstance(idx_node, Node) and is_op(idx_node, torch.ops.aten.index)):
            continue
        routing_branches.append(idx_node.args[0])
        experts = idx_node.args[1]
        experts_branches.extend(experts) if isinstance(
            experts, (list, tuple)
        ) else experts_branches.append(experts)
    return routing_branches, experts_branches


def _remove_dead_inplace_nodes_in_region(
    graph: torch.fx.Graph,
    start_boundary: torch.fx.Node,
    end_boundary: torch.fx.Node,
) -> bool:
    """
    Searches (via BFS) for a dead in-place node (index_add_) in the region
    between start_boundary and end_boundary. If one is found, it is removed from the graph.
    Returns True if a node was removed, False otherwise.
    """

    def target(n: torch.fx.Node) -> bool:
        return is_op(n, {torch.ops.aten.index_add_}) and len(n.users) == 0

    try:
        node_to_remove = bfs(start_boundary, target, attr_next="users", boundary=end_boundary)
        ad_logger.debug(f"Removing In-place Dead Node: {node_to_remove}")
        graph.erase_node(node_to_remove)
        return True
    except RuntimeError:
        return False
