from collections import defaultdict
from typing import Callable, Dict, Optional

import torch
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ...utils.node_utils import identify_regions_between_residuals, is_linear_op, is_op
from .._graph import canonicalize_graph


def match_moe_pattern(gm: GraphModule) -> GraphModule:
    graph = gm.graph

    ad_logger.info("MoE Pattern Matching")
    ad_logger.debug("Before MoE Pattern Matching: " + str(gm))
    # Preprocessing: Identify boundary nodes (e.g. residual connections) in the graph.
    boundary_nodes = identify_regions_between_residuals(gm)

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        node = start_boundary
        while node != end_boundary:
            node = node.next
            # Step 1: Identify topk nodes
            if not is_op(node, torch.ops.aten.topk):
                continue

            topk_node = node

            # Identify routing_weights and selected_experts from the output of topk node
            topk_output_mapping = {n.args[1]: n for n in topk_node.users}
            routing_weights, selected_experts = topk_output_mapping[0], topk_output_mapping[1]

            # Step 2: Identify normalized_routing_weights
            normalized_routing_weights = _find_normalized_routing_weights(
                routing_weights, boundary=end_boundary
            )

            # Step 3: Trace backwards within the region to find the gate linear node and extract hidden_states
            gate_linear_node = _find_gate_linear_node(topk_node, boundary=start_boundary)
            if gate_linear_node is None:
                continue
            hidden_states = gate_linear_node.args[0]

            # Step 4: Extract expert branch weights from the topk node (restricted by region).
            expert_weights = _extract_expert_weights_from_topk(topk_node, boundary=end_boundary)
            if not expert_weights:
                continue

            # Step 5: Identify the final GEMM output before the residual boundary.
            final_hidden_state_node = _find_final_hidden_state_node(end_boundary)
            if final_hidden_state_node is None:
                continue

            # Step 6: Insert the moe op into the graph.
            ad_logger.debug(
                f"""Found MoE Pattern: between boundary {start_boundary} and {end_boundary}.\n
                Capturing gate node: {gate_linear_node}, input hidden states node: {hidden_states},
                selected_experts node: {selected_experts}, routing_weights node: {normalized_routing_weights},
                expert weights : {expert_weights} """
            )
            with graph.inserting_before(final_hidden_state_node):
                w1_list = expert_weights["w1"]
                w2_list = expert_weights["w2"]
                w3_list = expert_weights["w3"]

                fused_moe_node = graph.call_function(
                    torch.ops.moe.torch_moe,
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

    gm = canonicalize_graph(gm)
    ad_logger.debug("After MoE Pattern Matching: " + str(gm))
    return gm


def fuse_moe(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Scan the FX graph and replace all calls to torch.ops.moe.torch_moe with
    torch.ops.moe.torch_fused_moe.
    """
    ad_logger.info("MoE fusion")
    ad_logger.debug("Before MoE fusion: " + str(gm))
    graph = gm.graph
    fused_key_counter = 0
    for node in list(graph.nodes):
        if not is_op(node, torch.ops.moe.torch_moe):
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
                torch.ops.moe.trtllm_fused_moe,
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

    gm = canonicalize_graph(gm)
    ad_logger.debug("After MoE fusion: " + str(gm))
    return gm


def _find_gate_linear_node(topk_node: Node, boundary: Optional[Node] = None) -> Optional[Node]:
    """
    Traverse backwards from the topk node to identify the gate linear op.
    Uses a BFS over all input nodes (via "all_input_nodes") starting from topk_node,
    returning the first node that qualifies as a linear op. The search stops if it
    reaches the boundary node.
    """
    return _bfs(
        topk_node, lambda node: is_linear_op(node), attr_next="all_input_nodes", boundary=boundary
    )


def _is_routing_weight(node: Node) -> bool:
    routing_ops = {
        torch.ops.aten.index,
        torch.ops.aten.select,
    }
    return is_op(node, routing_ops)


def _bfs(
    node: Node, target: Callable, attr_next: str = "users", boundary: Optional[Node] = None
) -> Node:
    queue = [node]
    visited = set()
    while queue:
        cur_node = queue.pop(0)
        if boundary is not None and cur_node == boundary:
            continue  # Skip the boundary node.
        if target(cur_node):
            return cur_node
        for next_node in getattr(cur_node, attr_next):
            if boundary is not None and next_node == boundary:
                continue  # Do not expand past the boundary.
            if next_node not in visited:
                visited.add(next_node)
                queue.append(next_node)
    raise RuntimeError(f"Could not find node with target condition {target}.")


def _find_normalized_routing_weights(
    routing_weights_node: Node, boundary: Optional[Node] = None
) -> Optional[Node]:
    """
    Starting from the routing_weights node, use BFS to locate a node that:
      1. Is a torch.ops.aten.to op,
      2. And its first argument is produced by a torch.ops.aten.div_ op.

    Returns:
        The node producing the normalized routing weights, or None if not found.
    """
    try:
        norm_node = _bfs(
            routing_weights_node,
            lambda n: is_op(n, torch.ops.aten.to)
            and len(n.args) > 0
            and is_op(n.args[0], torch.ops.aten.div_),
            attr_next="users",
            boundary=boundary,
        )
        return norm_node
    except RuntimeError:
        return None


def _get_linear_weight_from_branch(branch_node: Node):
    def target_fn(n: Node):
        return n.op == "call_function" and n.target == torch.ops.linear.simple.default

    linear_node = _bfs(branch_node, target_fn, attr_next="all_input_nodes")
    return linear_node.args[1] if linear_node is not None else None


def _extract_expert_weights_from_topk(
    topk_node: Node, boundary: Optional[Node] = None
) -> Dict[str, Node]:
    """
    This function locates multiplication nodes where one operand comes from a routing weight slice
    (using a heuristic) and the other is the output of an expert layer. From that expert branch, it
    extracts the weights:
      - w1: from the branch that goes through a silu op,
      - w3: from the direct branch,
      - w2: from the subsequent linear op applying the final transformation.

    Returns:
        A dictionary with keys "w1", "w2", and "w3", each mapping to a list of corresponding weight nodes.
    """
    candidate_expert_branches = []
    expert_weights = defaultdict(list)

    # Step 1. Find multiplication nodes that compute:
    #         expert_layer_output * routing_weights_slice
    # We scan the entire graph and look for aten.mul.Tensor nodes where one operand is a routing weight.
    node = topk_node.next
    while node != boundary and not is_op(node, torch.ops.aten.topk):
        if is_op(node, torch.ops.aten.mul):
            if len(node.args) < 2:
                continue
            arg0, arg1 = node.args[0], node.args[1]
            if _is_routing_weight(arg0) != _is_routing_weight(arg1):
                candidate_expert_branches.append(arg1 if _is_routing_weight(arg0) else arg0)

        node = node.next

    # Step 2. For each candidate expert branch, try to extract the three weights.
    for expert_branch in candidate_expert_branches:
        w1_node = None
        w2_node = None
        w3_node = None

        # The expert branch should eventually compute:
        #   (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        if is_linear_op(expert_branch):
            w2_node = expert_branch.args[1]
            inner = expert_branch.args[0]
        else:
            continue

        # The inner multiplication node should combine two branches:
        # one computed via F.silu(x @ w1.t()) and the other computed via (x @ w3.t()).
        if not is_op(inner, torch.ops.aten.mul):
            continue
        if len(inner.args) < 2:
            continue
        branch_a = inner.args[0]
        branch_b = inner.args[1]
        if is_op(branch_a, torch.ops.aten.silu):
            w1_node = _get_linear_weight_from_branch(branch_a)
            w3_node = _get_linear_weight_from_branch(branch_b)
        elif is_op(branch_b, torch.ops.aten.silu):
            w1_node = _get_linear_weight_from_branch(branch_b)
            w3_node = _get_linear_weight_from_branch(branch_a)
        else:
            continue

        if w1_node is not None and w2_node is not None and w3_node is not None:
            expert_weights["w1"].append(w1_node)
            expert_weights["w2"].append(w2_node)
            expert_weights["w3"].append(w3_node)
        else:
            continue

    return expert_weights


def _find_final_hidden_state_node(residual_node: Node) -> Optional[Node]:
    """
    Identify the final GEMM node (or its output) that computes the hidden_states
    before the residual addition.
    """

    for inp in residual_node.args:
        if isinstance(inp, Node) and is_op(
            inp, {torch.ops.aten.reshape, torch.ops.aten.mm, torch.ops.aten.matmul}
        ):
            return inp.args[0]
    return None


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
        node_to_remove = _bfs(start_boundary, target, attr_next="users", boundary=end_boundary)
        ad_logger.debug(f"Removing In-place Dead Node: {node_to_remove}")
        graph.erase_node(node_to_remove)
        return True
    except RuntimeError:
        return False
