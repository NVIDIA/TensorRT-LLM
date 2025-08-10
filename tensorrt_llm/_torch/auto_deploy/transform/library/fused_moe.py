from collections import defaultdict
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.node_utils import bfs, identify_regions_between_residuals, is_linear_op, is_op
from ...utils.quantization_utils import get_scales_and_type_from_node
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _insert_fused_moe_ops(gm: GraphModule) -> int:
    fused_key_counter = 0
    graph = gm.graph

    for node in list(graph.nodes):
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            continue

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
                # TODO(Fridah-nv): torch.ops.auto_deploy.trtllm_moe_fused for quantized models
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


def _extract_linear_parameters(linear_node: Node) -> tuple[Node, torch.Tensor, Optional[dict], str]:
    """
    Given a linear op node, extract the input tensor node, weight tensor,
    any quantization scales (if the op is quantized), and return a weight type.

    For a torch.ops.auto_deploy.torch_linear_simple.default op:
      - Returns (input_node, weight, None, "simple")

    For a torch.ops.auto_deploy.torch_quant_fp8_linear op:
      - Returns (input_node, weight, {"input_scale": input_scale, "weight_scale": weight_scale}, "fp8")
       For a torch.ops.auto_deploy.torch_quant_fp4_linear op:
      - Returns (input_node, weight, {"input_scale": input_scale, "weight_scale": weight_scale, "alpha": alpha}, "fp4")
    """
    input_node = linear_node.args[0]
    if is_op(linear_node, torch.ops.auto_deploy.torch_linear_simple):
        weight = linear_node.args[1]
        return input_node, weight, None, ""
    elif {
        is_op(linear_node, torch.ops.auto_deploy.torch_quant_fp4_linear),
        is_op(linear_node, torch.ops.auto_deploy.torch_quant_fp8_linear),
    }:
        weight = linear_node.args[1]
        scales, quant_type = get_scales_and_type_from_node(linear_node)
        return input_node, weight, scales, quant_type


def _match_expert_compute_pattern(start_boundary: Node, end_boundary: Node):
    """
    Match the expert compute pattern between the given boundaries.

    The expert compute pattern corresponds to:

        (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()

    For each expert, the function extracts the input node from the w1 branch and
    collects the weight parameters from three linear ops (w1, w3, and w2 branches).

    This function supports both:
      - torch.ops.auto_deploy.torch_linear_simple.default ops, and
      - torch.ops.auto_deploy.torch_quant_fp8_linear ops (also extracts quantization scales).
      - torch.ops.auto_deploy.torch_quant_fp4_linear ops (also extracts quantization scales).

    Returns:
        A tuple:
          (pattern_input_nodes, pattern_output_nodes, expert_weights, expert_scales, weight_type)

          - pattern_input_nodes: List of input nodes (x) used for the expert compute.
          - pattern_output_nodes: List of final expert output nodes (the linear op with weight w2).
          - expert_weights: Dict with keys "w1", "w2", "w3" mapping to lists of weight tensors.
          - expert_scales: Dict with keys "w1_input_scale", "w1_weight_scale", etc., containing scale tensors
                           (empty if weight_type is "simple").
          - weight_type: "fp8" if FP8 ops were used, "simple" otherwise.
    """
    pattern_input_nodes, pattern_output_nodes = [], []
    expert_weights = defaultdict(list)
    expert_scales = defaultdict(list)
    weight_type = "simple"  # default

    nodes = list(start_boundary.graph.nodes)
    region_nodes = nodes[nodes.index(start_boundary) + 1 : nodes.index(end_boundary)]

    for node in region_nodes:
        # Accept both simple and quantized linear ops.
        if not is_linear_op(node, include_quantization=True):
            continue

        final_linear = node
        if not final_linear.args or not isinstance(final_linear.args[0], Node):
            continue

        mul_node = final_linear.args[0]
        if not is_op(mul_node, torch.ops.aten.mul) or len(mul_node.args) < 2:
            continue

        arg_a, arg_b = mul_node.args[:2]
        silu_node = (
            arg_a
            if is_op(arg_a, torch.ops.aten.silu)
            else arg_b
            if is_op(arg_b, torch.ops.aten.silu)
            else None
        )
        if silu_node is None:
            continue

        if not (silu_node.args and is_linear_op(silu_node.args[0], include_quantization=True)):
            continue
        linear_w1_node = silu_node.args[0]

        # The other branch should be a linear op (w3 branch).
        linear_w3_node = arg_b if arg_a is silu_node else arg_a
        if not is_linear_op(linear_w3_node, include_quantization=True):
            continue
        if not (linear_w1_node.args and linear_w3_node.args):
            continue

        # Extract parameters from each linear op.
        input_node_w1, weight_w1, quant_params_w1, wt_type_w1 = _extract_linear_parameters(
            linear_w1_node
        )
        _, weight_w3, quant_params_w3, wt_type_w3 = _extract_linear_parameters(linear_w3_node)
        _, weight_w2, quant_params_w2, wt_type_w2 = _extract_linear_parameters(final_linear)

        if None in (weight_w1, weight_w3, weight_w2):
            continue

        # Ensure the weight type is consistent across branches.
        if wt_type_w1 != wt_type_w3 or wt_type_w1 != wt_type_w2:
            continue
        weight_type = wt_type_w1

        pattern_input_nodes.append(input_node_w1)
        pattern_output_nodes.append(final_linear)
        expert_weights["w1"].append(weight_w1)
        expert_weights["w3"].append(weight_w3)
        expert_weights["w2"].append(weight_w2)

        # TODO: sanity check that all experts have same weight type
        if weight_type == "fp8":
            expert_scales["w1_input_scale"].append(quant_params_w1["input_scale"])
            expert_scales["w1_weight_scale"].append(quant_params_w1["weight_scale"])
            expert_scales["w3_input_scale"].append(quant_params_w3["input_scale"])
            expert_scales["w3_weight_scale"].append(quant_params_w3["weight_scale"])
            expert_scales["w2_input_scale"].append(quant_params_w2["input_scale"])
            expert_scales["w2_weight_scale"].append(quant_params_w2["weight_scale"])
        elif weight_type == "fp4":
            expert_scales["w1_input_scale"].append(quant_params_w1["input_scale"])
            expert_scales["w1_weight_scale"].append(quant_params_w1["weight_scale"])
            expert_scales["w1_alpha"].append(quant_params_w1["alpha"])
            expert_scales["w3_input_scale"].append(quant_params_w3["input_scale"])
            expert_scales["w3_weight_scale"].append(quant_params_w3["weight_scale"])
            expert_scales["w3_alpha"].append(quant_params_w3["alpha"])
            expert_scales["w2_input_scale"].append(quant_params_w2["input_scale"])
            expert_scales["w2_weight_scale"].append(quant_params_w2["weight_scale"])
            expert_scales["w2_alpha"].append(quant_params_w2["alpha"])

    return pattern_input_nodes, pattern_output_nodes, expert_weights, expert_scales, weight_type


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
        if not is_op(idx_node, torch.ops.aten.index):
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
        graph.erase_node(node_to_remove)
        return True
    except RuntimeError:
        return False


@TransformRegistry.register("match_moe_pattern")
class MatchMoePattern(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph

        # Preprocessing: Identify boundary nodes (e.g. residual connections) in the graph.
        boundary_nodes = identify_regions_between_residuals(gm)

        num_moe_patterns = 0

        for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
            # Step 1: Identify Expert Compute pattern
            (
                pattern_input_nodes,
                pattern_output_nodes,
                expert_weights,
                expert_scales,
                weight_type,
            ) = _match_expert_compute_pattern(start_boundary, end_boundary)
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
            final_hidden_state_node = _find_final_hidden_state_node(
                pattern_output_nodes, end_boundary
            )
            if final_hidden_state_node is None:
                continue

            # Step 5: Insert the MoE op into the graph.
            with graph.inserting_before(final_hidden_state_node):
                w1_list = expert_weights["w1"]
                w2_list = expert_weights["w2"]
                w3_list = expert_weights["w3"]

                if weight_type == "fp8":
                    fused_moe_node = graph.call_function(
                        torch.ops.auto_deploy.torch_quant_fp8_moe,
                        args=(
                            hidden_states,
                            selected_experts,
                            normalized_routing_weights,
                            w1_list,
                            w2_list,
                            w3_list,
                            expert_scales["w1_input_scale"],
                            expert_scales["w2_input_scale"],
                            expert_scales["w3_input_scale"],
                            expert_scales["w1_weight_scale"],
                            expert_scales["w2_weight_scale"],
                            expert_scales["w3_weight_scale"],
                        ),
                    )
                elif weight_type == "fp4":
                    fused_moe_node = graph.call_function(
                        torch.ops.auto_deploy.torch_quant_fp4_moe,
                        args=(
                            hidden_states,
                            selected_experts,
                            normalized_routing_weights,
                            w1_list,
                            w2_list,
                            w3_list,
                            expert_scales["w1_input_scale"],
                            expert_scales["w2_input_scale"],
                            expert_scales["w3_input_scale"],
                            expert_scales["w1_weight_scale"],
                            expert_scales["w2_weight_scale"],
                            expert_scales["w3_weight_scale"],
                            expert_scales["w1_alpha"],
                            expert_scales["w2_alpha"],
                            expert_scales["w3_alpha"],
                        ),
                    )
                else:
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

        info = TransformInfo(
            skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("fuse_moe")
class FuseMoe(BaseTransform):
    """
    Scan the FX graph and replace all calls to torch.ops.auto_deploy.torch_moe with
    torch.ops.auto_deploy.trtllm_moe_fused.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _insert_fused_moe_ops(gm)

        info = TransformInfo(
            skipped=False, num_matches=fused_key_counter, is_clean=False, has_valid_shapes=False
        )
        return gm, info
