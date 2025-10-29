from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.node_utils import bfs, extract_op_args, identify_regions_between_residuals, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _insert_fused_moe_ops(gm: GraphModule) -> int:
    fused_key_counter = 0
    graph = gm.graph

    for node in graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            continue

        (mlp_style_val,) = extract_op_args(node, "mlp_style")

        hidden_states, selected_experts, routing_weights, w1_list, w2_list, w3_list = (
            extract_op_args(
                node,
                "x",
                "selected_experts",
                "routing_weights",
                "w1_weight",
                "w2_weight",
                "w3_weight",
            )
        )
        if mlp_style_val == "gated_mlp":
            fused_w_up_experts = torch.stack(
                [
                    torch.cat(
                        [gm.get_parameter(w3_node.target), gm.get_parameter(w1_node.target)], dim=-2
                    )
                    for w1_node, w3_node in zip(w1_list, w3_list)
                ],
                dim=0,
            )
            new_key_w_up = f"fused_moe_w3_w1_stacked_{fused_key_counter}"
            # TRTLLM fused MoE op supports gated MLP only.
            replacement_op = torch.ops.auto_deploy.trtllm_moe_fused

        elif mlp_style_val == "mlp":
            fused_w_up_experts = torch.stack([gm.get_parameter(n.target) for n in w1_list], dim=0)
            new_key_w_up = f"fused_moe_w1_stacked_{fused_key_counter}"
            # Triton fused MoE op supports mlp only.
            replacement_op = torch.ops.auto_deploy.triton_moe_fused

        else:
            raise ValueError(f"Unknown mlp_style: {mlp_style_val}")

        fused_w_down_experts = torch.stack([gm.get_parameter(n.target) for n in w2_list], dim=0)

        new_key_w_down = f"fused_moe_w2_stacked_{fused_key_counter}"
        fused_key_counter += 1
        param_w_up = torch.nn.Parameter(fused_w_up_experts)
        param_w_down = torch.nn.Parameter(fused_w_down_experts)
        gm.register_parameter(new_key_w_up, param_w_up)
        gm.register_parameter(new_key_w_down, param_w_down)

        with graph.inserting_before(node):
            new_node = graph.call_function(
                # TODO(Fridah-nv): torch.ops.auto_deploy.trtllm_moe_fused for quantized models
                replacement_op,
                args=(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    graph.get_attr(new_key_w_up),
                    graph.get_attr(new_key_w_down),
                ),
            )

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

        # Delete the unstacked weights immediately to save GPU memory
        # This will happen automatically after the graph is canonicalized, but for large models we'll run out of memory
        # during the transformation itself.
        gm.graph.eliminate_dead_code()
        gm.delete_all_unused_submodules()

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


def _extract_linear_parameters(
    linear_node: Node,
    target_op,
    scale_arg_indices: Dict[str, int],
) -> Tuple[Node, Node, Dict[str, Node]]:
    """
    Extract (input_node, weight_node, scales) from a *specific* linear op variant.

    Returns (None, None, {}) if `linear_node` is not the expected target_op.
    """
    if not is_op(linear_node, target_op):
        return None, None, {}

    # Expected argument layout:
    #   input, weight, (optional bias), then scale args at provided indices.
    if not linear_node.args or not isinstance(linear_node.args[0], Node):
        return None, None, {}
    input_node = linear_node.args[0]
    weight = linear_node.args[1]

    scales: Dict[str, Node] = {}
    for k, idx in scale_arg_indices.items():
        try:
            scales[k] = linear_node.args[idx]
        except Exception:
            return None, None, {}

    return input_node, weight, scales


def _match_expert_compute_pattern(
    start_boundary: Node,
    end_boundary: Node,
    target_op,
    scale_arg_indices: Dict[str, int],
):
    """
    Match the expert compute pattern between the given boundaries.

    The expert compute pattern corresponds to:

        (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()

    For each expert, the function extracts the input node from the w1 branch and
    collects the weight parameters from three linear ops (w1, w3, and w2 branches).

    This function supports both:
      - torch.ops.auto_deploy.torch_linear_simple.default ops, and
      - torch.ops.auto_deploy.torch_quant_fp8_linear ops (also extracts quantization scales).
      - torch.ops.auto_deploy.torch_quant_nvfp4_linear ops (also extracts quantization scales).

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

    nodes = list(start_boundary.graph.nodes)
    region_nodes = nodes[nodes.index(start_boundary) + 1 : nodes.index(end_boundary)]

    for node in region_nodes:
        if not is_op(node, target_op):
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

        if not (silu_node.args and is_op(silu_node.args[0], target_op)):
            continue
        linear_w1_node = silu_node.args[0]

        # The other branch should be a linear op (w3 branch).
        linear_w3_node = arg_b if arg_a is silu_node else arg_a
        if not is_op(linear_w3_node, target_op):
            continue
        if not (linear_w1_node.args and linear_w3_node.args):
            continue

        # Extract parameters from each linear op.
        input_node_w1, weight_w1, s_w1 = _extract_linear_parameters(
            linear_w1_node, target_op, scale_arg_indices
        )
        _, weight_w3, s_w3 = _extract_linear_parameters(
            linear_w3_node, target_op, scale_arg_indices
        )
        _, weight_w2, s_w2 = _extract_linear_parameters(final_linear, target_op, scale_arg_indices)

        if None in (weight_w1, weight_w3, weight_w2):
            continue

        pattern_input_nodes.append(input_node_w1)
        pattern_output_nodes.append(final_linear)
        expert_weights["w1"].append(weight_w1)
        expert_weights["w3"].append(weight_w3)
        expert_weights["w2"].append(weight_w2)

        # Collect scales per-branch with keys "w{1|2|3}_<scale_key>"
        for key, node_scale in s_w1.items():
            expert_scales[f"w1_{key}"].append(node_scale)
        for key, node_scale in s_w3.items():
            expert_scales[f"w3_{key}"].append(node_scale)
        for key, node_scale in s_w2.items():
            expert_scales[f"w2_{key}"].append(node_scale)

    return pattern_input_nodes, pattern_output_nodes, expert_weights, expert_scales


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


class MatchMoePattern(BaseTransform):
    """Base MoE pattern matcher; subclasses specify linear and fused MoE ops and scale layouts."""

    # Subclasses must implement:
    def target_op(self):  # linear op to match
        raise NotImplementedError

    def moe_op(self):  # fused MoE op to insert
        raise NotImplementedError

    def scale_arg_indices(self) -> Dict[str, int]:
        """Map scale names -> arg index in the matched linear op."""
        raise NotImplementedError

    def scale_keys(self) -> List[str]:
        """Order of scale keys to emit into fused MoE op (e.g., ['input_scale','weight_scale',...])."""
        raise NotImplementedError

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

        lin_op = self.target_op()
        scale_idx = self.scale_arg_indices()
        scale_keys = self.scale_keys()
        fused_moe = self.moe_op()

        for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
            # Step 1: Identify Expert Compute pattern
            (
                pattern_input_nodes,
                pattern_output_nodes,
                expert_weights,
                expert_scales,
            ) = _match_expert_compute_pattern(
                start_boundary,
                end_boundary,
                target_op=lin_op,
                scale_arg_indices=scale_idx,
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

                fused_args = [
                    hidden_states,
                    selected_experts,
                    normalized_routing_weights,
                    w1_list,
                    w2_list,
                    w3_list,
                ]

                # Append scales as: for each key -> (w1_key_list, w2_key_list, w3_key_list)
                for key in scale_keys:
                    fused_args.extend(
                        [
                            expert_scales[f"w1_{key}"],
                            expert_scales[f"w2_{key}"],
                            expert_scales[f"w3_{key}"],
                        ]
                    )

                fused_moe_node = graph.call_function(fused_moe, args=tuple(fused_args))

            final_hidden_state_node.replace_all_uses_with(fused_moe_node)
            graph.erase_node(final_hidden_state_node)

            while _remove_dead_inplace_nodes_in_region(gm.graph, start_boundary, end_boundary):
                gm.graph.eliminate_dead_code()

            num_moe_patterns += 1

        info = TransformInfo(
            skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
        )
        return gm, info


@TransformRegistry.register("match_moe_pattern")
class MatchSimpleMoePattern(MatchMoePattern):
    """Match and fuse simple (unquantized) MoE subgraph."""

    def target_op(self):
        return torch.ops.auto_deploy.torch_linear_simple

    def moe_op(self):
        return torch.ops.auto_deploy.torch_moe.default

    def scale_arg_indices(self) -> Dict[str, int]:
        return {}

    def scale_keys(self) -> List[str]:
        return []


@TransformRegistry.register("match_fp8_moe_pattern")
class MatchFP8MoePattern(MatchMoePattern):
    """Match and fuse FP8-quantized MoE subgraph."""

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_fp8_linear

    def moe_op(self):
        return torch.ops.auto_deploy.torch_quant_fp8_moe.default

    def scale_arg_indices(self) -> Dict[str, int]:
        return {"input_scale": 3, "weight_scale": 4}

    def scale_keys(self) -> List[str]:
        return ["input_scale", "weight_scale"]


@TransformRegistry.register("match_nvfp4_moe_pattern")
class MatchNVFP4MoePattern(MatchMoePattern):
    """Match and fuse NVFP4-quantized MoE subgraph."""

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_nvfp4_linear

    def moe_op(self):
        return torch.ops.auto_deploy.torch_quant_nvfp4_moe.default

    def scale_arg_indices(self) -> Dict[str, int]:
        return {"input_scale": 3, "weight_scale": 4, "alpha": 5}

    def scale_keys(self) -> List[str]:
        return ["input_scale", "weight_scale", "alpha"]


def _stack_fp8_moe_weights(gm: GraphModule) -> int:
    """
    Stack per-expert FP8 weights and scales by materializing stacked tensors as parameters.
    This is fast because we directly stack the tensor values (not graph nodes).
    Similar to _insert_fused_moe_ops but for quantized MoE.
    """
    fused_key_counter = 0
    graph = gm.graph

    for node in graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_quant_fp8_moe):
            continue

        # Extract weight and scale lists from args
        try:
            (
                hidden_states,
                selected_experts,
                routing_weights,
                w1_list,
                w2_list,
                w3_list,
                w1_input_scale,
                w2_input_scale,
                w3_input_scale,
                w1_weight_scale,
                w2_weight_scale,
                w3_weight_scale,
            ) = extract_op_args(
                node,
                "x",
                "selected_experts",
                "routing_weights",
                "w1_weight",
                "w2_weight",
                "w3_weight",
                "w1_input_scale",
                "w2_input_scale",
                "w3_input_scale",
                "w1_weight_scale",
                "w2_weight_scale",
                "w3_weight_scale",
            )
        except Exception:
            continue

        # Helper to get parameter or buffer
        def get_param_or_buffer(target):
            """Get parameter or buffer by target name."""
            try:
                return gm.get_parameter(target)
            except AttributeError:
                # It's a buffer, not a parameter
                parts = target.rsplit(".", 1)
                if len(parts) == 2:
                    mod = gm.get_submodule(parts[0])
                    return getattr(mod, parts[1])
                else:
                    return getattr(gm, target)

        # Stack the actual tensor values (fast, like in quantize_moe.py)
        w1_stacked = torch.stack([gm.get_parameter(n.target) for n in w1_list], dim=0)
        w2_stacked = torch.stack([gm.get_parameter(n.target) for n in w2_list], dim=0)
        w3_stacked = (
            torch.stack([gm.get_parameter(n.target) for n in w3_list], dim=0)
            if w3_list
            else torch.empty(0, device=w1_stacked.device, dtype=w1_stacked.dtype)
        )

        # Scales are buffers, not parameters
        w1_input_scale_stacked = torch.stack(
            [get_param_or_buffer(n.target) for n in w1_input_scale], dim=0
        )
        w2_input_scale_stacked = torch.stack(
            [get_param_or_buffer(n.target) for n in w2_input_scale], dim=0
        )
        w3_input_scale_stacked = (
            torch.stack([get_param_or_buffer(n.target) for n in w3_input_scale], dim=0)
            if w3_input_scale
            else torch.empty(
                0, device=w1_input_scale_stacked.device, dtype=w1_input_scale_stacked.dtype
            )
        )

        w1_weight_scale_stacked = torch.stack(
            [get_param_or_buffer(n.target) for n in w1_weight_scale], dim=0
        )
        w2_weight_scale_stacked = torch.stack(
            [get_param_or_buffer(n.target) for n in w2_weight_scale], dim=0
        )
        w3_weight_scale_stacked = (
            torch.stack([get_param_or_buffer(n.target) for n in w3_weight_scale], dim=0)
            if w3_weight_scale
            else torch.empty(
                0, device=w1_weight_scale_stacked.device, dtype=w1_weight_scale_stacked.dtype
            )
        )

        # Register stacked tensors as new parameters
        new_key_w1 = f"quant_moe_w1_stacked_{fused_key_counter}"
        new_key_w2 = f"quant_moe_w2_stacked_{fused_key_counter}"
        new_key_w3 = f"quant_moe_w3_stacked_{fused_key_counter}"
        new_key_w1_input_scale = f"quant_moe_w1_input_scale_stacked_{fused_key_counter}"
        new_key_w2_input_scale = f"quant_moe_w2_input_scale_stacked_{fused_key_counter}"
        new_key_w3_input_scale = f"quant_moe_w3_input_scale_stacked_{fused_key_counter}"
        new_key_w1_weight_scale = f"quant_moe_w1_weight_scale_stacked_{fused_key_counter}"
        new_key_w2_weight_scale = f"quant_moe_w2_weight_scale_stacked_{fused_key_counter}"
        new_key_w3_weight_scale = f"quant_moe_w3_weight_scale_stacked_{fused_key_counter}"

        fused_key_counter += 1

        # Register as parameters (not buffers, to match the original per-expert params)
        gm.register_parameter(new_key_w1, torch.nn.Parameter(w1_stacked, requires_grad=False))
        gm.register_parameter(new_key_w2, torch.nn.Parameter(w2_stacked, requires_grad=False))
        gm.register_parameter(new_key_w3, torch.nn.Parameter(w3_stacked, requires_grad=False))
        gm.register_parameter(
            new_key_w1_input_scale, torch.nn.Parameter(w1_input_scale_stacked, requires_grad=False)
        )
        gm.register_parameter(
            new_key_w2_input_scale, torch.nn.Parameter(w2_input_scale_stacked, requires_grad=False)
        )
        gm.register_parameter(
            new_key_w3_input_scale, torch.nn.Parameter(w3_input_scale_stacked, requires_grad=False)
        )
        gm.register_parameter(
            new_key_w1_weight_scale,
            torch.nn.Parameter(w1_weight_scale_stacked, requires_grad=False),
        )
        gm.register_parameter(
            new_key_w2_weight_scale,
            torch.nn.Parameter(w2_weight_scale_stacked, requires_grad=False),
        )
        gm.register_parameter(
            new_key_w3_weight_scale,
            torch.nn.Parameter(w3_weight_scale_stacked, requires_grad=False),
        )

        # Create new node with get_attr for stacked parameters
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.auto_deploy.triton_quant_fp8_moe,
                args=(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    graph.get_attr(new_key_w1),
                    graph.get_attr(new_key_w2),
                    graph.get_attr(new_key_w3),
                    graph.get_attr(new_key_w1_input_scale),
                    graph.get_attr(new_key_w2_input_scale),
                    graph.get_attr(new_key_w3_input_scale),
                    graph.get_attr(new_key_w1_weight_scale),
                    graph.get_attr(new_key_w2_weight_scale),
                    graph.get_attr(new_key_w3_weight_scale),
                ),
                kwargs=node.kwargs,
            )

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    # Clean up after processing all nodes
    # eliminate_dead_code will remove unused get_attr nodes, then delete_all_unused_submodules
    # will remove the parameters/buffers that are no longer referenced
    gm.graph.eliminate_dead_code()
    gm.delete_all_unused_submodules()

    return fused_key_counter


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


@TransformRegistry.register("fuse_fp8_moe")
class FuseFP8Moe(BaseTransform):
    """
    Stack per-expert FP8 MoE weights and scales to avoid runtime stacking overhead.
    This runs after weights are loaded, similar to FuseMoe for unquantized MoE.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _stack_fp8_moe_weights(gm)

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=False,
            has_valid_shapes=False,
        )
        return gm, info
