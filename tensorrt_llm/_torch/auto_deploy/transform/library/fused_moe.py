from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.node_utils import bfs, extract_op_args, identify_regions_between_residuals, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _insert_fused_moe_ops(gm: GraphModule, backend: Literal["auto", "trtllm", "triton"]) -> int:
    """Replace torch MoE ops with fused backend-specific implementations.

    Handles both:
    - Standard MoE (per-expert weight lists): torch_moe with apply_routing_on_input=False
    - Llama4 MoE (pre-stacked weight tensors): torch_moe with apply_routing_on_input=True

    For Llama4 stacked tensors, applies routing weights to input before the fused kernel.
    """
    fused_key_counter = 0
    graph = gm.graph
    backend = backend.lower()

    # Map backend to fused MoE op (handles both standard and Llama4 stacked tensor patterns)
    replacement_op = {
        "auto": torch.ops.auto_deploy.trtllm_moe_fused,
        "trtllm": torch.ops.auto_deploy.trtllm_moe_fused,
        "triton": torch.ops.auto_deploy.triton_moe_fused,
    }[backend]

    for node in graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_moe):
            continue

        # Detect if this is a stacked MoE (Llama4 pattern) or per-expert list (standard pattern)
        (apply_routing_val, w1_weight_list) = extract_op_args(
            node, "apply_routing_on_input", "w1_weight"
        )

        # Check if it's stacked format: single-element list with 3D tensor
        is_stacked_moe = False
        if apply_routing_val:
            # In FX graphs, w1_weight_list might be a Node representing a list() call
            list_content = None
            if isinstance(w1_weight_list, Node) and w1_weight_list.target is list:
                # Extract from list() call node
                if w1_weight_list.args:
                    list_content = w1_weight_list.args[0]
            elif isinstance(w1_weight_list, (list, tuple)):
                # Direct Python list
                list_content = w1_weight_list

            # Check if it's a single-element list with a 3D tensor
            if list_content is not None and len(list_content) == 1:
                w1_node = list_content[0]
                if isinstance(w1_node, Node) and w1_node.op == "get_attr":
                    try:
                        w1_tensor = gm.get_parameter(w1_node.target)
                        is_stacked_moe = w1_tensor.ndim == 3
                    except (AttributeError, KeyError):
                        pass

        if is_stacked_moe:
            # Stacked MoE (Llama4 pattern): only supports gated MLP
            (act_fn_val,) = extract_op_args(node, "act_fn")
            _process_llama4_stacked_moe_node(
                gm, graph, node, replacement_op, act_fn_val, fused_key_counter
            )
        else:
            # Standard MoE with per-expert weight lists
            (mlp_style_val, act_fn_val) = extract_op_args(node, "mlp_style", "act_fn")
            assert backend != "triton" or mlp_style_val == "mlp", (
                "Triton backend only supports mlp style."
            )
            _process_regular_moe_node(
                gm, graph, node, replacement_op, mlp_style_val, act_fn_val, fused_key_counter
            )

        fused_key_counter += 1

        # Delete the unstacked weights immediately to save GPU memory
        # This will happen automatically after the graph is canonicalized, but for large models we'll run out of memory
        # during the transformation itself.
        gm.graph.eliminate_dead_code()
        gm.delete_all_unused_submodules()

    return fused_key_counter


def _process_regular_moe_node(
    gm: GraphModule,
    graph: torch.fx.Graph,
    node: Node,
    replacement_op,
    mlp_style_val: str,
    act_fn_val: str,
    fused_key_counter: int,
) -> None:
    """Process a single torch_moe node with per-expert weight lists.

    Stacks weight parameters and creates a fused MoE node.
    The kernel applies routing weights to the output.
    """
    hidden_states, selected_experts, routing_weights, w1_list, w2_list, w3_list = extract_op_args(
        node,
        "x",
        "selected_experts",
        "routing_weights",
        "w1_weight",
        "w2_weight",
        "w3_weight",
    )

    # Stack weights based on MLP style
    if mlp_style_val == "gated_mlp":
        # For gated MLP, concatenate w3 and w1 then stack across experts
        fused_w_up_experts = torch.stack(
            [
                torch.cat(
                    [gm.get_parameter(w3_node.target), gm.get_parameter(w1_node.target)],
                    dim=-2,
                )
                for w1_node, w3_node in zip(w1_list, w3_list)
            ],
            dim=0,
        )
        new_key_w_up = f"fused_moe_w3_w1_stacked_{fused_key_counter}"
    elif mlp_style_val == "mlp":
        # For regular MLP, just stack w1
        fused_w_up_experts = torch.stack([gm.get_parameter(n.target) for n in w1_list], dim=0)
        new_key_w_up = f"fused_moe_w1_stacked_{fused_key_counter}"
    else:
        raise ValueError(f"Unknown mlp_style: {mlp_style_val}")

    # Stack w2/down weights
    fused_w_down_experts = torch.stack([gm.get_parameter(n.target) for n in w2_list], dim=0)
    new_key_w_down = f"fused_moe_w2_stacked_{fused_key_counter}"

    # Register the stacked weights as parameters
    param_w_up = torch.nn.Parameter(fused_w_up_experts)
    gm.register_parameter(new_key_w_up, param_w_up)

    param_w_down = torch.nn.Parameter(fused_w_down_experts)
    gm.register_parameter(new_key_w_down, param_w_down)

    # Create fused MoE node - kernel applies routing to output
    with graph.inserting_before(node):
        w_up_arg = graph.get_attr(new_key_w_up)
        w_down_arg = graph.get_attr(new_key_w_down)

        new_node = graph.call_function(
            replacement_op,
            args=(hidden_states, selected_experts, routing_weights, w_up_arg, w_down_arg),
            kwargs={
                "mlp_style": mlp_style_val,
                "act_fn": act_fn_val,
            },
        )

    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


def _process_llama4_stacked_moe_node(
    gm: GraphModule,
    graph: torch.fx.Graph,
    node: Node,
    replacement_op,
    act_fn_val: str,
    fused_key_counter: int,
) -> None:
    """Process a single Llama4 MoE node with pre-stacked weight tensors.

    Only supports gated MLP (SwiGLU-style) architecture.
    Converts Llama4 format weights to TRT-LLM format to standardize all downstream ops.
    Applies routing weights to INPUT before the fused kernel to prevent double multiplication.
    This is the Llama4 pattern where weights are already stacked across experts.
    Result: silu(input * routing_weight) - routing affects activation.
    """
    # torch_moe with stacked format: weights are in single-element lists
    hidden_states, selected_experts, routing_weights, w1_list, w2_list = extract_op_args(
        node,
        "x",
        "selected_experts",
        "routing_weights",
        "w1_weight",
        "w2_weight",
    )

    # Extract the single stacked tensor from each list
    # Handle both FX graph Nodes (list() calls) and direct Python lists
    def extract_from_list_arg(list_arg):
        if isinstance(list_arg, Node) and list_arg.target is list:
            # Extract from list() call node
            return list_arg.args[0][0] if list_arg.args else None
        elif isinstance(list_arg, (list, tuple)):
            # Direct Python list
            return list_arg[0]
        else:
            raise ValueError(f"Unexpected list format: {type(list_arg)}")

    w3_w1_stacked = extract_from_list_arg(w1_list)
    w2_stacked = extract_from_list_arg(w2_list)

    # Convert Llama4 format to TRT-LLM format if needed
    # This standardizes all downstream ops to only handle TRT-LLM format
    if w3_w1_stacked.op == "get_attr" and w2_stacked.op == "get_attr":
        gate_up_weight = gm.get_parameter(w3_w1_stacked.target)
        down_weight = gm.get_parameter(w2_stacked.target)

        # Detect format:
        # - Llama4: gate_up is (E, H, 2*I) and down is (E, I, H)
        # - TRT-LLM: gate_up is (E, 2*I, H) and down is (E, H, I)
        # If both have H in middle dimension, they're Llama4 format
        is_llama4 = gate_up_weight.shape[1] == down_weight.shape[2]

        if is_llama4:
            # Convert Llama4 (E, H, 2*I) -> TRT-LLM (E, 2*I, H)
            gate_up_trtllm = gate_up_weight.transpose(1, 2).contiguous()
            # Convert Llama4 (E, I, H) -> TRT-LLM (E, H, I)
            down_trtllm = down_weight.transpose(1, 2).contiguous()

            # Register converted weights
            new_key_w_up = f"llama4_to_trtllm_w3_w1_{fused_key_counter}"
            new_key_w_down = f"llama4_to_trtllm_w2_{fused_key_counter}"

            gm.register_parameter(new_key_w_up, torch.nn.Parameter(gate_up_trtllm))
            gm.register_parameter(new_key_w_down, torch.nn.Parameter(down_trtllm))

            # Store keys to create get_attr nodes later in insertion context
            needs_get_attr = True
            w_up_key = new_key_w_up
            w_down_key = new_key_w_down
        else:
            # Already TRT-LLM format, use directly
            needs_get_attr = False
            w_up_arg = w3_w1_stacked
            w_down_arg = w2_stacked
    else:
        # Not get_attr nodes (might be intermediate ops), use directly
        needs_get_attr = False
        w_up_arg = w3_w1_stacked
        w_down_arg = w2_stacked

    # Llama4 INPUT-SIDE routing: apply routing to INPUT before kernel
    # Cast BOTH input and routing_weights to weight dtype if needed
    # Critical: BFloat16 * Float32 → Float32 (type promotion) so we cast both to same dtype
    with graph.inserting_before(node):
        # Create get_attr nodes INSIDE insertion context for proper topological ordering
        if needs_get_attr:
            w_up_arg = graph.get_attr(w_up_key)
            w_down_arg = graph.get_attr(w_down_key)

        # Get weight dtype to ensure dtype consistency for Llama4 stacked tensors
        # The fused kernel requires input and weights to have matching dtypes
        weight_dtype = None
        if w_up_arg.op == "get_attr":
            try:
                weight_tensor = gm.get_parameter(w_up_arg.target)
                weight_dtype = weight_tensor.dtype
            except (AttributeError, KeyError):
                pass
        input_to_scale = hidden_states
        routing_to_scale = routing_weights

        if weight_dtype is not None and weight_dtype != torch.float32:
            input_to_scale = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(hidden_states,),
                kwargs={"dtype": weight_dtype},
            )
            routing_to_scale = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(routing_weights,),
                kwargs={"dtype": weight_dtype},
            )

        # Scale input: hidden_states = hidden_states * routing_weights (both same dtype now)
        scaled_input = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(input_to_scale, routing_to_scale),
        )

        # Pass ones to kernel to prevent it from multiplying routing again (already applied)
        ones_node = graph.call_function(
            torch.ops.aten.ones_like.default,
            args=(routing_weights,),
        )

        new_node = graph.call_function(
            replacement_op,
            args=(scaled_input, selected_experts, ones_node, w_up_arg, w_down_arg),
            kwargs={
                "act_fn": act_fn_val,
            },
        )

    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


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
        index_add_node, _ = bfs(
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
        node_to_remove, _ = bfs(start_boundary, target, attr_next="users", boundary=end_boundary)
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
            )[0].args[0]
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
            skipped=False,
            num_matches=num_moe_patterns,
            is_clean=num_moe_patterns == 0,
            has_valid_shapes=num_moe_patterns == 0,
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


class MatchBmmMoePatternConfig(TransformConfig):
    """Configuration for MatchBmmMoePattern transform."""

    pass


@TransformRegistry.register("match_bmm_moe_pattern")
class MatchBmmMoePattern(BaseTransform):
    """Match and fuse Llama4 MoE pattern with pre-stacked weight tensors.

    This pattern uses batch matrix multiply (BMM) operations for parallel expert computation
    with weights already stacked across the expert dimension.

    Only matches patterns where topk uses k=1 (single expert per token).
    """

    config: MatchBmmMoePatternConfig

    @classmethod
    def get_config_class(cls):
        return MatchBmmMoePatternConfig

    @staticmethod
    def _find_gate_up_bmm(final_bmm: Node) -> Optional[Tuple[Node, Node]]:
        """Find the MoE gate_up BMM and chunk node from the final BMM.

        BMM MoE pattern traces back: final_bmm <- mul(up, silu(gate)) <- chunk <- first_bmm (gate_up)

        Returns:
            Tuple of (first_bmm, gate_up_weight) or None if not found
        """
        # Input to final bmm should be mul(up, silu(gate))
        mul_node = final_bmm.args[0]
        if not isinstance(mul_node, Node) or not is_op(mul_node, torch.ops.aten.mul):
            return None

        if not mul_node.args or len(mul_node.args) < 2:
            return None

        # Find silu node (one of the mul inputs)
        arg_a, arg_b = mul_node.args[:2]
        silu_node = (
            arg_a
            if is_op(arg_a, torch.ops.aten.silu)
            else arg_b
            if is_op(arg_b, torch.ops.aten.silu)
            else None
        )
        if silu_node is None:
            return None

        up_node = arg_b if arg_a is silu_node else arg_a
        if not isinstance(up_node, Node):
            return None

        # silu input should be gate from chunk
        if not silu_node.args or not isinstance(silu_node.args[0], Node):
            return None
        gate_node = silu_node.args[0]

        # Both gate and up come from chunk (getitem nodes)
        if gate_node.op != "call_function" or up_node.op != "call_function":
            return None

        # Find the chunk node
        chunk_node = None
        if hasattr(gate_node, "args") and gate_node.args:
            potential_chunk = gate_node.args[0]
            if isinstance(potential_chunk, Node) and is_op(potential_chunk, torch.ops.aten.chunk):
                chunk_node = potential_chunk

        if chunk_node is None or not chunk_node.args or chunk_node.args[1] != 2:
            return None

        # chunk input is the first batched BMM for Llama4 (gate_up_proj)
        first_bmm = chunk_node.args[0]
        if not isinstance(first_bmm, Node) or not is_op(first_bmm, torch.ops.aten.bmm):
            return None

        if not first_bmm.args or len(first_bmm.args) < 2:
            return None

        # Llama4: gate_up_weight is pre-stacked [num_experts, hidden, 2*intermediate]
        gate_up_weight = first_bmm.args[1]
        if not isinstance(gate_up_weight, Node) or gate_up_weight.op != "get_attr":
            return None

        return (first_bmm, gate_up_weight)

    @staticmethod
    def _find_input_and_routing(batched_input: Node) -> Optional[Tuple[Node, Node]]:
        """Find the input hidden states and routing weights from batched input.

        BMM MoE pattern traces back: batched_input <- mul(repeat(input), routing) <- repeat <- input

        Only matches patterns where topk uses k=1 (single expert per token).

        Returns:
            Tuple of (input_hidden_states, topk_node) or None if not found
        """
        # batched_input comes from mul(repeated_input, routing_weights)
        if not batched_input.args or not isinstance(batched_input.args[0], Node):
            return None

        mul_routing = batched_input.args[0]
        if (
            not is_op(mul_routing, torch.ops.aten.mul)
            or not mul_routing.args
            or len(mul_routing.args) < 2
        ):
            return None

        # One arg is repeat (input), other is routing weights
        repeat_node = None
        routing_weight_node = None
        for arg in mul_routing.args[:2]:
            if isinstance(arg, Node):
                if is_op(arg, torch.ops.aten.repeat):
                    repeat_node = arg
                else:
                    routing_weight_node = arg

        if not repeat_node or not routing_weight_node:
            return None

        # Get original input from repeat
        if not repeat_node.args or not isinstance(repeat_node.args[0], Node):
            return None
        input_hidden_states = repeat_node.args[0]

        # Trace back from routing_weight to find topk
        try:
            topk_node, _ = bfs(
                routing_weight_node,
                lambda n: is_op(n, torch.ops.aten.topk),
                attr_next="all_input_nodes",
            )
            router_logits = topk_node.args[0] if topk_node.args else None
            if not router_logits:
                return None

            # Verify topk is using k=1 (only match single-expert-per-token routing)
            if len(topk_node.args) < 2:
                return None
            k_value = topk_node.args[1]
            if k_value != 1:
                return None

        except RuntimeError:
            return None

        return (input_hidden_states, topk_node)

    @staticmethod
    def _find_output_and_routing_flavor(final_bmm: Node) -> Optional[Tuple[Node, bool]]:
        """Find the output node and detect routing application method.

        Llama4 stacked MoE pattern traces forward: final_bmm -> view -> reshape -> sum [-> mul?]

        Returns:
            Tuple of (output_node, apply_routing_on_input) or None if not found
            apply_routing_on_input is True if routing is applied to input, False if applied to output
        """
        # Llama4 pattern: bmm -> view([-1, hidden]) -> reshape([num_experts, -1, hidden]) -> sum(dim=0)
        output_view = None
        for user in final_bmm.users:
            if is_op(user, torch.ops.aten.view):
                output_view = user
                break

        if not output_view:
            return None

        # Find reshape after view
        reshape_node = None
        for user in output_view.users:
            if is_op(user, torch.ops.aten.reshape):
                reshape_node = user
                break

        if not reshape_node:
            return None

        # Find sum after reshape
        sum_node = None
        for user in reshape_node.users:
            if is_op(user, torch.ops.aten.sum):
                sum_node = user
                break

        if not sum_node:
            return None

        # Detect routing application method: check if routing is applied after sum (OUTPUT)
        apply_routing_on_input = True  # Default for Llama4 (routing already applied before BMM)
        output_node = sum_node

        for user in sum_node.users:
            if is_op(user, torch.ops.aten.mul):
                # Found multiplication after sum - routing is applied to OUTPUT
                apply_routing_on_input = False
                output_node = user
                break

        return (output_node, apply_routing_on_input)

    @staticmethod
    def _match_bmm_moe_pattern(
        start_boundary: Node,
        end_boundary: Node,
    ):
        """
        Match the BMM MoE pattern (ONE pattern per layer, not per expert).

        This BMM MoE pattern uses batch matrix multiply (BMM) operations for parallel expert
        computation with pre-stacked weight tensors.

        Only matches patterns where topk uses k=1 (single expert per token).

        Supports TWO routing flavors:

        1. INPUT-SIDE routing (most common):
            - Pattern: mul(input, routing) -> bmm -> silu -> bmm -> sum
            - Result: silu(input * routing_weight) - routing affects activation
            - Routing multiplication happens BEFORE BMM operations

        2. OUTPUT-SIDE routing (alternative):
            - Pattern: bmm -> silu -> bmm -> sum -> mul(output, routing)
            - Result: silu(input) * routing_weight) - routing scales output
            - Routing multiplication happens AFTER sum

        The function auto-detects which flavor is present and returns metadata.

        The BMM MoE pattern corresponds to:
            repeated_input = repeat(input, [num_experts, 1])
            routing_weights = reshape(transpose(sigmoid(scatter(topk(router_logits)))))
            routed_input = mul(repeated_input, routing_weights)  # <-- INPUT-SIDE MULTIPLICATION
            batched_input = view(routed_input, [num_experts, -1, hidden])
            gate_up = bmm(batched_input, gate_up_proj)  # gate_up_proj: pre-stacked [num_experts, hidden, 2*inter]
            gate, up = chunk(gate_up, 2)
            output = bmm(up * silu(gate), down_proj)  # down_proj: pre-stacked [num_experts, inter, hidden]
            final_output = view(output, [-1, hidden])

        Returns:
            List of dicts, one per MoE layer found:
            {
                "input": Node,                      # Unmultiplied input tensor
                "router_logits": Node,              # Router logits tensor
                "gate_up_weight": Node,             # Stacked gate+up weights
                "down_weight": Node,                # Stacked down weights
                "output": Node,                     # Output node to replace (sum or mul)
                "topk": Node,                       # TopK node for routing
                "apply_routing_on_input": bool,     # True if routing on input, False if on output
            }
        """
        moe_layers = []

        nodes = list(start_boundary.graph.nodes)
        region_nodes = nodes[nodes.index(start_boundary) + 1 : nodes.index(end_boundary)]

        for node in region_nodes:
            # Look for the final bmm (down_proj) - this is the BATCHED bmm
            if not is_op(node, torch.ops.aten.bmm):
                continue

            final_bmm = node
            if not final_bmm.args or len(final_bmm.args) < 2:
                continue

            # Step 1: Get down_proj weight
            down_weight = final_bmm.args[1]
            if not isinstance(down_weight, Node) or down_weight.op != "get_attr":
                continue

            # Step 2: Find the first BMM (gate_up) by tracing back through chunk and mul
            result = MatchBmmMoePattern._find_gate_up_bmm(final_bmm)
            if result is None:
                continue
            first_bmm, gate_up_weight = result

            # Step 3: Get batched input and trace back to original input and routing
            batched_input = first_bmm.args[0]
            if not isinstance(batched_input, Node) or not is_op(batched_input, torch.ops.aten.view):
                continue

            result = MatchBmmMoePattern._find_input_and_routing(batched_input)
            if result is None:
                continue
            input_hidden_states, topk_node = result

            # Get router_logits for metadata
            router_logits = topk_node.args[0] if topk_node.args else None
            if not router_logits:
                continue

            # Step 4: Find output node and detect routing application method
            result = MatchBmmMoePattern._find_output_and_routing_flavor(final_bmm)
            if result is None:
                continue
            output_node, apply_routing_on_input = result

            # Step 5: Add the matched layer
            moe_layers.append(
                {
                    "input": input_hidden_states,
                    "router_logits": router_logits,
                    "gate_up_weight": gate_up_weight,
                    "down_weight": down_weight,
                    "output": output_node,
                    "topk": topk_node,
                    "apply_routing_on_input": apply_routing_on_input,
                }
            )

        return moe_layers

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
            # Step 1: Match BMM MoE patterns (one pattern per MoE layer)
            moe_layers = self._match_bmm_moe_pattern(start_boundary, end_boundary)

            if not moe_layers:
                continue

            # Process each MoE layer
            for layer_info in moe_layers:
                input_hidden_states = layer_info["input"]
                gate_up_weight = layer_info["gate_up_weight"]
                down_weight = layer_info["down_weight"]
                output_node = layer_info["output"]
                topk_node = layer_info["topk"]
                # Get routing application method from pattern matcher
                # Default to True (apply on input) which is the common Llama4 pattern
                input_routing = layer_info.get("apply_routing_on_input", True)

                # Step 2: Extract routing information
                # selected_experts: topk indices [tokens, top_k]
                # routing_weights: topk values [tokens, top_k]
                selected_experts = None
                routing_weights_node = None

                for user in topk_node.users:
                    if (
                        user.op == "call_function"
                        and hasattr(user.target, "__name__")
                        and user.target.__name__ == "getitem"
                    ):
                        if len(user.args) >= 2:
                            if user.args[1] == 1:  # indices
                                selected_experts = user
                            elif user.args[1] == 0:  # values
                                # For the fused MoE op, we use topk values directly
                                # (shape: tokens x top_k), NOT the scattered version
                                routing_weights_node = user

                if not selected_experts:
                    continue

                if not routing_weights_node:
                    continue

                # Find scatter and sigmoid nodes - we need the sigmoid output for routing weights!
                # The original pattern: sigmoid(scatter(topk_values)) normalizes routing to [0,1]
                # The fused op must use the same normalized values, not raw topk logits.
                scatter_node = None
                sigmoid_node = None

                # Find scatter operation
                for user in routing_weights_node.users:
                    if is_op(user, {torch.ops.aten.scatter, torch.ops.aten.scatter_}):
                        scatter_node = user
                        # Check if sigmoid exists after scatter (may have to.dtype in between)
                        current = user
                        for _ in range(2):  # Allow up to 2 hops
                            for next_user in current.users:
                                if is_op(next_user, torch.ops.aten.sigmoid):
                                    sigmoid_node = next_user
                                    break
                                elif is_op(next_user, torch.ops.aten.to):
                                    current = next_user
                                    break
                            if sigmoid_node:
                                break
                        break

                if not scatter_node:
                    continue

                if not sigmoid_node:
                    continue

                # Extract normalized routing weights from the sigmoid output
                # The sigmoid output has shape [tokens, num_experts] (scattered)
                # We need to gather it back to [tokens, top_k] using selected_experts indices
                # This gives us sigmoid(scatter(topk_values)) in the compact [tokens, top_k] format
                graph = gm.graph
                with graph.inserting_after(sigmoid_node):
                    # Create gather operation: routing_weights_normalized = sigmoid_output.gather(1, selected_experts)
                    routing_weights_normalized = graph.call_function(
                        torch.ops.aten.gather,
                        args=(sigmoid_node, 1, selected_experts),
                    )

                # Use the normalized routing weights instead of raw topk values
                routing_weights_node = routing_weights_normalized

                # Step 4: Apply MoE fusion
                # If input_routing is True: kernel applies routing to input
                # If input_routing is False: kernel applies routing to output
                apply_routing_on_input = input_routing

                # Wrap stacked tensors in single-element lists for torch_moe unified interface
                with graph.inserting_before(output_node):
                    # Create list nodes for stacked weights
                    w1_list_node = graph.call_function(
                        list,
                        args=([gate_up_weight],),
                    )
                    w2_list_node = graph.call_function(
                        list,
                        args=([down_weight],),
                    )
                    w3_list_node = graph.call_function(
                        list,
                        args=([],),  # Empty list for stacked gated MLP
                    )

                    fused_moe_node = graph.call_function(
                        torch.ops.auto_deploy.torch_moe,
                        args=(
                            input_hidden_states,
                            selected_experts,
                            routing_weights_node,
                            w1_list_node,
                            w2_list_node,
                            w3_list_node,
                        ),
                        kwargs={
                            "mlp_style": "gated_mlp",
                            "apply_routing_on_input": apply_routing_on_input,
                        },
                    )

                # Replace the output node with fused MoE
                output_node.replace_all_uses_with(fused_moe_node)
                graph.erase_node(output_node)

                # Clean up dead nodes
                gm.graph.eliminate_dead_code()

                # Clean up dead inplace nodes in the region
                while _remove_dead_inplace_nodes_in_region(gm.graph, start_boundary, end_boundary):
                    gm.graph.eliminate_dead_code()

                # Delete unused submodules/parameters
                gm.delete_all_unused_submodules()

                num_moe_patterns += 1

        info = TransformInfo(
            skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
        )
        return gm, info


def _stack_fp8_moe_weights(gm: GraphModule, backend: Literal["auto", "trtllm", "triton"]) -> int:
    """
    Stack per-expert FP8 weights and scales by materializing stacked tensors as parameters.
    This is fast because we directly stack the tensor values (not graph nodes).
    Similar to _insert_fused_moe_ops but for quantized MoE.
    """
    fused_key_counter = 0
    graph = gm.graph

    backend = backend.lower()
    replacement_op = {
        "auto": torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
        "trtllm": torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
        "triton": torch.ops.auto_deploy.triton_quant_fp8_moe,
    }[backend]

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

        w1_weight_scale_stacked = (
            torch.stack([get_param_or_buffer(n.target) for n in w1_weight_scale], dim=0)
            .to(torch.float32)
            .contiguous()
        )
        w2_weight_scale_stacked = (
            torch.stack([get_param_or_buffer(n.target) for n in w2_weight_scale], dim=0)
            .to(torch.float32)
            .contiguous()
        )
        w3_weight_scale_stacked = (
            (
                torch.stack([get_param_or_buffer(n.target) for n in w3_weight_scale], dim=0)
                if w3_weight_scale
                else torch.empty(
                    0, device=w1_weight_scale_stacked.device, dtype=w1_weight_scale_stacked.dtype
                )
            )
            .to(torch.float32)
            .contiguous()
        )
        assert torch.all(w1_input_scale_stacked[0] == w1_input_scale_stacked), (
            "All w1 scales should have the same value."
        )
        assert torch.all(w2_input_scale_stacked[0] == w2_input_scale_stacked), (
            "All w2 scales should have the same value."
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

        additional_nodes = []
        if backend == "trtllm":
            # For optimization reasons, we precompute a few additional arguments to the trtllm_quant_fp8_moe_fused op
            # to avoid computing them at runtime.
            gemm1_dequant = (w1_weight_scale_stacked * w1_input_scale_stacked[0]).squeeze()
            gemm2_act_quant = (1.0 / w2_input_scale_stacked[0]).to(torch.float32)
            gemm2_dequant = (w2_weight_scale_stacked * w2_input_scale_stacked[0]).squeeze()

            new_key_gemm1_dequant = f"quant_moe_gemm1_dequant_stacked_{fused_key_counter}"
            new_key_gemm2_act_quant = f"quant_moe_gemm2_act_quant_stacked_{fused_key_counter}"
            new_key_gemm2_dequant = f"quant_moe_gemm2_dequant_stacked_{fused_key_counter}"
            gm.register_parameter(
                new_key_gemm1_dequant,
                torch.nn.Parameter(gemm1_dequant, requires_grad=False),
            )
            gm.register_parameter(
                new_key_gemm2_act_quant,
                torch.nn.Parameter(gemm2_act_quant, requires_grad=False),
            )
            gm.register_parameter(
                new_key_gemm2_dequant,
                torch.nn.Parameter(gemm2_dequant, requires_grad=False),
            )
            additional_nodes = [
                new_key_gemm1_dequant,
                new_key_gemm2_act_quant,
                new_key_gemm2_dequant,
            ]

        # Create new node with get_attr for stacked parameters
        with graph.inserting_before(node):
            args = (
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
            )
            additional_args = (graph.get_attr(node) for node in additional_nodes)
            new_node = graph.call_function(
                replacement_op,
                args=(*args, *additional_args),
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


class FuseMoeConfig(TransformConfig):
    """Configuration for MoE fusion transform."""

    backend: str = Field(
        default="auto",
        description="Backend to use for MoE computation ('auto', 'trtllm' or 'triton'. default: 'auto').",
    )


@TransformRegistry.register("fuse_moe")
class FuseMoe(BaseTransform):
    """
    Scan the FX graph and replace all calls to torch.ops.auto_deploy.torch_moe with
    torch.ops.auto_deploy.trtllm_moe_fused.
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseMoeConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _insert_fused_moe_ops(gm, backend=self.config.backend)

        info = TransformInfo(
            skipped=False,
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info


class FuseFP8MoeConfig(TransformConfig):
    """Configuration for FP8 MoE fusion transform."""

    backend: str = Field(
        default="auto",
        description="Backend to use for FP8 MoE computation ('auto', 'trtllm' or 'triton'. default: 'auto').",
    )


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
            fused_key_counter = _stack_fp8_moe_weights(gm, backend=self.config.backend)

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info
