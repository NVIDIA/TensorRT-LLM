import math
from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.utils import ActivationType

from ...custom_ops.quant import TRTLLM_NVFP4_PACKING_FACTOR, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code, get_attr_by_name
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.module import get_submodule_of_param
from ...utils.node_utils import bfs, extract_op_args, identify_regions_between_residuals, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _bmm_moe_gate_up_split_hook(
    state_dict,
    prefix,
    *args,
    source_key: str,
    intermediate_size: int,
    w1_keys: List[str],
    w3_keys: List[str],
):
    """Hook to split gate_up_weight into all per-expert w1 and w3 weights.

    Args:
        source_key: Original stacked weight key (e.g., "gate_up_weight")
        intermediate_size: Intermediate dimension size
        w1_keys: List of target parameter keys for w1 weights
        w3_keys: List of target parameter keys for w3 weights
    """
    source_full_key = prefix + source_key

    if source_full_key in state_dict:
        stacked_tensor = state_dict[source_full_key]
        # Split on last dim: (E, H, 2I) -> 2x (E, H, I)
        w1_stacked, w3_stacked = stacked_tensor.split(intermediate_size, dim=2)
        # Transpose and contiguous in batch, then unbind into views
        w1_experts = w1_stacked.transpose(1, 2).contiguous().unbind(0)
        w3_experts = w3_stacked.transpose(1, 2).contiguous().unbind(0)
        for w1_key, w3_key, w1, w3 in zip(w1_keys, w3_keys, w1_experts, w3_experts):
            state_dict[prefix + w1_key] = w1
            state_dict[prefix + w3_key] = w3


def _bmm_moe_down_split_hook(
    state_dict,
    prefix,
    *args,
    source_key: str,
    w2_keys: List[str],
):
    """Hook to split down_weight into all per-expert w2 weights.

    Args:
        source_key: Original stacked weight key (e.g., "down_weight")
        w2_keys: List of target parameter keys for w2 weights
    """
    source_full_key = prefix + source_key

    if source_full_key in state_dict:
        stacked_tensor = state_dict[source_full_key]
        # Transpose and contiguous in batch, then unbind into views
        w2_experts = stacked_tensor.transpose(1, 2).contiguous().unbind(0)
        for w2_key, w2 in zip(w2_keys, w2_experts):
            state_dict[prefix + w2_key] = w2


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
        if is_op(node, torch.ops.auto_deploy.torch_moe):
            (is_gated_mlp, act_fn) = extract_op_args(node, "is_gated_mlp", "act_fn")

            # Standard MoE with per-expert weight lists
            assert backend != "triton" or not is_gated_mlp, (
                "Triton backend only supports mlp style."
            )
            _process_moe_node(
                gm, graph, node, replacement_op, is_gated_mlp, act_fn, fused_key_counter
            )

            fused_key_counter += 1

            # Delete the unstacked weights immediately to save GPU memory
            # This will happen automatically after the graph is canonicalized,
            # but for large models we'll run out of memory during the transformation itself.
            eliminate_dead_code(gm)
            delete_all_unused_submodules(gm)

    return fused_key_counter


def _process_moe_node(
    gm: GraphModule,
    graph: torch.fx.Graph,
    node: Node,
    replacement_op,
    is_gated_mlp: bool,
    act_fn: ActivationType,
    fused_key_counter: int,
) -> None:
    """Process a single torch_moe node with per-expert weight lists.

    Stacks weight parameters and creates a fused MoE node.
    The kernel applies routing weights to the output.
    """
    (
        hidden_states,
        selected_experts,
        routing_weights,
        w1_list,
        w2_list,
        w3_list,
        apply_routing_on_input,
    ) = extract_op_args(
        node,
        "x",
        "selected_experts",
        "routing_weights",
        "w1_weight",
        "w2_weight",
        "w3_weight",
        "apply_routing_on_input",
    )

    # Stack weights based on MLP style
    if is_gated_mlp:
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
    else:
        # For regular MLP, just stack w1
        fused_w_up_experts = torch.stack([gm.get_parameter(n.target) for n in w1_list], dim=0)
        new_key_w_up = f"fused_moe_w1_stacked_{fused_key_counter}"

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
        # Get weight dtype for casting - fused kernel requires activation dtype to match weight dtype
        weight_dtype = fused_w_up_experts.dtype

        if apply_routing_on_input:
            # Scale input: hidden_states = hidden_states * routing_weights
            hidden_states = graph.call_function(
                torch.ops.aten.mul.Tensor,
                args=(hidden_states, routing_weights),
            )

            # Pass ones to kernel to prevent it from multiplying routing again (already applied)
            routing_weights = graph.call_function(
                torch.ops.aten.ones_like.default,
                args=(routing_weights,),
            )

        # Kernel requires activation dtype to match weight dtype
        hidden_states = graph.call_function(
            torch.ops.aten.to,
            args=(hidden_states, weight_dtype),
        )

        new_node = graph.call_function(
            replacement_op,
            args=(hidden_states, selected_experts, routing_weights, w_up_arg, w_down_arg),
            kwargs={
                "is_gated_mlp": is_gated_mlp,
                "act_fn": act_fn,
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

    node_to_remove, _ = bfs(start_boundary, target, attr_next="users", boundary=end_boundary)
    if node_to_remove:
        graph.erase_node(node_to_remove)
        return True
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
                eliminate_dead_code(gm)

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
        topk_node, _ = bfs(
            routing_weight_node,
            lambda n: is_op(n, torch.ops.aten.topk),
            attr_next="all_input_nodes",
        )
        if topk_node:
            router_logits = topk_node.args[0] if topk_node.args else None
            if not router_logits:
                return None

            # Verify topk is using k=1 (only match single-expert-per-token routing)
            if len(topk_node.args) < 2:
                return None
            k_value = topk_node.args[1]
            if k_value != 1:
                return None
        else:
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

            # Get shapes from node metadata
            if not hasattr(gate_up_weight, "meta") or "val" not in gate_up_weight.meta:
                continue
            if not hasattr(down_weight, "meta") or "val" not in down_weight.meta:
                continue
            gate_up_shape = gate_up_weight.meta["val"].shape
            down_shape = down_weight.meta["val"].shape

            # Only support llama4 shaped weights for now
            if len(gate_up_shape) != len(down_shape) or len(gate_up_shape) != 3:
                continue

            # Llama4 expectation:
            # num_experts = gate_up_shape[0] == down_shape[0]
            # hidden_size = gate_up_shape[1] == down_shape[2]
            # gate_up_shape[2] == 2 * down_shape[1] (intermediate_size)
            if gate_up_shape[0] != down_shape[0]:
                continue

            if gate_up_shape[2] != 2 * down_shape[1]:
                continue

            if gate_up_shape[1] != down_shape[2]:
                continue

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

                # Materialize stacked tensors into per-expert parameters for torch_moe

                # Get the actual tensors from the graph nodes
                if gate_up_weight.op != "get_attr" or down_weight.op != "get_attr":
                    raise RuntimeError(
                        f"Expected get_attr nodes for BMM MoE weights, got {gate_up_weight.op} and {down_weight.op}"
                    )

                gate_up_tensor = gm.get_parameter(gate_up_weight.target)
                down_tensor = gm.get_parameter(down_weight.target)

                # Support only llama4 shaped weights for now

                if gate_up_tensor.shape[2] != 2 * down_tensor.shape[1]:
                    raise RuntimeError(
                        f"Expected gate_up_tensor.shape[2] == 2 * down_tensor.shape[1],"
                        f"got {gate_up_tensor.shape[2]} and {down_tensor.shape[1]}"
                    )

                # Get dimensions
                assert len(gate_up_tensor.shape) == 3, (
                    f"Expected gate_up_tensor.shape to have 3 dimensions, got {len(gate_up_tensor.shape)}"
                )
                assert len(down_tensor.shape) == 3, (
                    f"Expected down_tensor.shape to have 3 dimensions, got {len(down_tensor.shape)}"
                )
                num_experts = gate_up_tensor.shape[0]
                assert num_experts == down_tensor.shape[0], (
                    f"Expected num_experts == down_tensor.shape[0],"
                    f"got {num_experts} and {down_tensor.shape[0]}"
                )
                hidden_size = gate_up_tensor.shape[1]
                assert hidden_size == down_tensor.shape[2], (
                    f"Expected hidden_size == down_tensor.shape[2],"
                    f"got {hidden_size} and {down_tensor.shape[2]}"
                )
                intermediate_size = gate_up_tensor.shape[2] // 2
                assert intermediate_size == down_tensor.shape[1], (
                    f"Expected intermediate_size == down_tensor.shape[1],"
                    f"got {intermediate_size} and {down_tensor.shape[1]}"
                )

                # Store checkpoint keys for hooks
                gate_up_checkpoint_key = str(gate_up_weight.target)
                down_checkpoint_key = str(down_weight.target)

                # Split each stacked tensor into per-expert tensors and register as parameters
                # This creates get_attr nodes that sharding expects
                w1_keys = []
                w2_keys = []
                w3_keys = []

                for expert_idx in range(num_experts):
                    # Register each expert's weight as a separate parameter
                    w1_key = f"bmm_moe_w1_expert_{num_moe_patterns}_{expert_idx}"
                    w2_key = f"bmm_moe_w2_expert_{num_moe_patterns}_{expert_idx}"
                    w3_key = f"bmm_moe_w3_expert_{num_moe_patterns}_{expert_idx}"

                    w1_keys.append(w1_key)
                    w2_keys.append(w2_key)
                    w3_keys.append(w3_key)

                    w1_param = torch.nn.Parameter(
                        gate_up_tensor[expert_idx, :, :intermediate_size].transpose(0, 1)
                    )
                    w2_param = torch.nn.Parameter(down_tensor[expert_idx].transpose(0, 1))
                    w3_param = torch.nn.Parameter(
                        gate_up_tensor[expert_idx, :, intermediate_size:].transpose(0, 1)
                    )

                    gm.register_parameter(w1_key, w1_param)
                    gm.register_parameter(w2_key, w2_param)
                    gm.register_parameter(w3_key, w3_param)

                # Register checkpoint loading hooks - ONE per stacked weight
                # Hook for gate_up_weight: splits into all w1 and w3 expert weights
                gm._register_load_state_dict_pre_hook(
                    partial(
                        _bmm_moe_gate_up_split_hook,
                        source_key=gate_up_checkpoint_key,
                        intermediate_size=intermediate_size,
                        w1_keys=w1_keys,
                        w3_keys=w3_keys,
                    )
                )

                # Hook for down_weight: splits into all w2 expert weights
                gm._register_load_state_dict_pre_hook(
                    partial(
                        _bmm_moe_down_split_hook,
                        source_key=down_checkpoint_key,
                        w2_keys=w2_keys,
                    )
                )

                # Now create get_attr nodes for each expert weight
                # These must be created within the insertion context for proper graph ordering
                insertion_point = graph.find_nodes(op="get_attr")[0]
                with graph.inserting_before(insertion_point):
                    w1_nodes = [graph.get_attr(key) for key in w1_keys]
                    w2_nodes = [graph.get_attr(key) for key in w2_keys]
                    w3_nodes = [graph.get_attr(key) for key in w3_keys]

                with graph.inserting_before(output_node):
                    fused_moe_node = graph.call_function(
                        torch.ops.auto_deploy.torch_moe,
                        args=(
                            input_hidden_states,
                            selected_experts,
                            routing_weights_node,
                            w1_nodes,
                            w2_nodes,
                            w3_nodes,
                        ),
                        kwargs={
                            "is_gated_mlp": True,
                            "apply_routing_on_input": apply_routing_on_input,
                        },
                    )

                # Replace the output node with fused MoE
                output_node.replace_all_uses_with(fused_moe_node)
                graph.erase_node(output_node)

                # Clean up dead nodes
                eliminate_dead_code(gm)

                # Clean up dead inplace nodes in the region
                while _remove_dead_inplace_nodes_in_region(gm.graph, start_boundary, end_boundary):
                    eliminate_dead_code(gm)

                # Delete unused submodules/parameters
                delete_all_unused_submodules(gm)

                num_moe_patterns += 1

        info = TransformInfo(
            skipped=False, num_matches=num_moe_patterns, is_clean=False, has_valid_shapes=False
        )
        return gm, info


def remove_original_experts(gm: GraphModule, weight_lists: List[List[Node]]) -> None:
    """Remove original expert submodules after weights have been stacked.

    This function attempts to free GPU memory by deleting the original expert
    submodules whose weights have been replaced by fused/stacked versions.

    Args:
        gm: The GraphModule containing the expert submodules
        weight_lists: List of weight node lists (e.g., [w1_list, w2_list, w3_list])
    """
    # Flatten all weight lists/
    weight_lists_flat = [w for weights in weight_lists for w in weights]

    for w in weight_lists_flat:
        w_param = get_attr_by_name(gm, w.target)
        if w_param is not None:
            owner_module, owner_module_path, param_name = get_submodule_of_param(gm, w.target)
            owner_param = get_attr_by_name(owner_module, param_name)
            if owner_param is w_param:
                gm.delete_submodule(owner_module_path)
            else:
                # param w is not owned by owner_module, skip
                continue
        else:
            continue


def _stack_fp8_moe_weights(gm: GraphModule, backend: Literal["auto", "trtllm", "triton"]) -> int:
    """
    Stack per-expert FP8 weights and scales by materializing stacked tensors as parameters.
    This is fast because we directly stack the tensor values (not graph nodes).
    Similar to _insert_fused_moe_ops but for quantized MoE.
    """

    def _register_parameter(gm: GraphModule, target, value):
        gm.register_parameter(target, torch.nn.Parameter(value, requires_grad=False))

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

    def _extract_op_args(node):
        return extract_op_args(
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
            "is_gated_mlp",
        )

    def _stack(param_list, dim=0):
        return torch.stack(
            [get_param_or_buffer(element.target) for element in param_list], dim=dim
        ).contiguous()

    def _prepare_args_cutlass_format():
        if is_gated_mlp:
            # For gated MLP, concatenate w1 and w3 as [w3, w1]
            fc1_expert_weights = torch.cat(
                [w3_stacked, w1_stacked], dim=1
            ).contiguous()  # [E, 2*I, H]
            fc1_act_scale = torch.cat(
                [w3_input_scale_stacked, w1_input_scale_stacked], dim=1
            ).contiguous()
        else:
            fc1_expert_weights = w1_stacked
            fc1_act_scale = w1_input_scale_stacked

        fc2_expert_weights = w2_stacked

        # For optimization reasons, we precompute a few additional arguments to the trtllm_quant_fp8_moe_fused op
        # to avoid computing them at runtime.
        fc1_dequant = (w1_weight_scale_stacked * w1_input_scale_stacked[0]).squeeze()
        fc2_act_scale_recip = (1.0 / w2_input_scale_stacked[0]).to(torch.float32)
        fc2_dequant = (w2_weight_scale_stacked * w2_input_scale_stacked[0]).squeeze()

        new_key_fc1_dequant = f"quant_moe_fc1_dequant_stacked_{fused_key_counter}"
        new_key_fc2_act_scale_recip = f"quant_moe_fc2_act_scale_recip_stacked_{fused_key_counter}"
        new_key_fc2_dequant = f"quant_moe_fc2_dequant_stacked_{fused_key_counter}"
        new_key_fc1_expert_weights = f"quant_moe_w3_w1_stacked_{fused_key_counter}"
        new_key_fc2_expert_weights = f"quant_moe_w2_stacked_{fused_key_counter}"
        new_key_fc1_act_scale = f"quant_moe_w3_w1_input_scale_stacked_{fused_key_counter}"

        _register_parameter(gm, new_key_fc1_dequant, fc1_dequant)
        _register_parameter(gm, new_key_fc2_act_scale_recip, fc2_act_scale_recip)
        _register_parameter(gm, new_key_fc2_dequant, fc2_dequant)
        _register_parameter(gm, new_key_fc1_expert_weights, fc1_expert_weights)
        _register_parameter(gm, new_key_fc2_expert_weights, fc2_expert_weights)
        _register_parameter(gm, new_key_fc1_act_scale, fc1_act_scale)

        with graph.inserting_before(node):
            args = (
                hidden_states,
                selected_experts,
                routing_weights,
                graph.get_attr(new_key_fc1_expert_weights),
                graph.get_attr(new_key_fc2_expert_weights),
                graph.get_attr(new_key_fc1_act_scale),
                graph.get_attr(new_key_fc1_dequant),
                graph.get_attr(new_key_fc2_act_scale_recip),
                graph.get_attr(new_key_fc2_dequant),
            )
        return args

    def _prepare_args_triton_format():
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

        _register_parameter(gm, new_key_w1, w1_stacked)
        _register_parameter(gm, new_key_w2, w2_stacked)
        _register_parameter(gm, new_key_w3, w3_stacked)
        _register_parameter(gm, new_key_w1_input_scale, w1_input_scale_stacked)
        _register_parameter(gm, new_key_w2_input_scale, w2_input_scale_stacked)
        _register_parameter(gm, new_key_w3_input_scale, w3_input_scale_stacked)
        _register_parameter(gm, new_key_w1_weight_scale, w1_weight_scale_stacked)
        _register_parameter(gm, new_key_w2_weight_scale, w2_weight_scale_stacked)
        _register_parameter(gm, new_key_w3_weight_scale, w3_weight_scale_stacked)

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

        return args

    fused_key_counter = 0
    graph = gm.graph

    backend = backend.lower()
    replacement_op = {
        "auto": torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
        "trtllm": torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused,
        "triton": torch.ops.auto_deploy.triton_quant_fp8_moe,
    }[backend]

    matched_nodes = [
        node for node in graph.nodes if is_op(node, torch.ops.auto_deploy.torch_quant_fp8_moe)
    ]
    for node in matched_nodes:
        # Extract weight and scale lists from args
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
            is_gated_mlp,
        ) = _extract_op_args(node)

        # Stack the actual tensor values (fast, like in quantize_moe.py)
        w1_stacked = _stack(w1_list, dim=0)
        w2_stacked = _stack(w2_list, dim=0)
        w3_stacked = (
            _stack(w3_list, dim=0)
            if w3_list
            else torch.empty(0, device=w1_stacked.device, dtype=w1_stacked.dtype)
        )

        # Scales are buffers, not parameters
        w1_input_scale_stacked = _stack(w1_input_scale, dim=0)
        w2_input_scale_stacked = _stack(w2_input_scale, dim=0)
        w3_input_scale_stacked = (
            _stack(w3_input_scale, dim=0)
            if w3_input_scale
            else torch.empty(
                0, device=w1_input_scale_stacked.device, dtype=w1_input_scale_stacked.dtype
            )
        )
        assert torch.all(w1_input_scale_stacked[0] == w1_input_scale_stacked), (
            "All w1 scales should have the same value."
        )
        assert torch.all(w2_input_scale_stacked[0] == w2_input_scale_stacked), (
            "All w2 scales should have the same value."
        )

        w1_weight_scale_stacked = _stack(w1_weight_scale, dim=0).to(torch.float32)
        w2_weight_scale_stacked = _stack(w2_weight_scale, dim=0).to(torch.float32)
        w3_weight_scale_stacked = (
            (
                _stack(w3_weight_scale, dim=0)
                if w3_weight_scale
                else torch.empty(
                    0, device=w1_weight_scale_stacked.device, dtype=w1_weight_scale_stacked.dtype
                )
            )
            .to(torch.float32)
            .contiguous()
        )

        if backend == "trtllm":
            args = _prepare_args_cutlass_format()
        else:
            args = _prepare_args_triton_format()

        fused_key_counter += 1

        # Create new node with get_attr for stacked parameters
        with graph.inserting_before(node):
            new_node = graph.call_function(
                replacement_op,
                args,
                kwargs=node.kwargs,
            )

        node.replace_all_uses_with(new_node)
        input_nodes = node.all_input_nodes
        graph.erase_node(node)
        for input_node in input_nodes:
            if input_node.op == "get_attr" and len(input_node.users) == 0:
                graph.erase_node(input_node)
        remove_original_experts(gm, [w1_list, w2_list, w3_list])

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


def _stack_nvfp4_moe_weights(gm: GraphModule) -> int:
    def _register_parameter(gm: GraphModule, target, value):
        gm.register_parameter(target, torch.nn.Parameter(value, requires_grad=False))

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

    def _extract_op_args(node):
        return extract_op_args(
            node,
            "x",
            "selected_experts",
            "routing_weights",
            "w1_weight",
            "w2_weight",
            "w3_weight",
            "w1_input_scale",
            "w2_input_scale",
            "w1_weight_scale",
            "w2_weight_scale",
            "w3_weight_scale",
            "w1_alpha",
            "w2_alpha",
            "is_gated_mlp",
            "act_fn",
        )

    def _stack(param_list, dim=0, device=None, dtype=None):
        if param_list:
            return torch.stack(
                [get_param_or_buffer(element.target) for element in param_list], dim=dim
            ).contiguous()
        else:
            return torch.empty(0, device=device, dtype=dtype)

    def _prepare_args_cutlass_format_nvfp4():
        if is_gated_mlp:
            # For gated MLP, concatenate w1 and w3 as [w3, w1]
            fc1_expert_weights = torch.cat([w3_stacked, w1_stacked], dim=1).contiguous()
            # Expect w3 input scale and alpha to be the same as w1
            fc1_act_scale = w1_input_scale_stacked
            fc1_alpha_stacked = w1_alpha_stacked
            fc1_weight_blockscale_fp8_stacked = torch.cat(
                [w3_weight_blockscale_fp8_stacked, w1_weight_blockscale_fp8_stacked], dim=1
            ).contiguous()
        else:
            fc1_expert_weights = w1_stacked
            fc1_act_scale = w1_input_scale_stacked
            fc1_alpha_stacked = w1_alpha_stacked
            fc1_weight_blockscale_fp8_stacked = w1_weight_blockscale_fp8_stacked

        fc2_expert_weights = w2_stacked
        fc2_act_scale = w2_input_scale_stacked
        fc2_weight_blockscale_fp8_stacked = w2_weight_blockscale_fp8_stacked

        new_key_fc1_expert_weights = f"nvfp4_moe_w3_w1_stacked_{fused_key_counter}"
        new_key_fc2_expert_weights = f"nvfp4_moe_w2_stacked_{fused_key_counter}"

        new_key_fc1_weight_blockscale_fp8 = (
            f"nvfp4_moe_fc1_weight_blockscale_fp8_stacked_{fused_key_counter}"
        )
        new_key_fc2_weight_blockscale_fp8 = (
            f"nvfp4_moe_fc2_weight_blockscale_fp8_stacked_{fused_key_counter}"
        )
        new_key_fc1_act_scale = f"nvfp4_moe_w3_w1_input_scale_stacked_{fused_key_counter}"
        new_key_fc2_act_scale = f"nvfp4_moe_w2_input_scale_stacked_{fused_key_counter}"
        new_key_fc1_alpha = f"nvfp4_moe_w1_alpha_stacked_{fused_key_counter}"
        new_key_fc2_alpha = f"nvfp4_moe_w2_alpha_stacked_{fused_key_counter}"

        # Pad fc1_expert_weights to match the already padded scales
        fc1_pad_size = fc1_weight_blockscale_fp8_stacked.shape[1] - fc1_expert_weights.shape[1]
        if fc1_pad_size > 0:
            fc1_expert_weights = torch.nn.functional.pad(
                fc1_expert_weights, (0, 0, 0, fc1_pad_size), mode="constant", value=0
            )
            # Need to update fc2 scales and weights to match the padded size of fc1,
            # as they share the same intermediate dimension.
            target_intermediate = fc1_weight_blockscale_fp8_stacked.shape[1]
            TRTLLM_NVFP4_SCALING_VECTOR_NUM_ELEMENTS = TRTLLM_NVFP4_SCALING_VECTOR_SIZE
            TRTLLM_NVFP4_SCALING_BYTES_SIZE = (
                TRTLLM_NVFP4_SCALING_VECTOR_NUM_ELEMENTS // TRTLLM_NVFP4_PACKING_FACTOR
            )
            target_n_blocks = target_intermediate // TRTLLM_NVFP4_SCALING_VECTOR_NUM_ELEMENTS
            padded_target_n_blocks = (
                math.ceil(target_n_blocks / TRTLLM_NVFP4_SCALING_BYTES_SIZE)
                * TRTLLM_NVFP4_SCALING_BYTES_SIZE
            )
            fc2_blocks_pad = padded_target_n_blocks - fc2_weight_blockscale_fp8_stacked.shape[2]

            if fc2_blocks_pad > 0:
                # unswizzle fc2 scales
                fc2_blockscale_shape = list(fc2_weight_blockscale_fp8_stacked.shape)
                fc2_blockscale_shape[2] = padded_target_n_blocks
                fc2_weight_blockscale_fp8_stacked = torch.ops.trtllm.block_scale_interleave_reverse(
                    fc2_weight_blockscale_fp8_stacked.view(torch.uint8)
                )
                fc2_weight_blockscale_fp8_stacked = torch.nn.functional.pad(
                    fc2_weight_blockscale_fp8_stacked, (0, fc2_blocks_pad), mode="constant", value=0
                )
                fc2_weight_blockscale_fp8_stacked = (
                    torch.ops.trtllm.block_scale_interleave(fc2_weight_blockscale_fp8_stacked)
                    .view(torch.float8_e4m3fn)
                    .reshape(fc2_blockscale_shape)
                )
            fc2_expert_weights = torch.nn.functional.pad(
                fc2_expert_weights,
                (0, fc1_pad_size // TRTLLM_NVFP4_PACKING_FACTOR, 0, 0),
                mode="constant",
                value=0,
            ).view(torch.uint8)

        # FP4 weights are already packed as uint8, don't convert dtype
        _register_parameter(gm, new_key_fc1_expert_weights, fc1_expert_weights)
        _register_parameter(gm, new_key_fc2_expert_weights, fc2_expert_weights)
        _register_parameter(
            gm, new_key_fc1_weight_blockscale_fp8, fc1_weight_blockscale_fp8_stacked
        )
        _register_parameter(
            gm, new_key_fc2_weight_blockscale_fp8, fc2_weight_blockscale_fp8_stacked
        )
        _register_parameter(gm, new_key_fc1_act_scale, fc1_act_scale)
        _register_parameter(gm, new_key_fc2_act_scale, fc2_act_scale)
        _register_parameter(gm, new_key_fc1_alpha, fc1_alpha_stacked)
        _register_parameter(gm, new_key_fc2_alpha, w2_alpha_stacked)

        with graph.inserting_before(node):
            args = (
                hidden_states,
                selected_experts,
                routing_weights,
                graph.get_attr(new_key_fc1_expert_weights),
                graph.get_attr(new_key_fc2_expert_weights),
                graph.get_attr(new_key_fc1_weight_blockscale_fp8),
                graph.get_attr(new_key_fc2_weight_blockscale_fp8),
                graph.get_attr(new_key_fc1_act_scale),
                graph.get_attr(new_key_fc2_act_scale),
                graph.get_attr(new_key_fc1_alpha),
                graph.get_attr(new_key_fc2_alpha),
            )
        kwargs = {
            "is_gated_mlp": is_gated_mlp,
            "act_fn": act_fn,
        }
        return args, kwargs

    fused_key_counter = 0
    graph = gm.graph

    replacement_op = torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused
    replaced_op = torch.ops.auto_deploy.torch_quant_nvfp4_moe

    matched_nodes = [node for node in graph.nodes if is_op(node, replaced_op)]
    for node in matched_nodes:
        # Extract weight and scale lists from args
        (
            hidden_states,
            selected_experts,
            routing_weights,
            w1_list,
            w2_list,
            w3_list,
            w1_input_scale,
            w2_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
            w1_alpha,
            w2_alpha,
            is_gated_mlp,
            act_fn,
        ) = _extract_op_args(node)

        # Stack the actual tensor values (fast, like in quantize_moe.py)
        w1_stacked = _stack(w1_list, dim=0)
        w2_stacked = _stack(w2_list, dim=0)
        device, dtype = (w1_stacked.device, w1_stacked.dtype)
        w3_stacked = _stack(w3_list, dim=0, device=device, dtype=dtype)

        # Scales are buffers, not parameters
        w1_input_scale_stacked = _stack(w1_input_scale, dim=0)
        w2_input_scale_stacked = _stack(w2_input_scale, dim=0)

        # Use .view() not .to() to reinterpret bytes as float8, not value conversion
        w1_weight_blockscale_fp8_stacked = _stack(w1_weight_scale, dim=0).view(torch.float8_e4m3fn)
        w2_weight_blockscale_fp8_stacked = _stack(w2_weight_scale, dim=0).view(torch.float8_e4m3fn)
        w3_weight_blockscale_fp8_stacked = _stack(
            w3_weight_scale, dim=0, device=device, dtype=dtype
        ).view(torch.float8_e4m3fn)

        w1_alpha_stacked = _stack(w1_alpha, dim=0)
        w2_alpha_stacked = _stack(w2_alpha, dim=0)

        args, kwargs = _prepare_args_cutlass_format_nvfp4()

        fused_key_counter += 1

        # Create new node with get_attr for stacked parameters
        with graph.inserting_before(node):
            new_node = graph.call_function(
                replacement_op,
                args,
                kwargs=kwargs,
            )

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    # Clean up after processing all nodes
    # eliminate_dead_code will remove unused get_attr nodes, then delete_all_unused_submodules
    # will remove the parameters/buffers that are no longer referenced
    eliminate_dead_code(gm)
    delete_all_unused_submodules(gm)
    return fused_key_counter


@TransformRegistry.register("fuse_nvfp4_moe")
class FuseNVFP4Moe(BaseTransform):
    """
    Stack per-expert NVFP4 MoE weights and scales to avoid runtime stacking overhead.
    This runs after weights are loaded, similar to FuseFP8Moe.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _stack_nvfp4_moe_weights(gm)

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info
