import math
from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.quantization.utils.fp4_utils import (
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
)

from ...custom_ops.quantization.quant import (
    TRTLLM_NVFP4_PACKING_FACTOR,
    TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import (
    del_attr_by_name,
    delete_all_unused_submodules,
    eliminate_dead_code,
    get_attr_by_name,
)
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
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


def _fused_moe_gate_up_split_hook(
    state_dict,
    prefix,
    *args,
    source_key: str,
    intermediate_size: int,
    w1_keys: List[str],
    w3_keys: List[str],
):
    """Fallback hook to split fused gate_up tensor into per-expert w1/w3 tensors.

    This hook is intentionally ordering-agnostic w.r.t. checkpoint families:
    it assumes `source_key` has already been normalized to runtime/operator layout
    by model-specific hooks (if needed), then performs structural splitting only.
    """
    source_full_key = prefix + source_key
    if source_full_key not in state_dict:
        return

    stacked_tensor = state_dict[source_full_key]
    # Runtime/operator layout: [w3, w1] along dim=1 for shape (E, 2I, H)
    w3_stacked, w1_stacked = stacked_tensor.split(intermediate_size, dim=1)
    w3_experts = w3_stacked.contiguous().unbind(0)
    w1_experts = w1_stacked.contiguous().unbind(0)

    # Don't overwrite keys already populated by model-owned hooks.
    for w1_key, w1 in zip(w1_keys, w1_experts):
        dst = prefix + w1_key
        if dst not in state_dict:
            state_dict[dst] = w1
    for w3_key, w3 in zip(w3_keys, w3_experts):
        dst = prefix + w3_key
        if dst not in state_dict:
            state_dict[dst] = w3


def _fused_moe_down_split_hook(
    state_dict,
    prefix,
    *args,
    source_key: str,
    w2_keys: List[str],
):
    """Fallback hook to split fused down tensor into per-expert w2 tensors."""
    source_full_key = prefix + source_key
    if source_full_key not in state_dict:
        return

    stacked_tensor = state_dict[source_full_key]
    w2_experts = stacked_tensor.contiguous().unbind(0)
    for w2_key, w2 in zip(w2_keys, w2_experts):
        dst = prefix + w2_key
        if dst not in state_dict:
            state_dict[dst] = w2


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


def _replace_torch_moe_fused_ops(
    gm: GraphModule, backend: Literal["auto", "trtllm", "triton"]
) -> int:
    """Replace torch_moe_fused (pre-stacked ref impl) with backend-specific fused MoE.

    Unlike _insert_fused_moe_ops which handles per-expert weight lists,
    this handles models that already use pre-stacked 3D weight tensors directly
    (e.g. Qwen 3.5 MoE). No weight restructuring is needed -- it is a 1:1 op swap.
    """
    count = 0
    graph = gm.graph
    replacement_op = {
        "auto": torch.ops.auto_deploy.trtllm_moe_fused,
        "trtllm": torch.ops.auto_deploy.trtllm_moe_fused,
        "triton": torch.ops.auto_deploy.triton_moe_fused,
    }[backend.lower()]

    for node in graph.nodes:
        if is_op(node, torch.ops.auto_deploy.torch_moe_fused):
            with graph.inserting_before(node):
                new_node = graph.call_function(
                    replacement_op,
                    args=node.args,
                    kwargs={"is_gated_mlp": True, "act_fn": int(ActivationType.Silu)},
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            count += 1

    return count


def _split_torch_moe_fused_to_expert_lists(gm: GraphModule) -> int:
    """Rewrite torch_moe_fused nodes into torch_moe with per-expert weight lists.

    This runs before sharding so EP can operate on existing torch_moe list-based path.
    The transform keeps checkpoint-order policy out of transform logic by:
      - materializing expert-list parameters structurally, and
      - registering fallback structural split hooks on the owner module.
    Model-specific hooks can normalize checkpoint layout before these fallback hooks run.
    """
    graph = gm.graph
    converted = 0

    for node in list(graph.nodes):
        if not is_op(node, torch.ops.auto_deploy.torch_moe_fused):
            continue

        hidden_states, selected_experts, routing_weights, w3_w1_node, w2_node = node.args
        if not (
            isinstance(w3_w1_node, Node)
            and isinstance(w2_node, Node)
            and w3_w1_node.op == "get_attr"
            and w2_node.op == "get_attr"
        ):
            continue

        w3_w1_target = str(w3_w1_node.target)
        w2_target = str(w2_node.target)
        w3_w1_tensor = gm.get_parameter(w3_w1_target)
        w2_tensor = gm.get_parameter(w2_target)

        if len(w3_w1_tensor.shape) != 3 or len(w2_tensor.shape) != 3:
            continue

        num_experts = w3_w1_tensor.shape[0]
        intermediate_size = w3_w1_tensor.shape[1] // 2
        if w3_w1_tensor.shape[1] != 2 * intermediate_size:
            continue
        if w2_tensor.shape[0] != num_experts:
            continue

        owner_module, owner_path, source_name = get_submodule_of_param(gm, w3_w1_target)
        owner_module_w2, owner_path_w2, source_name_w2 = get_submodule_of_param(gm, w2_target)
        if owner_module is not owner_module_w2 or owner_path != owner_path_w2:
            continue

        # Materialize per-expert params under the owner module.
        w1_full_keys: list[str] = []
        w2_full_keys: list[str] = []
        w3_full_keys: list[str] = []
        w1_local_names: list[str] = []
        w2_local_names: list[str] = []
        w3_local_names: list[str] = []

        for expert_idx in range(num_experts):
            # Keep stable names so model-side hooks can target them.
            w1_name = f"w1_expert_{expert_idx}"
            w2_name = f"w2_expert_{expert_idx}"
            w3_name = f"w3_expert_{expert_idx}"

            if not hasattr(owner_module, w1_name):
                owner_module.register_parameter(
                    w1_name,
                    torch.nn.Parameter(w3_w1_tensor[expert_idx, intermediate_size:, :]),
                )
            if not hasattr(owner_module, w2_name):
                owner_module.register_parameter(
                    w2_name,
                    torch.nn.Parameter(w2_tensor[expert_idx, :, :]),
                )
            if not hasattr(owner_module, w3_name):
                owner_module.register_parameter(
                    w3_name,
                    torch.nn.Parameter(w3_w1_tensor[expert_idx, :intermediate_size, :]),
                )

            w1_local_names.append(w1_name)
            w2_local_names.append(w2_name)
            w3_local_names.append(w3_name)

            prefix = f"{owner_path}." if owner_path else ""
            w1_full_keys.append(prefix + w1_name)
            w2_full_keys.append(prefix + w2_name)
            w3_full_keys.append(prefix + w3_name)

        # Fallback structural split hooks (model hooks can pre-normalize layout).
        owner_module._register_load_state_dict_pre_hook(
            partial(
                _fused_moe_gate_up_split_hook,
                source_key=source_name,
                intermediate_size=intermediate_size,
                w1_keys=w1_local_names,
                w3_keys=w3_local_names,
            )
        )
        owner_module._register_load_state_dict_pre_hook(
            partial(
                _fused_moe_down_split_hook,
                source_key=source_name_w2,
                w2_keys=w2_local_names,
            )
        )

        with graph.inserting_before(node):
            w1_nodes = [graph.get_attr(k) for k in w1_full_keys]
            w2_nodes = [graph.get_attr(k) for k in w2_full_keys]
            w3_nodes = [graph.get_attr(k) for k in w3_full_keys]
            new_node = graph.call_function(
                torch.ops.auto_deploy.torch_moe,
                args=(
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    w1_nodes,
                    w2_nodes,
                    w3_nodes,
                ),
                kwargs={"is_gated_mlp": True, "act_fn": int(ActivationType.Silu)},
            )

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        converted += 1

        # Clean dead graph/state then explicitly remove unused fused params.
        eliminate_dead_code(gm)
        delete_all_unused_submodules(gm)

        for old_target in (w3_w1_target, w2_target):
            still_used = any(
                n.op == "get_attr" and str(n.target) == old_target for n in gm.graph.nodes
            )
            if not still_used:
                try:
                    del_attr_by_name(gm, old_target)
                except AttributeError:
                    pass

    return converted


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
        mapping_config,
        max_num_tokens,
    ) = extract_op_args(
        node,
        "x",
        "selected_experts",
        "routing_weights",
        "w1_weight",
        "w2_weight",
        "w3_weight",
        "apply_routing_on_input",
        "mapping_config",
        "max_num_tokens",
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

        # Build kwargs for new node - preserve all kwargs from original node
        # and override with known values
        fused_kwargs = dict(node.kwargs) if node.kwargs else {}
        fused_kwargs.update(
            {
                "is_gated_mlp": is_gated_mlp,
                "act_fn": act_fn,
                "mapping_config": mapping_config,
                "max_num_tokens": max_num_tokens,
                "apply_routing_on_input": apply_routing_on_input,
            }
        )

        new_node = graph.call_function(
            replacement_op,
            args=(hidden_states, selected_experts, routing_weights, w_up_arg, w_down_arg),
            kwargs=fused_kwargs,
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


def _stack_fp8_moe_weights(
    gm: GraphModule,
    backend: Literal["auto", "trtllm", "triton"],
    allow_different_input_scales: bool = False,
) -> int:
    """
    Stack per-expert FP8 weights and scales by materializing stacked tensors as parameters.
    This is fast because we directly stack the tensor values (not graph nodes).
    Similar to _insert_fused_moe_ops but for quantized MoE.

    Args:
        gm: The GraphModule to transform.
        backend: Backend to use ('auto', 'trtllm', or 'triton').
        allow_different_input_scales: If False (default), assert that all experts have identical
            input scales and fail if not. If True, allow different scales (use max for quantization).
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
        # We use max scale to handle different input scales per expert (if enabled).
        fc1_act_scale = fc1_act_scale.max()
        fc2_act_scale = w2_input_scale_stacked.max()
        fc1_dequant = (w1_weight_scale_stacked * w1_input_scale_stacked.max()).squeeze()
        fc2_act_scale_recip = (1.0 / fc2_act_scale).to(torch.float32)
        fc2_dequant = (w2_weight_scale_stacked * fc2_act_scale).squeeze()

        new_key_fc1_expert_weights = f"quant_moe_w3_w1_stacked_{fused_key_counter}"
        new_key_fc2_expert_weights = f"quant_moe_w2_stacked_{fused_key_counter}"
        new_key_fc1_act_scale = f"quant_moe_fc1_act_scale_{fused_key_counter}"
        new_key_fc1_dequant = f"quant_moe_fc1_dequant_stacked_{fused_key_counter}"
        new_key_fc2_act_scale_recip = f"quant_moe_fc2_act_scale_recip_stacked_{fused_key_counter}"
        new_key_fc2_dequant = f"quant_moe_fc2_dequant_stacked_{fused_key_counter}"

        _register_parameter(gm, new_key_fc1_expert_weights, fc1_expert_weights)
        _register_parameter(gm, new_key_fc2_expert_weights, fc2_expert_weights)
        _register_parameter(gm, new_key_fc1_act_scale, fc1_act_scale)
        _register_parameter(gm, new_key_fc1_dequant, fc1_dequant)
        _register_parameter(gm, new_key_fc2_act_scale_recip, fc2_act_scale_recip)
        _register_parameter(gm, new_key_fc2_dequant, fc2_dequant)

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
        new_key_w1_weight_scale = f"quant_moe_w1_weight_scale_stacked_{fused_key_counter}"
        new_key_w2_weight_scale = f"quant_moe_w2_weight_scale_stacked_{fused_key_counter}"
        new_key_w3_weight_scale = f"quant_moe_w3_weight_scale_stacked_{fused_key_counter}"
        w1_input_scale = w1_input_scale_stacked.max().reshape(1)
        w2_input_scale = w2_input_scale_stacked.max().reshape(1)
        # w3_input_scale: use max of w3 scales if present, else use empty tensor
        w3_input_scale = (
            w3_input_scale_stacked.max().reshape(1)
            if w3_input_scale_stacked.numel() > 0
            else torch.empty(1, device=w1_input_scale.device, dtype=w1_input_scale.dtype)
        )
        new_key_w1_input_scale = f"quant_moe_w1_input_scale_{fused_key_counter}"
        new_key_w2_input_scale = f"quant_moe_w2_input_scale_{fused_key_counter}"
        new_key_w3_input_scale = f"quant_moe_w3_input_scale_{fused_key_counter}"

        _register_parameter(gm, new_key_w1, w1_stacked)
        _register_parameter(gm, new_key_w2, w2_stacked)
        _register_parameter(gm, new_key_w3, w3_stacked)
        _register_parameter(gm, new_key_w1_input_scale, w1_input_scale)
        _register_parameter(gm, new_key_w2_input_scale, w2_input_scale)
        _register_parameter(gm, new_key_w3_input_scale, w3_input_scale)
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
        # Check if input scales are identical across experts
        w1_input_scales_identical = torch.all(
            w1_input_scale_stacked[0] == w1_input_scale_stacked
        ).item()
        w2_input_scales_identical = torch.all(
            w2_input_scale_stacked[0] == w2_input_scale_stacked
        ).item()

        if not w1_input_scales_identical or not w2_input_scales_identical:
            if not allow_different_input_scales:
                # Fail with assertion
                assert w1_input_scales_identical, (
                    "All w1 input scales should have the same value. "
                    "Set allow_different_input_scales=True to allow different scales (uses max)."
                )
                assert w2_input_scales_identical, (
                    "All w2 input scales should have the same value. "
                    "Set allow_different_input_scales=True to allow different scales (uses max)."
                )
            # Issue warning once and continue - max() will be used
            ad_logger.warning_once(
                "FP8 MoE: Input scales differ across experts. Using max(input_scale) for quantization. "
                "This may impact accuracy if scales differ significantly.",
                key="fp8_moe_different_input_scales",
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
    Scan the FX graph and replace all calls to torch.ops.auto_deploy.torch_moe and
    torch.ops.auto_deploy.torch_moe_fused with torch.ops.auto_deploy.trtllm_moe_fused.
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
            fused_key_counter += _replace_torch_moe_fused_ops(gm, backend=self.config.backend)

        info = TransformInfo(
            skipped=False,
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info


class SplitMoeFusedForShardingConfig(TransformConfig):
    """Configuration for converting torch_moe_fused to torch_moe pre-sharding."""

    pass


@TransformRegistry.register("split_moe_fused_for_sharding")
class SplitMoeFusedForSharding(BaseTransform):
    """Convert torch_moe_fused nodes to list-based torch_moe nodes before sharding."""

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return SplitMoeFusedForShardingConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = _split_torch_moe_fused_to_expert_lists(gm)
        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


class FuseFP8MoeConfig(TransformConfig):
    """Configuration for FP8 MoE fusion transform."""

    backend: str = Field(
        default="auto",
        description="Backend to use for FP8 MoE computation ('auto', 'trtllm' or 'triton'. default: 'auto').",
    )
    allow_different_input_scales: bool = Field(
        default=False,
        description=(
            "If False (default), assert that all experts have identical input scales and fail if not. "
            "If True, allow different per-expert input scales by using max(input_scale) for quantization. "
            "This matches TRT-LLM manual backend behavior but may impact accuracy if scales differ significantly."
        ),
    )


@TransformRegistry.register("fuse_fp8_moe")
class FuseFP8Moe(BaseTransform):
    """
    Stack per-expert FP8 MoE weights and scales to avoid runtime stacking overhead.
    This runs after weights are loaded, similar to FuseMoe for unquantized MoE.
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseFP8MoeConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _stack_fp8_moe_weights(
                gm,
                backend=self.config.backend,
                allow_different_input_scales=self.config.allow_different_input_scales,
            )

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info


def _stack_nvfp4_moe_weights(
    gm: GraphModule,
    allow_different_input_scales: bool = False,
) -> int:
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
            fc1_weight_blockscale_fp8_stacked = torch.cat(
                [w3_weight_blockscale_fp8_stacked, w1_weight_blockscale_fp8_stacked], dim=1
            ).contiguous()

            # Check if all w1 and w3 input scales are identical across experts
            all_scales_equal = (
                torch.all(w1_input_scale_stacked == w1_input_scale_stacked[0])
                and torch.all(w3_input_scale_stacked == w3_input_scale_stacked[0])
                and torch.all(w1_input_scale_stacked == w3_input_scale_stacked)
            )

            if all_scales_equal:
                # All scales are identical, no need for min() or alpha recomputation
                fc1_act_scale = w1_input_scale_stacked[0]
                fc1_alpha_stacked = w1_alpha_stacked
            else:
                if not allow_different_input_scales:
                    assert False, (
                        "FC1 input scales differ across experts (w1 and/or w3). "
                        "Set allow_different_input_scales=True to allow different scales (uses min)."
                    )
                # Issue warning once and continue - min() will be used
                ad_logger.warning_once(
                    "NVFP4 MoE: Input scales differ across experts. Using min(input_scale) for "
                    "FC1 quantization and recomputing alpha. This may impact accuracy if scales "
                    "differ significantly.",
                    key="nvfp4_moe_different_input_scales",
                )
                # Scales differ across experts - use global min scale and recompute alpha.
                # Use min() because NVFP4 scales are in kernel format (2688/amax):
                # smaller scale = larger amax = larger dynamic range.
                fc1_act_scale = torch.minimum(
                    w1_input_scale_stacked.min(), w3_input_scale_stacked.min()
                )
                # Recompute alpha using global input scale instead of per-expert input scale.
                # Formula: new_alpha = old_alpha * per_expert_input_scale / global_input_scale
                # This ensures alpha is consistent with the global fc1_act_scale used by the kernel.
                fc1_alpha_stacked = w1_alpha_stacked * w1_input_scale_stacked / fc1_act_scale
        else:
            fc1_expert_weights = w1_stacked
            fc1_weight_blockscale_fp8_stacked = w1_weight_blockscale_fp8_stacked

            # Check if all w1 input scales are identical across experts
            all_scales_equal = torch.all(w1_input_scale_stacked == w1_input_scale_stacked[0])

            if all_scales_equal:
                # All scales are identical, no need for min() or alpha recomputation
                fc1_act_scale = w1_input_scale_stacked[0]
                fc1_alpha_stacked = w1_alpha_stacked
            else:
                if not allow_different_input_scales:
                    assert False, (
                        "FC1 input scales differ across experts (w1). "
                        "Set allow_different_input_scales=True to allow different scales (uses min)."
                    )
                # Issue warning once and continue - min() will be used
                ad_logger.warning_once(
                    "NVFP4 MoE: Input scales differ across experts. Using min(input_scale) for "
                    "FC1 quantization and recomputing alpha. This may impact accuracy if scales "
                    "differ significantly.",
                    key="nvfp4_moe_different_input_scales",
                )
                # Scales differ across experts - use global min scale and recompute alpha
                fc1_act_scale = w1_input_scale_stacked.min()
                fc1_alpha_stacked = w1_alpha_stacked * w1_input_scale_stacked / fc1_act_scale

        fc2_expert_weights = w2_stacked
        # Keep fc2_act_scale per-expert (no global scale aggregation for fc2).
        # The kernel supports per-expert scales for fc2, and intermediate activations
        # naturally have different dynamic ranges per expert.
        fc2_act_scale = w2_input_scale_stacked
        # No alpha recomputation needed since fc2 uses per-expert input scales.
        fc2_alpha_stacked = w2_alpha_stacked
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
        _register_parameter(gm, new_key_fc2_alpha, fc2_alpha_stacked)

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
        # Preserve original kwargs (mapping_config, max_num_tokens) and override with known values
        kwargs = dict(node.kwargs) if node.kwargs else {}
        kwargs.update(
            {
                "is_gated_mlp": is_gated_mlp,
                "act_fn": act_fn,
            }
        )
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
            w3_input_scale,
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
        w3_input_scale_stacked = _stack(
            w3_input_scale,
            dim=0,
            device=w1_input_scale_stacked.device,
            dtype=w1_input_scale_stacked.dtype,
        )

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


class FuseNVFP4MoeConfig(TransformConfig):
    """Configuration for NVFP4 MoE fusion transform."""

    allow_different_input_scales: bool = Field(
        default=False,
        description=(
            "If False (default), assert that all experts have identical input scales and fail if not. "
            "If True, allow different per-expert input scales by using min(input_scale) for quantization. "
            "Note: NVFP4 uses min() (not max like FP8) because scales are in kernel format (2688/amax): "
            "smaller scale = larger amax = larger dynamic range. "
            "This may impact accuracy if scales differ significantly."
        ),
    )


@TransformRegistry.register("fuse_nvfp4_moe")
class FuseNVFP4Moe(BaseTransform):
    """
    Stack per-expert NVFP4 MoE weights and scales to avoid runtime stacking overhead.
    This runs after weights are loaded, similar to FuseFP8Moe.
    """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNVFP4MoeConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        with cuda_memory_tracker():
            fused_key_counter = _stack_nvfp4_moe_weights(
                gm,
                allow_different_input_scales=self.config.allow_different_input_scales,
            )

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info


def _shuffle_weight_for_trtllm_gen(
    weight: torch.Tensor,
    is_gated: bool,
    epilogue_tile_m: int = 128,
) -> torch.Tensor:
    """Apply weight shuffling required by TRTLLM-Gen kernel.

    Args:
        weight: Weight tensor to shuffle [E, N, K] or [N, K]
        is_gated: Whether this is a gated weight (w3_w1) - applies gated act reordering
        epilogue_tile_m: Tile size for epilogue (default 128)

    Returns:
        Shuffled weight tensor
    """
    # Handle batched (expert) weights
    if weight.ndim == 3:
        num_experts = weight.shape[0]
        shuffled_weights = []
        for e in range(num_experts):
            shuffled_weights.append(
                _shuffle_weight_for_trtllm_gen(weight[e], is_gated, epilogue_tile_m)
            )
        return torch.stack(shuffled_weights, dim=0)

    # Single weight tensor [N, K]
    device = weight.device

    # Get permute indices
    if is_gated:
        # For gated weights, apply gated act reordering first
        permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(weight)
    else:
        permute0 = torch.arange(weight.shape[0], dtype=torch.long)

    permute1 = get_shuffle_matrix_a_row_indices(weight, epilogue_tile_m=epilogue_tile_m)

    # Chain permutations
    combined_permute = permute0[permute1].to(device)

    # Apply shuffle
    shuffled_weight = torch.ops.trtllm.shuffle_matrix(weight, combined_permute)

    return shuffled_weight


def _stack_nvfp4_trtllm_gen_moe_weights(gm: GraphModule) -> int:
    """Stack and shuffle NVFP4 MoE weights for TRTLLM-Gen kernel (SM100+).

    This function transforms torch_quant_nvfp4_moe nodes to use the optimized
    fp4_block_scale_moe_runner kernel by:
    1. Stacking per-expert weights
    2. Applying weight shuffling required by TRTLLM-Gen
    3. Padding weights to required alignment
    """

    def _register_parameter(gm: GraphModule, target, value):
        gm.register_parameter(target, torch.nn.Parameter(value, requires_grad=False))

    def get_param_or_buffer(target):
        """Get parameter or buffer by target name."""
        try:
            return gm.get_parameter(target)
        except AttributeError:
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
            "w1_alpha",
            "w2_alpha",
            "w3_alpha",
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

    def _round_up(x, alignment):
        return (x + alignment - 1) // alignment * alignment

    fused_key_counter = 0
    graph = gm.graph

    replacement_op = torch.ops.auto_deploy.trtllm_nvfp4_trtllm_gen_moe_fused
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
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
            w1_alpha,
            w2_alpha,
            w3_alpha,
            is_gated_mlp,
            act_fn,
        ) = _extract_op_args(node)

        # Stack the actual tensor values
        w1_stacked = _stack(w1_list, dim=0)
        w2_stacked = _stack(w2_list, dim=0)
        device, dtype = (w1_stacked.device, w1_stacked.dtype)
        w3_stacked = _stack(w3_list, dim=0, device=device, dtype=dtype)

        # Get dimensions
        num_experts = w1_stacked.shape[0]
        # FP4 weights are packed, so actual dimensions are *2
        hidden_size = w1_stacked.shape[-1] * 2  # K dimension

        # Compute padded sizes (alignment 32 or 256 for large hidden sizes)
        # This is matching the manual path's "no padding for hidden_size <= 1024",
        # and should utilize memory more efficiently for large hidden sizes.
        weight_alignment = 256 if hidden_size > 1024 and hidden_size % 256 != 0 else 32
        input_hidden_alignment = weight_alignment

        # Concatenate w3 and w1 for gated MLP (w3_w1 stacked)
        if is_gated_mlp:
            # [E, I, H/2] + [E, I, H/2] -> [E, 2I, H/2]
            fc1_w_stacked = torch.cat([w3_stacked, w1_stacked], dim=1).contiguous()
        else:
            fc1_w_stacked = w1_stacked

        fc2_w_stacked = w2_stacked

        # Pad weights if necessary (fc1_w / fc2_w: weight tensor dims, GEMM N=output K=input)
        fc1_w_n_dim = fc1_w_stacked.shape[1]
        fc1_w_k_dim = fc1_w_stacked.shape[2] * 2  # Unpack
        fc1_w_n_padded = _round_up(fc1_w_n_dim, weight_alignment)
        fc1_w_k_padded = _round_up(fc1_w_k_dim, input_hidden_alignment)

        if fc1_w_n_padded > fc1_w_n_dim or fc1_w_k_padded > fc1_w_k_dim:
            ad_logger.debug(f"NVFP4 MoE (TRTLLM-Gen) fc1 weights before pad: {fc1_w_stacked.shape}")
            fc1_w_stacked = torch.nn.functional.pad(
                fc1_w_stacked,
                (0, (fc1_w_k_padded - fc1_w_k_dim) // 2, 0, fc1_w_n_padded - fc1_w_n_dim),
            )
            ad_logger.debug(f"NVFP4 MoE (TRTLLM-Gen) fc1 weights after pad:  {fc1_w_stacked.shape}")

        fc2_w_n_dim = fc2_w_stacked.shape[1]
        fc2_w_k_dim = fc2_w_stacked.shape[2] * 2
        fc2_w_n_padded = _round_up(fc2_w_n_dim, weight_alignment)
        fc2_w_k_padded = _round_up(fc2_w_k_dim, weight_alignment)

        if fc2_w_n_padded > fc2_w_n_dim or fc2_w_k_padded > fc2_w_k_dim:
            ad_logger.debug(f"NVFP4 MoE (TRTLLM-Gen) fc2 weights before pad: {fc2_w_stacked.shape}")
            fc2_w_stacked = torch.nn.functional.pad(
                fc2_w_stacked,
                (0, (fc2_w_k_padded - fc2_w_k_dim) // 2, 0, fc2_w_n_padded - fc2_w_n_dim),
            )
            ad_logger.debug(f"NVFP4 MoE (TRTLLM-Gen) fc2 weights after pad:  {fc2_w_stacked.shape}")

        # Apply weight shuffling to match TRTLLM-Gen layout
        fc1_shuffled = _shuffle_weight_for_trtllm_gen(fc1_w_stacked, is_gated=is_gated_mlp)
        fc2_shuffled = _shuffle_weight_for_trtllm_gen(fc2_w_stacked, is_gated=False)

        # Stack scales
        w1_input_scale_stacked = _stack(w1_input_scale, dim=0)
        w2_input_scale_stacked = _stack(w2_input_scale, dim=0)

        w1_weight_blockscale_stacked = _stack(w1_weight_scale, dim=0).view(torch.float8_e4m3fn)
        w2_weight_blockscale_stacked = _stack(w2_weight_scale, dim=0).view(torch.float8_e4m3fn)
        w3_weight_blockscale_stacked = _stack(
            w3_weight_scale, dim=0, device=device, dtype=dtype
        ).view(torch.float8_e4m3fn)

        w1_alpha_stacked = _stack(w1_alpha, dim=0)
        w2_alpha_stacked = _stack(w2_alpha, dim=0)

        # trtllmgen fp4_block_scale_moe kernel (and C++ fused_moe when it uses it) requires
        # gemm1_weights_scale dim 2 == hidden_size/16. If this checkpoint uses a different
        # scale layout (e.g. scaling_vector_size=32), skip fusing into trtllm_nvfp4_trtllm_gen_moe_fused
        # and leave as torch_quant_nvfp4_moe (per-expert linear path).
        expected_scale_k = hidden_size // 16
        if w1_weight_blockscale_stacked.shape[2] != expected_scale_k:
            continue
        else:
            ad_logger.debug_once(
                "NVFP4 MoE (TRTLLM-Gen): gemm1_weights_scale dim 2 (%s) == hidden_size/16 (%s). "
                "Fusing into trtllm_nvfp4_trtllm_gen_moe_fused. hidden_size: %s"
                % (w1_weight_blockscale_stacked.shape[2], expected_scale_k, hidden_size),
                key="nvfp4_moe_trtllm_gen_scale_layout_fusion",
            )

        # Concatenate scales for gated MLP
        if is_gated_mlp:
            fc1_weight_blockscale = torch.cat(
                [w3_weight_blockscale_stacked, w1_weight_blockscale_stacked], dim=1
            ).contiguous()
        else:
            fc1_weight_blockscale = w1_weight_blockscale_stacked

        fc2_weight_blockscale = w2_weight_blockscale_stacked

        # Kernel expects gemm1_weights_scale dim 1 == (gated ? 2*intermediate_size : intermediate_size),
        # dim 2 == args.hidden_size/16. gemm2_weights_scale dim 1 == args.hidden_size, dim 2 == intermediate_size/16.
        # Pad scales to match padded weights so the kernel shape checks pass.
        expected_fc1_scale_n = 2 * (fc1_w_n_padded // 2) if is_gated_mlp else fc1_w_n_padded
        if fc1_weight_blockscale.shape[1] < expected_fc1_scale_n:
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc1_scale before pad dim1: {fc1_weight_blockscale.shape}"
            )
            pad_n = expected_fc1_scale_n - fc1_weight_blockscale.shape[1]
            fc1_weight_blockscale = torch.nn.functional.pad(
                fc1_weight_blockscale, (0, 0, 0, pad_n), value=0.0
            )
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc1_scale after pad dim1:  {fc1_weight_blockscale.shape}"
            )
        expected_scale_k_padded = fc1_w_k_padded // 16
        if fc1_weight_blockscale.shape[2] < expected_scale_k_padded:
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc1_scale before pad dim2: {fc1_weight_blockscale.shape}"
            )
            pad_k = expected_scale_k_padded - fc1_weight_blockscale.shape[2]
            fc1_weight_blockscale = torch.nn.functional.pad(
                fc1_weight_blockscale, (0, pad_k), value=0.0
            )
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc1_scale after pad dim2:  {fc1_weight_blockscale.shape}"
            )
        intermediate_size_for_kernel = fc1_w_n_padded // 2 if is_gated_mlp else fc1_w_n_padded
        if fc2_weight_blockscale.shape[2] < (intermediate_size_for_kernel // 16):
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc2_scale before pad dim2: {fc2_weight_blockscale.shape}"
            )
            pad_k_fc2 = (intermediate_size_for_kernel // 16) - fc2_weight_blockscale.shape[2]
            fc2_weight_blockscale = torch.nn.functional.pad(
                fc2_weight_blockscale, (0, pad_k_fc2), value=0.0
            )
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc2_scale after pad dim2:  {fc2_weight_blockscale.shape}"
            )
        if fc2_weight_blockscale.shape[1] < fc1_w_k_padded:
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc2_scale before pad dim1: {fc2_weight_blockscale.shape}"
            )
            pad_h = fc1_w_k_padded - fc2_weight_blockscale.shape[1]
            fc2_weight_blockscale = torch.nn.functional.pad(
                fc2_weight_blockscale, (0, 0, 0, pad_h), value=0.0
            )
            ad_logger.debug(
                f"NVFP4 MoE (TRTLLM-Gen) fc2_scale after pad dim1:  {fc2_weight_blockscale.shape}"
            )

        # Shuffle and interleave scales to match TRTLLM-Gen layout (same as module path).
        # get_shuffle_matrix_sf_a_row_indices requires M % 128 == 0 and K % 4 == 0.
        epilogue_tile_m = 128
        num_elts_per_sf = 16

        _sf_permute_cache = {}

        def _shuffle_scale_stack(scale_3d: torch.Tensor, is_gated: bool) -> torch.Tensor:
            E, M, K = scale_3d.shape
            if M % 128 != 0 or K % 4 != 0:
                return scale_3d
            scale_3d_uint8 = scale_3d.view(torch.uint8)
            cache_key = (M, K, is_gated)
            if cache_key not in _sf_permute_cache:
                slice_2d = scale_3d_uint8[0]
                if is_gated:
                    permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(slice_2d.float()).to(
                        slice_2d.device
                    )
                else:
                    permute0 = torch.arange(M, dtype=torch.long, device=slice_2d.device)
                permute1 = get_shuffle_matrix_sf_a_row_indices(
                    slice_2d, epilogue_tile_m=epilogue_tile_m, num_elts_per_sf=num_elts_per_sf
                )
                if permute1.device != slice_2d.device:
                    permute1 = permute1.to(slice_2d.device)
                _sf_permute_cache[cache_key] = permute0[permute1]
            combined = _sf_permute_cache[cache_key]
            out_list = []
            for e in range(E):
                shuffled = torch.ops.trtllm.shuffle_matrix(scale_3d_uint8[e], combined)
                # block_scale_interleave returns 1D (rows_padded*cols_padded); reshape to (M, K)
                # so kernel receives 3D [E, M, K] with dim 2 == hidden_size/16.
                interleaved = torch.ops.trtllm.block_scale_interleave(shuffled)
                interleaved_2d = interleaved.reshape(M, K)
                out_list.append(interleaved_2d.view(torch.float8_e4m3fn))
            out = torch.stack(out_list, dim=0)
            return out

        # _shuffle_scale_stack returns input unchanged if M % 128 != 0 or K % 4 != 0
        fc1_weight_blockscale = _shuffle_scale_stack(fc1_weight_blockscale, is_gated=is_gated_mlp)
        fc2_weight_blockscale = _shuffle_scale_stack(fc2_weight_blockscale, is_gated=False)

        # Compute top_k from routing_weights shape
        # routing_weights has shape [num_tokens, top_k]
        if hasattr(routing_weights, "meta") and "val" in routing_weights.meta:
            top_k = routing_weights.meta["val"].shape[-1]
        else:
            # Default to 8 if we can't determine
            ad_logger.info(
                "NVFP4 MoE (TRTLLM-Gen) could not determine top_k, defaulting to top_k=8"
            )
            top_k = 8

        # Register new parameters
        new_key_fc1 = f"trtllm_gen_nvfp4_moe_fc1_stacked_{fused_key_counter}"
        new_key_fc2 = f"trtllm_gen_nvfp4_moe_fc2_stacked_{fused_key_counter}"
        new_key_fc1_scale = f"trtllm_gen_nvfp4_moe_fc1_scale_{fused_key_counter}"
        new_key_fc2_scale = f"trtllm_gen_nvfp4_moe_fc2_scale_{fused_key_counter}"
        new_key_fc1_act_scale = f"trtllm_gen_nvfp4_moe_fc1_act_scale_{fused_key_counter}"
        new_key_fc1_scale_c = f"trtllm_gen_nvfp4_moe_fc1_scale_c_{fused_key_counter}"
        new_key_fc1_alpha = f"trtllm_gen_nvfp4_moe_fc1_alpha_{fused_key_counter}"
        new_key_fc2_alpha = f"trtllm_gen_nvfp4_moe_fc2_alpha_{fused_key_counter}"

        _register_parameter(gm, new_key_fc1, fc1_shuffled)
        _register_parameter(gm, new_key_fc2, fc2_shuffled)
        _register_parameter(gm, new_key_fc1_scale, fc1_weight_blockscale)
        _register_parameter(gm, new_key_fc2_scale, fc2_weight_blockscale)
        _register_parameter(gm, new_key_fc1_act_scale, w1_input_scale_stacked)
        # Compute fc1_scale_c properly based on activation type
        # For SwiGLU: fc1_scale_c = fc2_input_scale * fc31_alpha
        # For Relu2: fc1_scale_c = fc2_input_scale
        # Note: fc31_alpha corresponds to w1_alpha (the dequant scale for gate_up layer)
        # and fc2_input_scale corresponds to w2_input_scale (input scale for down layer)
        if is_gated_mlp:
            # SwiGLU: combine fc2 input scale with fc31 dequant scale
            fc1_scale_c = w2_input_scale_stacked * w1_alpha_stacked
        else:
            # Relu2: just use fc2 input scale
            fc1_scale_c = w2_input_scale_stacked

        # Ensure correct shape and dtype
        fc1_scale_c = fc1_scale_c.to(dtype=torch.float32, device=device)
        _register_parameter(gm, new_key_fc1_scale_c, fc1_scale_c)
        _register_parameter(gm, new_key_fc1_alpha, w1_alpha_stacked)
        _register_parameter(gm, new_key_fc2_alpha, w2_alpha_stacked)

        # Create new node
        with graph.inserting_before(node):
            args = (
                hidden_states,
                selected_experts,
                routing_weights,
                graph.get_attr(new_key_fc1),
                graph.get_attr(new_key_fc2),
                graph.get_attr(new_key_fc1_scale),
                graph.get_attr(new_key_fc2_scale),
                graph.get_attr(new_key_fc1_act_scale),
                graph.get_attr(new_key_fc1_scale_c),
                graph.get_attr(new_key_fc1_alpha),
                graph.get_attr(new_key_fc2_alpha),
            )

            kwargs = {
                "num_experts": num_experts,
                "top_k": top_k,
                "intermediate_size": fc1_w_n_padded // 2 if is_gated_mlp else fc1_w_n_padded,
                "hidden_size": hidden_size,
                "is_gated_mlp": is_gated_mlp,
                "act_fn": act_fn,
            }

            new_node = graph.call_function(replacement_op, args, kwargs=kwargs)

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        fused_key_counter += 1

    # Clean up
    eliminate_dead_code(gm)
    delete_all_unused_submodules(gm)
    return fused_key_counter


@TransformRegistry.register("fuse_nvfp4_trtllm_gen_moe")
class FuseNVFP4TRTLLMGenMoe(BaseTransform):
    """
    Stack and shuffle per-expert NVFP4 MoE weights for TRTLLM-Gen kernel (SM100+ only).

    This transform is optimized for Blackwell architecture (SM100/SM103) and uses
    the fp4_block_scale_moe_runner kernel which requires shuffled weight format.
    On non-Blackwell GPUs, this transform is skipped and the standard fuse_nvfp4_moe
    transform will be used instead.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Only run on SM100+ (Blackwell)
        sm_version = get_sm_version()
        if sm_version not in {100, 103}:
            return gm, TransformInfo(
                skipped=True,
                num_matches=0,
                is_clean=True,
                has_valid_shapes=True,
            )

        with cuda_memory_tracker():
            fused_key_counter = _stack_nvfp4_trtllm_gen_moe_weights(gm)

        info = TransformInfo(
            skipped=(fused_key_counter == 0),
            num_matches=fused_key_counter,
            is_clean=fused_key_counter == 0,
            has_valid_shapes=fused_key_counter == 0,
        )
        return gm, info
