# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformation for fusing RMSNorm + FP8 quantization.

Matches patterns where flashinfer_rms_norm feeds into trtllm_quant_fp8_linear,
and replaces them with a fused Triton kernel (RMSNorm + FP8 quant) followed by
a GEMM-only op that takes pre-quantized FP8 input.

Also handles the case where fuse_add_rms_norm has already run, converting
flashinfer_rms_norm nodes to flashinfer_fused_add_rms_norm. In this case,
we match getitem(flashinfer_fused_add_rms_norm, 0) and replace with a fully
fused add+norm+quant kernel (triton_fused_add_rms_norm_quant_fp8).

This eliminates the DRAM round-trip between the normalization and quantization
steps by producing both BF16 (for residual / other consumers) and FP8 (for GEMM)
outputs in a single pass.
"""

import operator
from typing import List, Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import flashinfer_fused_add_rms_norm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _get_fp8_linear_args(node: Node):
    """Extract arguments from a trtllm_quant_fp8_linear node.

    The op can be called with positional or keyword args:
        op(input, weight_fp8, bias, input_scale=..., weight_scale=...)
    or:
        op(input, weight_fp8, None, input_scale=..., weight_scale=...)

    Returns:
        Tuple of (input, weight_fp8, bias, input_scale, weight_scale) nodes.
    """
    input_arg = node.args[0]
    weight_arg = node.args[1]
    bias_arg = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
    in_scale = node.args[3] if len(node.args) > 3 else node.kwargs.get("input_scale")
    w_scale = node.args[4] if len(node.args) > 4 else node.kwargs.get("weight_scale")
    return input_arg, weight_arg, bias_arg, in_scale, w_scale


def _resolve_get_attr(gm: GraphModule, node: Node) -> "torch.Tensor | None":
    """Resolve a get_attr node to its actual tensor value."""
    if node.op != "get_attr":
        return None
    try:
        atoms = node.target.split(".")
        mod = gm
        for atom in atoms:
            mod = getattr(mod, atom)
        return mod if isinstance(mod, torch.Tensor) else None
    except (AttributeError, TypeError):
        return None


def _same_scale_source(a: Node, b: Node, gm: "GraphModule | None" = None) -> bool:
    """Check if two FX nodes reference the same underlying scale value.

    FX tracing may create separate get_attr nodes for the same parameter
    (e.g., self.in_scale accessed 3 times yields in_scale, in_scale_1,
    in_scale_2). These are different Node objects but point to the same
    parameter via their `target` attribute.

    In real models, Q/K/V projections in the same layer each have their own
    input_scale attribute (e.g., q_proj.input_scale, k_proj.input_scale)
    but the actual tensor values are typically identical since they were
    calibrated on the same input distribution.  When a GraphModule is
    provided, we fall back to comparing actual tensor values.
    """
    if a is b:
        return True
    # Both are get_attr referencing the same module attribute
    if a.op == "get_attr" and b.op == "get_attr":
        if a.target == b.target:
            return True
        # Different attributes -- compare actual tensor values if gm available
        if gm is not None:
            val_a = _resolve_get_attr(gm, a)
            val_b = _resolve_get_attr(gm, b)
            if val_a is not None and val_b is not None:
                return torch.equal(val_a, val_b)
    return False


def _find_fp8_linear_consumers(
    norm_node: Node,
    gm: "GraphModule | None" = None,
) -> Tuple[List[Node], "Node | None"]:
    """Find trtllm_quant_fp8_linear consumers of a norm node.

    Returns:
        Tuple of (list of fp8_linear consumer nodes, shared input_scale node).
        If consumers have different input_scales, returns ([], None).
    """
    fp8_linear_users = []
    shared_scale = None

    for user in norm_node.users:
        if is_op(user, torch.ops.auto_deploy.trtllm_quant_fp8_linear):
            _, _, _, in_scale, _ = _get_fp8_linear_args(user)
            if shared_scale is None:
                shared_scale = in_scale
            elif not _same_scale_source(in_scale, shared_scale, gm):
                # Different input_scales -- cannot share a single FP8 output
                return [], None
            fp8_linear_users.append(user)

    return fp8_linear_users, shared_scale


def _get_out_dtype_str(norm_node: Node) -> str:
    """Determine the output dtype string from a norm node's metadata."""
    if "val" in norm_node.meta:
        val = norm_node.meta["val"]
        if hasattr(val, "dtype"):
            return str(val.dtype).replace("torch.", "")
    return "bfloat16"


def _find_last_input_node(graph, input_nodes: List[Node]) -> Node:
    """Find the topologically last node among the given input nodes.

    This determines the earliest safe insertion point where all inputs
    are guaranteed to be defined.
    """
    input_set = set(input_nodes)
    last_input = None
    for n in graph.nodes:
        if n in input_set:
            last_input = n
    return last_input


def _node_comes_before(node_a: Node, node_b: Node, graph) -> bool:
    """Check if node_a comes before node_b in topological order."""
    for n in graph.nodes:
        if n is node_a:
            return True
        if n is node_b:
            return False
    return False


def _get_norm_source_info(
    node: Node,
) -> Tuple[bool, str, Tuple, Optional[Node], Optional[Node]]:
    """Check if node represents a norm output and extract source info.

    Returns:
        (is_norm_output, source_type, norm_args, fused_node, add_out_node)
        - source_type: "direct" for flashinfer_rms_norm,
                       "fused" for getitem(flashinfer_fused_add_rms_norm, 0)
        - norm_args: (input, weight, eps) - the args needed for triton_rms_norm_quant_fp8
        - fused_node: the flashinfer_fused_add_rms_norm node (only for "fused" type)
        - add_out_node: the getitem node for add output (only for "fused" type)
    """
    # Case 1: Direct flashinfer_rms_norm
    if is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
        return True, "direct", node.args, None, None

    # Case 2: getitem(flashinfer_fused_add_rms_norm, 0)
    # Note: flashinfer_fused_add_rms_norm is a Python function wrapper (not a custom op),
    # so we check against the function object directly.
    if node.op == "call_function" and node.target == operator.getitem:
        source_node = node.args[0]
        idx = node.args[1]
        if idx == 0 and isinstance(source_node, Node):
            # Check if source is flashinfer_fused_add_rms_norm (Python function)
            if (
                source_node.op == "call_function"
                and source_node.target is flashinfer_fused_add_rms_norm
            ):
                # source_node.args = (add_rhs, add_lhs, weight, eps)
                add_rhs, add_lhs, weight, eps = source_node.args
                # Find the add_out node (getitem with index 1)
                add_out_node = None
                for user in source_node.users:
                    if (
                        user.op == "call_function"
                        and user.target == operator.getitem
                        and user.args[1] == 1
                    ):
                        add_out_node = user
                        break
                # The norm input is the add result, which we'll compute as add(add_lhs, add_rhs)
                # For triton_rms_norm_quant_fp8, we need (input, weight, eps)
                # We'll create the add node later; for now return the add operands
                return True, "fused", (add_lhs, add_rhs, weight, eps), source_node, add_out_node

    return False, "", (), None, None


@TransformRegistry.register("fuse_rmsnorm_quant_fp8")
class FuseRMSNormQuantFP8(BaseTransform):
    """Fuse RMSNorm + FP8 quantization into a single Triton kernel.

    Matches two patterns:

    1. Direct flashinfer_rms_norm:
        norm_out = flashinfer_rms_norm(x, weight, eps)
        linear_out = trtllm_quant_fp8_linear(norm_out, w_fp8, bias, in_scale, w_scale)

    2. Fused add+norm (after fuse_add_rms_norm has run):
        fused = flashinfer_fused_add_rms_norm(x, residual, weight, eps)
        norm_out = getitem(fused, 0)
        add_out = getitem(fused, 1)
        linear_out = trtllm_quant_fp8_linear(norm_out, w_fp8, ...)

    Replaces with:
        # For direct case:
        bf16_out, fp8_out = triton_rms_norm_quant_fp8(x, weight, eps, in_scale)

        # For fused case (fully fused add+norm+quant):
        bf16_out, fp8_out, add_out = triton_fused_add_rms_norm_quant_fp8(
            x, residual, weight, eps, in_scale
        )

        # Both cases:
        linear_out = trtllm_fp8_gemm(fp8_out, w_fp8, bias, in_scale, w_scale, dtype)
        (other consumers of norm_out use bf16_out)

    Handles multi-consumer case: when norm_out feeds multiple fp8_linear ops
    (e.g., Q, K, V projections), all share the fused FP8 output.

    Skips norms that have any consumer before the earliest fp8_linear user (e.g.
    MoE gate/view). Fusing those would leave two norms (bf16 + fp8); we skip instead.
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        cnt = 0

        # Track nodes we've already processed (for fused case, multiple getitems
        # reference the same fused node)
        processed_fused_nodes = set()

        for node in list(graph.nodes):
            # Check if this node represents a norm output
            is_norm, source_type, norm_info, fused_node, add_out_node = _get_norm_source_info(node)
            if not is_norm:
                continue

            # Skip if we've already processed this fused node
            if fused_node is not None and id(fused_node) in processed_fused_nodes:
                continue

            fp8_linear_users, shared_scale = _find_fp8_linear_consumers(node, gm)
            if not fp8_linear_users:
                continue

            # Determine output dtype from the norm node metadata
            out_dtype_str = _get_out_dtype_str(node)

            # Find the topologically earliest fp8_linear user and its scale
            fp8_user_set = set(fp8_linear_users)
            earliest_fp8_user = None
            for n in graph.nodes:
                if n in fp8_user_set:
                    earliest_fp8_user = n
                    break
            _, _, _, earliest_scale, _ = _get_fp8_linear_args(earliest_fp8_user)

            # Skip if any norm consumer other than fp8_linear appears before the
            # earliest fp8_linear user (e.g. MoE gate/view). Then we would keep the
            # original norm for those and add a fused norm for fp8 (two norms).
            # Only fuse when we can replace the norm entirely.
            has_other_consumer_before_fp8 = any(
                u not in fp8_user_set and _node_comes_before(u, earliest_fp8_user, graph)
                for u in node.users
            )
            if has_other_consumer_before_fp8:
                continue

            # --- Handle based on source type ---
            if source_type == "direct":
                # Direct flashinfer_rms_norm case
                norm_args = norm_info  # (input, weight, eps)

                with graph.inserting_before(earliest_fp8_user):
                    fused_quant_node = graph.call_function(
                        torch.ops.auto_deploy.triton_rms_norm_quant_fp8.default,
                        args=(*norm_args, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_quant_node, 0))
                    fp8_node = graph.call_function(operator.getitem, args=(fused_quant_node, 1))

                # Rewire remaining consumers of the original norm node
                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)

                # Only erase the original norm if it has no remaining users
                if len(node.users) == 0:
                    graph.erase_node(node)

            else:
                # Fused add+norm case: getitem(flashinfer_fused_add_rms_norm, 0)
                # Use the fully fused add+norm+quant kernel
                add_lhs, add_rhs, weight, eps = norm_info
                processed_fused_nodes.add(id(fused_node))

                with graph.inserting_before(earliest_fp8_user):
                    # Create fused add+norm+quant node
                    # triton_fused_add_rms_norm_quant_fp8(x, residual, weight, eps, scale)
                    #   -> (bf16_norm, fp8_norm, add_out)
                    # Note: add_rhs is x, add_lhs is residual in the original fused op
                    fused_all_node = graph.call_function(
                        torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8.default,
                        args=(add_rhs, add_lhs, weight, eps, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_all_node, 0))
                    fp8_node = graph.call_function(operator.getitem, args=(fused_all_node, 1))
                    add_node = graph.call_function(operator.getitem, args=(fused_all_node, 2))

                # Rewire consumers of the original norm_out (getitem 0) to bf16_node
                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)

                # Rewire consumers of add_out (getitem 1) to the new add_node
                if add_out_node is not None:
                    add_out_users = list(add_out_node.users.keys())
                    add_users_to_rewire = [
                        u for u in add_out_users if _node_comes_before(add_node, u, graph)
                    ]
                    for user in add_users_to_rewire:
                        user.replace_input_with(add_out_node, add_node)

                # Erase old nodes if they have no remaining users
                if len(node.users) == 0:
                    graph.erase_node(node)
                if add_out_node is not None and len(add_out_node.users) == 0:
                    graph.erase_node(add_out_node)
                if fused_node is not None and len(fused_node.users) == 0:
                    graph.erase_node(fused_node)

            # --- Replace each fp8_linear consumer with fp8_gemm ---
            for fp8_user in fp8_linear_users:
                _, weight_arg, bias_arg, in_scale, w_scale = _get_fp8_linear_args(fp8_user)

                with graph.inserting_after(fp8_user):
                    gemm_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fp8_gemm.default,
                        args=(fp8_node, weight_arg, bias_arg),
                        kwargs={
                            "input_scale": in_scale,
                            "weight_scale": w_scale,
                            "out_dtype": out_dtype_str,
                        },
                    )
                    fp8_user.replace_all_uses_with(gemm_node)
                graph.erase_node(fp8_user)
                cnt += 1

        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
