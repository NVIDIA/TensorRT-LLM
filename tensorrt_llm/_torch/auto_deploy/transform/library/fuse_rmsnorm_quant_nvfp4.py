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

"""Transformation for fusing RMSNorm + NVFP4 quantization.

Matches patterns where flashinfer_rms_norm feeds into torch_quant_nvfp4_linear,
and replaces them with the C++ fused kernel (Add + RMSNorm + NVFP4 quant)
followed by a GEMM-only op that takes pre-quantized FP4 input.

Also handles the case where fuse_add_rms_norm has already run, converting
flashinfer_rms_norm nodes to flashinfer_fused_add_rms_norm.  In this case,
we match getitem(flashinfer_fused_add_rms_norm, 0) and replace with a fully
fused add+norm+quant kernel (trtllm_fused_add_rms_norm_quant_nvfp4).

This eliminates the DRAM round-trip between the normalisation and quantisation
steps by producing both BF16 (for residual / other consumers) and packed FP4
(for GEMM) outputs in a single pass.
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


def _get_nvfp4_linear_args(node: Node):
    """Extract arguments from a torch_quant_nvfp4_linear node.

    After fuse_nvfp4_linear the node is called as:
        op(input, weight_fp4, bias=…, input_scale=…, weight_scale=…, alpha=…)

    Returns:
        (input, weight_fp4, bias, input_scale, weight_scale, alpha) nodes.
    """
    input_arg = node.args[0]
    weight_arg = node.args[1]
    bias_arg = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
    in_scale = node.args[3] if len(node.args) > 3 else node.kwargs.get("input_scale")
    w_scale = node.args[4] if len(node.args) > 4 else node.kwargs.get("weight_scale")
    alpha = node.args[5] if len(node.args) > 5 else node.kwargs.get("alpha")
    return input_arg, weight_arg, bias_arg, in_scale, w_scale, alpha


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from fuse_rmsnorm_quant_fp8 to keep the modules
# self-contained; a future refactor could extract them into a common utility).
# ---------------------------------------------------------------------------


def _resolve_get_attr(gm: GraphModule, node: Node) -> "torch.Tensor | None":
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
    if a is b:
        return True
    if a.op == "get_attr" and b.op == "get_attr":
        if a.target == b.target:
            return True
        if gm is not None:
            val_a = _resolve_get_attr(gm, a)
            val_b = _resolve_get_attr(gm, b)
            if val_a is not None and val_b is not None:
                return torch.equal(val_a, val_b)
    return False


def _find_nvfp4_linear_consumers(
    norm_node: Node,
    gm: "GraphModule | None" = None,
) -> Tuple[List[Node], "Node | None"]:
    """Find torch_quant_nvfp4_linear consumers of a norm node.

    Returns:
        (list of nvfp4_linear consumer nodes, shared input_scale node).
        If consumers have different input_scales, returns ([], None).
    """
    nvfp4_linear_users = []
    shared_scale = None

    for user in norm_node.users:
        if is_op(user, torch.ops.auto_deploy.torch_quant_nvfp4_linear):
            _, _, _, in_scale, _, _ = _get_nvfp4_linear_args(user)
            if shared_scale is None:
                shared_scale = in_scale
            elif not _same_scale_source(in_scale, shared_scale, gm):
                return [], None
            nvfp4_linear_users.append(user)

    return nvfp4_linear_users, shared_scale


def _get_out_dtype_str(norm_node: Node) -> str:
    if "val" in norm_node.meta:
        val = norm_node.meta["val"]
        if hasattr(val, "dtype"):
            return str(val.dtype).replace("torch.", "")
    return "bfloat16"


def _node_comes_before(node_a: Node, node_b: Node, graph) -> bool:
    for n in graph.nodes:
        if n is node_a:
            return True
        if n is node_b:
            return False
    return False


def _get_norm_source_info(
    node: Node,
) -> Tuple[bool, str, Tuple, Optional[Node], Optional[Node]]:
    """Classify a node as a norm output and extract source info.

    Returns:
        (is_norm_output, source_type, norm_args, fused_node, add_out_node)
    """
    # Case 1: Direct flashinfer_rms_norm
    if is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
        return True, "direct", node.args, None, None

    # Case 2: getitem(flashinfer_fused_add_rms_norm, 0)
    if node.op == "call_function" and node.target == operator.getitem:
        source_node = node.args[0]
        idx = node.args[1]
        if idx == 0 and isinstance(source_node, Node):
            if (
                source_node.op == "call_function"
                and source_node.target is flashinfer_fused_add_rms_norm
            ):
                add_rhs, add_lhs, weight, eps = source_node.args
                add_out_node = None
                for user in source_node.users:
                    if (
                        user.op == "call_function"
                        and user.target == operator.getitem
                        and user.args[1] == 1
                    ):
                        add_out_node = user
                        break
                return True, "fused", (add_lhs, add_rhs, weight, eps), source_node, add_out_node

    return False, "", (), None, None


@TransformRegistry.register("fuse_rmsnorm_quant_nvfp4")
class FuseRMSNormQuantNVFP4(BaseTransform):
    """Fuse RMSNorm + NVFP4 quantization into the C++ warp-specialised kernel.

    Matches two patterns:

    1. Direct flashinfer_rms_norm:
        norm_out = flashinfer_rms_norm(x, weight, eps)
        linear_out = torch_quant_nvfp4_linear(norm_out, w_fp4, bias,
                                               input_scale, weight_scale, alpha)

    2. Fused add+norm (after fuse_add_rms_norm has run):
        fused = flashinfer_fused_add_rms_norm(x, residual, weight, eps)
        norm_out = getitem(fused, 0)
        add_out  = getitem(fused, 1)
        linear_out = torch_quant_nvfp4_linear(norm_out, ...)

    Replaces with:
        # Direct case:
        bf16_out, fp4_out, sf_out = trtllm_rms_norm_quant_nvfp4(
            x, weight, eps, input_scale)

        # Fused case:
        bf16_out, fp4_out, sf_out, add_out = trtllm_fused_add_rms_norm_quant_nvfp4(
            x, residual, weight, eps, input_scale)

        # Both cases:
        linear_out = trtllm_nvfp4_gemm(fp4_out, w_fp4, sf_out,
                                        weight_scale, alpha, bias, dtype)
        (other consumers of norm_out use bf16_out)
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
        processed_fused_nodes: set = set()

        for node in list(graph.nodes):
            is_norm, source_type, norm_info, fused_node, add_out_node = _get_norm_source_info(node)
            if not is_norm:
                continue

            if fused_node is not None and id(fused_node) in processed_fused_nodes:
                continue

            nvfp4_linear_users, shared_scale = _find_nvfp4_linear_consumers(node, gm)
            if not nvfp4_linear_users:
                continue

            out_dtype_str = _get_out_dtype_str(node)

            # Find topologically earliest nvfp4_linear user
            nvfp4_user_set = set(nvfp4_linear_users)
            earliest_nvfp4_user = None
            for n in graph.nodes:
                if n in nvfp4_user_set:
                    earliest_nvfp4_user = n
                    break
            _, _, _, earliest_scale, _, _ = _get_nvfp4_linear_args(earliest_nvfp4_user)

            # Skip if any norm consumer other than nvfp4_linear appears before the
            # earliest nvfp4_linear user.
            has_other_consumer_before = any(
                u not in nvfp4_user_set and _node_comes_before(u, earliest_nvfp4_user, graph)
                for u in node.users
            )
            if has_other_consumer_before:
                continue

            # --- Handle based on source type ---
            if source_type == "direct":
                norm_args = norm_info  # (input, weight, eps)

                with graph.inserting_before(earliest_nvfp4_user):
                    fused_quant_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_rms_norm_quant_nvfp4.default,
                        args=(*norm_args, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_quant_node, 0))
                    fp4_node = graph.call_function(operator.getitem, args=(fused_quant_node, 1))
                    sf_node = graph.call_function(operator.getitem, args=(fused_quant_node, 2))

                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)

                if len(node.users) == 0:
                    graph.erase_node(node)

            else:
                # Fused add+norm case
                add_lhs, add_rhs, weight, eps = norm_info
                processed_fused_nodes.add(id(fused_node))

                with graph.inserting_before(earliest_nvfp4_user):
                    fused_all_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fused_add_rms_norm_quant_nvfp4.default,
                        args=(add_rhs, add_lhs, weight, eps, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_all_node, 0))
                    fp4_node = graph.call_function(operator.getitem, args=(fused_all_node, 1))
                    sf_node = graph.call_function(operator.getitem, args=(fused_all_node, 2))
                    add_node = graph.call_function(operator.getitem, args=(fused_all_node, 3))

                # Rewire norm_out consumers to bf16
                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)

                # Rewire add_out consumers
                if add_out_node is not None:
                    add_out_users = list(add_out_node.users.keys())
                    add_users_to_rewire = [
                        u for u in add_out_users if _node_comes_before(add_node, u, graph)
                    ]
                    for user in add_users_to_rewire:
                        user.replace_input_with(add_out_node, add_node)

                # Erase old nodes if unused
                if len(node.users) == 0:
                    graph.erase_node(node)
                if add_out_node is not None and len(add_out_node.users) == 0:
                    graph.erase_node(add_out_node)
                if fused_node is not None and len(fused_node.users) == 0:
                    graph.erase_node(fused_node)

            # --- Replace each nvfp4_linear consumer with nvfp4_gemm ---
            for nvfp4_user in nvfp4_linear_users:
                _, weight_arg, bias_arg, _in_scale, w_scale, alpha = _get_nvfp4_linear_args(
                    nvfp4_user
                )

                with graph.inserting_after(nvfp4_user):
                    gemm_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_nvfp4_gemm.default,
                        args=(fp4_node, weight_arg, sf_node, w_scale, alpha),
                        kwargs={
                            "bias": bias_arg,
                            "out_dtype": out_dtype_str,
                        },
                    )
                    nvfp4_user.replace_all_uses_with(gemm_node)
                graph.erase_node(nvfp4_user)
                cnt += 1

        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
