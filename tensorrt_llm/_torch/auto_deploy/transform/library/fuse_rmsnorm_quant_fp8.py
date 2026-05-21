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

"""Transformation for fusing RMSNorm + FP8 quantization."""

import operator
from typing import List, Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import flashinfer_fused_add_rms_norm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    extract_op_args,
    extract_output_tuple,
    is_any_view_op,
    is_op,
    is_trivial_passthrough_user,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _is_supported_fp8_linear(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.trtllm_quant_fp8_linear) or is_op(
        node, torch.ops.auto_deploy.torch_quant_fp8_linear
    )


def _extract_fp8_linear_args(node: Node):
    return extract_op_args(node, "input", "weight_fp8", "bias", "input_scale", "weight_scale")


def _same_scale_node(lhs: Node, rhs: Node) -> bool:
    if lhs is rhs:
        return True
    return lhs.op == "get_attr" and rhs.op == "get_attr" and lhs.target == rhs.target


def _collect_grouped_fp8_linear_users(
    source_node: Node,
    seed_user: Node,
    seed_scale: Node,
    processed_users: set[int],
) -> List[Node]:
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(source_node)
    if not traversal_ok:
        return []

    grouped_users: List[Node] = []
    for user in terminal_users:
        if not _is_supported_fp8_linear(user) or id(user) in processed_users:
            continue

        input_arg, _, _, input_scale, _ = _extract_fp8_linear_args(user)
        if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
            continue

        user_source, _ = _unwrap_post_norm_nodes(input_arg)
        if user_source is not source_node:
            continue
        if not _same_scale_node(seed_scale, input_scale):
            continue

        grouped_users.append(user)

    if seed_user not in grouped_users:
        grouped_users.append(seed_user)

    return grouped_users


def _is_view_like(node: Node) -> bool:
    return is_any_view_op(node)


def _unwrap_post_norm_nodes(node: Node) -> Tuple[Node, list[Node]]:
    current = node
    post_nodes: list[Node] = []
    while isinstance(current, Node) and _is_view_like(current):
        post_nodes.append(current)
        current = current.args[0]
    return current, post_nodes


def _reapply_post_norm_nodes(graph, current: Node, post_nodes: list[Node]) -> Node:
    for post_node in reversed(post_nodes):
        current = graph.call_function(
            post_node.target,
            args=(current, *post_node.args[1:]),
            kwargs=post_node.kwargs,
        )
        current.meta.update(post_node.meta)
    return current


def _extract_dtype_from_meta(node: Node) -> Optional[torch.dtype]:
    val = node.meta.get("val")
    if hasattr(val, "dtype"):
        return val.dtype

    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "dtype"):
        return tensor_meta.dtype

    return None


def _get_out_dtype_str(norm_node: Node) -> Optional[str]:
    dtype = _extract_dtype_from_meta(norm_node)
    if dtype is None:
        return None
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"{norm_node.format_node()} has unsupported dtype {dtype}; "
            "expected float16/bfloat16/float32."
        )
    return str(dtype).replace("torch.", "")


def _get_norm_source_info(
    node: Node,
) -> Tuple[bool, str, Tuple, Optional[Node], Optional[Node], Optional[Node], Optional[Node]]:
    if is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
        return True, "direct", node.args, None, None, None, None

    if is_op(node, torch.ops.auto_deploy.torch_rmsnorm):
        norm_input = node.args[0]
        pre_norm_node = norm_input if isinstance(norm_input, Node) else None
        if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.to.dtype):
            norm_input = norm_input.args[0]

        if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.add.Tensor):
            add_lhs, add_rhs = norm_input.args
            return (
                True,
                "raw_add",
                (add_lhs, add_rhs, node.args[1], node.args[2]),
                None,
                norm_input,
                pre_norm_node,
                None,
            )

        return (
            True,
            "direct",
            (node.args[0], node.args[1], node.args[2]),
            None,
            None,
            pre_norm_node,
            None,
        )

    if node.op == "call_function" and node.target == operator.getitem:
        source_node = node.args[0]
        idx = node.args[1]
        if idx == 0 and isinstance(source_node, Node):
            if (
                source_node.op == "call_function"
                and source_node.target is flashinfer_fused_add_rms_norm
            ):
                add_rhs, add_lhs, weight, eps = source_node.args
                _, add_out_node = extract_output_tuple(source_node, count=2)
                return (
                    True,
                    "fused",
                    (add_lhs, add_rhs, weight, eps),
                    source_node,
                    add_out_node,
                    None,
                    None,
                )

    return False, "", (), None, None, None, None


@TransformRegistry.register("fuse_rmsnorm_quant_fp8")
class FuseRMSNormQuantFP8(BaseTransform):
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
        processed_fused_nodes = set()
        processed_fp8_users: set[int] = set()
        original_nodes = list(graph.nodes)
        node_order = {n: i for i, n in enumerate(original_nodes)}

        for node in original_nodes:
            if not _is_supported_fp8_linear(node) or id(node) in processed_fp8_users:
                continue

            input_arg, _, _, input_scale, _ = _extract_fp8_linear_args(node)
            if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
                continue

            norm_node, _ = _unwrap_post_norm_nodes(input_arg)
            (
                is_norm,
                source_type,
                norm_info,
                fused_node,
                add_out_node,
                pre_norm_node,
                _unused_norm_source,
            ) = _get_norm_source_info(norm_node)
            if not is_norm:
                continue
            if fused_node is not None and id(fused_node) in processed_fused_nodes:
                continue

            fp8_linear_users = _collect_grouped_fp8_linear_users(
                norm_node, node, input_scale, processed_fp8_users
            )
            if not fp8_linear_users:
                continue
            # Safety: only fuse when all terminal consumers on the norm data path
            # are FP8 linears in this group. Mixed-consumer patterns can otherwise
            # split one logical norm into mismatched producer paths.
            terminal_users, traversal_ok = collect_terminal_users_through_passthrough(norm_node)
            if not traversal_ok:
                continue
            fp8_user_set = set(fp8_linear_users)
            if any(user not in fp8_user_set for user in terminal_users):
                continue

            out_dtype_str = _get_out_dtype_str(norm_node)
            if out_dtype_str is None:
                continue
            earliest_fp8_user = min(fp8_linear_users, key=lambda n: node_order.get(n, float("inf")))
            earliest_fp8_user_idx = node_order.get(earliest_fp8_user)
            if earliest_fp8_user_idx is None:
                continue
            _, _, _, earliest_scale, _ = _extract_fp8_linear_args(earliest_fp8_user)

            has_other_consumer_before_fp8 = any(
                u not in fp8_user_set
                and not is_trivial_passthrough_user(u)
                and node_order.get(u, float("inf")) < earliest_fp8_user_idx
                for u in norm_node.users
            )
            if has_other_consumer_before_fp8:
                continue

            if source_type == "direct":
                norm_args = norm_info
                with graph.inserting_before(earliest_fp8_user):
                    fused_quant_node = graph.call_function(
                        torch.ops.auto_deploy.triton_rms_norm_quant_fp8.default,
                        args=(*norm_args, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_quant_node, 0))
                    fp8_node = graph.call_function(operator.getitem, args=(fused_quant_node, 1))

                remaining_users = list(norm_node.users.keys())
                users_to_rewire = [
                    u
                    for u in remaining_users
                    if node_order.get(u, float("inf")) >= earliest_fp8_user_idx
                ]
                for user in users_to_rewire:
                    user.replace_input_with(norm_node, bf16_node)
                if len(norm_node.users) == 0:
                    graph.erase_node(norm_node)
            else:
                add_lhs, add_rhs, weight, eps = norm_info
                if fused_node is not None:
                    processed_fused_nodes.add(id(fused_node))

                with graph.inserting_before(earliest_fp8_user):
                    fused_all_node = graph.call_function(
                        torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8.default,
                        args=(add_rhs, add_lhs, weight, eps, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_all_node, 0))
                    fp8_node = graph.call_function(operator.getitem, args=(fused_all_node, 1))
                    add_node = graph.call_function(operator.getitem, args=(fused_all_node, 2))

                remaining_users = list(norm_node.users.keys())
                users_to_rewire = [
                    u
                    for u in remaining_users
                    if node_order.get(u, float("inf")) >= earliest_fp8_user_idx
                ]
                for user in users_to_rewire:
                    user.replace_input_with(norm_node, bf16_node)

                if add_out_node is not None:
                    add_out_users = list(add_out_node.users.keys())
                    add_users_to_rewire = [
                        u
                        for u in add_out_users
                        if node_order.get(u, float("inf")) >= earliest_fp8_user_idx
                    ]
                    for user in add_users_to_rewire:
                        user.replace_input_with(add_out_node, add_node)

                if len(norm_node.users) == 0:
                    graph.erase_node(norm_node)
                if add_out_node is not None and len(add_out_node.users) == 0:
                    graph.erase_node(add_out_node)
                if fused_node is not None and len(fused_node.users) == 0:
                    graph.erase_node(fused_node)

            if (
                pre_norm_node is not None
                and pre_norm_node is not add_out_node
                and len(pre_norm_node.users) == 0
            ):
                graph.erase_node(pre_norm_node)

            for fp8_user in fp8_linear_users:
                input_arg, weight_arg, bias_arg, in_scale, w_scale = _extract_fp8_linear_args(
                    fp8_user
                )
                with graph.inserting_before(fp8_user):
                    fp8_input = fp8_node
                    if isinstance(input_arg, Node):
                        source_arg, post_nodes = _unwrap_post_norm_nodes(input_arg)
                        if source_arg is norm_node:
                            fp8_input = _reapply_post_norm_nodes(graph, fp8_node, post_nodes)
                    gemm_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default,
                        args=(fp8_input, weight_arg, bias_arg),
                        kwargs={
                            "input_scale": in_scale,
                            "weight_scale": w_scale,
                            "out_dtype": out_dtype_str,
                        },
                    )
                    fp8_user.replace_all_uses_with(gemm_node)
                graph.erase_node(fp8_user)
                processed_fp8_users.add(id(fp8_user))
                cnt += 1

        if cnt > 0:
            eliminate_dead_code(gm)
        gm.recompile()
        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
