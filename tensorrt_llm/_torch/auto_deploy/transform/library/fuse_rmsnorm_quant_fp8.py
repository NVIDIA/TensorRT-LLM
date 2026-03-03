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
from ...utils.node_utils import (
    extract_op_args,
    extract_output_tuple,
    get_shared_input_scale_for_fp8_linears,
    is_op,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _find_fp8_linear_consumers(
    norm_node: Node,
) -> Tuple[List[Node], "Node | None"]:
    return get_shared_input_scale_for_fp8_linears(norm_node.users)


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
    if is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
        return True, "direct", node.args, None, None

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
                return True, "fused", (add_lhs, add_rhs, weight, eps), source_node, add_out_node

    return False, "", (), None, None


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

        for node in list(graph.nodes):
            is_norm, source_type, norm_info, fused_node, add_out_node = _get_norm_source_info(node)
            if not is_norm:
                continue
            if fused_node is not None and id(fused_node) in processed_fused_nodes:
                continue

            fp8_linear_users, _ = _find_fp8_linear_consumers(node)
            if not fp8_linear_users:
                continue

            out_dtype_str = _get_out_dtype_str(node)
            fp8_user_set = set(fp8_linear_users)

            earliest_fp8_user = None
            for n in graph.nodes:
                if n in fp8_user_set:
                    earliest_fp8_user = n
                    break

            _, _, _, earliest_scale, _ = extract_op_args(
                earliest_fp8_user, "input", "weight_fp8", "bias", "input_scale", "weight_scale"
            )

            has_other_consumer_before_fp8 = any(
                u not in fp8_user_set and _node_comes_before(u, earliest_fp8_user, graph)
                for u in node.users
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

                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)
                if len(node.users) == 0:
                    graph.erase_node(node)
            else:
                add_lhs, add_rhs, weight, eps = norm_info
                processed_fused_nodes.add(id(fused_node))

                with graph.inserting_before(earliest_fp8_user):
                    fused_all_node = graph.call_function(
                        torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8.default,
                        args=(add_rhs, add_lhs, weight, eps, earliest_scale),
                    )
                    bf16_node = graph.call_function(operator.getitem, args=(fused_all_node, 0))
                    fp8_node = graph.call_function(operator.getitem, args=(fused_all_node, 1))
                    add_node = graph.call_function(operator.getitem, args=(fused_all_node, 2))

                remaining_users = list(node.users.keys())
                users_to_rewire = [
                    u for u in remaining_users if _node_comes_before(bf16_node, u, graph)
                ]
                for user in users_to_rewire:
                    user.replace_input_with(node, bf16_node)

                if add_out_node is not None:
                    add_out_users = list(add_out_node.users.keys())
                    add_users_to_rewire = [
                        u for u in add_out_users if _node_comes_before(add_node, u, graph)
                    ]
                    for user in add_users_to_rewire:
                        user.replace_input_with(add_out_node, add_node)

                if len(node.users) == 0:
                    graph.erase_node(node)
                if add_out_node is not None and len(add_out_node.users) == 0:
                    graph.erase_node(add_out_node)
                if fused_node is not None and len(fused_node.users) == 0:
                    graph.erase_node(fused_node)

            for fp8_user in fp8_linear_users:
                _, weight_arg, bias_arg, in_scale, w_scale = extract_op_args(
                    fp8_user, "input", "weight_fp8", "bias", "input_scale", "weight_scale"
                )
                with graph.inserting_after(fp8_user):
                    gemm_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default,
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
