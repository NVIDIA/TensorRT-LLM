# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fuse Torch-backend NVFP4 norm-quant kernels into AutoDeploy graphs."""

import operator
from typing import List, Tuple, Type

import torch
from torch.fx import GraphModule, Node

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ...custom_ops.quantization.quant import TRTLLM_NVFP4_SCALING_VECTOR_SIZE
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.node_utils import (
    collect_terminal_users_through_passthrough,
    extract_op_args,
    extract_output_tuple,
    is_dtype_cast_op,
    is_op,
    unwrap_input_through_passthrough,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _is_supported_nvfp4_linear(node: Node) -> bool:
    return is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear)


def _extract_nvfp4_linear_args(node: Node):
    return extract_op_args(
        node, "input", "weight_fp4", "bias", "input_scale", "weight_scale", "alpha"
    )


def _same_scale_node(lhs: Node, rhs: Node) -> bool:
    if lhs is rhs:
        return True
    return lhs.op == "get_attr" and rhs.op == "get_attr" and lhs.target == rhs.target


def _unwrap_post_norm_nodes(node: Node) -> Tuple[Node, list[Node]]:
    return unwrap_input_through_passthrough(node, allow_dtype_cast=True)


def _has_unsupported_post_norm_nodes(post_nodes: list[Node]) -> bool:
    return any(not is_dtype_cast_op(node) for node in post_nodes)


def _extract_dtype_from_meta(node: Node) -> torch.dtype | None:
    val = node.meta.get("val")
    if hasattr(val, "dtype"):
        return val.dtype

    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "dtype"):
        return tensor_meta.dtype

    return None


def _get_out_dtype(linear_node: Node, source_node: Node) -> torch.dtype:
    input_arg, _, _, _, _, _ = _extract_nvfp4_linear_args(linear_node)
    input_dtype = _extract_dtype_from_meta(input_arg) if isinstance(input_arg, Node) else None
    return (
        _extract_dtype_from_meta(linear_node)
        or input_dtype
        or _extract_dtype_from_meta(source_node)
        or torch.bfloat16
    )


def _get_last_dim_from_meta(node: Node) -> int | None:
    val = node.meta.get("val")
    if hasattr(val, "shape") and len(val.shape) > 0:
        return int(val.shape[-1])

    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "shape") and len(tensor_meta.shape) > 0:
        return int(tensor_meta.shape[-1])

    return None


def _get_shape_from_meta(node: Node) -> tuple | None:
    val = node.meta.get("val")
    if hasattr(val, "shape"):
        return tuple(val.shape)

    tensor_meta = node.meta.get("tensor_meta")
    if hasattr(tensor_meta, "shape"):
        return tuple(tensor_meta.shape)

    return None


def _new_empty_from_meta(source_node: Node, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    val = source_node.meta.get("val")
    if hasattr(val, "new_empty"):
        return val.new_empty(shape, dtype=dtype)
    return torch.empty(shape, dtype=dtype, device="meta")


def _set_tensor_val_meta(
    node: Node,
    source_node: Node,
    shape: tuple,
    dtype: torch.dtype,
) -> None:
    node.meta["val"] = _new_empty_from_meta(source_node, shape, dtype)
    node.meta.pop("tensor_meta", None)


def _set_nvfp4_quant_meta(fp4_node: Node, scale_node: Node, source_node: Node) -> None:
    source_shape = _get_shape_from_meta(source_node)
    if source_shape is None:
        return

    fp4_shape, scale_shape = fp4_utils.get_fp4_shape(source_shape, TRTLLM_NVFP4_SCALING_VECTOR_SIZE)
    _set_tensor_val_meta(fp4_node, source_node, tuple(fp4_shape), torch.uint8)
    _set_tensor_val_meta(scale_node, source_node, (scale_shape,), torch.uint8)


def _supports_trtllm_fused_add_rmsnorm_quant_nvfp4(node: Node) -> bool:
    hidden_size = _get_last_dim_from_meta(node)
    if hidden_size is None:
        return False
    return 2048 <= hidden_size <= 16384 and hidden_size % 16 == 0


def _get_arg_defined_before(
    graph,
    arg: Node,
    insertion_node: Node,
    node_order: dict[Node, int],
) -> Node | None:
    if node_order.get(arg, -1) < node_order.get(insertion_node, float("inf")):
        return arg
    if arg.op != "get_attr":
        return None

    # Model parameters/buffers are order-independent logically, but FX lint requires
    # the get_attr node to appear before every consumer.
    with graph.inserting_before(insertion_node):
        cloned_arg = graph.get_attr(arg.target)
        cloned_arg.meta.update(arg.meta)
    return cloned_arg


def _collect_grouped_nvfp4_linear_users(
    source_node: Node,
    seed_user: Node,
    seed_scale: Node,
    processed_users: set[int],
) -> List[Node]:
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return []

    grouped_users: List[Node] = []
    for user in terminal_users:
        if not _is_supported_nvfp4_linear(user) or id(user) in processed_users:
            continue

        input_arg, _, _, input_scale, _, _ = _extract_nvfp4_linear_args(user)
        if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
            continue

        user_source, post_nodes = _unwrap_post_norm_nodes(input_arg)
        if user_source is not source_node or _has_unsupported_post_norm_nodes(post_nodes):
            continue
        if not _same_scale_node(seed_scale, input_scale):
            continue

        grouped_users.append(user)

    if seed_user not in grouped_users:
        grouped_users.append(seed_user)

    return grouped_users


def _all_terminal_users_are_grouped(source_node: Node, grouped_users: List[Node]) -> bool:
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return False
    grouped_user_set = set(grouped_users)
    return all(user in grouped_user_set for user in terminal_users)


def _has_terminal_users_outside_group(source_node: Node, grouped_users: List[Node]) -> bool:
    terminal_users, traversal_ok = collect_terminal_users_through_passthrough(
        source_node, allow_dtype_cast=True
    )
    if not traversal_ok:
        return True
    grouped_user_set = set(grouped_users)
    return any(user not in grouped_user_set for user in terminal_users)


def _is_getitem(node: Node, idx: int) -> bool:
    return node.op == "call_function" and node.target == operator.getitem and node.args[1] == idx


def _extract_nonquant_allreduce_norm(node: Node):
    if not _is_getitem(node, 0):
        return None

    source_node = node.args[0]
    if not isinstance(source_node, Node) or not is_op(
        source_node, torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm
    ):
        return None

    tensor, residual, norm_weight, eps, strategy = source_node.args
    _, residual_out = extract_output_tuple(source_node, count=2)
    return source_node, tensor, residual, norm_weight, eps, strategy, residual_out


def _extract_add_rmsnorm(node: Node):
    if not is_op(
        node,
        [
            torch.ops.auto_deploy.flashinfer_rms_norm,
            torch.ops.auto_deploy.torch_rmsnorm,
            torch.ops.auto_deploy.triton_rms_norm,
        ],
    ):
        return None

    norm_input, norm_weight, eps = node.args
    pre_norm_cast = None
    if isinstance(norm_input, Node) and is_dtype_cast_op(norm_input):
        pre_norm_cast = norm_input
        norm_input = norm_input.args[0]

    if not isinstance(norm_input, Node) or not is_op(norm_input, torch.ops.aten.add.Tensor):
        return None

    add_lhs, add_rhs = norm_input.args[:2]
    if not isinstance(add_lhs, Node) or not isinstance(add_rhs, Node):
        return None

    return norm_input, pre_norm_cast, add_lhs, add_rhs, norm_weight, eps


def _extract_gated_rmsnorm(node: Node):
    if not is_op(
        node,
        [
            torch.ops.auto_deploy.torch_rmsnorm_gated,
            torch.ops.auto_deploy.triton_rmsnorm_gated,
        ],
    ):
        return None

    x, weight, gate, eps, group_size, norm_before_gate = extract_op_args(
        node, "x", "weight", "gate", "eps", "group_size", "norm_before_gate"
    )
    if gate is None or norm_before_gate:
        return None
    return x, weight, gate, eps, group_size


def _insert_prequant_linear(
    graph,
    linear_node: Node,
    fp4_input: Node,
    scale_factors: Node,
    out_dtype: torch.dtype,
) -> Node:
    _, weight_fp4, bias, _, weight_scale, alpha = _extract_nvfp4_linear_args(linear_node)
    return graph.call_function(
        torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default,
        args=(fp4_input, weight_fp4, scale_factors, weight_scale, alpha),
        kwargs={"bias": bias, "out_dtype": out_dtype},
    )


@TransformRegistry.register("fuse_rmsnorm_quant_nvfp4")
class FuseRMSNormQuantNVFP4(BaseTransform):
    """Fuse NVFP4 quantization into RMSNorm producers where TRT-LLM kernels exist."""

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
        processed_nvfp4_users: set[int] = set()
        original_nodes = list(graph.nodes)
        node_order = {n: i for i, n in enumerate(original_nodes)}

        for node in original_nodes:
            if not _is_supported_nvfp4_linear(node) or id(node) in processed_nvfp4_users:
                continue

            input_arg, _, _, input_scale, _, _ = _extract_nvfp4_linear_args(node)
            if not isinstance(input_arg, Node) or not isinstance(input_scale, Node):
                continue

            norm_node, post_nodes = _unwrap_post_norm_nodes(input_arg)
            if _has_unsupported_post_norm_nodes(post_nodes):
                continue

            nvfp4_linear_users = _collect_grouped_nvfp4_linear_users(
                norm_node, node, input_scale, processed_nvfp4_users
            )
            if not nvfp4_linear_users:
                continue

            earliest_user = min(nvfp4_linear_users, key=lambda n: node_order.get(n, float("inf")))

            num_matches = self._try_fuse_allreduce_rmsnorm_quant(
                norm_node,
                input_scale,
                nvfp4_linear_users,
                earliest_user,
                node_order,
                processed_nvfp4_users,
            )
            if num_matches is not None:
                cnt += num_matches
                continue

            num_matches = self._try_fuse_add_rmsnorm_quant(
                norm_node,
                input_scale,
                nvfp4_linear_users,
                node_order,
                processed_nvfp4_users,
            )
            if num_matches is not None:
                cnt += num_matches
                continue

            num_matches = self._try_fuse_gated_rmsnorm_quant(
                norm_node,
                input_scale,
                nvfp4_linear_users,
                earliest_user,
                processed_nvfp4_users,
            )
            if num_matches is None:
                continue
            cnt += num_matches

        if cnt > 0:
            eliminate_dead_code(gm)
            gm.graph.lint()
        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=True,
            has_valid_shapes=True,
        )
        return gm, info

    def _try_fuse_allreduce_rmsnorm_quant(
        self,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: List[Node],
        earliest_user: Node,
        node_order: dict[Node, int],
        processed_nvfp4_users: set[int],
    ) -> int | None:
        allreduce_info = _extract_nonquant_allreduce_norm(norm_node)
        if allreduce_info is None:
            return None

        (
            allreduce_node,
            tensor,
            residual,
            norm_weight,
            eps,
            strategy,
            residual_out_node,
        ) = allreduce_info
        graph = norm_node.graph
        needs_norm_output = _has_terminal_users_outside_group(norm_node, nvfp4_linear_users)
        insertion_node = earliest_user
        if needs_norm_output:
            insertion_node = min(
                list(norm_node.users), key=lambda n: node_order.get(n, float("inf"))
            )

        fused_input_scale = _get_arg_defined_before(graph, input_scale, insertion_node, node_order)
        if fused_input_scale is None:
            return 0

        with graph.inserting_before(insertion_node):
            if needs_norm_output:
                fused_quant = graph.call_function(
                    torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_out_quant_nvfp4.default,
                    args=(tensor, residual, norm_weight, fused_input_scale, eps, strategy),
                )
                new_norm_out = graph.call_function(operator.getitem, args=(fused_quant, 0))
                fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 1))
                scale_node = graph.call_function(operator.getitem, args=(fused_quant, 2))
                new_residual_out = graph.call_function(operator.getitem, args=(fused_quant, 3))
                new_norm_out.meta.update(norm_node.meta)
            else:
                fused_quant = graph.call_function(
                    torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_quant_nvfp4.default,
                    args=(tensor, residual, norm_weight, fused_input_scale, eps, strategy),
                )
                fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 0))
                scale_node = graph.call_function(operator.getitem, args=(fused_quant, 1))
                new_residual_out = graph.call_function(operator.getitem, args=(fused_quant, 2))
            if residual_out_node is not None:
                new_residual_out.meta.update(residual_out_node.meta)
            _set_nvfp4_quant_meta(fp4_node, scale_node, norm_node)

        if needs_norm_output:
            norm_node.replace_all_uses_with(new_norm_out)
        if residual_out_node is not None:
            residual_out_node.replace_all_uses_with(new_residual_out)

        num_matches = self._replace_nvfp4_linears(
            nvfp4_linear_users,
            fp4_node,
            scale_node,
            norm_node,
            processed_nvfp4_users,
        )

        if residual_out_node is not None and len(residual_out_node.users) == 0:
            graph.erase_node(residual_out_node)
        if len(norm_node.users) == 0:
            graph.erase_node(norm_node)
        if len(allreduce_node.users) == 0:
            graph.erase_node(allreduce_node)
        return num_matches

    def _try_fuse_add_rmsnorm_quant(
        self,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: List[Node],
        node_order: dict[Node, int],
        processed_nvfp4_users: set[int],
    ) -> int | None:
        add_norm_info = _extract_add_rmsnorm(norm_node)
        if add_norm_info is None:
            return None

        if not _supports_trtllm_fused_add_rmsnorm_quant_nvfp4(norm_node):
            hidden_size = _get_last_dim_from_meta(norm_node)
            ad_logger.debug(
                "fuse_rmsnorm_quant_nvfp4: skipping add+norm at "
                f"{norm_node.name}, hidden_size={hidden_size} outside supported "
                "[2048, 16384] range or not divisible by 16"
            )
            return 0

        add_node, pre_norm_cast, add_lhs, add_rhs, norm_weight, eps = add_norm_info
        graph = norm_node.graph
        needs_norm_output = _has_terminal_users_outside_group(norm_node, nvfp4_linear_users)
        insertion_candidates = list(add_node.users) + list(norm_node.users)
        if pre_norm_cast is not None:
            insertion_candidates.extend(list(pre_norm_cast.users))
        insertion_node = min(
            insertion_candidates,
            key=lambda n: node_order.get(n, float("inf")),
        )
        fused_input_scale = _get_arg_defined_before(graph, input_scale, insertion_node, node_order)
        fused_norm_weight = _get_arg_defined_before(graph, norm_weight, insertion_node, node_order)
        if fused_input_scale is None or fused_norm_weight is None:
            return 0

        with graph.inserting_before(insertion_node):
            if needs_norm_output:
                fused_quant = graph.call_function(
                    torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_out_quant_nvfp4.default,
                    args=(add_lhs, add_rhs, fused_norm_weight, fused_input_scale, eps),
                )
                new_norm_out = graph.call_function(operator.getitem, args=(fused_quant, 0))
                fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 1))
                new_residual_out = graph.call_function(operator.getitem, args=(fused_quant, 2))
                scale_node = graph.call_function(operator.getitem, args=(fused_quant, 3))
                new_norm_out.meta.update(norm_node.meta)
            else:
                fused_quant = graph.call_function(
                    torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_quant_nvfp4.default,
                    args=(add_lhs, add_rhs, fused_norm_weight, fused_input_scale, eps),
                )
                fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 0))
                new_residual_out = graph.call_function(operator.getitem, args=(fused_quant, 1))
                scale_node = graph.call_function(operator.getitem, args=(fused_quant, 2))
            new_residual_out.meta.update(add_node.meta)
            _set_nvfp4_quant_meta(fp4_node, scale_node, norm_node)

        if needs_norm_output:
            norm_node.replace_all_uses_with(new_norm_out)

        num_matches = self._replace_nvfp4_linears(
            nvfp4_linear_users,
            fp4_node,
            scale_node,
            norm_node,
            processed_nvfp4_users,
        )

        add_node.replace_all_uses_with(new_residual_out)

        if len(norm_node.users) == 0:
            graph.erase_node(norm_node)
        if pre_norm_cast is not None and len(pre_norm_cast.users) == 0:
            graph.erase_node(pre_norm_cast)
        if len(add_node.users) == 0:
            graph.erase_node(add_node)
        return num_matches

    def _try_fuse_gated_rmsnorm_quant(
        self,
        norm_node: Node,
        input_scale: Node,
        nvfp4_linear_users: List[Node],
        earliest_user: Node,
        processed_nvfp4_users: set[int],
    ) -> int | None:
        gated_info = _extract_gated_rmsnorm(norm_node)
        if gated_info is None:
            return None
        if not _all_terminal_users_are_grouped(norm_node, nvfp4_linear_users):
            return 0

        x, weight, gate, eps, group_size = gated_info
        graph = norm_node.graph
        with graph.inserting_before(earliest_user):
            fused_quant = graph.call_function(
                torch.ops.auto_deploy.trtllm_fused_gated_rmsnorm_quant_nvfp4.default,
                args=(x, gate, weight, input_scale, eps, group_size),
            )
            fp4_node = graph.call_function(operator.getitem, args=(fused_quant, 0))
            scale_node = graph.call_function(operator.getitem, args=(fused_quant, 1))
            _set_nvfp4_quant_meta(fp4_node, scale_node, norm_node)

        num_matches = self._replace_nvfp4_linears(
            nvfp4_linear_users,
            fp4_node,
            scale_node,
            norm_node,
            processed_nvfp4_users,
        )

        if len(norm_node.users) == 0:
            graph.erase_node(norm_node)
        return num_matches

    def _replace_nvfp4_linears(
        self,
        nvfp4_linear_users: List[Node],
        fp4_node: Node,
        scale_node: Node,
        source_node: Node,
        processed_nvfp4_users: set[int],
    ) -> int:
        cnt = 0
        graph = fp4_node.graph
        for nvfp4_user in nvfp4_linear_users:
            out_dtype = _get_out_dtype(nvfp4_user, source_node)
            with graph.inserting_before(nvfp4_user):
                gemm_node = _insert_prequant_linear(
                    graph, nvfp4_user, fp4_node, scale_node, out_dtype
                )
                gemm_node.meta.update(nvfp4_user.meta)
                nvfp4_user.replace_all_uses_with(gemm_node)
            graph.erase_node(nvfp4_user)
            processed_nvfp4_users.add(id(nvfp4_user))
            cnt += 1
        return cnt
