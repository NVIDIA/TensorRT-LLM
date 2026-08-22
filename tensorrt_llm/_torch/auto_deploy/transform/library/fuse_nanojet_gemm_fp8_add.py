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

"""Fuse an e4m3-fed FP8 projection and the residual add reading it into one nanojet GEMM."""

from collections import Counter
from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ....nanojet_utils import ensure_tune_configs
from ...custom_ops.linear.nanojet_gemm_fp8_add_inplace import register
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.nanojet_graph import get_attr_tensor, is_fp8_linear, per_tensor_scale, set_val_meta
from ...utils.node_utils import extract_op_args, is_op, unwrap_input_through_passthrough
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


def _sole_residual_add(node: Node, order: dict):
    """The ``add`` this projection feeds and the accumulator it may safely write into.

    Safe only when nothing reads the accumulator after this add, since the op writes into it.
    """
    if len(node.users) != 1:
        return None
    add_node = next(iter(node.users))
    if not is_op(add_node, torch.ops.aten.add.Tensor) or len(add_node.args) != 2:
        return None
    left, right = add_node.args
    if left is node:
        other = right
    elif right is node:
        other = left
    else:
        return None
    if not isinstance(other, Node):
        return None
    value = other.meta.get("val")
    if value is None or value.dtype != torch.bfloat16:
        return None
    # Writing into a graph input would corrupt a tensor this pass has no claim on.
    if other.op == "placeholder":
        return None
    # Accumulating into its own activation reads and overwrites one buffer.
    if other is node.args[0]:
        return None
    # The GEMM writes into the accumulator, so the shapes must already match.
    projection = node.meta.get("val")
    if projection is None or tuple(projection.shape) != tuple(value.shape):
        return None
    add_position = order.get(add_node, -1)
    if any(order.get(user, -1) > add_position for user in other.users if user is not add_node):
        return None
    return add_node, other


def _is_fp8(node) -> bool:
    """Whether the value reaching this projection is e4m3."""
    source, _ = unwrap_input_through_passthrough(node)
    value = source.meta.get("val") if isinstance(source, Node) else None
    return value is not None and value.dtype == torch.float8_e4m3fn


class FuseNanojetGemmFP8AddConfig(TransformConfig):
    """Configuration for folding an FP8 projection into its residual add."""


@TransformRegistry.register("fuse_nanojet_gemm_fp8_add")
class FuseNanojetGemmFP8Add(BaseTransform):
    """Fuse an e4m3-fed FP8 linear and the residual add reading it into one nanojet GEMM."""

    config: FuseNanojetGemmFP8AddConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNanojetGemmFP8AddConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled or not register():
            return gm, TransformInfo(skipped=True, num_matches=0)

        ensure_tune_configs(factory)

        graph = gm.graph
        num_matches = 0
        rejected: Counter = Counter()

        order = {n: i for i, n in enumerate(graph.nodes)}
        for node in reversed(list(graph.nodes)):
            if not is_fp8_linear(node):
                continue
            if extract_op_args(node, "bias")[0] is not None:  # bias needs folding
                rejected["bias"] += 1
                continue
            activation = node.args[0] if node.args else None
            if not isinstance(activation, Node):
                rejected["activation-not-node"] += 1
                continue
            if not _is_fp8(activation):
                rejected["activation-dtype"] += 1
                continue
            weight_node = extract_op_args(node, "weight_fp8")[0]
            weight = get_attr_tensor(gm, weight_node)
            if weight is None or weight.dtype != torch.float8_e4m3fn:
                rejected["weight-dtype"] += 1
                continue
            input_scale = per_tensor_scale(gm, extract_op_args(node, "input_scale")[0])
            weight_scale = per_tensor_scale(gm, extract_op_args(node, "weight_scale")[0])
            if input_scale is None or weight_scale is None:
                rejected["scales"] += 1
                continue

            residual = _sole_residual_add(node, order)
            if residual is None:
                rejected["no-residual-add"] += 1
                continue
            add_node, accumulator = residual
            # Node order is what tells FX "accumulated, then read".
            with graph.inserting_before(add_node):
                graph.call_function(
                    torch.ops.auto_deploy.nanojet_gemm_fp8_add_inplace.default,
                    args=(activation, weight_node, accumulator, input_scale, weight_scale),
                )
            set_val_meta(accumulator, add_node)
            add_node.replace_all_uses_with(accumulator)
            graph.erase_node(add_node)
            order = {n: i for i, n in enumerate(graph.nodes)}
            num_matches += 1

        if num_matches:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(
                f"fuse_nanojet_gemm_fp8_add: {num_matches} FP8 projections folded into "
                "their residual add"
            )
        if rejected:
            ad_logger.info(
                "fuse_nanojet_gemm_fp8_add rejected: "
                + ", ".join(f"{reason} x{count}" for reason, count in rejected.items())
            )

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
