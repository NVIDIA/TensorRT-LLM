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

"""Graph transform to fuse MoE softmax → top-k → renormalize routing.

Detects the standard MoE routing pattern used by Qwen3.5 (and similar models):
    routing_weights = softmax(router_logits, dtype=float32)
    routing_weights, indices = topk(routing_weights, k)
    routing_weights = routing_weights / routing_weights.sum(keepdim=True)

and replaces it with a single fused Triton kernel:
    routing_weights, indices = triton_fused_topk_softmax(router_logits, k)

This leverages the mathematical equivalence:
    topk(softmax(x)); x /= x.sum()  ≡  softmax(topk(x))

The fused kernel avoids computing softmax over all experts (e.g. 256), instead
finding top-k from raw logits and computing softmax only over the k selected values.
"""

import operator
from typing import Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

# Importing this module registers the torch.ops.auto_deploy.triton_fused_topk_softmax op.
import tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.triton_routing  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import eliminate_dead_code
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

# ---------------------------------------------------------------------------
# Pattern-detection helpers
# ---------------------------------------------------------------------------


def _get_single_getitem_user(node: Node, index: int) -> Optional[Node]:
    """Return the unique ``operator.getitem(node, index)`` user, or *None*."""
    for user in node.users:
        if user.op == "call_function" and user.target is operator.getitem and user.args[1] == index:
            return user
    return None


def _trace_back_through_softmax(node: Node) -> Optional[Node]:
    """Trace backwards from *node* to find the raw logits before softmax.

    Handles the common aten decompositions produced by ``torch.export``:

    1. ``aten.softmax.int(logits, dim)``  — no dtype cast
    2. ``aten.softmax.int(logits, dim, dtype)``  — with dtype cast
    3. ``aten._softmax.default(logits, dim, half_to_float)``
    4. ``aten._to_copy(logits, dtype=float32) → aten._softmax.default(…)``

    Returns the original logits tensor (before any softmax / dtype cast) or
    *None* if the input does not originate from a softmax.
    """
    _softmax_ops = (
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
    )
    if not is_op(node, _softmax_ops):
        return None

    softmax_input = node.args[0]

    # For aten._softmax.default, check for a preceding dtype cast
    if is_op(node, torch.ops.aten._softmax.default):
        if isinstance(softmax_input, Node) and is_op(
            softmax_input, torch.ops.aten._to_copy.default
        ):
            if len(softmax_input.users) == 1:
                return softmax_input.args[0]

    # For both variants the first arg is the logits
    return softmax_input


def _find_renormalization_node(values_node: Node) -> Optional[Node]:
    """Return the ``aten.div`` node that renormalizes *values_node*, or *None*.

    Looks for the pattern::

        sum_val = aten.sum.dim_IntList(values, [dim], keepdim=True)
        renorm = aten.div.Tensor(values, sum_val)
    """
    for user in values_node.users:
        if is_op(user, torch.ops.aten.div.Tensor) and user.args[0] is values_node:
            divisor = user.args[1]
            if (
                isinstance(divisor, Node)
                and is_op(divisor, torch.ops.aten.sum.dim_IntList)
                and divisor.args[0] is values_node
            ):
                return user
        elif is_op(user, torch.ops.aten.sum.dim_IntList) and user.args[0] is values_node:
            for sum_user in user.users:
                if (
                    is_op(sum_user, torch.ops.aten.div.Tensor)
                    and sum_user.args[0] is values_node
                    and sum_user.args[1] is user
                ):
                    return sum_user
    return None


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


@TransformRegistry.register("match_moe_routing_pattern")
class MatchMoeRoutingPattern(BaseTransform):
    """Match softmax → topk → renormalize and replace with a fused Triton op.

    This transform detects the 3-op MoE routing pattern::

        routing_weights = softmax(logits, dtype=float32)
        routing_weights, indices = topk(routing_weights, k)
        routing_weights /= routing_weights.sum(keepdim=True)

    and replaces it with::

        routing_weights, indices = triton_fused_topk_softmax(logits, k)

    The fused kernel exploits the equivalence
    ``topk(softmax(x)) / Σ  ≡  softmax(topk(x))`` and avoids computing
    softmax over all experts.
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
        num_matches = 0

        for node in list(graph.nodes):
            # ---- Step 1: find an aten.topk node ----------------------------
            if not is_op(node, torch.ops.aten.topk.default):
                continue

            topk_node = node
            topk_input = topk_node.args[0]
            top_k = topk_node.args[1]  # int literal

            if not isinstance(topk_input, Node) or not isinstance(top_k, int):
                continue

            # ---- Step 2: verify that topk input is a softmax ---------------
            original_logits = _trace_back_through_softmax(topk_input)
            if original_logits is None:
                continue

            # ---- Step 3: locate getitem[0] (values) and getitem[1] (indices)
            values_node = _get_single_getitem_user(topk_node, 0)
            indices_node = _get_single_getitem_user(topk_node, 1)
            if values_node is None or indices_node is None:
                continue

            # ---- Step 4: verify values are renormalized --------------------
            renorm_node = _find_renormalization_node(values_node)
            if renorm_node is None:
                continue

            # ---- Step 5: all conditions met — insert fused op --------------
            ad_logger.info(f"Matched MoE routing pattern: softmax → topk(k={top_k}) → renormalize")

            with graph.inserting_before(topk_node):
                fused_node = graph.call_function(
                    torch.ops.auto_deploy.triton_fused_topk_softmax,
                    args=(original_logits, top_k),
                )
                fused_weights = graph.call_function(operator.getitem, args=(fused_node, 0))
                fused_indices = graph.call_function(operator.getitem, args=(fused_node, 1))

            # Replace all downstream uses
            renorm_node.replace_all_uses_with(fused_weights)
            indices_node.replace_all_uses_with(fused_indices)

            num_matches += 1

        if num_matches > 0:
            eliminate_dead_code(gm)
            gm.recompile()
            ad_logger.info(f"Fused {num_matches} MoE routing pattern(s).")

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
