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

Also detects the noaux_tc routing pattern used by DeepSeek-V3 / NemotronH /
GLM4-MoE / Kimi-K2, replacing it with ``torch.ops.trtllm.noaux_tc_op``.
"""

import operator
from typing import Optional, Tuple, Type

import torch
from torch.fx import GraphModule, Node

# Importing this module registers the torch.ops.auto_deploy.triton_fused_topk_softmax op.
from ...custom_ops.fused_moe import triton_routing  # noqa: F401
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

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


# ---------------------------------------------------------------------------
# noaux_tc routing pattern helpers
# ---------------------------------------------------------------------------

_TOPK_OPS = (torch.ops.aten.topk.default,)
_VIEW_OPS = (torch.ops.aten.view.default, torch.ops.aten.reshape.default)
_ADD_OPS = (torch.ops.aten.add.Tensor,)


def _scalar_int(node_or_value) -> Optional[int]:
    """Return *node_or_value* as a Python int if it is a literal, else None."""
    if isinstance(node_or_value, int):
        return node_or_value
    return None


def _find_bias_add_after_sigmoid(sigmoid_node: Node) -> Optional[Tuple[Node, Node]]:
    """Find ``scores + bias`` user of *sigmoid_node*; return (add_node, bias_node)."""
    for user in sigmoid_node.users:
        if not is_op(user, _ADD_OPS):
            continue
        a, b = user.args[0], user.args[1]
        if a is sigmoid_node and isinstance(b, Node):
            return user, b
        if b is sigmoid_node and isinstance(a, Node):
            return user, a
    return None


def _find_group_topk(scores_with_bias: Node) -> Optional[Tuple[Node, int]]:
    """Find the ``topk(view(scores_with_bias, ...), k=2)`` user; return (node, n_group)."""
    for user in scores_with_bias.users:
        view_node = user if is_op(user, _VIEW_OPS) else None
        if view_node is None:
            continue
        # view shape can be the second arg (list of ints/Nodes)
        shape = view_node.args[1] if len(view_node.args) > 1 else None
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            continue
        n_group = _scalar_int(shape[-2])
        if n_group is None:
            continue
        for vu in view_node.users:
            if is_op(vu, _TOPK_OPS) and _scalar_int(vu.args[1]) == 2:
                return vu, n_group
    return None


def _find_outer_topk(masked_node: Node) -> Optional[Tuple[Node, int]]:
    """Find ``topk(masked, k=top_k)`` user of *masked_node*; return (node, top_k)."""
    for user in masked_node.users:
        candidate = user
        if is_op(candidate, _TOPK_OPS):
            top_k = _scalar_int(candidate.args[1])
            if top_k is not None:
                return candidate, top_k
        # allow one view in between
        if is_op(candidate, _VIEW_OPS):
            for vu in candidate.users:
                if is_op(vu, _TOPK_OPS):
                    top_k = _scalar_int(vu.args[1])
                    if top_k is not None:
                        return vu, top_k
    return None


def _descends_from(node: Node, target: Node, max_depth: int = 10) -> bool:
    """Return True if *target* is reachable from *node*'s input ancestry within max_depth hops."""
    if not isinstance(node, Node) or not isinstance(target, Node):
        return False
    visited = set()
    frontier = [(node, 0)]
    while frontier:
        n, d = frontier.pop()
        if n is target:
            return True
        if d >= max_depth or n in visited:
            continue
        visited.add(n)
        for inp in n.all_input_nodes:
            frontier.append((inp, d + 1))
    return False


def _find_gather_from_indices(indices_node: Node, scores_node: Node) -> Optional[Node]:
    """Find ``aten.gather.default(scores_node, dim, indices_node)`` user of *indices_node*."""
    for user in indices_node.users:
        if not is_op(user, torch.ops.aten.gather.default):
            continue
        if len(user.args) >= 3 and user.args[0] is scores_node and user.args[2] is indices_node:
            return user
    return None


def _is_sum_of(node, cur: Node) -> bool:
    return (
        isinstance(node, Node)
        and is_op(node, torch.ops.aten.sum.dim_IntList)
        and node.args[0] is cur
    )


def _is_normalize_divisor(divisor, cur: Node) -> bool:
    """Accept ``sum(cur, ...)`` or its epsilon-stabilized form ``sum(...) + eps``."""
    if _is_sum_of(divisor, cur):
        return True
    if isinstance(divisor, Node) and is_op(divisor, torch.ops.aten.add.Tensor):
        a, b = divisor.args[0], divisor.args[1]
        for sum_cand, eps_cand in ((a, b), (b, a)):
            if _is_sum_of(sum_cand, cur) and isinstance(eps_cand, (int, float)):
                return True
    return False


def _walk_div_then_mul(start: Node) -> Tuple[Node, float]:
    """Walk forward through optional ``div.Tensor(self, sum)`` then ``mul.Tensor(self, scalar)``.

    Returns ``(final_node, routed_scaling_factor)``. If no scale is found, the
    factor defaults to ``1.0`` and *final_node* is the last node reached on the
    chain (gather, or div if no mul, etc.).
    """
    cur = start
    # optional norm: div(cur, sum(cur, ..., keepdim=True) [+ eps])
    for user in cur.users:
        if (
            is_op(user, torch.ops.aten.div.Tensor)
            and user.args[0] is cur
            and _is_normalize_divisor(user.args[1], cur)
        ):
            cur = user
            break
    # optional scalar multiply
    for user in cur.users:
        if is_op(user, torch.ops.aten.mul.Tensor) and user.args[0] is cur:
            scalar = user.args[1]
            if isinstance(scalar, (int, float)):
                return user, float(scalar)
    return cur, 1.0


@TransformRegistry.register("match_noaux_tc_pattern")
class MatchNoAuxTCPattern(BaseTransform):
    """Match the noaux_tc MoE routing chain and replace with a fused trtllm op.

    This transform detects the DeepSeek-V3 style routing pattern::

        sigmoid → +bias → group top-k → mask → top-k → gather → [norm] → scale

    and replaces it with::

        topk_weights, topk_idx = trtllm.noaux_tc_op(
            router_logits, bias, n_group, topk_group, top_k, routed_scaling_factor
        )

    The fused kernel performs sigmoid, bias correction, group-based top-k
    selection, gather, normalization and scaling in a single CUDA kernel.
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
            # ---- anchor: aten.sigmoid -> add(bias) ------------------------
            if not is_op(node, torch.ops.aten.sigmoid.default):
                continue
            sigmoid_node = node
            router_logits = sigmoid_node.args[0]
            if not isinstance(router_logits, Node):
                continue

            bias_add = _find_bias_add_after_sigmoid(sigmoid_node)
            if bias_add is None:
                continue
            scores_with_bias_node, bias_node = bias_add

            # ---- group top-k(k=2) ----------------------------------------
            inner = _find_group_topk(scores_with_bias_node)
            if inner is None:
                continue
            inner_topk, n_group = inner

            inner_values = _get_single_getitem_user(inner_topk, 0)
            if inner_values is None:
                continue

            # ---- sum -> outer topk(k=topk_group) -------------------------
            sum_node = None
            for u in inner_values.users:
                if is_op(u, torch.ops.aten.sum.dim_IntList):
                    sum_node = u
                    break
            if sum_node is None:
                continue

            outer_grp_topk = None
            for u in sum_node.users:
                if is_op(u, _TOPK_OPS):
                    outer_grp_topk = u
                    break
            if outer_grp_topk is None:
                continue
            topk_group = _scalar_int(outer_grp_topk.args[1])
            if topk_group is None:
                continue

            # ---- final masked top-k(k=top_k) -----------------------------
            # Only accept a masked_node whose mask input descends from outer_grp_topk;
            # otherwise an unrelated branch consuming scores_with_bias could be picked.
            masked_node = None
            for u in scores_with_bias_node.users:
                if not is_op(
                    u,
                    (
                        torch.ops.aten.where.self,
                        torch.ops.aten.masked_fill.Scalar,
                        torch.ops.aten.mul.Tensor,
                    ),
                ):
                    continue
                if not _descends_from(u, outer_grp_topk):
                    continue
                masked_node = u
                break
            if masked_node is None:
                continue

            outer_topk = _find_outer_topk(masked_node)
            if outer_topk is None:
                continue
            final_topk_node, top_k = outer_topk

            final_indices = _get_single_getitem_user(final_topk_node, 1)
            if final_indices is None:
                continue

            # ---- weights branch: gather(scores, -1, topk_idx) [/ sum] * scale --
            gather_node = _find_gather_from_indices(final_indices, sigmoid_node)
            if gather_node is None:
                continue
            weights_tail, routed_scaling_factor = _walk_div_then_mul(gather_node)

            # ---- emit fused noaux_tc_op ---------------------------------
            ad_logger.info(
                "Matched noaux_tc routing pattern: "
                f"n_group={n_group}, topk_group={topk_group}, top_k={top_k}, "
                f"scale={routed_scaling_factor}"
            )

            with graph.inserting_before(sigmoid_node):
                fused = graph.call_function(
                    torch.ops.trtllm.noaux_tc_op,
                    args=(
                        router_logits,
                        bias_node,
                        n_group,
                        topk_group,
                        top_k,
                        routed_scaling_factor,
                    ),
                )
                fused_weights = graph.call_function(operator.getitem, args=(fused, 0))
                fused_indices = graph.call_function(operator.getitem, args=(fused, 1))

            final_indices.replace_all_uses_with(fused_indices)
            weights_tail.replace_all_uses_with(fused_weights)

            num_matches += 1

        if num_matches > 0:
            eliminate_dead_code(gm)
            gm.recompile()
            ad_logger.info(f"Fused {num_matches} noaux_tc routing pattern(s).")

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
