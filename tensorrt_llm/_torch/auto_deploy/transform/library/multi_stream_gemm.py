# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generalized multi-stream transform for parallelizing fp8 GEMMs.

When multiple fp8 linear (GEMM) operations share the same input tensor, they
can execute concurrently on separate CUDA streams.  This transform identifies
such *fork points* in the FX graph and moves the **largest** GEMM (estimated by
weight shape) to the auxiliary CUDA stream while the remaining GEMMs stay on the
main stream.

The overlap benefit comes from the GPU pipeline: the main-stream GEMMs and the
aux-stream GEMM execute concurrently on the GPU, reducing the total wall-clock
time compared to sequential execution.

This is a generalization of the pattern used in ``multi_stream_mla_attn.py``
(which is MLA-specific) and can handle arbitrary fork-and-join patterns of
fp8 linear ops.

Example fork points that benefit from this transform:
  - **Linear attention layers** (4 fp8 linears: in_proj_qkv, z, b, a)
  - **Standard MHA layers** (3 fp8 linears: q_proj, k_proj, v_proj)
"""

import math
from typing import Callable, Dict, List, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import create_derived_custom_op, get_attr_by_name
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import (
    _make_aux_stream_impl,
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    record_event_passthrough,
    wait_aux_stream_passthrough,
)
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ---------------------------------------------------------------------------
# Supported linear op targets.  Extend this list to cover additional
# quantised or unquantised linear variants.
# ---------------------------------------------------------------------------
_SUPPORTED_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.trtllm_finegrained_fp8_linear,
]

# Multi-stream passthrough functions used by other transforms.  If any user of
# a fork point is one of these, we skip the fork point to avoid conflicts.
_MULTI_STREAM_OPS = [
    begin_aux_stream_passthrough,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
    record_event_passthrough,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_supported_linear(node: Node) -> bool:
    """Return ``True`` if *node* is a call to one of the supported linear ops."""
    return is_op(node, _SUPPORTED_LINEAR_OPS)


def _is_multi_stream_op(node: Node) -> bool:
    """Return ``True`` if *node* is a multi-stream passthrough function call."""
    if node.op != "call_function":
        return False
    return node.target in _MULTI_STREAM_OPS


def _estimate_weight_size(gm: GraphModule, linear_node: Node) -> int:
    """Estimate the GEMM cost of a linear node from its weight shape.

    For a linear with weight ``[N, K]``, the cost is proportional to ``N * K``
    (since the M dimension is shared across all linears at the same fork point).

    The weight is ``args[1]`` for all supported linear ops.  We first try the
    node's meta information (``node.meta["val"].shape``), falling back to
    accessing the actual tensor from the graph module.

    Returns:
        An integer proportional to the GEMM cost (product of weight dimensions).
    """
    weight_node = linear_node.args[1]

    # Try meta shape first (available after shape propagation).
    val = weight_node.meta.get("val") if hasattr(weight_node, "meta") else None
    if val is not None and hasattr(val, "shape") and len(val.shape) >= 2:
        return math.prod(val.shape)

    # Fallback: access the actual tensor via the get_attr path.
    if weight_node.op == "get_attr":
        try:
            weight_tensor = get_attr_by_name(gm, weight_node.target)
            return weight_tensor.numel()
        except AttributeError:
            pass

    # If we cannot determine the size, return 0 (this linear will not be
    # selected as the largest).
    ad_logger.warning(
        f"Could not estimate weight size for linear node {linear_node.name}; "
        "it will not be considered as the largest GEMM."
    )
    return 0


def _create_aux_op(base_op: Callable) -> Callable:
    """Create an ``_aux`` variant of a linear op that runs on the auxiliary CUDA stream.

    Uses a custom ``make_fake`` that delegates to the base op's registered fake
    so that output shapes are computed correctly (linear output shape != input shape).
    """
    return create_derived_custom_op(
        base_op,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _find_gemm_fork_points(
    gm: GraphModule,
    supported_ops: List[Callable],
) -> List[Tuple[Node, List[Node]]]:
    """Find fork points where 2+ supported linear ops share the same input.

    Returns a list of ``(fork_point, [linear_users])`` tuples.  Fork points
    that already have multi-stream ops among their users are skipped to avoid
    conflicts with other multi-stream transforms (e.g. ``multi_stream_moe``).
    """
    results: List[Tuple[Node, List[Node]]] = []

    for node in gm.graph.nodes:
        # Collect direct supported-linear users of this node.
        linear_users = [u for u in node.users if is_op(u, supported_ops)]
        if len(linear_users) < 2:
            continue

        # Skip if any user of this fork point is already a multi-stream op.
        if any(_is_multi_stream_op(u) for u in node.users):
            ad_logger.debug(f"Skipping fork point {node.name}: already has multi-stream ops.")
            continue

        results.append((node, linear_users))

    return results


def _move_users_after(graph, target_node: Node) -> None:
    """Move any transitive users of *target_node* that precede it to after it.

    After inserting an aux node at a late position in the graph and replacing
    uses of the original node, some former downstream nodes may violate
    topological order (they reference *target_node* but appear before it in
    the graph's linked list).  This function restores topological order by
    moving those nodes to just after *target_node* while preserving their
    relative order.

    This is safe because:
    - Moved nodes originally depended on the (now-erased) largest linear, so
      their non-aux inputs all appear before the original linear position,
      which is before *target_node*.
    - Moving them forward (to a later position) cannot place them before any
      of their other inputs.
    """
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    target_pos = node_order[target_node]

    # BFS to find all transitive users that appear before target_node.
    nodes_to_move: List[Node] = []
    visited: set = set()
    queue = list(target_node.users.keys())

    while queue:
        n = queue.pop(0)
        if n in visited or n.op == "output":
            continue
        visited.add(n)
        if node_order.get(n, float("inf")) < target_pos:
            nodes_to_move.append(n)
            queue.extend(n.users.keys())

    if not nodes_to_move:
        return

    # Sort by original order to maintain relative dependencies.
    nodes_to_move.sort(key=lambda n: node_order[n])

    # Move each node to right after target_node (or the previously moved node).
    anchor = target_node
    for n in nodes_to_move:
        anchor.append(n)
        anchor = n


def _parallelize_largest_gemm(
    gm: GraphModule,
    supported_ops: List[Callable],
) -> Tuple[GraphModule, int]:
    """Move the largest GEMM at each fork point to the auxiliary CUDA stream.

    For each fork point with 2+ supported linear users:

    1. Estimate weight size for each linear to identify the largest.
    2. Insert ``record_event_passthrough(fork_point)`` before the earliest
       non-largest linear to record the main-stream event (data is ready).
    3. Create an ``_aux`` variant of the largest linear's op.
    4. Insert the aux node **after** the latest non-largest linear in graph
       order so that the GPU pipeline can overlap the main-stream GEMMs with
       the aux-stream GEMM.
    5. Wire the aux node's hidden-state input to the ``record_event_passthrough``
       output (data dependency ensures event recording precedes aux dispatch).
    6. Replace all uses of the original largest linear with the aux node and
       erase the original.
    7. Move any downstream nodes of the original largest linear that now
       precede the aux node in graph order to after it (restoring topological
       order without sacrificing GPU overlap).
    """
    fork_points = _find_gemm_fork_points(gm, supported_ops)
    if not fork_points:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    # Create aux ops lazily for whatever linear op types are found.
    op_dict: Dict[Callable, Callable] = {}

    num_replaced = 0

    for fork_point, linear_users in fork_points:
        # ---- Step 1: Identify the largest linear by weight size. ----
        sizes = {ln: _estimate_weight_size(gm, ln) for ln in linear_users}
        largest = max(linear_users, key=lambda ln: sizes[ln])
        remaining = [ln for ln in linear_users if ln is not largest]

        if not remaining:
            # Shouldn't happen (we require 2+ linears), but guard anyway.
            continue

        # Sort remaining linears by their position in the graph.
        remaining.sort(key=lambda n: node_order.get(n, 0))
        earliest_remaining = remaining[0]
        latest_remaining = remaining[-1]

        ad_logger.info(
            f"Fork point {fork_point.name}: moving {largest.name} "
            f"(weight size {sizes[largest]}) to aux stream; "
            f"{len(remaining)} linear(s) stay on main stream."
        )

        # ---- Step 2: Insert record_event_passthrough. ----
        # Placed before the earliest remaining linear so the main-stream event
        # is recorded *before* any main-stream GEMMs are dispatched.
        with graph.inserting_before(earliest_remaining):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(fork_point,),
            )

        # ---- Step 3: Create aux op lazily. ----
        if largest.target not in op_dict:
            op_dict[largest.target] = _create_aux_op(largest.target)

        # ---- Step 4: Insert aux node after the latest remaining linear. ----
        # This ensures all main-stream GEMMs are dispatched to the GPU before
        # the aux node submits its work + wait, enabling overlap.
        new_args = tuple(rec_node if arg is fork_point else arg for arg in largest.args)

        with graph.inserting_after(latest_remaining):
            aux_node = graph.call_function(
                op_dict[largest.target],
                args=new_args,
                kwargs=largest.kwargs,
            )

        # ---- Step 5 & 6: Replace uses and erase original. ----
        largest.replace_all_uses_with(aux_node)
        graph.erase_node(largest)

        # ---- Step 7: Restore topological order. ----
        # The downstream nodes of the original largest linear (e.g. view,
        # reshape, split) may now appear *before* aux_node in graph order
        # because aux_node was inserted after the latest remaining linear.
        # Move those nodes to after aux_node so the graph is valid.
        _move_users_after(graph, aux_node)

        num_replaced += 1

    return gm, num_replaced


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------


@TransformRegistry.register("multi_stream_gemm")
class MultiStreamGemm(BaseTransform):
    """Multi-stream parallelization of fp8 GEMMs sharing the same input.

    For each fork point where 2+ fp8 linear ops share the same input tensor,
    the largest GEMM (by weight shape) is moved to the auxiliary CUDA stream
    so it executes concurrently with the remaining GEMMs on the main stream.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Ensure aux stream and events are set up for the current device.
        cuda_stream_manager.add_device(torch.cuda.current_device())

        gm, num_matches = _parallelize_largest_gemm(gm, _SUPPORTED_LINEAR_OPS)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
