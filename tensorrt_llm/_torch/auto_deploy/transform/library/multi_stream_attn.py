# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Multi-stream MLA transform: overlaps Q and KV paths on separate CUDA streams.

Applies up to two optimizations, tried in priority order:

**Pattern 0 — Full KV path overlap (unfused GEMMs)**:

When Q_a and KV_a projections are separate GEMMs (fuse_gemms_mixed_children
disabled), the ENTIRE KV path (GEMM + AllGather) is placed on the auxiliary
CUDA stream using begin/end_aux_stream_passthrough, overlapping with the
heavier Q path (GEMM + AllGather + LayerNorm + Q_b_proj) on main.

This eliminates the narrow→contiguous copies that fused GEMMs require and
gives better overlap since both KV GEMM and KV AllGather run on aux.

Match (in the original FX graph — any op in all_gather_ops()):
    fork_point → Q_a_proj → ... (Q chain)
              → KV_a_proj → <any AllGather op> → ...

Rewrite (the matched AllGather is rebuilt on the aux stream with
``workspace_id=_AUX_WORKSPACE_ID``; symm-mem strategies use a distinct
workspace via this id, NCCL strategies just ignore it):

                       fork_point (input layernorm out)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
              main stream                    aux stream
              ───────────                    ──────────
              Q_a_proj                       begin_aux
              Q_AllGather                    KV_a_proj
              Q_LayerNorm                    KV_AllGather (workspace_id=1)
              Q_b_proj                       end_aux
                  │                               │
                  └──────────► wait_aux ◄─────────┘
                                  │
                          downstream MLA

GPU timeline:
    Main: [Q_GEMM] → [Q_AllGather] → [Q_LayerNorm] → [Q_b_proj] → [wait_aux]
    Aux:  [KV_GEMM] → [KV_AllGather (aux ws)] → done

**Pattern 1 — Projection-only overlap**:

Moves only the KV projection linear onto the auxiliary CUDA stream; the rest
of the KV chain (split, rms_norm, view) stays on main.  The aux variant is
created via _make_aux_stream_impl, which records/waits events internally
instead of using the begin/end_aux passthroughs.

                       fork_point
                            │
                  ┌─────────┴─────────────────┐
                  ▼                           ▼
              main stream                 aux stream
              ───────────                 ──────────
              record_event                wait_event(main)
              Q_a_proj                    KV_a_proj
              ...                         record_event(aux)
              Q_b_proj                        │
                  │                           │
                  └────► wait_event(aux) ◄────┘
                            │
                   (KV split / rms_norm
                    continues on main)

GPU timeline:
    Main: [record_event] → [Q_a_proj] → [...] → [Q_b_proj] → [wait_aux_event]
    Aux:                   [KV_a_proj] → done
"""

from collections import deque
from typing import Callable, List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import create_derived_custom_op, eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import (
    _make_aux_stream_impl,
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    record_event_passthrough,
    wait_aux_stream_passthrough,
)
from ...utils.node_utils import (
    all_gather_ops,
    is_fake_quantized_linear_op,
    is_finegrained_fp8_linear_op,
    is_op,
)
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ===========================================================================
# Shared helpers
# ===========================================================================

_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]


# Distinct symm-mem workspace slot for the aux KV path. The unified
# *_dist_all_gather op routes workspace_id != 0 to a separate ProcessGroup
# (and therefore a separate symm_mem workspace), so a concurrent main-stream
# allgather on workspace_id=0 cannot clobber its buffer.
_AUX_WORKSPACE_ID = 1

# DSv3 MLA fused a-projection split sizes: q_a_lora (1536) and kv_a_with_mqa (576).
_DSV3_Q_A_OUT = 1536
_DSV3_KV_A_OUT = 576


def _is_narrow_op(n: Node) -> bool:
    """True for either narrow representation: aten.narrow.default or the
    torch.narrow built-in (emitted by the generic fuse_gemms split)."""
    return n.op == "call_function" and n.target in (
        torch.narrow,
        torch.ops.aten.narrow.default,
    )


def _is_linear(node: Node) -> bool:
    """Return ``True`` if *node* is any kind of linear op (regular or quantized)."""
    return (
        is_op(node, _LINEAR_OPS)
        or is_fake_quantized_linear_op(node)
        or is_finegrained_fp8_linear_op(node)
    )


def _has_downstream_linear(start: Node, max_depth: int = 3) -> bool:
    """BFS from *start* through its users and return ``True`` if a linear op is reachable.

    The search only follows *user* edges (downstream in the data-flow graph)
    and stops after *max_depth* hops.  ``start`` itself is **not** checked.
    """
    visited: set[Node] = {start}
    queue: deque[Tuple[Node, int]] = deque()

    for user in start.users:
        queue.append((user, 1))

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if _is_linear(node):
            return True

        if depth < max_depth:
            for user in node.users:
                queue.append((user, depth + 1))

    return False


def _get_output_feature_dim(node: Node) -> int:
    """Get the last dimension (output features) from a node's meta shape."""
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape") and len(val.shape) > 0:
        dim = val.shape[-1]
        return int(dim)
    return 0


def _build_aux_stream_all_gather_args(kv_ag: Node, new_input: Node) -> tuple:
    """Build all_gather args for the aux-stream KV path, backend-aware.

    ``trtllm_dist_all_gather`` signature:
        (tensor, strategy, dim=0, sizes=None, workspace_id=0)

    ``torch_dist_all_gather`` signature:
        (tensor, dim=0, sizes=None)
    """
    _trtllm_ag_op = getattr(torch.ops.auto_deploy, "trtllm_dist_all_gather", None)
    if _trtllm_ag_op is not None and is_op(kv_ag, _trtllm_ag_op):
        ag_strategy = kv_ag.args[1] if len(kv_ag.args) > 1 else "AUTO"
        ag_dim = kv_ag.args[2] if len(kv_ag.args) > 2 else -1
        ag_sizes = kv_ag.args[3] if len(kv_ag.args) > 3 else None
        return (new_input, ag_strategy, ag_dim, ag_sizes, _AUX_WORKSPACE_ID)

    # torch_dist_all_gather — preserve dim/sizes, no strategy or workspace_id.
    return (new_input,) + tuple(kv_ag.args[1:])


def _find_downstream_node(
    start: Node, predicate: Callable[[Node], bool], max_depth: int = 2
) -> Optional[Node]:
    """BFS to find the first downstream node matching *predicate*."""
    visited: set[Node] = {start}
    queue: deque[Tuple[Node, int]] = deque()
    for u in start.users:
        queue.append((u, 1))
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if predicate(node):
            return node
        if depth < max_depth:
            for u in node.users:
                queue.append((u, depth + 1))
    return None


# ===========================================================================
# Pattern 0: Full KV path on aux stream (unfused GEMMs)
# ===========================================================================


def _find_mla_qkv_pairs(gm: GraphModule) -> List[Tuple[Node, Node, Node]]:
    """Find MLA ``(fork_point, q_linear, kv_linear)`` triples.

    Identifies fork points where exactly 2 linears share the same input.
    The linear with the larger output dimension is Q_a, the smaller is KV_a.
    """
    results: List[Tuple[Node, Node, Node]] = []
    for node in gm.graph.nodes:
        linear_users = [u for u in node.users if _is_linear(u)]
        if len(linear_users) != 2:
            continue

        sizes = [_get_output_feature_dim(lin) for lin in linear_users]
        if sizes[0] <= 0 or sizes[1] <= 0 or sizes[0] == sizes[1]:
            continue

        if sizes[0] > sizes[1]:
            q_lin, kv_lin = linear_users[0], linear_users[1]
        else:
            q_lin, kv_lin = linear_users[1], linear_users[0]

        results.append((node, q_lin, kv_lin))

    return results


def _execute_kv_path_in_aux_stream(gm: GraphModule, world_size: int) -> Tuple[GraphModule, int]:
    """Move KV projection + AllGather onto the auxiliary CUDA stream.

    When Q and KV projections are separate (unfused) GEMMs, this places the
    entire KV path on the aux stream via begin/end_aux_stream_passthrough,
    overlapping with the heavier Q path on main.  The KV AllGather is
    re-emitted with ``workspace_id=_AUX_WORKSPACE_ID`` so symm-mem strategies
    use a distinct ProcessGroup/workspace and do not conflict with the
    main-stream AllGather.

    Returns ``(gm, num_matches)``.
    """
    if world_size <= 1:
        return gm, 0

    triples = _find_mla_qkv_pairs(gm)
    if not triples:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    num_matches = 0

    for fork_point, q_linear, kv_linear in triples:
        kv_ag = _find_downstream_node(
            kv_linear,
            lambda n: is_op(n, all_gather_ops()),
            max_depth=2,
        )
        if kv_ag is None:
            ad_logger.warning(f"No AllGather found downstream of {kv_linear.name}, skipping")
            continue

        ad_logger.info(
            f"Multi-stream MLA pattern 0 (unfused): "
            f"Q={q_linear.name} (dim={_get_output_feature_dim(q_linear)}), "
            f"KV={kv_linear.name} (dim={_get_output_feature_dim(kv_linear)}), "
            f"KV_AG={kv_ag.name} (fork={fork_point.name})"
        )

        # --- Move KV linear's get_attr args before q_linear ---
        # FP8 linears reference weight/scale_inv get_attr nodes that may sit
        # between q_linear and kv_linear in graph order.  Moving them earlier
        # is always safe (get_attr nodes have no data-flow inputs).
        q_pos = node_order.get(q_linear, 0)
        for arg in kv_linear.args:
            if isinstance(arg, Node) and arg.op == "get_attr":
                if node_order.get(arg, -1) >= q_pos:
                    q_linear.prepend(arg)
        for arg in kv_linear.kwargs.values():
            if isinstance(arg, Node) and arg.op == "get_attr":
                if node_order.get(arg, -1) >= q_pos:
                    q_linear.prepend(arg)

        # --- Build new KV path BEFORE q_linear in graph order ---
        with graph.inserting_before(q_linear):
            begin_node = graph.call_function(
                begin_aux_stream_passthrough,
                args=(fork_point,),
            )
            begin_node.meta["val"] = fork_point.meta.get("val")

            new_kv_args = tuple(begin_node if arg is fork_point else arg for arg in kv_linear.args)
            new_kv_gemm = graph.call_function(
                kv_linear.target, args=new_kv_args, kwargs=kv_linear.kwargs
            )
            for k, v in kv_linear.meta.items():
                new_kv_gemm.meta[k] = v

            new_kv_ag = graph.call_function(
                kv_ag.target,
                args=_build_aux_stream_all_gather_args(kv_ag, new_kv_gemm),
            )
            for k, v in kv_ag.meta.items():
                new_kv_ag.meta[k] = v

            end_node = graph.call_function(
                end_aux_stream_passthrough,
                args=(new_kv_ag,),
            )
            end_node.meta["val"] = kv_ag.meta.get("val")

        # --- Insert wait_aux before the earliest consumer of old kv_ag ---
        kv_ag_users = sorted(
            list(kv_ag.users.keys()),
            key=lambda n: node_order.get(n, float("inf")),
        )
        if kv_ag_users:
            earliest_user = kv_ag_users[0]
            with graph.inserting_before(earliest_user):
                wait_node = graph.call_function(
                    wait_aux_stream_passthrough,
                    args=(end_node,),
                )
                wait_node.meta["val"] = end_node.meta.get("val")
            kv_ag.replace_all_uses_with(wait_node)
        else:
            kv_ag.replace_all_uses_with(end_node)

        num_matches += 1

    if num_matches > 0:
        eliminate_dead_code(gm)

    return gm, num_matches


# ===========================================================================
# Pattern 3 (V12): Full KV path on aux WITHOUT AllGather (sharding-excluded case)
# ===========================================================================


def _execute_kv_path_in_aux_stream_no_allgather(
    gm: GraphModule, world_size: int
) -> Tuple[GraphModule, int]:
    """V12 pattern: full KV path on aux when KV linear has no AllGather downstream.

    Triggers when:
      - Unfused Q/KV GEMMs (two linears share fork point), AND
      - KV linear has NO AllGather downstream (e.g., yaml ``exclude_shard_node_filter``
        excludes ``kv_a_proj_with_mqa`` from tensor-parallel sharding).

    This is the PT-pattern equivalent for AD: KV path (kv_linear + split +
    kv_a_layernorm + view) all on aux stream, in parallel with Q path on main.

    Pattern 2 (post fuse_dsv3) wraps only the small KV-cone after the fused
    GEMM narrow. Pattern 3 wraps the larger KV chain starting from kv_linear,
    which is closer to what PT does in attention.py:1696 and avoids the CUDA
    driver folding small aux chunks back into main.
    """
    triples = _find_mla_qkv_pairs(gm)
    if not triples:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    num_matches = 0

    for fork_point, q_linear, kv_linear in triples:
        # Skip if kv_linear has an AllGather downstream — that's pattern 0's job.
        kv_ag = _find_downstream_node(kv_linear, lambda n: is_op(n, all_gather_ops()), max_depth=2)
        if kv_ag is not None:
            continue

        # BFS forward from kv_linear to find the mla_op and the KV chain nodes.
        mla_op: Optional[Node] = None
        kv_chain_set = {kv_linear}
        queue: deque[Node] = deque([kv_linear])
        while queue:
            n = queue.popleft()
            tgt_str = str(getattr(n, "target", ""))
            if "trtllm_mla" in tgt_str or "torch_mla" in tgt_str:
                mla_op = n
                continue
            for u in n.users:
                if u not in kv_chain_set:
                    kv_chain_set.add(u)
                    queue.append(u)
        if mla_op is None:
            continue

        kv_chain = [n for n in kv_chain_set if n is not kv_linear and n is not mla_op]
        kv_outputs = [n for n in kv_chain if mla_op in n.users]
        if not kv_outputs:
            continue

        last_kv_op = max(kv_outputs, key=lambda n: node_order.get(n, 0))

        ad_logger.info(
            "Multi-stream MLA pattern 3 (no-AG full KV path): "
            f"Q={q_linear.name}, KV={kv_linear.name}, mla_op={mla_op.name}, last_kv={last_kv_op.name}"
        )

        # Step 1: begin_aux before kv_linear, wrapping fork_point as input.
        with graph.inserting_before(kv_linear):
            begin_node = graph.call_function(begin_aux_stream_passthrough, args=(fork_point,))
            begin_node.meta["val"] = fork_point.meta.get("val")
        kv_linear.args = tuple(begin_node if a is fork_point else a for a in kv_linear.args)

        # Step 2: end_aux right after last_kv_op (rewire mla_op).
        with graph.inserting_after(last_kv_op):
            end_node = graph.call_function(end_aux_stream_passthrough, args=(last_kv_op,))
            end_node.meta["val"] = last_kv_op.meta.get("val")
        mla_op.args = tuple(end_node if a is last_kv_op else a for a in mla_op.args)

        # Step 3: wait_aux before mla_op on a Q-side input.
        q_inputs_to_mla = [
            u
            for u in mla_op.all_input_nodes
            if u not in kv_outputs and u.op != "get_attr" and u is not end_node
        ]
        if not q_inputs_to_mla:
            continue
        q_input_for_wait = sorted(q_inputs_to_mla, key=lambda x: node_order.get(x, 0))[-1]
        with graph.inserting_before(mla_op):
            wait_node = graph.call_function(wait_aux_stream_passthrough, args=(q_input_for_wait,))
            wait_node.meta["val"] = q_input_for_wait.meta.get("val")
        mla_op.args = tuple(wait_node if a is q_input_for_wait else a for a in mla_op.args)

        num_matches += 1

    if num_matches > 0:
        eliminate_dead_code(gm)

    return gm, num_matches


# ===========================================================================
# Pattern 1: Projection overlap (fallback)
# ===========================================================================


def _find_kv_proj_linears(gm: GraphModule) -> List[Tuple[Node, Node]]:
    """Find (fork_point, kv_linear) pairs suitable for aux-stream execution.

    A *fork point* is a node that directly feeds two or more supported linear
    ops.  Among these linears the one that does **not** lead to another linear
    within a small BFS depth is the KV projection candidate (the lighter
    branch).

    Returns a list of ``(fork_point, kv_linear_node)`` tuples.
    """
    results: List[Tuple[Node, Node]] = []

    for node in gm.graph.nodes:
        # Collect direct linear users of this node.
        linear_users = [u for u in node.users if _is_linear(u)]
        if len(linear_users) < 2:
            continue

        # Separate into "has downstream linear" (Q-like) and "does not" (KV-like).
        kv_candidates = [ln for ln in linear_users if not _has_downstream_linear(ln)]
        q_candidates = [ln for ln in linear_users if _has_downstream_linear(ln)]

        if not kv_candidates or not q_candidates:
            continue

        # Pick the KV candidate(s).  In MLA there is exactly one per fork point.
        for kv_linear in kv_candidates:
            results.append((node, kv_linear))

    return results


def _create_aux_linear_op(base_op: Callable) -> Callable:
    """Create an ``_aux`` variant of a linear op that runs on the auxiliary CUDA stream."""
    return create_derived_custom_op(
        base_op,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _execute_kv_proj_in_aux_stream(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Replace KV projection linears with aux-stream variants.

    For each matched ``(fork_point, kv_linear)`` the rewriter:

    1. Inserts ``record_event_passthrough(fork_point)`` so the main-stream
       event is recorded *before* the Q-chain kernels are submitted.
    2. Replaces the KV linear's target with its ``_aux`` variant and wires the
       ``record_event_passthrough`` output as the hidden-state input
       (creating a true data dependency).

    The remaining KV-chain ops (split, rms_norm, view) stay on the main
    stream — they are lightweight and run after the aux wait that is built
    into the derived op.

    Aux-stream variants are created lazily — only for base ops that actually
    appear in the matched KV positions.
    """
    pairs = _find_kv_proj_linears(gm)
    if not pairs:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    # Create aux ops lazily for whatever linear op types are found.
    ops_in_graph = {kv_linear.target for _, kv_linear in pairs}
    op_dict = {op: _create_aux_linear_op(op) for op in ops_in_graph}

    num_replaced = 0

    for fork_point, kv_linear in pairs:
        # Find the Q-chain linear(s) so we can insert the event record
        # *before* the earliest Q-chain op in graph order.
        q_linears = [u for u in fork_point.users if _is_linear(u) and u is not kv_linear]
        earliest_q = min(q_linears, key=lambda n: node_order.get(n, 0))

        # Insert record_event_passthrough right before the first Q-chain
        # linear so the event is recorded before Q kernels hit the GPU.
        with graph.inserting_before(earliest_q):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(fork_point,),
            )

        # Replace KV linear with its aux-stream variant.  The hidden-state
        # input (args[0]) is rewired to ``rec_node`` to create a data
        # dependency that ensures the event is recorded first.
        new_args = tuple(rec_node if arg is fork_point else arg for arg in kv_linear.args)

        with graph.inserting_after(kv_linear):
            new_node = graph.call_function(
                op_dict[kv_linear.target], args=new_args, kwargs=kv_linear.kwargs
            )

        kv_linear.replace_all_uses_with(new_node)
        graph.erase_node(kv_linear)
        num_replaced += 1

    return gm, num_replaced


# ===========================================================================
# Pattern 2: Fused-graph KV-cone overlap (post a-projection fusion)
# ===========================================================================


def _execute_kv_cone_in_aux_stream_fused(gm: GraphModule) -> Tuple[GraphModule, int]:
    """V8 — wrap KV-cone with begin/end/wait_aux (assumes KV-cone is in graph
    order BEFORE Q-cone, achieved by modeling_deepseek.py forward reorder).

    Layout AFTER user-code reorder (modeling_deepseek.py calls KV path first)::

        dsv3, narrow_kv, narrow_q  (main, views = free)
        split_with_sizes_1   ┐
        getitem×2            │  KV-cone (already in graph order before Q)
        kv_a_layernorm       │
        aten.to.dtype        │
        view_1               ┘
        q_a_layernorm  ┐
        q_b_proj       │  Q-cone
        view → split   │
        getitem×2      ┘
        trtllm_mla_with_cache

    This transform wraps the KV-cone segment with begin_aux/end_aux markers,
    and inserts wait_aux before the MLA op.  No node reordering is needed.
    Same pattern as multi_stream_moe.

    On the piecewise CG path begin/end/wait_aux are no-ops (disable_multi_stream
    from commit c38f0d24); monolithic CG (decode) has multi_stream enabled
    so the stream switch yields actual Q/KV overlap.
    """
    from collections import deque

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    num_wrapped = 0

    # Anchor structurally: the fused DSv3 a-projection is any node feeding two
    # narrows of length 1536 (q_a) and 576 (kv_a).  This is producer-agnostic --
    # matches both the dedicated dsv3_fused_a_gemm op and the generic fuse_gemms
    # torch_linear_simple -- and identifies q/kv by LENGTH (args[3]) rather than
    # start offset, so it is robust to the [q;kv] vs [kv;q] cat order.
    fused_nodes = []
    for n in graph.nodes:
        nws = [u for u in n.users if _is_narrow_op(u) and len(u.args) >= 4]
        lens = {int(u.args[3]) for u in nws}
        if _DSV3_Q_A_OUT in lens and _DSV3_KV_A_OUT in lens:
            fused_nodes.append(n)
    if not fused_nodes:
        return gm, 0

    for dsv3 in fused_nodes:
        narrows = [u for u in dsv3.users if _is_narrow_op(u) and len(u.args) >= 4]
        narrow_kv = next((n for n in narrows if int(n.args[3]) == _DSV3_KV_A_OUT), None)
        narrow_q = next((n for n in narrows if int(n.args[3]) == _DSV3_Q_A_OUT), None)
        if narrow_kv is None or narrow_q is None:
            continue

        # BFS forward from narrow_kv: collect KV cone, find trtllm_mla join.
        mla_op = None
        kv_cone: List[Node] = []
        visited = {narrow_kv}
        queue = deque([narrow_kv])
        while queue:
            n = queue.popleft()
            if "trtllm_mla" in str(getattr(n, "target", "")):
                mla_op = n
                continue
            if n is not narrow_kv:
                kv_cone.append(n)
            for u in n.users:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        if mla_op is None or not kv_cone:
            continue

        kv_cone.sort(key=lambda x: node_order.get(x, 0))
        kv_outputs = [n for n in kv_cone if mla_op in n.users]
        if not kv_outputs:
            continue
        first_kv_compute = kv_cone[0]
        last_kv_op = kv_cone[-1]

        # Sanity: KV cone must come BEFORE Q-cone in graph order (= user code
        # has already been reordered to call KV first).  If Q comes first,
        # this V8 path is not applicable — would need V7-style reorder.
        q_users = sorted(
            (u for u in narrow_q.users if u is not mla_op),
            key=lambda x: node_order.get(x, float("inf")),
        )
        if q_users and node_order.get(q_users[0], 0) < node_order.get(first_kv_compute, 0):
            ad_logger.debug(
                f"KV cone {first_kv_compute.name} is AFTER Q anchor {q_users[0].name} "
                "in graph order — V8 layout invariant violated. Skipping."
            )
            continue

        # Pick an MLA-op Q-side input to wrap with wait_aux (stream-level sync;
        # one call gates the entire aux stream).  Use last input that isn't
        # in kv_outputs / a buffer.
        q_inputs_to_mla = [
            u for u in mla_op.all_input_nodes if u not in kv_outputs and u.op != "get_attr"
        ]
        if not q_inputs_to_mla:
            continue
        wait_input_for_wait = sorted(q_inputs_to_mla, key=lambda x: node_order.get(x, 0))[-1]

        ad_logger.info(
            f"Multi-stream MLA pattern 2 (V8, KV-first user code): dsv3={dsv3.name}, "
            f"narrow_kv={narrow_kv.name}, kv_cone=[{','.join(n.name for n in kv_cone)}], "
            f"last_kv={last_kv_op.name}, q_for_wait={wait_input_for_wait.name}, "
            f"mla_op={mla_op.name}"
        )

        # Step 1: Insert begin_aux just before first_kv_compute, with narrow_kv
        # as input (semantics: aux waits for main's narrow_kv = dsv3, i.e. only
        # dsv3 must complete before KV-cone starts on aux).
        # V8: shared CUDA event (default event_id=-1) — matches the wait_aux below.
        with graph.inserting_before(first_kv_compute):
            begin_node = graph.call_function(begin_aux_stream_passthrough, args=(narrow_kv,))
            begin_node.meta["val"] = narrow_kv.meta.get("val")
        first_kv_compute.args = tuple(
            begin_node if a is narrow_kv else a for a in first_kv_compute.args
        )

        # Step 2: Insert end_aux right after last_kv_op (switch back to main
        # so Q-cone enqueues on main).
        with graph.inserting_after(last_kv_op):
            end_node = graph.call_function(end_aux_stream_passthrough, args=(last_kv_op,))
            end_node.meta["val"] = last_kv_op.meta.get("val")
        # Rewire mla_op's reference to last_kv_op → end_node.
        mla_op.args = tuple(end_node if a is last_kv_op else a for a in mla_op.args)

        # Step 3: Insert wait_aux just before mla_op on the chosen input (single
        # event wait — gates entire aux stream).
        with graph.inserting_before(mla_op):
            wait_node = graph.call_function(
                wait_aux_stream_passthrough, args=(wait_input_for_wait,)
            )
            wait_node.meta["val"] = wait_input_for_wait.meta.get("val")
        mla_op.args = tuple(wait_node if a is wait_input_for_wait else a for a in mla_op.args)

        num_wrapped += 1

    if num_wrapped > 0:
        eliminate_dead_code(gm)

    return gm, num_wrapped


# ===========================================================================
# Transform class
# ===========================================================================


@TransformRegistry.register("multi_stream_mla_attn")
class MultiStreamMLAAttn(BaseTransform):
    """Multi-stream Q/KV parallelism for MLA attention blocks.

    Pattern 0: Full KV path overlap for unfused Q/KV GEMMs (begin/end aux).
    Pattern 1: Overlaps KV projection linear with Q projection chain (fallback).
    Pattern 2: Fused-graph (post a-projection fusion) KV-cone overlap.

    Pattern 0 is tried first.  If it finds nothing, pattern 2 is tried (handles
    the post-V4 fused graph).  Pattern 1 is the legacy fallback.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        n_unfused = 0
        n_fused = 0
        n_proj = 0
        n_no_ag = 0
        # Pattern 0: full KV path on aux (unfused GEMMs with AllGather)
        gm, n_unfused = _execute_kv_path_in_aux_stream(gm, shared_config.world_size)
        ad_logger.info(f"Multi-stream MLA pattern 0 (unfused KV path): {n_unfused} matches")

        if n_unfused == 0:
            # Pattern 3 (V12): full KV path on aux for unfused GEMMs WITHOUT AllGather
            gm, n_no_ag = _execute_kv_path_in_aux_stream_no_allgather(gm, shared_config.world_size)
            ad_logger.info(f"Multi-stream MLA pattern 3 (no-AG full KV path): {n_no_ag} matches")
            if n_no_ag == 0:
                # Pattern 2: fused-graph KV-cone overlap (fuse_gemms + op-level dsv3 select).
                gm, n_fused = _execute_kv_cone_in_aux_stream_fused(gm)
                ad_logger.info(f"Multi-stream MLA pattern 2 (fused KV cone): {n_fused} matches")
                if n_fused == 0:
                    # Pattern 1: legacy projection overlap fallback
                    gm, n_proj = _execute_kv_proj_in_aux_stream(gm)
                    ad_logger.info(f"Multi-stream MLA pattern 1 (projection): {n_proj} matches")

        total = n_unfused + n_no_ag + n_fused + n_proj
        info = TransformInfo(
            skipped=False,
            num_matches=total,
            is_clean=total == 0,
            has_valid_shapes=total == 0,
        )
        return gm, info
