"""Multi-stream MLA transform: overlaps Q and KV paths on separate CUDA streams.

Applies up to three optimizations, tried in priority order:

**Phase 0 — Full KV path overlap (unfused GEMMs)**:

When Q_a and KV_a projections are separate GEMMs (fuse_gemms_mixed_children
disabled), the ENTIRE KV path (GEMM + AllGather) is placed on the auxiliary
CUDA stream using begin/end_aux_stream_passthrough, overlapping with the
heavier Q path (GEMM + AllGather + LayerNorm + Q_b_proj) on main.

This eliminates the narrow→contiguous copies that fused GEMMs require and
gives better overlap since both KV GEMM and KV AllGather run on aux.

GPU timeline:
    Main: [Q_GEMM] → [Q_AllGather] → [Q_LayerNorm] → [Q_b_proj] → [wait_aux]
    Aux:  [KV_GEMM] → [KV_AllGather_NCCL] → done

**Phase 1 — Projection overlap (fallback for non-quantized graphs)**:

Moves the KV projection linear onto the auxiliary CUDA stream so it executes
concurrently with the Q chain on the main stream.

**Phase 2 — AllGather overlap (fused GEMMs)**:

After GEMM fusion (fuse_gemms_mixed_children), fused projections produce:

    fused_gemm -> narrow(q_a) -> contiguous -> allgather_q(dim=-1)
               -> narrow(kv_a) -> contiguous -> allgather_kv(dim=-1)

This phase runs the smaller (KV) AllGather on an auxiliary CUDA stream
concurrently with the larger (Q) AllGather on the main stream.

GPU timeline:
    Main: [...narrow_kv+contig...][record_event][allgather_q ████████][wait_aux]
    Aux:                          [wait_main ✓ ][allgather_kv ███][record_aux]
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
from ...utils.node_utils import is_fake_quantized_linear_op, is_finegrained_fp8_linear_op, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ===========================================================================
# Shared helpers
# ===========================================================================

_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]

_ALL_GATHER_OPS = {
    torch.ops.auto_deploy.symm_mem_all_gather,
    torch.ops.auto_deploy.trtllm_dist_all_gather,
    torch.ops.auto_deploy.torch_dist_all_gather,
    torch.ops.auto_deploy.symm_mem_all_gather_torch,
}


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
# Phase 0: Full KV path on aux stream (unfused GEMMs)
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
    overlapping with the heavier Q path on main.  The KV AllGather uses NCCL
    (trtllm_dist_all_gather) to avoid shared-workspace conflicts with
    symm_mem_all_gather on main.

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
            lambda n: is_op(n, _ALL_GATHER_OPS),
            max_depth=2,
        )
        if kv_ag is None:
            ad_logger.warning(f"No AllGather found downstream of {kv_linear.name}, skipping")
            continue

        ag_dim = kv_ag.args[1] if len(kv_ag.args) > 1 else -1

        ad_logger.info(
            f"Multi-stream MLA phase 0 (unfused): "
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
            begin_node = graph.call_function(begin_aux_stream_passthrough, args=(fork_point,))
            begin_node.meta["val"] = fork_point.meta.get("val")

            new_kv_args = tuple(begin_node if arg is fork_point else arg for arg in kv_linear.args)
            new_kv_gemm = graph.call_function(
                kv_linear.target, args=new_kv_args, kwargs=kv_linear.kwargs
            )
            for k, v in kv_linear.meta.items():
                new_kv_gemm.meta[k] = v

            new_kv_ag = graph.call_function(
                torch.ops.auto_deploy.trtllm_dist_all_gather,
                args=(new_kv_gemm, ag_dim),
            )
            for k, v in kv_ag.meta.items():
                new_kv_ag.meta[k] = v

            end_node = graph.call_function(end_aux_stream_passthrough, args=(new_kv_ag,))
            end_node.meta["val"] = kv_ag.meta.get("val")

        # --- Insert wait_aux before the earliest consumer of old kv_ag ---
        kv_ag_users = sorted(
            list(kv_ag.users.keys()),
            key=lambda n: node_order.get(n, float("inf")),
        )
        if kv_ag_users:
            earliest_user = kv_ag_users[0]
            with graph.inserting_before(earliest_user):
                wait_node = graph.call_function(wait_aux_stream_passthrough, args=(end_node,))
                wait_node.meta["val"] = end_node.meta.get("val")
            kv_ag.replace_all_uses_with(wait_node)
        else:
            kv_ag.replace_all_uses_with(end_node)

        num_matches += 1

    if num_matches > 0:
        eliminate_dead_code(gm)

    return gm, num_matches


# ===========================================================================
# Phase 1: Projection overlap (fallback)
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
# Phase 2: AllGather overlap (fused GEMMs)
# ===========================================================================


@dataclass
class _AllGatherChain:
    """Describes one allgather node and its predecessor narrow+contiguous chain."""

    ag_node: Node
    ag_op: object
    source: Node
    narrow_node: Node
    narrow_dim: int
    offset: int
    size: int
    gather_dim: int
    contig_node: Optional[Node]


def _trace_back(ag_node: Node) -> Optional[_AllGatherChain]:
    """Trace backwards from an allgather: ag <- contiguous(opt) <- narrow <- source."""
    if len(ag_node.args) < 1:
        return None

    gather_dim = ag_node.args[1] if len(ag_node.args) > 1 else 0
    ag_op = ag_node.target
    prev = ag_node.args[0]

    contig_node = None
    if isinstance(prev, Node) and prev.op == "call_method" and prev.target == "contiguous":
        contig_node = prev
        prev = prev.args[0]

    if not isinstance(prev, Node):
        return None
    if not (prev.op == "call_function" and prev.target is torch.narrow):
        return None

    narrow_node = prev
    source = narrow_node.args[0]
    narrow_dim = narrow_node.args[1]
    offset = narrow_node.args[2]
    size = narrow_node.args[3]

    if not isinstance(source, Node):
        return None

    if narrow_dim != gather_dim:
        return None

    return _AllGatherChain(
        ag_node=ag_node,
        ag_op=ag_op,
        source=source,
        narrow_node=narrow_node,
        narrow_dim=narrow_dim,
        offset=offset,
        size=size,
        gather_dim=gather_dim,
        contig_node=contig_node,
    )


def _validate_group(chains: List[_AllGatherChain]) -> bool:
    """Check that a group of chains can be safely multi-streamed."""
    if len(chains) < 2:
        return False

    first = chains[0]
    for c in chains[1:]:
        if c.gather_dim != first.gather_dim:
            return False
        if c.narrow_dim != first.narrow_dim:
            return False

    sorted_chains = sorted(chains, key=lambda c: c.offset)
    expected_offset = 0
    for c in sorted_chains:
        if c.offset != expected_offset:
            return False
        expected_offset += c.size

    return True


def _create_aux_ag_op() -> Callable:
    """Create an _aux variant of trtllm_dist_all_gather that runs on aux stream."""
    return create_derived_custom_op(
        torch.ops.auto_deploy.trtllm_dist_all_gather,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _rewrite_group_multi_stream(
    gm: GraphModule,
    chains: List[_AllGatherChain],
    aux_ag_op: Callable,
) -> None:
    """Rewrite a pair of sibling allgathers for multi-stream overlap.

    The larger allgather (Q) stays on the main stream. The smaller allgather
    (KV) is moved to the aux stream using NCCL. The KV narrow+contiguous are
    moved before the Q allgather so that the main-stream event can be recorded
    after the KV input tensor is ready, enabling full overlap.
    """
    chains_sorted = sorted(chains, key=lambda c: c.size, reverse=True)
    main_chain = chains_sorted[0]
    aux_chain = chains_sorted[1]

    graph = gm.graph
    source = main_chain.source

    # Step 1: Create new KV narrow+contiguous BEFORE the main allgather.
    # This ensures kv_input is ready on main stream before recording the event.
    with graph.inserting_before(main_chain.ag_node):
        new_narrow = graph.call_function(
            torch.narrow,
            args=(source, aux_chain.narrow_dim, aux_chain.offset, aux_chain.size),
        )
        narrow_val = aux_chain.narrow_node.meta.get("val")
        if narrow_val is not None:
            new_narrow.meta["val"] = narrow_val

        if aux_chain.contig_node is not None:
            new_contig = graph.call_method("contiguous", args=(new_narrow,))
            contig_val = aux_chain.contig_node.meta.get("val")
            if contig_val is not None:
                new_contig.meta["val"] = contig_val
            kv_input = new_contig
        else:
            kv_input = new_narrow

        rec_node = graph.call_function(record_event_passthrough, args=(kv_input,))
        rec_node.meta["val"] = kv_input.meta.get("val")

    # Step 2: Create aux allgather AFTER main allgather in graph order.
    # CPU dispatches: allgather_q on main → then _aux switches to aux stream.
    # GPU: allgather_q runs on main while allgather_kv runs on aux concurrently.
    with graph.inserting_after(main_chain.ag_node):
        new_ag = graph.call_function(aux_ag_op, args=(rec_node, aux_chain.gather_dim))
        ag_val = aux_chain.ag_node.meta.get("val")
        if ag_val is not None:
            new_ag.meta["val"] = ag_val

    # Step 3: Replace original KV allgather with the new aux allgather.
    aux_chain.ag_node.replace_all_uses_with(new_ag)


def _execute_kv_allgather_in_aux_stream(
    gm: GraphModule, world_size: int
) -> Tuple[GraphModule, int]:
    """Move KV AllGather ops to the auxiliary CUDA stream."""
    if world_size <= 1:
        return gm, 0

    aux_ag_op = _create_aux_ag_op()

    groups: Dict[Node, List[_AllGatherChain]] = defaultdict(list)
    for node in gm.graph.nodes:
        if not is_op(node, _ALL_GATHER_OPS):
            continue
        chain = _trace_back(node)
        if chain is not None:
            groups[chain.source].append(chain)

    num_matches = 0
    for source, chains in groups.items():
        if not _validate_group(chains):
            continue
        if len(chains) != 2:
            continue

        sorted_by_size = sorted(chains, key=lambda c: c.size, reverse=True)
        ad_logger.info(
            f"Multi-stream MLA allgather: Q={sorted_by_size[0].ag_node.name} "
            f"(size={sorted_by_size[0].size}) on main, "
            f"KV={sorted_by_size[1].ag_node.name} "
            f"(size={sorted_by_size[1].size}) on aux "
            f"(source={source.name})"
        )
        _rewrite_group_multi_stream(gm, chains, aux_ag_op)
        num_matches += 1

    if num_matches > 0:
        eliminate_dead_code(gm)

    return gm, num_matches


# ===========================================================================
# Transform class
# ===========================================================================


@TransformRegistry.register("multi_stream_mla_attn")
class MultiStreamMLAAttn(BaseTransform):
    """Multi-stream Q/KV parallelism for MLA attention blocks.

    Phase 0: Full KV path overlap for unfused Q/KV GEMMs (begin/end aux).
    Phase 1: Overlaps KV projection linear with Q projection chain (fallback).
    Phase 2: Overlaps KV AllGather with Q AllGather after GEMM fusion (fallback).

    Phase 0 is tried first; if it matches (unfused graph), phases 1 & 2 are
    skipped.  If phase 0 finds nothing (fused graph), phases 1 & 2 run as
    fallback.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        # Phase 0: full KV path on aux (unfused GEMMs)
        gm, n_unfused = _execute_kv_path_in_aux_stream(gm, shared_config.world_size)
        ad_logger.info(f"Multi-stream MLA phase 0 (unfused KV path): {n_unfused} matches")

        if n_unfused > 0:
            total = n_unfused
        else:
            # Fallback: Phase 1 (projection overlap) + Phase 2 (allgather overlap)
            gm, n_proj = _execute_kv_proj_in_aux_stream(gm)
            ad_logger.info(f"Multi-stream MLA phase 1 (projection): {n_proj} matches")

            gm, n_ag = _execute_kv_allgather_in_aux_stream(gm, shared_config.world_size)
            ad_logger.info(f"Multi-stream MLA phase 2 (allgather): {n_ag} matches")

            total = n_proj + n_ag

        info = TransformInfo(
            skipped=False,
            num_matches=total,
            is_clean=total == 0,
            has_valid_shapes=total == 0,
        )
        return gm, info
