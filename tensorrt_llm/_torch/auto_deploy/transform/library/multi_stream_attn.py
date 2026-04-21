"""Multi-stream MLA transform: overlaps Q and KV paths on separate CUDA streams.

Applies two independent optimizations within a single transform:

**Phase 1 — Projection overlap**:

In DeepSeek-style MLA (Multi-head Latent Attention), the input layernorm output
forks into two independent projection chains that merge at the RoPE + attention op:

  - **Q chain** (heavier): q_a_proj -> rms_norm -> q_b_proj -> view -> split
  - **KV chain** (lighter): kv_a_proj_with_mqa -> split -> rms_norm + view

The Q chain is ~9x heavier than the KV chain.  This phase moves the KV
projection linear onto the auxiliary CUDA stream so it executes concurrently
with the Q chain on the main stream.

**Phase 2 — AllGather overlap**:

After GEMM fusion (fuse_gemms_mixed_children), fused projections produce:

    fused_gemm -> narrow(q_a) -> contiguous -> allgather_q(dim=-1)
               -> narrow(kv_a) -> contiguous -> allgather_kv(dim=-1)

This phase runs the smaller (KV) AllGather on an auxiliary CUDA stream
concurrently with the larger (Q) AllGather on the main stream, overlapping
their latency.  The KV AllGather uses NCCL (trtllm_dist_all_gather) on the
aux stream to avoid shared-workspace conflicts with symm_mem_all_gather on
the main stream.

To achieve GPU overlap the graph is reordered so that the KV narrow+contiguous
execute BEFORE the Q AllGather, then the main-stream event is recorded:

    source -> narrow_kv -> contiguous_kv -> record_event(main)
           -> narrow_q  -> contiguous_q  -> allgather_q          [main]
           -> trtllm_dist_all_gather_aux(rec_node, dim)          [aux]

GPU timeline:
    Main: [...narrow_kv+contig...][record_event][allgather_q ████████][wait_aux]
    Aux:                          [wait_main ✓ ][allgather_kv ███][record_aux]

Must run AFTER fuse_gemms_mixed_children in the post_load_fusion stage.
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
    cuda_stream_manager,
    record_event_passthrough,
)
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ===========================================================================
# Phase 1: Projection overlap
# ===========================================================================

_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]


def _is_linear(node: Node) -> bool:
    """Return ``True`` if *node* is a call to one of the supported linear ops."""
    return is_op(node, _LINEAR_OPS)


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
# Phase 2: AllGather overlap
# ===========================================================================

_ALL_GATHER_OPS = {
    torch.ops.auto_deploy.symm_mem_all_gather,
    torch.ops.auto_deploy.trtllm_dist_all_gather,
    torch.ops.auto_deploy.torch_dist_all_gather,
    torch.ops.auto_deploy.symm_mem_all_gather_torch,
}


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

    Phase 1: Overlaps KV projection linear with Q projection chain.
    Phase 2: Overlaps KV AllGather with Q AllGather (after GEMM fusion).
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        # Phase 1: projection overlap
        gm, n_proj = _execute_kv_proj_in_aux_stream(gm)
        ad_logger.info(f"Multi-stream MLA phase 1 (projection): {n_proj} matches")

        # Phase 2: allgather overlap
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
