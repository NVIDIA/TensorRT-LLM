"""Transform for multi-stream execution of Q and KV projection chains in MLA attention.

In DeepSeek-style MLA (Multi-head Latent Attention), the input layernorm output
forks into two independent projection chains that merge at the RoPE + attention op:

  - **Q chain** (heavier): q_a_proj -> rms_norm -> q_b_proj -> view -> split
  - **KV chain** (lighter): kv_a_proj_with_mqa -> split -> rms_norm + view

The Q chain is ~9x heavier than the KV chain.  This transform moves the KV
projection linear onto the auxiliary CUDA stream so it executes concurrently
with the Q chain on the main stream.
"""

from collections import deque
from typing import Callable, List, Tuple

import torch
from torch.fx import GraphModule, Node

# Reuse CachedSequenceInterface for the _apply signature.
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import create_derived_custom_op
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# Reuse the stream infrastructure from the MoE multi-stream transform.
from .multi_stream_moe import (
    ModelFactory,
    _make_aux_stream_impl,
    cuda_stream_manager,
    record_event_passthrough,
)

# ---------------------------------------------------------------------------
# Supported linear op targets.  Extend this list to cover quantised variants.
# ---------------------------------------------------------------------------
_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    op_dict = {op: _create_aux_op(op) for op in ops_in_graph}

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
                kwargs={"device": torch.cuda.current_device()},
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


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------


@TransformRegistry.register("multi_stream_mla_attn")
class MultiStreamMLAAttn(BaseTransform):
    """Multi-stream Q/KV projection parallelism for MLA attention blocks."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Ensure aux stream and events are set up for the current device.
        cuda_stream_manager.add_device(torch.cuda.current_device())

        gm, num_matches = _execute_kv_proj_in_aux_stream(gm)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        with open("/tmp/after_multi_stream_mla_attn.txt", "w") as f:
            f.write(str(gm.graph))
        return gm, info
