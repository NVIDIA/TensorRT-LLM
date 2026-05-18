# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QK-norm + RoPE pattern matcher: replace the captured Diffusers single-stream
attention site's (QK-norm + RoPE) subgraph with one call to
`torch.ops.visgen_auto.dit_qk_norm_rope` (which wraps TRT-LLM's hand-fused
`fused_dit_qk_norm_rope` kernel).

This pass runs *after* `_fuse_same_input_linears` in ``fusion.py``, which
already merges the Q/K/V projections into a single fused linear. The
captured pattern per single-stream attention site:

    fused_qkv = linear(x, W_qkv [, b_qkv])    # output of QKV-GEMM fusion
    q_n = narrow(fused_qkv, -1, 0, q_dim)
    k_n = narrow(fused_qkv, -1, q_dim, k_dim)
    v_n = narrow(fused_qkv, -1, q_dim + k_dim, v_dim)
    q_unf = unflatten(q_n, -1, (H, D))
    k_unf = unflatten(k_n, -1, (H, D))
    v_unf = unflatten(v_n, -1, (H, D))
    q_norm = rms_norm(q_unf, [D], norm_q.weight, eps)
    k_norm = rms_norm(k_unf, [D], norm_k.weight, eps)
    # ... RoPE math (~30 ops, ends with q_rope and k_rope) ...
    q_p = permute(q_rope, [0, 2, 1, 3])       # (B, S, H, D) -> (B, H, S, D)
    k_p = permute(k_rope, [0, 2, 1, 3])
    v_p = permute(v_unf, [0, 2, 1, 3])
    sdpa = visgen_auto.sdpa(q_p, k_p, v_p, ...)

Rewrite: replace the entire cluster with

    qkv_norm_rope = visgen_auto.dit_qk_norm_rope(
        fused_qkv, cos, sin, norm_q.weight, norm_k.weight,
        H_q, H_k, H_v, D, eps, interleave=True)
    q_p = getitem(qkv_norm_rope, 0)
    k_p = getitem(qkv_norm_rope, 1)
    v_p = getitem(qkv_norm_rope, 2)
    sdpa = visgen_auto.sdpa(q_p, k_p, v_p, ...)

Double-stream blocks (those with an intermediate `cat` that joins encoder
and hidden Q/K/V before RoPE) are handled by the separate
``fuse_qk_norm_rope_dual_stream`` matcher in this same module, which
configures the kernel's dual-stream parameters (`q_add_weight`,
`k_add_weight`, `num_txt_tokens`).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional

import torch
import torch.fx as fx

from tensorrt_llm.logger import logger

from . import ops  # noqa: F401 — ensures torch.ops.visgen_auto.sdpa is registered

_SDPA_TARGETS = (torch.ops.visgen_auto.sdpa.default,)


@dataclass
class _SingleStreamSite:
    """Resolved structural info for one single-stream attention site."""

    sdpa_node: fx.Node
    fused_qkv_linear: fx.Node  # the linear node producing packed QKV
    qkv_start: int  # offset of Q (start of QKV slice) in the linear output
    norm_q_weight: fx.Node  # get_attr for norm_q.weight
    norm_k_weight: fx.Node  # get_attr for norm_k.weight
    cos_node: fx.Node  # cos tensor consumed by RoPE
    sin_node: fx.Node  # sin tensor consumed by RoPE
    num_heads_q: int
    num_heads_k: int
    num_heads_v: int
    head_dim: int
    eps: float


def _walk_back_skipping(node: fx.Node, skip_targets: set) -> Optional[fx.Node]:
    """Walk backwards through nodes whose target is in ``skip_targets`` (single-input
    transparent ops), returning the first node *not* in ``skip_targets``.
    """
    cur = node
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if cur.op != "call_function" or cur.target not in skip_targets:
            return cur
        if not cur.args:
            return None
        cur = cur.args[0] if isinstance(cur.args[0], fx.Node) else None
    return None


_TRANSPARENT_OPS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
    torch.ops.aten._to_copy.default,
}


def _trace_back_to_narrow(start: fx.Node) -> Optional[fx.Node]:
    """Skip transparent ops back until reaching ``torch.narrow``."""
    cur = start
    for _ in range(20):
        if cur is None or not isinstance(cur, fx.Node):
            return None
        if cur.op == "call_function" and cur.target is torch.narrow:
            return cur
        if cur.op == "call_function" and cur.target in _TRANSPARENT_OPS:
            cur = cur.args[0] if cur.args else None
            continue
        return None
    return None


def _trace_back_to_rms_norm(start: fx.Node) -> Optional[fx.Node]:
    """Walk backwards from `start` through RoPE math + transparent ops, returning
    the first ``aten.rms_norm`` ancestor reachable through allowed-op chains.

    Uses BFS because `mul.Tensor(q, cos)` and `add.Tensor(q*cos, q_rot*sin)`
    each have two tensor inputs and the rms_norm ancestor can be on either
    branch. Bounded depth + visited-set guards against runaway. Bails out
    (returns None) on any non-allowed op (e.g. ``aten.cat`` for double-stream).
    """
    rope_math = {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.stack.default,
        torch.ops.aten.unbind.int,
        operator.getitem,
        torch.ops.aten.unsqueeze.default,
    } | _TRANSPARENT_OPS

    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, fx.Node) or id(cur) in visited:
            continue
        visited.add(id(cur))
        if len(visited) > 200:
            return None
        if cur.op != "call_function":
            continue
        if cur.target is torch.ops.aten.rms_norm.default:
            return cur
        if cur.target not in rope_math:
            # Skip this branch (e.g. cat for cos/sin source, or cat for
            # double-stream Q joint concat). Single-stream sites still
            # have Q reachable through other branches; double-stream
            # sites correctly fail to find rms_norm because Q goes
            # *through* cat, which we don't traverse.
            continue
        for n in _node_inputs(cur):
            stack.append(n)
    return None


def _is_visgen_fused_qkv(linear_node: fx.Node) -> bool:
    """True if the linear's weight came from the QKV-GEMM fusion pass."""
    if not (
        linear_node.op == "call_function" and linear_node.target is torch.ops.aten.linear.default
    ):
        return False
    if len(linear_node.args) < 2:
        return False
    w = linear_node.args[1]
    return (
        isinstance(w, fx.Node)
        and w.op == "get_attr"
        and str(w.target).startswith("_visgen_auto_fused_w_")
    )


_ROPE_ROTATE_OPS = {
    torch.ops.aten.neg.default,
    torch.ops.aten.stack.default,
    torch.ops.aten.unbind.int,
    operator.getitem,
}


def _node_inputs(node: fx.Node):
    """Yield every fx.Node that appears in `node.args` (including inside lists/tuples)."""
    for a in node.args:
        if isinstance(a, fx.Node):
            yield a
        elif isinstance(a, (list, tuple)):
            for sub in a:
                if isinstance(sub, fx.Node):
                    yield sub


def _has_q_ancestor(start: fx.Node) -> bool:
    """BFS through transparent + RoPE-rotate + cat ops looking for an `rms_norm` ancestor.

    Returns True if reachable — meaning this argument is the Q-side of a
    RoPE mul (either `q * cos` directly or `rotate_half(q) * sin`).
    `aten.cat.default` is included because double-stream attention has
    Q traversing a joint encoder+hidden cat between rms_norm and the
    RoPE add.
    """
    allowed = _TRANSPARENT_OPS | _ROPE_ROTATE_OPS | {torch.ops.aten.cat.default}
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, fx.Node) or id(cur) in visited:
            continue
        visited.add(id(cur))
        if len(visited) > 200:
            return False
        if cur.op != "call_function":
            continue
        if cur.target is torch.ops.aten.rms_norm.default:
            return True
        if cur.target in allowed:
            for n in _node_inputs(cur):
                stack.append(n)
    return False


def _find_cos_sin_sources(rope_add_node: fx.Node) -> Optional[tuple[fx.Node, fx.Node]]:
    """From an RoPE-result add node (q*cos + q_rot*sin), return (cos_src, sin_src).

    Captured pattern (Flux1, after type promotion to FP32):
        add(mul(q_cast, cos_cast), mul(q_rot_cast, sin_cast))
    The first mul corresponds to ``x * cos`` (Flux's `apply_rotary_emb`
    convention: ``out = x.float() * cos + x_rotated.float() * sin``), so
    its non-Q arg is cos. The second mul's non-Q arg is sin.

    Q-side identification is by tracing through transparent + RoPE-rotate
    ops to an `rms_norm` ancestor.
    """
    if rope_add_node.op != "call_function" or rope_add_node.target is not torch.ops.aten.add.Tensor:
        return None
    if len(rope_add_node.args) < 2:
        return None

    non_q_sides: list[Optional[fx.Node]] = [None, None]
    for i, mul_node in enumerate(rope_add_node.args[:2]):
        if not (
            isinstance(mul_node, fx.Node)
            and mul_node.op == "call_function"
            and mul_node.target is torch.ops.aten.mul.Tensor
        ):
            return None
        if len(mul_node.args) < 2:
            return None
        # Identify which mul-arg is the Q-side (has rms_norm ancestor).
        q_side_idx = None
        for j, arg in enumerate(mul_node.args[:2]):
            if isinstance(arg, fx.Node) and _has_q_ancestor(arg):
                q_side_idx = j
                break
        if q_side_idx is None:
            return None
        other_idx = 1 - q_side_idx
        other = mul_node.args[other_idx]
        if not isinstance(other, fx.Node):
            return None
        non_q_sides[i] = other

    # By Flux convention: first mul is ``x * cos``, second is ``x_rot * sin``.
    return non_q_sides[0], non_q_sides[1]


def _walk_back_to(start: fx.Node, sentinels: set) -> Optional[fx.Node]:
    cur = start
    for _ in range(40):
        if cur is None or not isinstance(cur, fx.Node):
            return None
        if cur.op == "call_function" and cur.target in sentinels:
            return cur
        if cur.op == "call_function" and cur.target in _TRANSPARENT_OPS:
            cur = cur.args[0] if cur.args else None
            continue
        return None
    return None


def _resolve_site(gm: fx.GraphModule, sdpa: fx.Node) -> Optional[_SingleStreamSite]:
    """Resolve a single-stream attention site; return None if not matching."""
    q_in, k_in, v_in = sdpa.args[0], sdpa.args[1], sdpa.args[2]

    # Q path must include an rms_norm + a RoPE add. Trace to the RoPE add first.
    q_rope_add = _trace_back_first_op(q_in, torch.ops.aten.add.Tensor)
    k_rope_add = _trace_back_first_op(k_in, torch.ops.aten.add.Tensor)
    if q_rope_add is None or k_rope_add is None:
        return None

    cs = _find_cos_sin_sources(q_rope_add)
    if cs is None:
        return None
    cos_node, sin_node = cs

    # Trace from RoPE add → rms_norm (skipping RoPE math)
    q_rms = _trace_back_to_rms_norm(q_rope_add)
    k_rms = _trace_back_to_rms_norm(k_rope_add)
    if q_rms is None or k_rms is None:
        return None

    # rms_norm inputs trace to narrow (after unflatten/etc.)
    q_narrow = _trace_back_to_narrow(q_rms.args[0])
    k_narrow = _trace_back_to_narrow(k_rms.args[0])
    v_narrow = _trace_back_to_narrow(v_in)
    if not (q_narrow and k_narrow and v_narrow):
        return None

    if q_narrow.args[0] is not k_narrow.args[0] or v_narrow.args[0] is not q_narrow.args[0]:
        return None

    fused_qkv = q_narrow.args[0]
    if not isinstance(fused_qkv, fx.Node) or not _is_visgen_fused_qkv(fused_qkv):
        return None

    # Double-stream sites are rejected implicitly: `_trace_back_to_rms_norm`
    # only traverses allowed ops and does NOT walk through `aten.cat`, so on
    # double-stream blocks (where Q goes through a joint cat between rms_norm
    # and the RoPE add) the rms_norm walker returns None.

    # Per-narrow sizes give us (q_dim, k_dim, v_dim). narrow args: (input, dim, start, length).
    q_dim = int(q_narrow.args[3])
    k_dim = int(k_narrow.args[3])
    v_dim = int(v_narrow.args[3])
    q_start = int(q_narrow.args[2])
    k_start = int(k_narrow.args[2])
    v_start = int(v_narrow.args[2])
    # Require QKV contiguous in the linear output. Flux1 layouts seen:
    #  - transformer_block (double): linear is QKV-only, q_start=0
    #  - single_transformer_block: linear is [MLP | Q | K | V], q_start = mlp_dim
    if k_start != q_start + q_dim or v_start != q_start + q_dim + k_dim:
        return None
    head_dim = int(q_rms.args[1][0]) if len(q_rms.args) > 1 else None
    if head_dim is None or q_dim % head_dim != 0:
        return None
    num_heads_q = q_dim // head_dim
    num_heads_k = k_dim // head_dim
    num_heads_v = v_dim // head_dim

    eps = float(q_rms.args[3]) if len(q_rms.args) > 3 else 1e-6

    return _SingleStreamSite(
        sdpa_node=sdpa,
        fused_qkv_linear=fused_qkv,
        qkv_start=q_start,
        norm_q_weight=q_rms.args[2],
        norm_k_weight=k_rms.args[2],
        cos_node=cos_node,
        sin_node=sin_node,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=eps,
    )


def _trace_back_first_op(start: fx.Node, target_op) -> Optional[fx.Node]:
    cur = start
    for _ in range(20):
        if cur is None or not isinstance(cur, fx.Node):
            return None
        if cur.op == "call_function" and cur.target is target_op:
            return cur
        if cur.op == "call_function" and cur.target in _TRANSPARENT_OPS:
            cur = cur.args[0] if cur.args else None
            continue
        return None
    return None


def _trace_back_to_chunk_split(start: fx.Node) -> Optional[tuple]:
    """Walk back from `start` looking for `aten.chunk.default(qkv_src, 3, -1)`.

    Tolerated on the way: `_TRANSPARENT_OPS`, `aten.unflatten.int` (the (H,D)
    reshape that follows the chunk pick), and a single `operator.getitem`
    (the chunk-index pick). Returns ``(chunk_node, chunk_idx)`` if the chain
    reaches a 3-way chunk on the last dim, else None.

    Used by `_resolve_split_chunk_site` for FLUX.2 single-stream blocks
    whose attention site is structured as:
        linear → split_with_sizes([qkv, mlp], -1) → getitem(0)
              → chunk(3, -1) → getitem(k) → unflatten([H, D]) → rms_norm → ...
    rather than FLUX.1's `_fuse_same_input_linears`-produced narrow chain.
    """
    cur = start
    chunk_idx = None
    for _ in range(20):
        if not isinstance(cur, fx.Node) or cur.op != "call_function":
            return None
        if cur.target is torch.ops.aten.chunk.default:
            return (cur, chunk_idx)
        if cur.target is operator.getitem:
            # First getitem we see should be the chunk-index pick.
            if chunk_idx is None and len(cur.args) >= 2:
                parent = cur.args[0]
                if (
                    isinstance(parent, fx.Node)
                    and parent.op == "call_function"
                    and parent.target is torch.ops.aten.chunk.default
                ):
                    chunk_idx = cur.args[1]
            cur = cur.args[0] if cur.args else None
            continue
        if cur.target is torch.ops.aten.unflatten.int:
            cur = cur.args[0] if cur.args else None
            continue
        if cur.target in _TRANSPARENT_OPS:
            cur = cur.args[0] if cur.args else None
            continue
        return None
    return None


def _resolve_split_chunk_site(gm: fx.GraphModule, sdpa: fx.Node) -> Optional[_SingleStreamSite]:
    """Resolve a single-stream attention site that uses a Diffusers-native
    combined QKV linear (e.g. FLUX.2's ``to_qkv_mlp_proj``), not the
    ``_fuse_same_input_linears``-produced narrow chain that FLUX.1 uses.

    The combined linear's output is split into a QKV portion and an MLP
    portion via ``aten.split_with_sizes([qkv_total, mlp_total], -1)``; the
    QKV portion is then ``aten.chunk(3, -1)``-ed into Q/K/V. We accept
    chunk-input forms:
      - ``getitem(split_with_sizes(linear, [...], -1), idx)`` (FLUX.2),
      - or ``linear.default`` directly (chunk-only, no MLP split).

    Returns a ``_SingleStreamSite`` carrying the chunk-input as the
    ``fused_qkv_linear`` (since it's already (B, S, qkv_total) shaped); the
    rewrite step's ``torch.narrow(fused_qkv_linear, -1, 0, qkv_total)`` is
    a free view in this case.
    """
    q_in, k_in, v_in = sdpa.args[0], sdpa.args[1], sdpa.args[2]

    q_rope_add = _trace_back_first_op(q_in, torch.ops.aten.add.Tensor)
    k_rope_add = _trace_back_first_op(k_in, torch.ops.aten.add.Tensor)
    if q_rope_add is None or k_rope_add is None:
        return None
    cs = _find_cos_sin_sources(q_rope_add)
    if cs is None:
        return None
    cos_node, sin_node = cs

    q_rms = _trace_back_to_rms_norm(q_rope_add)
    k_rms = _trace_back_to_rms_norm(k_rope_add)
    if q_rms is None or k_rms is None:
        return None

    q_chunk_info = _trace_back_to_chunk_split(q_rms.args[0])
    k_chunk_info = _trace_back_to_chunk_split(k_rms.args[0])
    v_chunk_info = _trace_back_to_chunk_split(v_in)
    if not (q_chunk_info and k_chunk_info and v_chunk_info):
        return None

    q_chunk, q_idx = q_chunk_info
    k_chunk, k_idx = k_chunk_info
    v_chunk, v_idx = v_chunk_info
    if q_chunk is not k_chunk or q_chunk is not v_chunk:
        return None
    if (q_idx, k_idx, v_idx) != (0, 1, 2):
        return None
    if len(q_chunk.args) < 2 or q_chunk.args[1] != 3:
        return None

    chunk_input = q_chunk.args[0]
    if not isinstance(chunk_input, fx.Node) or chunk_input.op != "call_function":
        return None

    qkv_total: Optional[int] = None
    if chunk_input.target is operator.getitem and len(chunk_input.args) >= 2:
        sws_node, sws_idx = chunk_input.args[0], chunk_input.args[1]
        if not (
            isinstance(sws_node, fx.Node)
            and sws_node.op == "call_function"
            and sws_node.target is torch.ops.aten.split_with_sizes.default
        ):
            return None
        sizes = sws_node.args[1]
        if not isinstance(sizes, (list, tuple)) or not isinstance(sws_idx, int):
            return None
        if sws_idx < 0 or sws_idx >= len(sizes):
            return None
        qkv_total = int(sizes[sws_idx])
        # The split's underlying linear must exist (used by the qkv branch);
        # we don't need its node directly because `chunk_input` is already
        # the qkv-shaped tensor.
        if not (
            isinstance(sws_node.args[0], fx.Node)
            and sws_node.args[0].op == "call_function"
            and sws_node.args[0].target is torch.ops.aten.linear.default
        ):
            return None
    elif chunk_input.target is torch.ops.aten.linear.default:
        # Direct chunk-of-linear (rare; no MLP co-mixed in the linear output).
        w = chunk_input.args[1] if len(chunk_input.args) > 1 else None
        if not isinstance(w, fx.Node) or w.op != "get_attr":
            return None
        try:
            qkv_total = int(gm.get_parameter(w.target).shape[0])
        except (AttributeError, IndexError):
            return None
    else:
        return None

    if qkv_total is None:
        return None

    head_dim_arg = q_rms.args[1] if len(q_rms.args) > 1 else None
    if not isinstance(head_dim_arg, (list, tuple)) or not head_dim_arg:
        return None
    head_dim = int(head_dim_arg[0])
    chunk_size = qkv_total // 3
    if chunk_size * 3 != qkv_total:
        return None
    num_heads = chunk_size // head_dim
    if num_heads * head_dim != chunk_size:
        return None

    eps = float(q_rms.args[3]) if len(q_rms.args) > 3 else 1e-6

    return _SingleStreamSite(
        sdpa_node=sdpa,
        # `chunk_input` is already (B, S, qkv_total)-shaped — the rewrite's
        # `narrow(fused_qkv_linear, -1, 0, qkv_total)` is a free view.
        fused_qkv_linear=chunk_input,
        qkv_start=0,
        norm_q_weight=q_rms.args[2],
        norm_k_weight=k_rms.args[2],
        cos_node=cos_node,
        sin_node=sin_node,
        num_heads_q=num_heads,
        num_heads_k=num_heads,
        num_heads_v=num_heads,
        head_dim=head_dim,
        eps=eps,
    )


def _path_contains_op(start: fx.Node, end: fx.Node, target_op) -> bool:
    visited = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur is end or id(cur) in visited:
            continue
        visited.add(id(cur))
        if cur.op == "call_function" and cur.target is target_op:
            return True
        for a in cur.args:
            if isinstance(a, fx.Node):
                stack.append(a)
    return False


def fuse_qk_norm_rope(gm: fx.GraphModule) -> int:
    """Replace single-stream QK-norm+RoPE clusters with `dit_qk_norm_rope` calls.

    Returns the number of single-stream attention sites rewritten.
    """
    g = gm.graph
    target_op = torch.ops.visgen_auto.dit_qk_norm_rope.default

    sdpa_nodes = [n for n in g.nodes if n.op == "call_function" and n.target in _SDPA_TARGETS]

    n_rewritten = 0
    n_skipped = 0
    for sdpa in sdpa_nodes:
        # Try FLUX.1-style first (fused-QKV linear from
        # `_fuse_same_input_linears` + narrow). Fall back to FLUX.2-style
        # (Diffusers-native combined `to_qkv_mlp_proj` linear that goes
        # through `split_with_sizes → getitem(0) → chunk(3, -1)`).
        site = _resolve_site(gm, sdpa) or _resolve_split_chunk_site(gm, sdpa)
        if site is None:
            n_skipped += 1
            continue

        qkv_total = (site.num_heads_q + site.num_heads_k + site.num_heads_v) * site.head_dim
        with g.inserting_before(sdpa):
            # Slice the fused linear output to just the QKV portion. Flux.1
            # single_transformer_block packs the linear as [MLP | Q | K | V]
            # so qkv_start > 0; double_transformer_block packs as [Q | K | V]
            # alone (qkv_start == 0).
            qkv_only = g.call_function(
                torch.narrow,
                args=(site.fused_qkv_linear, -1, site.qkv_start, qkv_total),
            )
            op_call = g.call_function(
                target_op,
                args=(
                    qkv_only,
                    site.cos_node,
                    site.sin_node,
                    site.norm_q_weight,
                    site.norm_k_weight,
                    site.num_heads_q,
                    site.num_heads_k,
                    site.num_heads_v,
                    site.head_dim,
                    site.eps,
                    True,  # interleave: Flux uses adjacent-pair rotation
                ),
            )
            q_out = g.call_function(operator.getitem, args=(op_call, 0))
            k_out = g.call_function(operator.getitem, args=(op_call, 1))
            v_out = g.call_function(operator.getitem, args=(op_call, 2))

        sdpa.replace_input_with(sdpa.args[0], q_out)
        sdpa.replace_input_with(sdpa.args[1], k_out)
        sdpa.replace_input_with(sdpa.args[2], v_out)

        n_rewritten += 1

    if n_rewritten:
        # `_assert_tensor_metadata` is treated as having side effects by FX
        # DCE — it'll keep the entire old RoPE+norm chain alive even though
        # SDPA no longer consumes it. Strip these asserts so DCE can collapse
        # the old chain. The asserts are export-time shape/dtype sanity
        # checks; removing them at this point is safe.
        for n in list(g.nodes):
            if (
                n.op == "call_function"
                and n.target is torch.ops.aten._assert_tensor_metadata.default
            ):
                g.erase_node(n)
        g.eliminate_dead_code()
        g.lint()
        gm.recompile()

    logger.info(
        f"VisGen-Auto qk_rope_fusion: rewrote {n_rewritten} single-stream sites; "
        f"skipped {n_skipped} (double-stream or non-matching)"
    )
    return n_rewritten


# ---------------------------------------------------------------------------
# Double-stream (joint-attention) matcher — FLUX.1/.2 dual_transformer_blocks
# ---------------------------------------------------------------------------


@dataclass
class _DoubleStreamSite:
    sdpa_node: fx.Node
    encoder_linear: fx.Node  # fused QKV linear for encoder (txt) side
    hidden_linear: fx.Node  # fused QKV linear for hidden (img) side
    qkv_start: int  # offset of Q in each linear's output
    encoder_norm_q_weight: fx.Node  # norm_added_q.weight
    encoder_norm_k_weight: fx.Node  # norm_added_k.weight
    hidden_norm_q_weight: fx.Node  # norm_q.weight
    hidden_norm_k_weight: fx.Node  # norm_k.weight
    cos_node: fx.Node
    sin_node: fx.Node
    num_heads_q: int
    num_heads_k: int
    num_heads_v: int
    head_dim: int
    eps: float


def _trace_forward_to_rms_norm(narrow_node: fx.Node) -> Optional[fx.Node]:
    """BFS forward through unflatten/transparent ops from a narrow to find its rms_norm."""
    frontier = {narrow_node}
    for _ in range(6):
        nxt = set()
        for n in frontier:
            for u in n.users:
                if u.op != "call_function":
                    continue
                if u.target is torch.ops.aten.rms_norm.default:
                    return u
                if u.target in _TRANSPARENT_OPS:
                    nxt.add(u)
        if not nxt:
            return None
        frontier = nxt
    return None


def _find_linear_qkv_rms_norms(linear: fx.Node) -> Optional[dict]:
    """Find Q/K narrows under a fused QKV linear and the rms_norms consuming them.

    Returns ``{qkv_start, q_dim, k_dim, v_dim, head_dim, eps, norm_q, norm_k}`` or None.
    """
    narrows = []
    for u in linear.users:
        if (
            u.op == "call_function"
            and u.target is torch.narrow
            and len(u.args) >= 4
            and isinstance(u.args[2], int)
            and isinstance(u.args[3], int)
        ):
            narrows.append((u, int(u.args[2]), int(u.args[3])))
    if len(narrows) < 3:
        return None
    narrows.sort(key=lambda x: x[1])

    # Find a Q,K,V triple where the offsets line up consecutively
    for i in range(len(narrows) - 2):
        (qn, q_start, q_dim), (kn, k_start, k_dim), (_vn, v_start, v_dim) = (
            narrows[i],
            narrows[i + 1],
            narrows[i + 2],
        )
        if k_start != q_start + q_dim or v_start != q_start + q_dim + k_dim:
            continue
        norm_q = _trace_forward_to_rms_norm(qn)
        norm_k = _trace_forward_to_rms_norm(kn)
        if norm_q is None or norm_k is None:
            continue
        head_dim = int(norm_q.args[1][0]) if len(norm_q.args) > 1 else None
        if head_dim is None or q_dim % head_dim != 0:
            continue
        eps = float(norm_q.args[3]) if len(norm_q.args) > 3 else 1e-6
        return {
            "qkv_start": q_start,
            "q_dim": q_dim,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "head_dim": head_dim,
            "eps": eps,
            "norm_q": norm_q,
            "norm_k": norm_k,
        }
    return None


def _resolve_double_stream_site(gm: fx.GraphModule, sdpa: fx.Node) -> Optional[_DoubleStreamSite]:
    """Resolve a double-stream attention site; return None if not matching."""
    q_in, _k_in, v_in = sdpa.args[0], sdpa.args[1], sdpa.args[2]

    # V never has rms_norm or RoPE. V path traces to a cat (joint encoder+hidden).
    v_cat = _trace_back_first_op(v_in, torch.ops.aten.cat.default)
    if v_cat is None:
        return None
    if not isinstance(v_cat.args[0], (list, tuple)) or len(v_cat.args[0]) != 2:
        return None
    # Flux convention: cat([encoder_v, hidden_v], dim=...).
    encoder_v_input, hidden_v_input = v_cat.args[0]

    encoder_v_narrow = _trace_back_to_narrow(encoder_v_input)
    hidden_v_narrow = _trace_back_to_narrow(hidden_v_input)
    if not (encoder_v_narrow and hidden_v_narrow):
        return None
    encoder_linear = encoder_v_narrow.args[0]
    hidden_linear = hidden_v_narrow.args[0]
    if not isinstance(encoder_linear, fx.Node) or not isinstance(hidden_linear, fx.Node):
        return None
    if encoder_linear is hidden_linear:
        return None  # not a dual-stream pattern
    if not (_is_visgen_fused_qkv(encoder_linear) and _is_visgen_fused_qkv(hidden_linear)):
        return None

    enc = _find_linear_qkv_rms_norms(encoder_linear)
    hid = _find_linear_qkv_rms_norms(hidden_linear)
    if not (enc and hid):
        return None
    # Require symmetric shape across the two sides
    for key in ("qkv_start", "q_dim", "k_dim", "v_dim", "head_dim"):
        if enc[key] != hid[key]:
            return None

    q_rope_add = _trace_back_first_op(q_in, torch.ops.aten.add.Tensor)
    if q_rope_add is None:
        return None
    cs = _find_cos_sin_sources(q_rope_add)
    if cs is None:
        return None
    cos_node, sin_node = cs

    return _DoubleStreamSite(
        sdpa_node=sdpa,
        encoder_linear=encoder_linear,
        hidden_linear=hidden_linear,
        qkv_start=enc["qkv_start"],
        encoder_norm_q_weight=enc["norm_q"].args[2],
        encoder_norm_k_weight=enc["norm_k"].args[2],
        hidden_norm_q_weight=hid["norm_q"].args[2],
        hidden_norm_k_weight=hid["norm_k"].args[2],
        cos_node=cos_node,
        sin_node=sin_node,
        num_heads_q=enc["q_dim"] // enc["head_dim"],
        num_heads_k=enc["k_dim"] // enc["head_dim"],
        num_heads_v=enc["v_dim"] // enc["head_dim"],
        head_dim=enc["head_dim"],
        eps=enc["eps"],
    )


def fuse_qk_norm_rope_dual_stream(graph_module: fx.GraphModule) -> int:
    """Replace double-stream attention clusters with `dit_qk_norm_rope` calls
    using the kernel's dual-stream mode (`q_add_weight` + `num_txt_tokens`).
    """
    g = graph_module.graph
    target_op = torch.ops.visgen_auto.dit_qk_norm_rope.default

    sdpa_nodes = [n for n in g.nodes if n.op == "call_function" and n.target in _SDPA_TARGETS]

    n_rewritten = 0
    n_skipped = 0
    for sdpa in sdpa_nodes:
        # Skip SDPA sites already rewritten as single-stream (their args are
        # getitem of an existing dit_qk_norm_rope call).
        q_input = sdpa.args[0]
        if (
            isinstance(q_input, fx.Node)
            and q_input.op == "call_function"
            and q_input.target is operator.getitem
        ):
            parent = q_input.args[0] if q_input.args else None
            if (
                isinstance(parent, fx.Node)
                and parent.op == "call_function"
                and parent.target is target_op
            ):
                continue

        site = _resolve_double_stream_site(graph_module, sdpa)
        if site is None:
            n_skipped += 1
            continue

        qkv_total = (site.num_heads_q + site.num_heads_k + site.num_heads_v) * site.head_dim
        with g.inserting_before(sdpa):
            enc_qkv = g.call_function(
                torch.narrow,
                args=(site.encoder_linear, -1, site.qkv_start, qkv_total),
            )
            hid_qkv = g.call_function(
                torch.narrow,
                args=(site.hidden_linear, -1, site.qkv_start, qkv_total),
            )
            # Joint concat along seq dim — encoder first per Flux convention.
            joint_qkv = g.call_function(
                torch.ops.aten.cat.default,
                args=([enc_qkv, hid_qkv], 1),
            )
            # `num_txt_tokens` is the per-batch encoder seq length, taken
            # symbolically from the encoder linear's output shape.
            num_txt_tokens = g.call_function(
                torch.ops.aten.sym_size.int,
                args=(enc_qkv, 1),
            )
            op_call = g.call_function(
                target_op,
                args=(
                    joint_qkv,
                    site.cos_node,
                    site.sin_node,
                    site.hidden_norm_q_weight,
                    site.hidden_norm_k_weight,
                    site.num_heads_q,
                    site.num_heads_k,
                    site.num_heads_v,
                    site.head_dim,
                    site.eps,
                    True,  # interleave
                ),
                kwargs={
                    "norm_q_add_weight": site.encoder_norm_q_weight,
                    "norm_k_add_weight": site.encoder_norm_k_weight,
                    "num_txt_tokens": num_txt_tokens,
                },
            )
            q_out = g.call_function(operator.getitem, args=(op_call, 0))
            k_out = g.call_function(operator.getitem, args=(op_call, 1))
            v_out = g.call_function(operator.getitem, args=(op_call, 2))

        sdpa.replace_input_with(sdpa.args[0], q_out)
        sdpa.replace_input_with(sdpa.args[1], k_out)
        sdpa.replace_input_with(sdpa.args[2], v_out)
        n_rewritten += 1

    if n_rewritten:
        for n in list(g.nodes):
            if (
                n.op == "call_function"
                and n.target is torch.ops.aten._assert_tensor_metadata.default
            ):
                g.erase_node(n)
        g.eliminate_dead_code()
        g.lint()
        graph_module.recompile()

    logger.info(
        f"VisGen-Auto qk_rope_fusion (dual): rewrote {n_rewritten} double-stream sites; "
        f"skipped {n_skipped}"
    )
    return n_rewritten
