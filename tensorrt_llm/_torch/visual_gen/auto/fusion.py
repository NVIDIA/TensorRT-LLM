# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiT-specific FX fusion passes for the auto path.

QKV-GEMM fusion: any group of ``aten.linear`` calls that share the same
input gets merged into one fused linear + ``torch.narrow`` slices. In a
Diffusers Flux attention block this turns 3 separate Q/K/V projections
into 1 fused projection (the same shape the handwritten ``qkv_proj``
gives), which is the dominant per-block perf term.

The implementation mirrors AutoDeploy's `_insert_fused_gemm` shape but
lives outside the AD ``BaseTransform`` scaffolding so we don't have to
stub ``ModelFactory`` / ``CachedSequenceInterface`` just to call one
graph rewrite.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.fx as fx
import torch.nn as nn

from tensorrt_llm.logger import logger


def _fuse_same_input_linears(gm: fx.GraphModule) -> int:
    """Fuse ``aten.linear`` calls that share the same input activation.

    Pattern (per attention site, captured from Diffusers Flux):
        x = ... (normed hidden state)
        q = linear(x, W_q [, b_q])    # (B, S, num_heads * head_dim)
        k = linear(x, W_k [, b_k])
        v = linear(x, W_v [, b_v])

    Rewrite:
        W_qkv = concat([W_q, W_k, W_v], dim=0)              # baked at fusion time
        b_qkv = concat([b_q, b_k, b_v], dim=0) if present
        qkv   = linear(x, W_qkv [, b_qkv])
        q = narrow(qkv, -1, 0, out_q)
        k = narrow(qkv, -1, out_q, out_k)
        v = narrow(qkv, -1, out_q + out_k, out_v)

    The fused weight and bias are registered as new parameters on `gm` so
    the cost is paid once at rewrite time, not on every forward.

    Returns the number of fused groups (one per attention-projection site).
    """
    g = gm.graph

    # Group linears by their input activation node.
    by_input: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    for n in g.nodes:
        if n.op != "call_function" or n.target is not torch.ops.aten.linear.default:
            continue
        if len(n.args) < 2:
            continue
        by_input[n.args[0]].append(n)

    n_fused_groups = 0
    fused_idx = 0

    for input_node, linears in by_input.items():
        if len(linears) < 2:
            continue

        # All weights/biases must be addressable via get_attr (i.e., come from
        # gm.named_parameters), and the bias state must be consistent across
        # the group. Mixed bias/no-bias groups are skipped to keep the pass safe.
        weight_nodes: list[fx.Node] = []
        bias_nodes: list[fx.Node | None] = []
        skip = False
        for ln in linears:
            w = ln.args[1] if len(ln.args) > 1 else None
            b = ln.args[2] if len(ln.args) > 2 else None
            if not isinstance(w, fx.Node) or w.op != "get_attr":
                skip = True
                break
            if b is not None and (not isinstance(b, fx.Node) or b.op != "get_attr"):
                skip = True
                break
            weight_nodes.append(w)
            bias_nodes.append(b)
        if skip:
            continue
        if not (all(b is None for b in bias_nodes) or all(b is not None for b in bias_nodes)):
            continue

        try:
            weights = [gm.get_parameter(w.target) for w in weight_nodes]
        except AttributeError:
            continue

        dtypes = {w.dtype for w in weights}
        if len(dtypes) != 1:
            logger.debug(
                f"VisGen-Auto fusion: skipping mixed-dtype group on {input_node.name}: {dtypes}"
            )
            continue
        in_features = {w.shape[1] for w in weights}
        if len(in_features) != 1:
            logger.debug(
                f"VisGen-Auto fusion: skipping mixed in_features group on {input_node.name}: "
                f"{in_features}"
            )
            continue

        out_sizes = [int(w.shape[0]) for w in weights]
        fused_weight = torch.cat(weights, dim=0).contiguous()
        fused_w_name = f"_visgen_auto_fused_w_{fused_idx}"
        gm.register_parameter(fused_w_name, nn.Parameter(fused_weight, requires_grad=False))

        fused_b_name: str | None = None
        if bias_nodes[0] is not None:
            biases = [gm.get_parameter(b.target) for b in bias_nodes]
            fused_bias = torch.cat(biases, dim=0).contiguous()
            fused_b_name = f"_visgen_auto_fused_b_{fused_idx}"
            gm.register_parameter(fused_b_name, nn.Parameter(fused_bias, requires_grad=False))

        first = linears[0]
        with g.inserting_before(first):
            w_node = g.get_attr(fused_w_name)
            b_node = g.get_attr(fused_b_name) if fused_b_name else None
            fused_args = (
                (input_node, w_node, b_node) if b_node is not None else (input_node, w_node)
            )
            fused_linear = g.call_function(
                torch.ops.aten.linear.default,
                args=fused_args,
            )
            offset = 0
            for orig, size in zip(linears, out_sizes):
                narrow_node = g.call_function(
                    torch.narrow,
                    args=(fused_linear, -1, offset, size),
                )
                orig.replace_all_uses_with(narrow_node)
                offset += size

        for orig in linears:
            g.erase_node(orig)

        n_fused_groups += 1
        fused_idx += 1

    if n_fused_groups > 0:
        g.lint()
        gm.recompile()

    logger.info(f"VisGen-Auto fusion: same-input-linear groups fused = {n_fused_groups}")
    return n_fused_groups
