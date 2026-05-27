# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Fuse DeepSeek-V3 MLA q_a_proj + kv_a_proj_with_mqa into
``auto_deploy::dsv3_fused_a_gemm`` (which dispatches the trtllm custom kernel).

DeepSeek-V3 MLA has two adjacent rank-down projections that share the same
input (post-input_layernorm hidden, 7168-wide):
  - q_a_proj: 7168 -> 1536, bf16
  - kv_a_proj_with_mqa: 7168 -> 576 (kv_lora_rank=512 + qk_rope_head_dim=64), bf16

The PyTorch backend concatenates these weights and dispatches a single
``trtllm::dsv3_fused_a_gemm_op`` (custom bf16 kernel for num_tokens in [1,16]).
AutoDeploy by default keeps them as two separate ``torch_linear_simple`` calls
that go through cuBLAS splitK GEMMs.  This transform mirrors the PT pattern.

Runs at ``post_load_fusion`` (after weight materialization, before
``multi_stream_mla_attn`` at ``compile``).  Once fused, the MLA layer's
fork point no longer has two linear users so ``multi_stream_mla_attn``
pattern 0 finds no match — intentional; the dsv3 kernel removes the need
for aux-stream KV overlap.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_weight_name, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

_LINEAR_OPS = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]

# DSv3 fixed shapes — the custom kernel is only instantiated for these.
_HD_IN = 7168
_Q_A_OUT = 1536
_KV_A_OUT = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
_FUSED_OUT = _Q_A_OUT + _KV_A_OUT  # 2112


def _is_linear(node: Node) -> bool:
    return is_op(node, _LINEAR_OPS)


def _find_dsv3_a_pair(gm: GraphModule) -> List[Tuple[Node, Node, Node]]:
    """Return ``(fork_point, q_a_linear, kv_a_linear)`` triples to fuse.

    Identifies fork points whose linear users include weights named
    ``*q_a_proj.weight`` and ``*kv_a_proj_with_mqa.weight`` with the exact
    DSv3 dimensions and bf16 dtype.  Other fork shapes are skipped.
    """
    results: List[Tuple[Node, Node, Node]] = []

    for fork in gm.graph.nodes:
        linear_users = [u for u in fork.users if _is_linear(u)]
        if len(linear_users) < 2:
            continue

        q_a_node: Optional[Node] = None
        kv_a_node: Optional[Node] = None
        for ln in linear_users:
            wname = extract_weight_name(ln)
            if not isinstance(wname, str):
                continue
            if wname.endswith("q_a_proj.weight"):
                q_a_node = ln
            elif wname.endswith("kv_a_proj_with_mqa.weight"):
                kv_a_node = ln

        if q_a_node is None or kv_a_node is None:
            continue

        try:
            q_w = gm.get_parameter(extract_weight_name(q_a_node))
            kv_w = gm.get_parameter(extract_weight_name(kv_a_node))
        except AttributeError:
            continue

        if q_w.dtype != torch.bfloat16 or kv_w.dtype != torch.bfloat16:
            ad_logger.info(
                f"Skipping DSv3 a-gemm fuse: non-bf16 weights (q_a={q_w.dtype}, kv_a={kv_w.dtype})"
            )
            continue
        if q_w.shape != (_Q_A_OUT, _HD_IN) or kv_w.shape != (_KV_A_OUT, _HD_IN):
            ad_logger.info(
                f"Skipping DSv3 a-gemm fuse: weight shapes "
                f"q_a={tuple(q_w.shape)}, kv_a={tuple(kv_w.shape)} "
                f"do not match ({_Q_A_OUT},{_HD_IN}) / ({_KV_A_OUT},{_HD_IN})"
            )
            continue

        results.append((fork, q_a_node, kv_a_node))

    return results


def _insert_fused_dsv3_a(
    gm: GraphModule, idx: int, fork: Node, q_a_node: Node, kv_a_node: Node
) -> bool:
    """Materialize the fused weight and rewrite the graph for one MLA pair."""
    q_wname = extract_weight_name(q_a_node)
    kv_wname = extract_weight_name(kv_a_node)
    q_w = gm.get_parameter(q_wname)
    kv_w = gm.get_parameter(kv_wname)

    # Concatenate on dim-0 (out_features axis): (Q_A_OUT + KV_A_OUT, HD_IN).
    # The auto_deploy::dsv3_fused_a_gemm wrapper passes weight.t() to the
    # trtllm op for the required column-major mat_b stride.
    fused_weight = torch.cat([q_w.data, kv_w.data], dim=0).contiguous()
    key_fused = f"fused_dsv3_a_weight_{idx}"
    setattr(gm, key_fused, nn.Parameter(fused_weight, requires_grad=False))
    ad_logger.info(
        f"Fused DSv3 a-projection weights: {q_wname} + {kv_wname} "
        f"-> {key_fused} (shape={tuple(fused_weight.shape)}, dtype={fused_weight.dtype})"
    )

    graph = gm.graph
    fork_val = fork.meta.get("val")
    has_meta = fork_val is not None
    fork_dtype = fork_val.dtype if has_meta else torch.bfloat16

    with graph.inserting_before(q_a_node):
        get_w = graph.get_attr(key_fused, torch.Tensor)
        if has_meta:
            get_w.meta["val"] = torch.empty((_FUSED_OUT, _HD_IN), dtype=fork_dtype, device="meta")

        fused_call = graph.call_function(
            torch.ops.auto_deploy.dsv3_fused_a_gemm.default,
            args=(fork, get_w),
        )
        if has_meta:
            fused_call.meta["val"] = torch.empty(
                (*fork_val.shape[:-1], _FUSED_OUT), dtype=fork_dtype, device="meta"
            )

        q_a_narrow = graph.call_function(
            torch.ops.aten.narrow.default,
            args=(fused_call, -1, 0, _Q_A_OUT),
        )
        if has_meta:
            q_a_narrow.meta["val"] = torch.empty(
                (*fork_val.shape[:-1], _Q_A_OUT), dtype=fork_dtype, device="meta"
            )
        kv_a_narrow = graph.call_function(
            torch.ops.aten.narrow.default,
            args=(fused_call, -1, _Q_A_OUT, _KV_A_OUT),
        )
        if has_meta:
            kv_a_narrow.meta["val"] = torch.empty(
                (*fork_val.shape[:-1], _KV_A_OUT), dtype=fork_dtype, device="meta"
            )

    q_a_node.replace_all_uses_with(q_a_narrow)
    kv_a_node.replace_all_uses_with(kv_a_narrow)
    graph.erase_node(q_a_node)
    graph.erase_node(kv_a_node)
    return True


@TransformRegistry.register("fuse_dsv3_a_gemm")
class FuseDsv3AGemm(BaseTransform):
    """Fuse q_a_proj + kv_a_proj_with_mqa into ``auto_deploy::dsv3_fused_a_gemm``.

    Runs at ``post_load_fusion`` after weights are materialized but before
    ``multi_stream_mla_attn`` (at ``compile``).  Once this transform runs,
    ``multi_stream_mla_attn`` pattern 0 finds no fork point with two linear
    users so it skips; that is intentional — the dsv3 kernel replaces the
    overlap benefit by collapsing two GEMMs into one fast call.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        triples = _find_dsv3_a_pair(gm)
        if not triples:
            ad_logger.info("No DSv3 MLA a-projection pairs found; skipping fuse_dsv3_a_gemm.")
            return gm, TransformInfo(skipped=True, detail="no DSv3 pair")

        n_fused = 0
        for idx, (fork, q_a_node, kv_a_node) in enumerate(triples):
            try:
                if _insert_fused_dsv3_a(gm, idx, fork, q_a_node, kv_a_node):
                    n_fused += 1
            except Exception as e:
                ad_logger.warning(
                    f"fuse_dsv3_a_gemm: failed to fuse layer {idx} "
                    f"(q_a={q_a_node.name}, kv_a={kv_a_node.name}): {e}"
                )

        if n_fused > 0:
            gm.graph.eliminate_dead_code()
            delete_all_unused_submodules(gm)
            gm.recompile()

        ad_logger.info(f"fuse_dsv3_a_gemm: fused {n_fused} MLA a-projection pairs.")
        return gm, TransformInfo(
            skipped=(n_fused == 0),
            num_matches=n_fused,
            is_clean=False,
            has_valid_shapes=False,
        )
