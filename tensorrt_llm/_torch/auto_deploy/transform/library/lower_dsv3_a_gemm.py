# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Backend-op selection for the DeepSeek-V3 MLA fused a-projection.

The generic ``fuse_gemms`` transform already merges the two adjacent rank-down
projections that share the post-input_layernorm hidden (7168-wide)::

    q_a_proj:            7168 -> 1536  (bf16)
    kv_a_proj_with_mqa:  7168 -> 576   (kv_lora_rank 512 + qk_rope_head_dim 64)

into a single ``torch_linear_simple`` over a fused ``(2112, 7168)`` weight,
split back with two narrows.  That fused GEMM goes through cuBLAS.

This transform is the *backend selector*: it leaves the fusion to
``fuse_gemms`` and only swaps the producer op of the fused ``(2112, 7168)``
bf16 linear to ``auto_deploy::dsv3_fused_a_gemm``, which dispatches the
trtllm min-latency kernel for num_tokens in [1,16] and falls back to cuBLAS
otherwise.  No weights are built and no narrows are touched.

Runs at ``post_load_fusion`` *after* ``fuse_rope_into_trtllm_mla`` so it never
interferes with rope weight handling.
"""

from typing import List, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

_LINEAR_OPS = [torch.ops.auto_deploy.torch_linear_simple, torch.ops.aten.linear]

# DSv3 fixed shapes — the custom kernel is only instantiated for these.
_HD_IN = 7168
_Q_A_OUT = 1536
_KV_A_OUT = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
_FUSED_OUT = _Q_A_OUT + _KV_A_OUT  # 2112


def _is_narrow_op(n: Node) -> bool:
    """Match either narrow representation: the generic fuse_gemms split emits
    the ``torch.narrow`` built-in; other passes emit ``aten.narrow.default``."""
    return getattr(n, "target", None) in (torch.narrow, torch.ops.aten.narrow.default)


def _find_fused_a_nodes(gm: GraphModule) -> List[Node]:
    """Return fused a-projection linear nodes: a ``(2112, 7168)`` bf16 linear
    feeding two narrows of length 1536 (q_a) and 576 (kv_a)."""
    results: List[Node] = []
    for n in gm.graph.nodes:
        if not is_op(n, _LINEAR_OPS) or len(n.args) < 2:
            continue
        w = n.args[1]
        wval = getattr(w, "meta", {}).get("val") if isinstance(w, Node) else None
        if (
            wval is None
            or tuple(wval.shape) != (_FUSED_OUT, _HD_IN)
            or wval.dtype != torch.bfloat16
        ):
            continue
        lens = {int(u.args[3]) for u in n.users if _is_narrow_op(u) and len(u.args) >= 4}
        if _Q_A_OUT in lens and _KV_A_OUT in lens:
            results.append(n)
    return results


def _swap_to_dsv3(gm: GraphModule, node: Node) -> None:
    graph = gm.graph
    inp, weight = node.args[0], node.args[1]
    with graph.inserting_before(node):
        new_node = graph.call_function(
            torch.ops.auto_deploy.dsv3_fused_a_gemm.default, args=(inp, weight)
        )
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)


@TransformRegistry.register("lower_dsv3_a_gemm")
class LowerDsv3AGemm(BaseTransform):
    """Swap the fused DSv3 MLA a-projection linear to ``dsv3_fused_a_gemm``.

    Pure backend selection on the node produced by ``fuse_gemms`` — assumes
    the fusion already happened; does not fuse anything itself.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        nodes = _find_fused_a_nodes(gm)
        if not nodes:
            ad_logger.info("No fused DSv3 a-projection found; skipping lower_dsv3_a_gemm.")
            return gm, TransformInfo(skipped=True, detail="no fused DSv3 a-gemm node")

        for node in nodes:
            _swap_to_dsv3(gm, node)

        gm.graph.eliminate_dead_code()
        gm.recompile()
        ad_logger.info(
            f"lower_dsv3_a_gemm: routed {len(nodes)} fused a-projection(s) to dsv3 kernel."
        )
        return gm, TransformInfo(
            skipped=False,
            num_matches=len(nodes),
            is_clean=False,
            has_valid_shapes=False,
        )
