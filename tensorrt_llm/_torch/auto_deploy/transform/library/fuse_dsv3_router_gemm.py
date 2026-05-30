# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Lower the DeepSeek-V3 MoE gate to ``auto_deploy::dsv3_router_gemm``.

The reference gate computes logits in fp32 (``hidden.float() @ w.float().t()``),
which AutoDeploy lowers to a slow fp32 cuBLAS GEMV. The PyTorch backend instead
runs a bf16 GEMM with fp32 output via ``trtllm::dsv3_router_gemm_op`` (the
dsv3RouterGemm min-latency kernel). This transform mirrors that:

  to(fp32) -> torch_linear_simple(w_fp32[256,7168]) -> noaux_tc_op
    =>  dsv3_router_gemm(hidden_bf16, w_bf16[256,7168]) -> noaux_tc_op   (logits stay fp32)

Runs at ``post_load_fusion`` (after weight materialization), like fuse_dsv3_a_gemm.
Only the GEMM precision changes (bf16 inputs, fp32 output logits) — the routing
(noaux_tc_op) is untouched; matches the PT backend.
"""

from typing import Tuple

import torch
from torch import nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_weight_name, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

_LINEAR_OPS = [torch.ops.auto_deploy.torch_linear_simple, torch.ops.aten.linear]

# dsv3RouterGemm kernel is instantiated only for these fixed dims.
_N_EXPERTS = 256
_HD_IN = 7168


def _feeds_noaux(node: Node) -> bool:
    """True if *node*'s output is consumed by trtllm.noaux_tc_op (the router)."""
    for u in node.users:
        if is_op(u, torch.ops.trtllm.noaux_tc_op):
            return True
    return False


@TransformRegistry.register("fuse_dsv3_router_gemm")
class FuseDsv3RouterGemm(BaseTransform):
    """Replace the fp32 MoE-gate GEMM with the bf16 ``auto_deploy::dsv3_router_gemm``."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        n = 0
        for ln in list(graph.nodes):
            if not is_op(ln, _LINEAR_OPS) or not _feeds_noaux(ln):
                continue
            wname = extract_weight_name(ln)
            if not isinstance(wname, str):
                continue
            try:
                w = gm.get_parameter(wname)
            except AttributeError:
                continue
            if tuple(w.shape) != (_N_EXPERTS, _HD_IN):
                continue

            # bf16 input: peel a leading dtype-cast (the fp32 router cast) if present.
            x = ln.args[0]
            if isinstance(x, Node) and is_op(x, torch.ops.aten.to) and isinstance(x.args[0], Node):
                x_bf16 = x.args[0]
            else:
                x_bf16 = x

            # Materialize a bf16 router weight (matches the PT backend).
            key = f"dsv3_router_weight_{n}"
            setattr(
                gm, key, nn.Parameter(w.data.to(torch.bfloat16).contiguous(), requires_grad=False)
            )

            xv = x_bf16.meta.get("val")
            lv = ln.meta.get("val")
            with graph.inserting_before(ln):
                get_w = graph.get_attr(key, torch.Tensor)
                if xv is not None:
                    get_w.meta["val"] = torch.empty(
                        (_N_EXPERTS, _HD_IN), dtype=torch.bfloat16, device="meta"
                    )
                router = graph.call_function(
                    torch.ops.auto_deploy.dsv3_router_gemm.default, args=(x_bf16, get_w)
                )
                if lv is not None:
                    router.meta["val"] = lv
            ln.replace_all_uses_with(router)
            graph.erase_node(ln)
            n += 1
            ad_logger.info(
                f"fuse_dsv3_router_gemm: lowered gate {wname} -> dsv3_router_gemm ({key})"
            )

        if n > 0:
            gm.graph.eliminate_dead_code()
            delete_all_unused_submodules(gm)
            gm.recompile()

        return gm, TransformInfo(
            skipped=(n == 0), num_matches=n, is_clean=False, has_valid_shapes=False
        )
