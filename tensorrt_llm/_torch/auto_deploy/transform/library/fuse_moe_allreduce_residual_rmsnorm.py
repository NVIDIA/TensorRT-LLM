# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fuse the MoE all-reduce into the residual-add + RMSNorm (one fused op).

Folds a standalone all-reduce + a separate residual+norm into the fused
allreduce_residual_rmsnorm op. Algebraically identical; same op the attention
path already uses. Must run after the multi-stream pass builds the merge.

BEFORE (per layer):                          AFTER:
  routed --> dist_all_reduce                   shared --> wait_aux
       |--> view                                    |--> add(., residual)
  shared --> wait_aux --> add(., shared)       routed --> view
       |--> triton_fused_add_rms_norm(             \\__> fused_allreduce_residual_rmsnorm
                merged, residual)                       (routed, shared+residual)
  => dist_all_reduce + add + norm            => one fused op (AR epilogue does add+norm)
"""

from typing import Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import wait_aux_stream_passthrough
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _trace_ar_path(node: Node):
    """If *node* is wait_aux_passthrough(view(trtllm_dist_all_reduce(routed, strat))),
    return (routed, view_shape, strategy); else None."""
    # wait_aux passthrough (a @torch._dynamo.disable python function call_function)
    if not (node.op == "call_function" and node.target is wait_aux_stream_passthrough):
        return None
    view = node.args[0] if node.args else None
    if not (isinstance(view, Node) and is_op(view, torch.ops.aten.view)):
        return None
    ar = view.args[0]
    if not (isinstance(ar, Node) and is_op(ar, torch.ops.auto_deploy.trtllm_dist_all_reduce)):
        return None
    routed = ar.args[0]
    strategy = ar.args[1] if len(ar.args) > 1 else None
    return routed, view.args[1], strategy


@TransformRegistry.register("fuse_moe_allreduce_residual_rmsnorm")
class FuseMoEAllreduceResidualRMSNorm(BaseTransform):
    """Fold the MoE all-reduce into the (residual-add + RMSNorm) fused op (PT-style)."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if getattr(gm, "is_draft", False) or shared_config.dist_config is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        strategy = shared_config.dist_config.allreduce_strategy

        graph = gm.graph
        tfarn = torch.ops.auto_deploy.triton_fused_add_rms_norm
        num = 0

        for norm_node in list(graph.nodes):
            if not is_op(norm_node, tfarn):
                continue
            merged, prev_residual, weight, eps = (
                norm_node.args[0],
                norm_node.args[1],
                norm_node.args[2],
                norm_node.args[3],
            )
            if not (isinstance(merged, Node) and is_op(merged, torch.ops.aten.add.Tensor)):
                continue

            # one add input must be the AR path (wait_aux->view->AR), the other the shared expert
            a, b = merged.args[0], merged.args[1]
            traced = _trace_ar_path(a) if isinstance(a, Node) else None
            shared = b
            if traced is None and isinstance(b, Node):
                traced = _trace_ar_path(b)
                shared = a
            if traced is None:
                continue
            routed, view_shape, _ar_strategy = traced
            if not isinstance(shared, Node):
                continue

            ad_logger.info(
                f"[V26] fusing MoE all-reduce into residual+rmsnorm at {norm_node.name} "
                f"(routed={getattr(routed, 'name', routed)})"
            )

            with graph.inserting_before(norm_node):
                shared_sync = graph.call_function(wait_aux_stream_passthrough, args=(shared,))
                shared_plus_res = graph.call_function(
                    torch.ops.aten.add.Tensor, args=(shared_sync, prev_residual)
                )
                routed_3d = graph.call_function(
                    torch.ops.aten.view.default, args=(routed, view_shape)
                )
                fused = graph.call_function(
                    torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm.default,
                    args=(routed_3d, shared_plus_res, weight, eps, strategy),
                )
                fused.meta.update(norm_node.meta)

            # fused returns (normed, residual_out) — same tuple shape as triton_fused_add_rms_norm
            norm_node.replace_all_uses_with(fused)
            num += 1

        if num > 0:
            eliminate_dead_code(gm)

        info = TransformInfo(
            skipped=False,
            num_matches=num,
            is_clean=num == 0,
            has_valid_shapes=num == 0,
        )
        return gm, info
