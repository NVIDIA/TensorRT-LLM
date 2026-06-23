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
"""Transformations for fusing collective operations.

This module registers TRT-LLM backend patterns only. Fusion is only applied
when TRT-LLM is available (MPI mode) since it provides optimized fused kernels.
The torch backend (demollm mode) does not benefit from fusion.
"""

from functools import partial
from typing import Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import wait_aux_stream_passthrough
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# TODO: This is an overly simplified model that works well for vanilla Llama models.
# However, we eventually want to consider more sophisticated patterns such as
# * all_reduce(lin1(x) + lin2(x))
# * version above with fused GEMMs (i.e. with a split node)
# * all_reduce(pointwise_op(linear(x)))
# * ...


# ============================================================================
# Pattern Template Factory Functions
# ============================================================================


_RMSNORM_OPS = {
    "torch_rmsnorm": torch.ops.auto_deploy.torch_rmsnorm,
    "triton_rms_norm": torch.ops.auto_deploy.triton_rms_norm,
}


def _make_allreduce_residual_rmsnorm_pattern(
    add_order: str = "residual_first",
    strategy: str = "AUTO",
    rmsnorm_op_name: str = "torch_rmsnorm",
):
    """Factory function to create pattern functions for allreduce+residual+rmsnorm fusion.

    Args:
        add_order: Either "residual_first" (residual + x) or "x_first" (x + residual)
        strategy: AllReduce strategy to use in the pattern
        rmsnorm_op_name: Which rmsnorm op to match ("torch_rmsnorm" or "triton_rms_norm")

    Returns:
        A pattern function that can be used with register_ad_pattern
    """
    rmsnorm_op = _RMSNORM_OPS[rmsnorm_op_name]

    def pattern_fn(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
    ):
        """Pattern: trtllm_dist_all_reduce(x) -> add residual -> rmsnorm

        Reference PyTorch composition:
            y = trtllm_dist_all_reduce(x)
            z = residual + y  (or y + residual)
            normed = rmsnorm_op(z, weight, eps)
        Returns (normed, z)
        """
        hidden_states = torch.ops.auto_deploy.trtllm_dist_all_reduce(x, strategy)

        # Handle addition order
        if add_order == "residual_first":
            add = residual + hidden_states
        else:  # x_first
            add = hidden_states + residual

        normed = rmsnorm_op(add, weight, eps)

        return normed, add

    return pattern_fn


def _allreduce_residual_rmsnorm_replacement(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, strategy: str
):
    """Replacement using TRT-LLM fused kernel."""
    return torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm(
        x, residual, weight, eps, strategy
    )


# ============================================================================
# Transform Implementation
# ============================================================================


@TransformRegistry.register("fuse_allreduce_residual_rmsnorm")
class FuseAllreduceResidualRMSNorm(BaseTransform):
    """Fuse (allreduce + residual add + RMSNorm) into one fused op with tuple output.

    This transform only applies when TRT-LLM ops are used (MPI mode), as it provides
    optimized fused kernels. The torch backend (demollm mode) does not benefit from
    this fusion and uses unfused operations.

    Note: This transform expects torch_rmsnorm ops in the graph, which are created
    by the match_rmsnorm_pattern transform that runs earlier in the pipeline.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Collectives fusion depends on sharding (reads _sharding_transform_container).
        # Draft models are not sharded, so skip them.
        if getattr(gm, "is_draft", False):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        if shared_config.dist_config is not None:
            # Primary production path: DistConfig built by LlmArgs.init_dist_config
            # with allreduce_strategy populated from YAML.
            strategy = shared_config.dist_config.allreduce_strategy
        elif hasattr(gm, "_sharding_transform_container"):
            # Heuristic-pipeline fallback: entered only by external invocations
            # that construct InferenceOptimizer without a dist_config kwarg
            # (e.g. tests/unittest/auto_deploy/multigpu/transformations/library/
            # test_allreduce_residual_rmsnorm_fusion.py).
            strategy = gm._sharding_transform_container.config.allreduce_strategy.name
        else:
            ad_logger.warning("No dist config found, skipping allreduce-residual-rmsnorm fusion")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        ad_logger.info(f"allreduce strategy selected = {strategy!r}")

        # ============================================================================
        # Instantiate Pattern Functions
        # ============================================================================

        patterns = ADPatternMatcherPass()

        # Dummy shapes for tracing
        bsz, hidden = 8, 512
        dummy_args = [
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # x
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # residual
            torch.randn(hidden, device="meta", dtype=torch.bfloat16),  # weight
            0.1253,  # eps
        ]
        scalar_workaround = {"eps": 0.1253}

        for rmsnorm_op_name in _RMSNORM_OPS:
            for add_order in ("residual_first", "x_first"):
                pattern = _make_allreduce_residual_rmsnorm_pattern(
                    add_order=add_order, strategy=strategy, rmsnorm_op_name=rmsnorm_op_name
                )
                register_ad_pattern(
                    search_fn=pattern,
                    replace_fn=partial(_allreduce_residual_rmsnorm_replacement, strategy=strategy),
                    patterns=patterns,
                    dummy_args=dummy_args,
                    scalar_workaround=scalar_workaround,
                )

        num_matches = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


def _trace_ar_path(node: Node):
    """If *node* is wait_aux_passthrough(view(trtllm_dist_all_reduce(routed, strat))),
    return (routed, view_shape, strategy, wait_kwargs); else None.

    wait_kwargs (device/event_id) is forwarded to the relocated wait_aux so the
    fused path waits on the same aux event the original wait did."""
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
    return routed, view.args[1], strategy, dict(node.kwargs)


@TransformRegistry.register("fuse_moe_allreduce_residual_rmsnorm")
class FuseMoEAllreduceResidualRMSNorm(BaseTransform):
    """Fold the MoE all-reduce into the (residual-add + RMSNorm) fused op (PT-style).

    Sibling of FuseAllreduceResidualRMSNorm: both emit the same fused
    dist.trtllm_fused_allreduce_residual_rmsnorm op. The attention path is a
    pattern match; the MoE path is a graph walk because the all-reduce is hidden
    behind the multi-stream wait_aux passthrough and a shared-expert add. Folds a
    standalone all-reduce + a separate residual+norm into one fused op
    (algebraically identical). Must run after the multi-stream pass builds the merge.

    BEFORE (per layer):                          AFTER:
      routed --> dist_all_reduce                   shared --> wait_aux
           |--> view                                    |--> add(., residual)
      shared --> wait_aux --> add(., shared)       routed --> view
           |--> triton_fused_add_rms_norm(             \\__> fused_allreduce_residual_rmsnorm
                    merged, residual)                       (routed, shared+residual)
      => dist_all_reduce + add + norm            => one fused op (AR epilogue does add+norm)
    """

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
        default_strategy = shared_config.dist_config.allreduce_strategy

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
            routed, view_shape, ar_strategy, wait_kwargs = traced
            if not isinstance(shared, Node):
                continue
            # Fused op inherits the AR node's own strategy so it stays identical to
            # the standalone all-reduce it replaces (falls back to the dist default).
            fused_strategy = ar_strategy if ar_strategy is not None else default_strategy

            ad_logger.info(
                f"[V26] fusing MoE all-reduce into residual+rmsnorm at {norm_node.name} "
                f"(routed={getattr(routed, 'name', routed)})"
            )

            with graph.inserting_before(norm_node):
                shared_sync = graph.call_function(
                    wait_aux_stream_passthrough, args=(shared,), kwargs=wait_kwargs
                )
                shared_plus_res = graph.call_function(
                    torch.ops.aten.add.Tensor, args=(shared_sync, prev_residual)
                )
                routed_3d = graph.call_function(
                    torch.ops.aten.view.default, args=(routed, view_shape)
                )
                fused = graph.call_function(
                    torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm.default,
                    args=(routed_3d, shared_plus_res, weight, eps, fused_strategy),
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
