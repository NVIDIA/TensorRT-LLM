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

import operator
from functools import partial
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
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
            # Legacy fallback: entered only by external invocations that construct
            # InferenceOptimizer without a dist_config kwarg (e.g.
            # tests/unittest/auto_deploy/multigpu/transformations/library/
            # test_allreduce_residual_rmsnorm_fusion.py). Will be removed together
            # with the legacy sharding pipeline (sharding.py).
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


# ============================================================================
# fuse_allreduce_residual_rmsnorm + downstream FP8 quant
# ============================================================================


def _get_single_norm_consumer(fused_node: Node) -> Optional[Node]:
    """Return the unique downstream consumer of fused_node[0] when it's a single
    trtllm_quant_fp8_linear with matching scalar input_scale; else None.

    fused_node has 2 outputs: (norm, residual). We need to confirm getitem(0) is
    the only consumer of the norm output, and that consumer is a single FP8
    linear node.
    """
    norm_getitem: Optional[Node] = None
    for user in fused_node.users:
        if user.op == "call_function" and user.target is operator.getitem and len(user.args) >= 2:
            idx = user.args[1]
            if idx == 0:
                if norm_getitem is not None:
                    return None
                norm_getitem = user

    if norm_getitem is None:
        return None
    if len(norm_getitem.users) != 1:
        return None
    consumer = next(iter(norm_getitem.users))
    if not is_op(consumer, torch.ops.auto_deploy.trtllm_quant_fp8_linear):
        return None
    return consumer


@TransformRegistry.register("fuse_allreduce_residual_rmsnorm_quant_fp8")
class FuseAllreduceResidualRMSNormQuantFP8(BaseTransform):
    """Fuse (allreduce + residual + RMSNorm + FP8 quant) into one fused op.

    Pattern (post ``fuse_allreduce_residual_rmsnorm``)::

        fused = trtllm_fused_allreduce_residual_rmsnorm(x, residual, w, eps, strategy)
        norm = fused[0]  # bf16
        residual_out = fused[1]  # bf16
        linear_out = trtllm_quant_fp8_linear(norm, w_fp8, bias, input_scale, ...)

    The downstream FP8 linear internally calls ``static_quantize_e4m3_per_tensor``
    to quantize ``norm`` to FP8 before the matmul. By folding that quantize into
    the allreduce kernel (``AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8``) we
    save one kernel launch per allreduce site - meaningful at small-batch
    decode where launch count dominates allreduce wall time.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        num_matches = 0

        fused_op_target = torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm.default
        fused_quant_op_target = (
            torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_quant_fp8.default
        )

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not fused_op_target:
                continue

            linear_consumer = _get_single_norm_consumer(node)
            if linear_consumer is None:
                continue

            # Extract scale from linear; quant_fp8_linear args:
            # (input, weight_fp8, bias, input_scale, weight_scale, ...)
            if len(linear_consumer.args) < 4:
                continue
            input_scale = linear_consumer.args[3]
            if not isinstance(input_scale, Node):
                continue

            x_arg, residual_arg, weight_arg, eps_arg, strategy_arg = node.args

            # Insert before the FP8 linear consumer. ``input_scale`` is a
            # ``get_attr`` node defined upstream of the linear (the linear is
            # what consumes it), so this position is safe for all operands and
            # downstream of the original fused node's residual-only users.
            with graph.inserting_before(linear_consumer):
                fused_quant = graph.call_function(
                    fused_quant_op_target,
                    args=(
                        x_arg,
                        residual_arg,
                        weight_arg,
                        input_scale,
                        eps_arg,
                        strategy_arg,
                    ),
                )
                new_norm = graph.call_function(operator.getitem, args=(fused_quant, 0))
                new_residual = graph.call_function(operator.getitem, args=(fused_quant, 1))

            # Rewire all users of the original getitem(0)/(1) to the new
            # equivalents. Users of getitem(0) must consume the now-FP8 norm.
            for user in list(node.users):
                if user.op != "call_function" or user.target is not operator.getitem:
                    continue
                idx = user.args[1]
                replacement = new_norm if idx == 0 else new_residual
                user.replace_all_uses_with(replacement)
                graph.erase_node(user)

            # The lone consumer of the (now-FP8) norm is the FP8 linear. Tell
            # it the output dtype explicitly so the op resolves the bf16
            # accumulator without a bias hint.
            new_kwargs = dict(linear_consumer.kwargs)
            new_kwargs.setdefault("out_dtype", "bfloat16")
            linear_consumer.kwargs = new_kwargs

            graph.erase_node(node)
            num_matches += 1

        if num_matches > 0:
            gm.recompile()

        info = TransformInfo(
            skipped=num_matches == 0,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
