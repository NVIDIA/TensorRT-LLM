"""Transformations for fusing collective operations.

This module registers TRT-LLM backend patterns only. Fusion is only applied
when TRT-LLM is available (MPI mode) since it provides optimized fused kernels.
The torch backend (demollm mode) does not benefit from fusion.
"""

from functools import partial
from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
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


def _make_allreduce_residual_rmsnorm_pattern(
    add_order: str = "residual_first", strategy: str = "AUTO"
):
    """Factory function to create pattern functions for allreduce+residual+torch_rmsnorm fusion.

    This pattern matches the graph after match_rmsnorm_pattern has replaced
    RMSNorm patterns with torch_rmsnorm ops.

    Args:
        add_order: Either "residual_first" (residual + x) or "x_first" (x + residual)
        strategy: AllReduce strategy to use in the pattern

    Returns:
        A pattern function that can be used with register_ad_pattern
    """

    def pattern_fn(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
    ):
        """Pattern: trtllm_dist_all_reduce(x) -> add residual -> torch_rmsnorm

        Reference PyTorch composition:
            y = trtllm_dist_all_reduce(x)
            z = residual + y  (or y + residual)
            normed = torch_rmsnorm(z, weight, eps)
        Returns (normed, z)
        """
        hidden_states = torch.ops.auto_deploy.trtllm_dist_all_reduce(x, strategy)

        # Handle addition order
        if add_order == "residual_first":
            add = residual + hidden_states
        else:  # x_first
            add = hidden_states + residual

        # Use torch_rmsnorm op (already replaced by match_rmsnorm_pattern)
        normed = torch.ops.auto_deploy.torch_rmsnorm(add, weight, eps)

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

        # ============================================================================
        # Instantiate Pattern Functions
        # ============================================================================

        # Get the allreduce strategy from shared_config
        strategy = shared_config.sharding_transform_container.config.allreduce_strategy.name

        # TRT-LLM backend (MPI mode) - two patterns for different addition orders
        _allreduce_residual_rmsnorm_pattern_trtllm = _make_allreduce_residual_rmsnorm_pattern(
            add_order="residual_first", strategy=strategy
        )
        _allreduce_residual_rmsnorm_pattern2_trtllm = _make_allreduce_residual_rmsnorm_pattern(
            add_order="x_first", strategy=strategy
        )

        # Register TRT-LLM backend patterns only (no torch backend fusion)
        # Pattern 1: residual + allreduce(x)
        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern_trtllm,
            replace_fn=partial(_allreduce_residual_rmsnorm_replacement, strategy=strategy),
            patterns=patterns,
            dummy_args=dummy_args,
            scalar_workaround=scalar_workaround,
        )

        # Pattern 2: allreduce(x) + residual
        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern2_trtllm,
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
