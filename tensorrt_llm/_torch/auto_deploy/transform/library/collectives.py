"""Transformations for fusing collective operations.

This module implements multi-pattern registration: both torch and trtllm patterns
are registered, and the matcher finds whichever is present in the graph.
"""

from typing import Callable, Tuple

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
    all_reduce_op: Callable, add_order: str = "residual_first"
):
    """Factory function to create pattern functions for different backends.

    Args:
        all_reduce_op: The all_reduce op to use (torch_dist or trtllm_dist variant)
        add_order: Either "residual_first" (residual + x) or "x_first" (x + residual)

    Returns:
        A pattern function that can be used with register_ad_pattern
    """

    def pattern_fn(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
    ):
        """Pattern: all_reduce(x) -> add residual -> RMSNorm

        Reference PyTorch composition:
            y = all_reduce(x)
            z = residual + y  (or y + residual)
            normed = RMSNorm(z, weight, eps)
        Returns (normed, z)
        """
        input_dtype = x.dtype
        hidden_states = all_reduce_op(x)

        # Handle addition order
        if add_order == "residual_first":
            add = residual + hidden_states
        else:  # x_first
            add = hidden_states + residual

        hidden_states = add.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)

        normed = weight * hidden_states.to(input_dtype)

        return normed, add

    return pattern_fn


def _make_allreduce_residual_rmsnorm_replacement(fused_op: Callable):
    """Factory function to create replacement functions for different backends.

    Args:
        fused_op: The fused op to use (torch or trtllm variant)

    Returns:
        A replacement function that can be used with register_ad_pattern
    """

    def replacement_fn(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
        """Replacement using backend-specific fused op."""
        return fused_op(x, residual, weight, eps)

    return replacement_fn


# ============================================================================
# Instantiate Pattern and Replacement Functions
# ============================================================================

# TRT-LLM backend (MPI mode)
_allreduce_residual_rmsnorm_pattern_trtllm = _make_allreduce_residual_rmsnorm_pattern(
    torch.ops.auto_deploy.trtllm_dist_all_reduce, add_order="residual_first"
)
_allreduce_residual_rmsnorm_pattern2_trtllm = _make_allreduce_residual_rmsnorm_pattern(
    torch.ops.auto_deploy.trtllm_dist_all_reduce, add_order="x_first"
)
_allreduce_residual_rmsnorm_repl_trtllm = _make_allreduce_residual_rmsnorm_replacement(
    torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm
)


# ============================================================================
# Transform Implementation
# ============================================================================


@TransformRegistry.register("fuse_allreduce_residual_rmsnorm")
class FuseAllreduceResidualRMSNorm(BaseTransform):
    """Fuse (allreduce + residual add + RMSNorm) into one fused op with tuple output.

    This transform uses multi-pattern registration: both torch and trtllm patterns
    are registered, and the pattern matcher will find whichever is present in the graph.
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

        op_ignore_types = {torch.ops.aten.to.dtype: (torch.dtype,)}
        scalar_workaround = {"eps": 0.1253}

        # Register only trtllm patterns
        # TRT-LLM backend patterns (residual + x)
        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern_trtllm,
            replace_fn=_allreduce_residual_rmsnorm_repl_trtllm,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
            scalar_workaround=scalar_workaround,
        )

        # TRT-LLM backend patterns (x + residual)
        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern2_trtllm,
            replace_fn=_allreduce_residual_rmsnorm_repl_trtllm,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
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
