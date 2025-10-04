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


def _allreduce_residual_rmsnorm_pattern(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
):
    """
    Reference PyTorch composition of:
        y = all_reduce(x)
        z = residual + y
        normed = RMSNorm(z, weight, eps)
    Returns (normed, z)
    """

    input_dtype = x.dtype
    hidden_states = torch.ops.auto_deploy.torch_dist_all_reduce(x)
    add = residual + hidden_states

    hidden_states = add.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)

    normed = weight * hidden_states.to(input_dtype)

    return normed, add


def _allreduce_residual_rmsnorm_pattern2(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
):
    """
    Reference PyTorch composition of:
        y = all_reduce(x)
        z = y + residual
        normed = RMSNorm(z, weight, eps)
    Returns (normed, z)
    """

    input_dtype = x.dtype
    hidden_states = torch.ops.auto_deploy.torch_dist_all_reduce(x)
    add = hidden_states + residual

    hidden_states = add.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)

    normed = weight * hidden_states.to(input_dtype)

    return normed, add


def _allreduce_residual_rmsnorm_repl(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
):
    return torch.ops.dist.fused_allreduce_residual_rmsnorm(x, residual, weight, eps)


@TransformRegistry.register("fuse_allreduce_residual_rmsnorm")
class FuseAllreduceResidualRMSNorm(BaseTransform):
    """Fuse (allreduce + residual add + RMSNorm) into one fused op with tuple output."""

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

        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern,
            replace_fn=_allreduce_residual_rmsnorm_repl,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types={torch.ops.aten.to.dtype: (torch.dtype,)},
            scalar_workaround={"eps": 0.1253},
        )
        register_ad_pattern(
            search_fn=_allreduce_residual_rmsnorm_pattern2,
            replace_fn=_allreduce_residual_rmsnorm_repl,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types={torch.ops.aten.to.dtype: (torch.dtype,)},
            scalar_workaround={"eps": 0.1253},
        )

        num_matches = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=False, has_valid_shapes=False
        )
        return gm, info
