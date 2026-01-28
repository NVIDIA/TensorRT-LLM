"""Graph transform to optimize L2Norm execution using FLA Triton kernels."""

from typing import Literal, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface

# It is important to import ADPatternMatcherPass from pattern_matcher.py, not from torch._inductor.pattern_matcher
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_BACKEND_OPS = {
    "fla": torch.ops.auto_deploy.fla_l2norm.default,
    "torch": torch.ops.auto_deploy.torch_l2norm.default,
}


def _l2_norm_pattern(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Implements the L2Norm pattern for pattern matching.

    L2 normalization: x / sqrt(sum(x^2) + eps)

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor.
    """
    input_dtype = data.dtype
    data = data.to(torch.float32)
    sum_sq = (data * data).sum(dim=-1, keepdim=True)
    data = data * torch.rsqrt(sum_sq + eps)
    return data.to(input_dtype)


def _l2_norm_pattern_no_dtype_cast(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Implements the L2Norm pattern without dtype casting for pattern matching.

    Some models may already operate in float32 and skip the dtype cast.

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor.
    """
    sum_sq = (data * data).sum(dim=-1, keepdim=True)
    return data * torch.rsqrt(sum_sq + eps)


def _l2_norm_to_torch_l2norm(data: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace L2Norm pattern with torch_l2norm op (standardized representation).

    Args:
        data: Input tensor to normalize.
        eps: Small constant for numerical stability.

    Returns:
        L2 normalized tensor using torch_l2norm.
    """
    return torch.ops.auto_deploy.torch_l2norm(data, eps)


@TransformRegistry.register("match_l2norm_pattern")
class MatchL2NormPattern(BaseTransform):
    """Matches L2Norm patterns in the graph and replaces them with torch_l2norm op.

    This transform runs in the pattern_matcher stage and standardizes L2Norm patterns
    to use torch_l2norm op, which can later be fused to a specific backend in the
    post_load_fusion stage.

    Args:
        gm: Input graph module to transform.

    Returns:
        Transformed graph module with standardized torch_l2norm operations.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        bs = 2
        hidden_size = 512

        def dummy_args(input_dtype: torch.dtype, eps: float = 1e-6):
            return [
                torch.randn(bs, hidden_size, device="cuda", dtype=input_dtype),
                eps,
            ]

        configs = [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]

        search_fns = [
            _l2_norm_pattern,
            _l2_norm_pattern_no_dtype_cast,
        ]
        for search_fn in search_fns:
            for input_dtype in configs:
                register_ad_pattern(
                    search_fn=search_fn,
                    replace_fn=_l2_norm_to_torch_l2norm,
                    patterns=patterns,
                    dummy_args=dummy_args(input_dtype),
                    op_ignore_types={},
                    scalar_workaround={"eps": 1e-6},
                    skip_duplicates=True,
                )

        cnt = patterns.apply(graph)

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info


class FuseL2NormConfig(TransformConfig):
    """Configuration for the L2Norm fusion transform."""

    backend: Literal["torch", "fla"] = Field(
        default="fla",
        description="Backend to use for L2Norm computation ('fla' or 'torch').",
    )


@TransformRegistry.register("fuse_l2norm")
class FuseL2Norm(BaseTransform):
    """Fuses torch_l2norm ops with the selected backend implementation.

    This transform runs in the post_load_fusion stage and replaces torch_l2norm ops
    with the specified backend implementation (fla or torch).

    Args:
        gm: Input graph module to transform.
        backend: Backend to use for L2Norm computation ("fla" or "torch").

    Returns:
        Transformed graph module with backend-specific L2Norm operations.
    """

    config: FuseL2NormConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseL2NormConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        target_op = _BACKEND_OPS[self.config.backend]
        cnt = 0

        for node in list(graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_l2norm):
                with graph.inserting_after(node):
                    new_node: Node = graph.call_function(
                        target_op,
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                    new_node.meta = node.meta.copy()
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
                    cnt += 1

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info
