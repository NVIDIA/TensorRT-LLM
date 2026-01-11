"""Graph transform to optimize RMSNorm execution using FlashInfer."""

from typing import Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.rms_norm import gated_rms_norm_ref
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
    "flashinfer": torch.ops.auto_deploy.flashinfer_rms_norm,
    "triton": torch.ops.auto_deploy.triton_rms_norm,
    "torch": torch.ops.auto_deploy.torch_rmsnorm,
}


def _rms_norm_pattern(data: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Implements the RMSNorm pattern for pattern matching.

    Args:
        data: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor.
    """
    input_dtype = data.dtype
    data = data.to(torch.float32)
    variance = data.pow(2).mean(-1, keepdim=True)
    data = data * torch.rsqrt(variance + eps)
    return weight * data.to(input_dtype)


def _rms_norm_pattern_float32_weights(
    data: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Implements the RMSNorm pattern for pattern matching.

    Args:
        data: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor.
    """
    input_dtype = data.dtype
    data = data.to(torch.float32)
    variance = data.pow(2).mean(-1, keepdim=True)
    data = data * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * data).to(input_dtype)


def _rms_norm_to_torch_rmsnorm(
    data: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Replace RMSNorm pattern with torch_rmsnorm op (standardized representation).

    Args:
        data: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using torch_rmsnorm.
    """
    return torch.ops.auto_deploy.torch_rmsnorm(data, weight, eps)


def _rms_norm_replacement(
    data: torch.Tensor, weight: torch.Tensor, eps: float, backend: str
) -> torch.Tensor:
    """Backend-specific rms_norm implementation.

    Args:
        data: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.
        backend: Backend to use for RMSNorm computation ("flashinfer" or "triton").

    Returns:
        Normalized and scaled tensor using the specified backend implementation.
    """

    assert backend.lower() in _BACKEND_OPS, (
        f"Invalid {backend=}; must be one of {list(_BACKEND_OPS)}"
    )
    return _BACKEND_OPS[backend.lower()](data, weight, eps)


@TransformRegistry.register("match_rmsnorm_pattern")
class MatchRMSNormPattern(BaseTransform):
    """Matches RMSNorm patterns in the graph and replaces them with torch_rmsnorm op.

    This transform runs in the pattern_matcher stage and standardizes RMSNorm patterns
    to use torch_rmsnorm op, which can later be fused to a specific backend in the
    post_load_fusion stage.

    Args:
        gm: Input graph module to transform.

    Returns:
        Transformed graph module with standardized torch_rmsnorm operations.
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        # Pattern matching for regular RMSNorm
        bs = 2
        hidden_size = 512

        def dummy_args(input_dtype: torch.dtype, weight_dtype: torch.dtype, eps: float = 1e-6):
            return [
                torch.randn(bs, hidden_size, device="cuda", dtype=input_dtype),
                torch.randn(hidden_size, device="cuda", dtype=weight_dtype),
                eps,
            ]

        # Define configurations for different data types
        configs = [
            (torch.bfloat16, torch.bfloat16),
            (torch.float16, torch.float16),
            (torch.float32, torch.float32),
        ]

        # Register patterns for each configuration - replace with torch_rmsnorm
        search_fns = [
            _rms_norm_pattern,
            _rms_norm_pattern_float32_weights,
        ]
        for search_fn in search_fns:
            for input_dtype, weight_dtype in configs:
                register_ad_pattern(
                    search_fn=search_fn,
                    replace_fn=_rms_norm_to_torch_rmsnorm,
                    patterns=patterns,
                    dummy_args=dummy_args(input_dtype, weight_dtype),
                    op_ignore_types={},
                    scalar_workaround={"eps": 1e-6},
                )

        # Pattern matching for gated RMSNorm
        B, S, H = 2, 3, 4096
        group_size = 512
        eps = 1e-5

        def make_dummy_args_gated(group_size: int, eps: float) -> list:
            x = torch.randn(B, S, H, dtype=torch.float32)
            w = torch.randn(H, dtype=torch.float32)
            g = torch.randn(B, S, H, dtype=torch.float32)
            return [x, w, g, eps, group_size]

        op_ignore_types = {
            torch.ops.aten.reshape.default: (int, list, tuple),
            torch.ops.aten.view.default: (int, list, tuple),
            torch.ops.aten.mean.dim: (list, tuple),
            torch.ops.aten.to.dtype: (torch.dtype,),
        }

        # Register pattern for gated RMSNorm - replace with torch_rmsnorm_gated
        register_ad_pattern(
            search_fn=_gated_rmsnorm_pattern_ref,
            replace_fn=_gated_rmsnorm_to_torch_rmsnorm_gated,
            patterns=patterns,
            dummy_args=make_dummy_args_gated(group_size, eps),
            op_ignore_types=op_ignore_types,
            scalar_workaround={"eps": eps, "group_size": group_size},
            skip_duplicates=True,
        )

        cnt = patterns.apply(graph)

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info


class FuseRMSNormConfig(TransformConfig):
    """Configuration for the RMSNorm fusion transform."""

    rmsnorm_backend: str = Field(
        default="flashinfer",
        description="Backend to use for RMSNorm computation ('flashinfer', 'triton', or 'torch').",
    )
    gated_rmsnorm_backend: str = Field(
        default="triton",
        description="Backend to use for gated RMSNorm computation (currently only 'triton').",
    )


@TransformRegistry.register("fuse_rmsnorm")
class FuseRMSNorm(BaseTransform):
    """Fuses torch_rmsnorm ops with the selected backend implementation.

    This transform runs in the post_load_fusion stage and replaces torch_rmsnorm ops
    with the specified backend implementation (flashinfer, triton, or torch).

    Args:
        gm: Input graph module to transform.
        rmsnorm_backend: Backend to use for regular RMSNorm computation ("flashinfer", "triton", or "torch").
        gated_rmsnorm_backend: Backend to use for gated RMSNorm computation (currently only "triton").

    Returns:
        Transformed graph module with backend-specific RMSNorm operations.
    """

    config: FuseRMSNormConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseRMSNormConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Validate rmsnorm_backend
        if self.config.rmsnorm_backend.lower() not in _BACKEND_OPS:
            raise ValueError(
                f"Invalid rmsnorm_backend, must be one of {list(_BACKEND_OPS)}, got {self.config.rmsnorm_backend}"
            )

        # Validate gated_rmsnorm_backend (currently only triton is supported)
        if self.config.gated_rmsnorm_backend.lower() != "triton":
            raise ValueError(
                f"""Invalid gated_rmsnorm_backend, currently only 'triton' is supported,
                got {self.config.gated_rmsnorm_backend}"""
            )

        graph = gm.graph
        backend = self.config.rmsnorm_backend.lower()
        target_op = _BACKEND_OPS[backend]
        cnt = 0

        # Replace torch_rmsnorm ops with the selected backend
        for node in list(graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_rmsnorm):
                # Replace with the selected backend op
                with graph.inserting_after(node):
                    new_node: Node = graph.call_function(
                        target_op,
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
                    cnt += 1

        # Replace torch_rmsnorm_gated ops with triton_rmsnorm_gated
        for node in list(graph.nodes):
            if is_op(node, torch.ops.auto_deploy.torch_rmsnorm_gated):
                # Replace with triton_rmsnorm_gated op
                with graph.inserting_after(node):
                    new_node: Node = graph.call_function(
                        torch.ops.auto_deploy.triton_rmsnorm_gated,
                        args=node.args,
                        kwargs=node.kwargs,
                    )
                    node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
                    cnt += 1

        gm.recompile()

        info = TransformInfo(
            skipped=False, num_matches=cnt, is_clean=cnt == 0, has_valid_shapes=cnt == 0
        )

        return gm, info


def _gated_rmsnorm_pattern_ref(
    x: torch.Tensor,  # [B, S, H]
    weight: torch.Tensor,  # [H]
    gate: torch.Tensor,  # [B, S, H]
    eps: float = 1e-5,
    group_size: int = 512,
) -> torch.Tensor:
    y = gated_rms_norm_ref(
        x,
        weight,
        bias=None,
        z=gate,
        eps=eps,
        group_size=group_size,
        norm_before_gate=False,
        upcast=True,
    )

    return y


def _gated_rmsnorm_to_torch_rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    group_size: int,
) -> torch.Tensor:
    """Replace gated RMSNorm pattern with torch_rmsnorm_gated op (standardized representation).

    Args:
        x: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        gate: Gate tensor for gated normalization.
        eps: Small constant for numerical stability.
        group_size: Size of groups for grouped normalization.

    Returns:
        Normalized and gated tensor using torch_rmsnorm_gated.
    """
    return torch.ops.auto_deploy.torch_rmsnorm_gated(
        x, weight, gate, float(eps), int(group_size), False
    )
