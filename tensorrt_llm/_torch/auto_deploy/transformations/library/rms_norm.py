"""Graph transform to optimize RMSNorm execution using FlashInfer."""

from functools import partial

import torch
from torch.fx import GraphModule

from ...utils.logger import ad_logger

# It is important to import ADPatternMatcherPass from pattern_matcher.py, not from torch._inductor.pattern_matcher
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from .._graph import canonicalize_graph

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


def fuse_rmsnorm(gm: GraphModule, backend: str = "triton") -> None:
    """Matches and replaces RMSNorm patterns in the graph with FlashInfer or Triton implementation.

    This function sets up pattern matching to identify RMSNorm operations in the graph
    and replaces them with optimized implementations. It uses dummy tensors to register
    the pattern matching rules.

    Args:
        gm: Input graph module to transform.
        backend: Backend to use for RMSNorm computation ("flashinfer" or "triton").

    Returns:
        Transformed graph module with optimized RMSNorm operations.
    """
    if backend.lower() not in _BACKEND_OPS:
        raise ValueError(f"Invalid backend, must be one of {list(_BACKEND_OPS)}, got {backend}")
    ad_logger.info(f"Starting RMSNorm pattern matching with backend: {backend}")

    graph = gm.graph
    patterns = ADPatternMatcherPass()

    # Create dummy tensors for pattern matching
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

    # Register patterns for each configuration
    for input_dtype, weight_dtype in configs:
        register_ad_pattern(
            search_fn=_rms_norm_pattern,
            replace_fn=partial(_rms_norm_replacement, backend=backend),
            patterns=patterns,
            dummy_args=dummy_args(input_dtype, weight_dtype),
            op_ignore_types={},
            scalar_workaround={"eps": 1e-6},
        )

    cnt = patterns.apply(graph)
    ad_logger.info(f"RMSNorm pattern count: {cnt}")
    canonicalize_graph(gm)
    ad_logger.debug("RMSNorm pattern matching completed.")
