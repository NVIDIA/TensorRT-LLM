"""Graph transform to optimize RMSNorm execution using FlashInfer."""

from functools import partial

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.fx import GraphModule

from ...utils.logger import ad_logger
from ...utils.pattern_matcher import register_pattern
from .._graph import canonicalize_graph


def _rms_norm_pattern(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Implements the RMSNorm pattern for pattern matching.

    Args:
        hidden_states: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def _rms_norm_replacement(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float, backend: str
) -> torch.Tensor:
    """Replacement function that uses the FlashInfer RMSNorm implementation.

    Args:
        hidden_states: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.
        backend: Backend to use for RMSNorm computation ("flashinfer" or "triton").

    Returns:
        Normalized and scaled tensor using the specified backend implementation.
    """
    BACKEND_OPS = {
        "flashinfer": torch.ops.auto_deploy.rms_norm_flashinfer,
        "triton": torch.ops.auto_deploy.rms_norm_triton,
    }
    return BACKEND_OPS[backend.lower()](hidden_states, weight, eps)


def match_rms_norm_with_pm(gm: GraphModule, backend: str = "triton") -> GraphModule:
    """Matches and replaces RMSNorm patterns in the graph with FlashInfer implementation.

    This function sets up pattern matching to identify RMSNorm operations in the graph
    and replaces them with optimized implementations. It uses dummy tensors to register
    the pattern matching rules.

    Args:
        gm: Input graph module to transform.
        backend: Backend to use for RMSNorm computation ("flashinfer" or "triton").

    Returns:
        Transformed graph module with optimized RMSNorm operations.
    """
    ad_logger.info("Starting RMSNorm pattern matching...")
    VALID_BACKENDS = {"flashinfer", "triton"}
    if backend.lower() not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend, must be one of {VALID_BACKENDS}, got {backend}")
    graph = gm.graph
    patterns = PatternMatcherPass()

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
        register_pattern(
            search_fn=_rms_norm_pattern,
            replace_fn=partial(_rms_norm_replacement, backend=backend),
            patterns=patterns,
            dummy_args=dummy_args(input_dtype, weight_dtype),
            op_ignore_types={},
            scalar_workaround={"eps": 1e-6},
        )

    cnt = patterns.apply(graph)
    ad_logger.info(f"RMSNorm pattern count: {cnt}")
    gm = canonicalize_graph(gm)
    ad_logger.debug("RMSNorm pattern matching completed.")
    return gm
