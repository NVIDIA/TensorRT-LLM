"""
Build moe_align CUDA extension eagerly with a persistent build directory
(same workflow as agent_ops/load_moe.py).
"""

import os

import torch
from torch.utils.cpp_extension import load

# Recommend explicit arch list so NVCC targets the right GPUs. You can override via env.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.environ.get("AD_CACHE_DIR") or os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
    "ad_cache",
)
BUILD_DIR = os.path.join(CACHE_ROOT, "auto_deploy", "fused_moe", "moe_align")
try:
    os.makedirs(BUILD_DIR, exist_ok=True)
except PermissionError:
    import tempfile

    # Fallback to the system temp dir while maintaining a stable subfolder layout
    BUILD_DIR = os.path.join(
        tempfile.gettempdir(), "ad_cache", "auto_deploy", "fused_moe", "moe_align"
    )
    os.makedirs(BUILD_DIR, exist_ok=True)

moe_align_ext = load(
    name="moe_align_ext",
    sources=[os.path.join(THIS_DIR, "moe_align_kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        # Optional: "-Xptxas=-v",
    ],
    verbose=True,
    with_cuda=True,
    build_directory=BUILD_DIR,
    is_python_module=True,
)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    """
    Wrapper for the CUDA moe_align_block_size function.

    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Args:
        topk_ids: Tensor of shape [total_tokens, top_k] with expert indices
        num_experts: Total number of experts
        block_size: Block size for matrix multiplication
        sorted_token_ids: Output tensor for sorted token IDs
        expert_ids: Output tensor for expert IDs per block
        num_tokens_post_pad: Output tensor for total tokens after padding
    """
    # Basic validation
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()
    for t, name in [
        (sorted_token_ids, "sorted_token_ids"),
        (expert_ids, "expert_ids"),
        (num_tokens_post_pad, "num_tokens_post_pad"),
    ]:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if (
        sorted_token_ids.dtype != torch.int32
        or expert_ids.dtype != torch.int32
        or num_tokens_post_pad.dtype != torch.int32
    ):
        raise TypeError("sorted_token_ids, expert_ids, num_tokens_post_pad must be int32 tensors")

    moe_align_ext.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad
    )


def moe_align_cutedsl(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    """
    Wrapper for the CuTeDSL-optimized moe_align_cutedsl CUDA function.

    Sorts tokens by expert and outputs token IDs, sorted weights, and cu_seqlens
    directly for use with CuTeDSL's grouped GEMM API.

    Args:
        topk_ids: Tensor of shape [M, top_k] with expert indices (flattened)
        topk_weights: Tensor of shape [M, top_k] with routing weights (flattened)
        num_experts: Total number of experts
        top_k: Number of experts per token
        sorted_token_ids: Output tensor [capacity] for token IDs sorted by expert
        sorted_weights: Output tensor [capacity] for weights sorted by expert
        cu_seqlens: Output tensor [num_experts + 1] for cumulative sequence lengths
        num_tokens_post_pad: Output scalar tensor [1] for total valid tokens
    """
    # Basic validation
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")
    if not topk_weights.is_cuda:
        raise ValueError("topk_weights must be a CUDA tensor")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")

    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    for t, name in [
        (sorted_token_ids, "sorted_token_ids"),
        (cu_seqlens, "cu_seqlens"),
        (num_tokens_post_pad, "num_tokens_post_pad"),
    ]:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.dtype != torch.int32:
            raise TypeError(f"{name} must be int32 tensor")

    if not sorted_weights.is_cuda or not sorted_weights.is_contiguous():
        raise ValueError("sorted_weights must be a contiguous CUDA tensor")
    if sorted_weights.dtype != topk_weights.dtype:
        raise TypeError("sorted_weights must have same dtype as topk_weights")

    moe_align_ext.moe_align_cutedsl(
        topk_ids,
        topk_weights,
        num_experts,
        top_k,
        sorted_token_ids,
        sorted_weights,
        cu_seqlens,
        num_tokens_post_pad,
    )
