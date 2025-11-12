"""
AOT-compiled moe_align CUDA kernel.

The moe_align kernel is now compiled ahead-of-time (AOT) as part of the main
TensorRT-LLM build instead of being JIT-compiled on first use. This reduces
startup time and avoids compilation overhead.

The kernel implementation is in:
- cpp/tensorrt_llm/kernels/moeAlignKernels.cu
- cpp/tensorrt_llm/kernels/moeAlignKernels.h

The torch binding is in:
- cpp/tensorrt_llm/thop/moeAlignOp.cpp
"""

import torch


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
):
    """
    Wrapper for the AOT-compiled moe_align_block_size function.

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

    # Call the AOT-compiled kernel via torch ops
    torch.ops.trtllm.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad
    )
