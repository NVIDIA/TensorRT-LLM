"""
Triton implementation of the Fused MOE ops. Inspired by vLLM's triton MOE implementation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_mlp_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Minimal unquantized fused MoE GEMM kernel used twice per MLP:
      1) Y1 = X @ W_up^T  (no routing weights applied)
      2) Y2 = Act(Y1) @ W_down^T  (optionally apply routing weights)

    A layout: (M, K)
    B layout: (E, N, K)  — per-expert weight, contiguous on last dim
    C layout: (M, top_k, N)
    sorted_token_ids contains flattened indices in [0, M*top_k) for routed tokens,
    grouped by expert and padded to multiples of BLOCK_SIZE_M.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Bounds check: EM might not be a multiple of BLOCK_SIZE_M
    # so offs_token_id can exceed EM-1. Load with mask to avoid out-of-bounds.
    token_id_mask = offs_token_id < EM
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id, mask=token_id_mask, other=num_valid_tokens
    )
    token_mask = offs_token < num_valid_tokens

    # Clamp offs_token to valid range to avoid out-of-bounds pointer arithmetic
    # Padding tokens have value >= num_valid_tokens and will be masked out
    # Clamp to last valid token instead of 0 to avoid cache/memory issues
    max_valid_token = num_valid_tokens - 1
    offs_token_clamped = tl.where(token_mask, offs_token, max_valid_token)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token_clamped,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token_clamped[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token_clamped[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_mlp_moe_kernel_w8a8(
    # Pointers to matrices (A in FP8, B in FP8)
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Scale pointers
    a_scale_ptr,
    b_scale_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    token_id_mask = offs_token_id < EM
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id, mask=token_id_mask, other=num_valid_tokens
    )
    token_mask = offs_token < num_valid_tokens

    # Clamp offs_token to valid range to avoid out-of-bounds pointer arithmetic
    # Padding tokens have value >= num_valid_tokens and will be masked out
    # Clamp to last valid token instead of 0 to avoid cache/memory issues
    max_valid_token = num_valid_tokens - 1
    offs_token_clamped = tl.where(token_mask, offs_token, max_valid_token)

    # Expert id for this block (one expert per M-tile)
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token_clamped,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token_clamped[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Load tensor-wise scales before loop
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_experts)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        # Use acc= for FP8 fast accumulation (matches vLLM)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scales after K-loop
    accumulator = (accumulator * a_scale * b_scale).to(compute_type)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token_clamped[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _default_kernel_config(M: int, E: int, N: int, K: int, top_k: int) -> dict:
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }


def _pack_routed_tokens(
    topk_ids: torch.Tensor,
    M: int,
    E: int,
    top_k: int,
    block_size_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast CUDA-based token packing using vLLM's moe_align_block_size kernel.

    This implementation uses CUB's BlockScan and optimized atomic operations for
    fast token sorting by expert ID. It's much faster than PyTorch's torch.argsort.

    Ported from vLLM: https://github.com/vllm-project/vllm

    The kernel aligns token distribution across experts to be compatible with
    block_size for matrix multiplication, padding as needed.

    Returns:
      - sorted_token_ids: [capacity] tensor with sorted token IDs (padded)
      - expert_ids: [num_blocks] tensor with expert ID per block (0 for pad)
      - num_tokens_post_padded: tensor containing actual number of tokens after
        padding (kept as tensor for CUDA graph compatibility)
    """
    device = topk_ids.device
    T = M * top_k

    # Calculate capacity with padding (vLLM approach)
    max_num_tokens_padded = T + E * (block_size_m - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size_m - 1) // block_size_m

    # Prepare output tensors
    sorted_token_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    # Flatten topk_ids to [M*top_k] and ensure int32
    topk_ids_flat = topk_ids.reshape(-1).to(torch.int32)

    # Lazy import to avoid JIT compilation during module import
    try:
        from .load_moe_align import moe_align_block_size
    except Exception as e:
        raise ImportError(
            f"Failed to load moe_align_block_size CUDA extension. "
            f"Error: {e}. Make sure CUDA toolkit and nvcc are available."
        ) from e

    moe_align_block_size(
        topk_ids_flat, E, block_size_m, sorted_token_ids, expert_ids, num_tokens_post_pad
    )

    # Convert to int64 for compatibility with Triton kernel
    sorted_token_ids = sorted_token_ids.to(torch.int64)

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _invoke_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
):
    """Unified kernel launcher for both unquantized and FP8 W8A8 MoE kernels."""
    assert B.ndim == 3 and C.ndim == 3
    EM = sorted_token_ids.numel()
    if EM == 0:
        return
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    def _grid(META):
        return (
            triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
        )

    num_tokens = A.size(0) * top_k
    common_args = [
        A,
        B,
        C,
        topk_weights if topk_weights is not None else C,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
    ]

    common_kwargs = {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "top_k": top_k,
        "compute_type": compute_type,
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }

    if a_scale is not None and b_scale is not None:
        # FP8 W8A8 path
        fused_mlp_moe_kernel_w8a8[_grid](*common_args, a_scale, b_scale, **common_kwargs)
    else:
        # Unquantized path
        fused_mlp_moe_kernel[_grid](*common_args, **common_kwargs)


def _get_compute_type(dtype: torch.dtype):
    """Get Triton compute type from torch dtype."""
    if dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _fused_moe_mlp_relu2(
    hidden_states: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Fused MoE 2-layer MLP with ReLU^2 activation using Triton."""
    M, H = hidden_states.shape
    E, inter_size, _ = w_up.shape
    top_k = topk_ids.shape[1]

    config = _default_kernel_config(M, E, inter_size, H, top_k)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _pack_routed_tokens(
        topk_ids, M, E, top_k, config["BLOCK_SIZE_M"]
    )

    cache1 = hidden_states.new_empty((M, top_k, inter_size))
    cache3 = hidden_states.new_empty((M, top_k, H))
    compute_type = _get_compute_type(hidden_states.dtype)

    # GEMM 1: hidden @ w_up^T
    _invoke_kernel(
        hidden_states.contiguous(),
        w_up.contiguous(),
        cache1,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k,
        config,
        compute_type,
    )

    # Activation: ReLU^2
    act = torch.square(F.relu(cache1.view(-1, inter_size)))

    # GEMM 2: act @ w_down^T
    _invoke_kernel(
        act,
        w_down.contiguous(),
        cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type,
    )

    return cache3.sum(dim=1)


@torch.library.custom_op("auto_deploy::triton_moe_fused", mutates_args=())
def triton_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    """Triton unquantized MoE with 2-layer MLP and ReLU^2 activation."""
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])
    topk_ids = selected_experts.to(torch.int32).contiguous()
    topk_weights = routing_weights.to(torch.float32).contiguous()

    out2d = _fused_moe_mlp_relu2(x2d, w1_stacked_weight, w2_stacked_weight, topk_ids, topk_weights)
    return out2d.view(x_shape)


@triton_fused_moe.register_fake
def triton_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def _quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 with clamping (matches torch_quant_fp8_linear)."""
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.library.custom_op("auto_deploy::triton_quant_fp8_moe", mutates_args=())
def triton_quant_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,  # [E, I, H] stacked FP8 weights
    w2_weight: torch.Tensor,  # [E, H, I] stacked FP8 weights
    w3_weight: torch.Tensor,  # unused for mlp style
    w1_input_scale: torch.Tensor,  # [E] stacked input scales
    w2_input_scale: torch.Tensor,  # [E] stacked input scales
    w3_input_scale: torch.Tensor,  # unused
    w1_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w2_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w3_weight_scale: torch.Tensor,  # unused
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    """Triton FP8 W8A8 MoE with 2-layer MLP and ReLU^2 activation."""
    if mlp_style != "mlp":
        raise NotImplementedError("triton_quant_fp8_moe currently supports mlp_style=='mlp' only")

    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])
    topk_ids = selected_experts.to(torch.int32).contiguous()
    topk_weights = routing_weights.to(torch.float32).contiguous()

    # Weights are already stacked [E, ...] - just ensure contiguous and extract scales
    w1_q = w1_weight.contiguous()
    w2_q = w2_weight.contiguous()
    a1_scale = w1_input_scale[0].to(torch.float32).reshape(1).contiguous()
    a2_scale = w2_input_scale[0].to(torch.float32).reshape(1).contiguous()
    b1_scale = w1_weight_scale.to(torch.float32).contiguous()
    b2_scale = w2_weight_scale.to(torch.float32).contiguous()

    # Setup
    M, H = x2d.shape
    E, inter_size, _ = w1_q.shape
    top_k = topk_ids.shape[1]
    config = _default_kernel_config(M, E, inter_size, H, top_k)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _pack_routed_tokens(
        topk_ids, M, E, top_k, config["BLOCK_SIZE_M"]
    )
    compute_type = _get_compute_type(x2d.dtype)

    # Quantize input and allocate caches
    x_a8 = _quantize_fp8(x2d, a1_scale)
    cache1 = x2d.new_empty((M, top_k, inter_size))
    cache3 = x2d.new_empty((M, top_k, H))

    # GEMM 1: FP8 input @ FP8 w_up^T → BF16
    _invoke_kernel(
        x_a8,
        w1_q,
        cache1,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k,
        config,
        compute_type,
        a_scale=a1_scale,
        b_scale=b1_scale,
    )

    # Activation: ReLU^2, then quantize
    act = torch.square(F.relu(cache1.view(-1, inter_size)))
    act_a8 = _quantize_fp8(act, a2_scale)

    # GEMM 2: FP8 activation @ FP8 w_down^T → BF16
    _invoke_kernel(
        act_a8,
        w2_q,
        cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type,
        a_scale=a2_scale,
        b_scale=b2_scale,
    )

    return cache3.sum(dim=1).view(x_shape)


@triton_quant_fp8_moe.register_fake
def triton_quant_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w3_input_scale: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight_scale: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)
