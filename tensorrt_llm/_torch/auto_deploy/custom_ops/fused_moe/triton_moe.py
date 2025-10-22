"""
Triton implementation of the Fused MOE ops. Inspired by vLLM's triton MOE implementation.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ...utils.logger import ad_logger


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


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/vllm-project/vllm/blob/b4fda58a2d0e458e0186e4caa4354b3d07153c70/vllm/model_executor/layers/fused_moe/fused_moe.py#L828
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    # note that we prioritize user defined config
    # user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    user_defined_config_folder = "."
    if user_defined_config_folder is not None:
        user_defined_config_file_path = os.path.join(user_defined_config_folder, json_file_name)
        config_file_paths.append(user_defined_config_file_path)

    ad_folder = "triton_fused_moe_configs"
    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), ad_folder, json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                ad_logger.info("Using configuration from %s for MoE layer.", config_file_path)
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    ad_logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! Config file not found at %s"),
        config_file_paths,
    )
    return None


def _get_kernel_config(
    M: int, E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> dict:
    configs = get_moe_configs(E, N, dtype=None)
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config (closest batch size)
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Else use the default config
        config = _default_kernel_config(M, E)
    return config


def _default_kernel_config(M: int, E: int) -> dict:
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
    num_tokens_post_padded: torch.Tensor,  # Changed to tensor for CUDA graph compatibility
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type,
):
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
    fused_mlp_moe_kernel[_grid](
        A,
        B,
        C,
        topk_weights if topk_weights is not None else C,
        sorted_token_ids,
        expert_ids,
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
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def fused_mlp_relu2_unquantized(
    hidden_states: torch.Tensor,  # [M, H]
    w_up: torch.Tensor,  # [E, I, H]
    w_down: torch.Tensor,  # [E, H, I]
    topk_ids: torch.Tensor,  # [M, top_k]
    topk_weights: torch.Tensor,  # [M, top_k]
    *,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    Fast unquantized MoE MLP with ReLU^2 activation between two per-expert GEMMs.

    Requirements:
      - w_up: (E, I, H) with last dim contiguous
      - w_down: (E, H, I) with last dim contiguous
      - hidden_states: (M, H), topk_ids/topk_weights: (M, top_k)
    """
    assert hidden_states.ndim == 2
    assert w_up.ndim == 3 and w_down.ndim == 3
    assert topk_ids.ndim == 2 and topk_weights.ndim == 2
    M, H = hidden_states.shape
    E, inter_size, H_up = w_up.shape
    E2, H_down, inter_size2 = w_down.shape
    assert E == E2 and H == H_up and H == H_down and inter_size == inter_size2
    top_k = topk_ids.shape[1]

    # Ensure memory layout compatible with kernel expectations
    A = hidden_states.contiguous()
    B1 = w_up.contiguous()  # (E, I, H)
    B2 = w_down.contiguous()  # (E, H, I)

    # Kernel config (use a single BLOCK_SIZE_M for both GEMMs)
    config = _get_kernel_config(M, E, inter_size2, H, top_k)

    # Token routing packing (group-by-expert, pad to BLOCK_SIZE_M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _pack_routed_tokens(
        topk_ids,
        M,
        E,
        top_k,
        config["BLOCK_SIZE_M"],
    )

    # Workspaces
    cache1 = A.new_empty((M, top_k, inter_size))
    cache2 = A.new_empty((M * top_k, inter_size))
    cache3 = A.new_empty((M, top_k, H))

    # Compute type
    if A.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif A.dtype == torch.float16:
        compute_type = tl.float16
    elif A.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported dtype for hidden_states: {A.dtype}")

    # GEMM 1: X @ W_up^T → cache1 (no routing weights here)
    _invoke_kernel(
        A,
        B1,
        cache1,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=compute_type,
    )

    # Activation (ReLU^2) without gating/multiplication
    cache2 = torch.square(F.relu(cache1.view(-1, inter_size)))

    # GEMM 2: Act(cache1) @ W_down^T → cache3 (apply routing weights)
    _invoke_kernel(
        cache2,
        B2,
        cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,  # ensure offs_token maps to flattened rows (m*top_k + n)
        config=config,
        compute_type=compute_type,
    )

    # Sum across top-k per token
    out = cache3.sum(dim=1)
    return out


@torch.library.custom_op("auto_deploy::triton_moe_fused", mutates_args=())
def triton_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Triton implementation of the Fused MOE ops for Nemotron-6 models

    Each expert has two weight matrices and squared ReLU activation between them.
    """
    x_shape = x.shape
    hidden_size = x_shape[-1]
    x2d = x.view(-1, hidden_size)

    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int32)

    # Expect selected_experts/routing_weights to be [M, top_k]
    topk_ids = selected_experts.contiguous()
    topk_weights = routing_weights.contiguous()
    assert topk_ids.dim() == 2 and topk_weights.dim() == 2, (
        f"Expected 2D routing tensors, got {topk_ids.shape} and {topk_weights.shape}"
    )
    assert topk_ids.shape[0] == x2d.shape[0], (
        f"Token count mismatch: tokens={x2d.shape[0]} ids={topk_ids.shape[0]}"
    )

    out2d = fused_mlp_relu2_unquantized(
        x2d,
        w1_stacked_weight,
        w2_stacked_weight,
        topk_ids,
        topk_weights,
        apply_router_weight_on_input=False,
    )
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
