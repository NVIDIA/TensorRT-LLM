# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA Corporation. All rights reserved.
"""
Triton FP8 block-scale MoE forward pass for SM120.
"""

import torch
import triton
import triton.language as tl

_BLOCK_SHAPE = [128, 128]  # [group_k, group_n] for FP8 block-scale
_BLOCK_SIZE_M = 16


@triton.jit
def _moe_histogram_kernel(
    flat_ids_ptr,
    counts_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Build a histogram of selected experts."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    ids = tl.load(flat_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    ones = tl.full([BLOCK_N], 1, dtype=tl.int32)
    tl.atomic_add(counts_ptr + ids, ones, mask=mask)


@triton.jit
def _int32_add(a, b):
    return a + b


@triton.jit
def _moe_prefix_kernel(
    counts_ptr,
    out_off_ptr,
    num_post_pad_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Compute exclusive offsets from padded expert counts."""
    e = tl.arange(0, NUM_EXPERTS)
    counts = tl.load(counts_ptr + e).to(tl.int32)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    # Inclusive scan; subtract self for exclusive prefix.
    inc = tl.associative_scan(padded, 0, _int32_add)
    out_off = inc - padded
    tl.store(out_off_ptr + e, out_off)
    tl.store(num_post_pad_ptr, tl.sum(padded).to(tl.int32))


@triton.jit
def _moe_expert_ids_kernel(
    counts_ptr,
    out_off_ptr,
    expert_ids_ptr,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill the expert id for each padded token block."""
    e = tl.program_id(0)
    if e >= num_experts:
        return
    cnt = tl.load(counts_ptr + e).to(tl.int32)
    off = tl.load(out_off_ptr + e).to(tl.int32)
    padded = ((cnt + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    start_blk = off // BLOCK_SIZE
    n_blk = padded // BLOCK_SIZE
    for i in range(n_blk):
        tl.store(expert_ids_ptr + start_blk + i, e)


@triton.jit
def _moe_scatter_kernel(
    flat_ids_ptr,
    counters_ptr,
    sorted_token_ids_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    """Scatter token indices into their expert ranges."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    ids = tl.load(flat_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    ones = tl.full([BLOCK_N], 1, dtype=tl.int32)
    # Atomic add returns the slot reserved for this token.
    pos = tl.atomic_add(counters_ptr + ids, ones, mask=mask)
    tl.store(sorted_token_ids_ptr + pos, offs.to(tl.int32), mask=mask)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert pairs by expert and pad each expert's row-count to
    a multiple of block_size.

    The implementation is CUDA-graph compatible: all buffer sizes are Python
    integers and there are no GPU-to-CPU synchronizations.

    Args:
        topk_ids: (num_tokens, top_k) int64 expert IDs.
        block_size: BLOCK_SIZE_M used in the GEMM kernel.
        num_experts: Total number of experts.

    Returns:
        Tuple of sorted token IDs, block expert IDs, and the padded token count.
    """
    device = topk_ids.device
    flat_ids = topk_ids.reshape(-1).to(torch.int32)
    N = flat_ids.shape[0]  # Python int, no GPU sync

    max_tokens_static = N + num_experts * block_size
    num_m_blocks_static = N // block_size + num_experts

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    out_offsets = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorted_token_ids = torch.full((max_tokens_static,), N, dtype=torch.int32, device=device)
    # `expert_ids_out` does not need to be zero-initialized: _moe_expert_ids_kernel
    # writes positions [0, total_n_blocks) exhaustively (per-expert ranges are
    # back-to-back via the prefix scan). The allocated tail is never read.
    expert_ids_out = torch.empty((num_m_blocks_static,), dtype=torch.int32, device=device)
    num_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    _BLOCK_N = 32

    if N > 0:
        _moe_histogram_kernel[triton.cdiv(N, _BLOCK_N),](flat_ids, counts, N, BLOCK_N=_BLOCK_N)

    _moe_prefix_kernel[1,](
        counts,
        out_offsets,
        num_post_pad,
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
    )

    _moe_expert_ids_kernel[num_experts,](
        counts,
        out_offsets,
        expert_ids_out,
        num_experts=num_experts,
        BLOCK_SIZE=block_size,
    )

    if N > 0:
        counters = out_offsets.clone()
        _moe_scatter_kernel[triton.cdiv(N, _BLOCK_N),](
            flat_ids, counters, sorted_token_ids, N, BLOCK_N=_BLOCK_N
        )

    return sorted_token_ids, expert_ids_out, num_post_pad


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    compute_type: tl.constexpr,
):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _build_moe_gemm_autotune_configs():
    """Config grid for `_fused_moe_bf16act_fp8w_kernel`.

    Constraints:
        - BLOCK_SIZE_M is FIXED at 16. The MoE alignment kernels
          (`_moe_expert_ids_kernel` etc.) lay out `expert_ids_out` with one
          expert per 16-token chunk (the `_BLOCK_SIZE_M` constant). The GEMM
          loads `expert_ids[pid_m]` assuming the same chunk size, so
          mismatched values produce wrong outputs (autotune cannot detect
          this; it benchmarks runtime, not numerics).
        - BLOCK_SIZE_K <= group_k (=128): the kernel loads ONE scale per
          K-tile, so larger K-tiles would alias multiple block scales.
    """
    configs = []
    for BN in (64, 128, 256):
        for BK in (64, 128):
            for GROUP_M in (1, 4, 8):
                for num_stages in (3, 4):
                    for num_warps in (4, 8):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_M": 16,
                                    "BLOCK_SIZE_N": BN,
                                    "BLOCK_SIZE_K": BK,
                                    "GROUP_SIZE_M": GROUP_M,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton.autotune(
    configs=_build_moe_gemm_autotune_configs(),
    key=["N", "K"],
    # FUSE_TOPK_REDUCE uses atomic_add into c_ptr, so autotune trials must start
    # from zero instead of accumulating on the previous trial's output.
    reset_to_zero=["c_ptr"],
)
@triton.jit
def _fused_moe_bf16act_fp8w_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_TOPK_REDUCE: tl.constexpr,
    REDUCE_TOPK: tl.constexpr,
):
    """
    Fused MoE GEMM with BF16 activations and FP8 block-scaled weights.

    For weight-only quantized models: activations never go through FP8, so
    there is no per-layer quantization error. Each FP8 weight tile is
    dequanted to BF16 inside the K-loop using the per-block scale:
        C[m, n] += dot(A_bf16[m, :], B_fp8[:, n].to(bf16)) * B_scale[n//gn, k//gk]
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        if FUSE_TOPK_REDUCE:
            return
        _write_zeros_to_output(
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
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    offs_bsn = offs_bn // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        b = b_fp8.to(tl.bfloat16)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

        accumulator += tl.dot(a, b, out_dtype=tl.float32) * b_scale[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if FUSE_TOPK_REDUCE:
        out_row = offs_token // REDUCE_TOPK
        c_ptrs = c_ptr + stride_cm * out_row[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_bf16act_fp8w_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    fuse_topk_reduce: bool = False,
    reduce_topk: int = 1,
) -> None:
    """
    Launch _fused_moe_bf16act_fp8w_kernel.

    Tensor layouts (default, fuse_topk_reduce=False):
        A       : (num_tokens_or_expanded, K)  BF16, contiguous
        B       : (E, N, K)                    FP8, contiguous
        C       : (batch, top_k_orig, N)       BF16, stride(1)=N, stride(2)=1
        B_scale : (E, N//128, K//128)           F32
        topk_weights: (num_tokens, top_k_orig) F32  (flat-indexable by offs_token)

    When fuse_topk_reduce=True (used for the second MoE GEMM), each program
    atomically accumulates its expert-weighted contribution into output row
    `offs_token // reduce_topk`, eliminating the standalone reduction kernel.
    The caller is responsible for zero-initializing C, which then has shape
    (num_tokens, N).
    """
    EM = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    group_n, group_k = _BLOCK_SHAPE[1], _BLOCK_SHAPE[0]

    # C is 3D by default and 2D when fusing the top-k reduce.
    if fuse_topk_reduce:
        stride_cm, stride_cn = C.stride(0), C.stride(1)
    else:
        stride_cm, stride_cn = C.stride(1), C.stride(2)

    def grid(META):
        return (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _fused_moe_bf16act_fp8w_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        A.shape[0] * top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        stride_cm,
        stride_cn,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        group_n=group_n,
        group_k=group_k,
        top_k=top_k,
        compute_type=tl.bfloat16,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_TOPK_REDUCE=fuse_topk_reduce,
        REDUCE_TOPK=reduce_topk,
    )


@triton.jit
def _gated_activation_kernel(
    ic1_ptr,
    ic2_ptr,
    M,
    INTER,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Compute ic2[m, i] = activation(gate) * up."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < INTER)

    row_stride = 2 * INTER
    up_ptrs = ic1_ptr + offs_m[:, None] * row_stride + offs_i[None, :]
    gate_ptrs = up_ptrs + INTER

    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    if ACTIVATION == 0:
        act = gate * (1.0 / (1.0 + tl.exp(-gate)))
    else:
        act = 0.5 * gate * (1.0 + tl.erf(gate * 0.7071067811865475))

    out = (act * up).to(tl.bfloat16)
    out_ptrs = ic2_ptr + offs_m[:, None] * INTER + offs_i[None, :]
    tl.store(out_ptrs, out, mask=mask)


def _invoke_gated_activation_kernel(
    ic1: torch.Tensor,
    ic2: torch.Tensor,
    activation: int,
) -> None:
    """Launch `_gated_activation_kernel`.

    Args:
        ic1: (M, 2*INTER) bfloat16, contiguous (up half | gate half).
        ic2: (M, INTER)   bfloat16, contiguous (output, may be uninitialized).
        activation: 0 for silu/swiglu, 1 for gelu/geglu (exact).
    """
    assert ic1.is_contiguous() and ic2.is_contiguous()
    assert ic1.dtype == torch.bfloat16 and ic2.dtype == torch.bfloat16
    M, two_i = ic1.shape
    M2, inter = ic2.shape
    assert M == M2 and two_i == 2 * inter

    BLOCK_M = 16
    BLOCK_I = 128 if inter >= 128 else triton.next_power_of_2(inter)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(inter, BLOCK_I))
    _gated_activation_kernel[grid](
        ic1,
        ic2,
        M,
        inter,
        BLOCK_M=BLOCK_M,
        BLOCK_I=BLOCK_I,
        ACTIVATION=activation,
    )


def run_triton_fp8_block_scale_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w3_w1: torch.Tensor,
    w3_w1_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    activation_type: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Complete MoE forward pass using Triton FP8 block-scale GEMMs.

    This is a drop-in replacement for torch.ops.trtllm.fused_moe() for
    SM120 + FP8_BLOCK_SCALES, matching the weight layout produced by
    DeepSeekFP8BlockScalesFusedMoEMethod.

    Args:
        x: (T, H) BF16 input activations.
        token_selected_experts: (T, K) int64 expert IDs per token.
        token_final_scales: (T, K) F32 routing weights.
        w3_w1: (E, 2I, H) FP8 gate+up projection weights.
        w3_w1_scales: (E, Nb, Kb) F32 block scales for w3_w1.
        w2: (E, H, I) FP8 down-projection weights.
        w2_scales: (E, Nb, Kb) F32 block scales for w2.
        activation_type: ActivationType enum value.
        output_dtype: Optional output dtype.

    Returns:
        (T, H) output tensor.
    """
    from tensorrt_llm._torch.modules.fused_moe.interface import ActivationType

    num_tokens, hidden = x.shape
    num_experts, gate_up_size, _ = w3_w1.shape
    intermediate = gate_up_size // 2
    top_k = token_selected_experts.shape[1]
    device = x.device

    act_int = int(activation_type)
    swiglu = int(ActivationType.Swiglu)
    geglu = int(ActivationType.Geglu)
    silu = int(ActivationType.Silu)
    gelu = int(ActivationType.Gelu)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        token_selected_experts, _BLOCK_SIZE_M, num_experts
    )

    ic1 = torch.empty((num_tokens, top_k, gate_up_size), dtype=torch.bfloat16, device=device)
    _invoke_bf16act_fp8w_moe_kernel(
        x.contiguous(),
        w3_w1,
        ic1,
        w3_w1_scales,
        token_final_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
    )

    if act_int in (swiglu, silu):
        activation_id = 0
    elif act_int in (geglu, gelu):
        activation_id = 1
    else:
        raise ValueError(f"Unsupported activation_type={act_int} in Triton FP8 block-scale MoE")

    ic2 = torch.empty((num_tokens * top_k, intermediate), dtype=torch.bfloat16, device=device)
    _invoke_gated_activation_kernel(
        ic1.view(num_tokens * top_k, gate_up_size),
        ic2,
        activation_id,
    )

    # The kernel constexpr `top_k=1` here means each ic2 row is already one
    # (token, expert) pair (not to be confused with the model's `top_k`).
    fuse_topk_reduce = top_k > 1
    if fuse_topk_reduce:
        output = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        gemm2_out = output
    else:
        # The unfused path uses the 3D C layout expected by the GEMM wrapper.
        gemm2_out = torch.empty((num_tokens, 1, hidden), dtype=torch.bfloat16, device=device)

    _invoke_bf16act_fp8w_moe_kernel(
        ic2,
        w2,
        gemm2_out,
        w2_scales,
        token_final_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        fuse_topk_reduce=fuse_topk_reduce,
        reduce_topk=top_k,
    )

    if not fuse_topk_reduce:
        output = gemm2_out.squeeze(1)

    if output_dtype is not None and output.dtype != output_dtype:
        output = output.to(output_dtype)
    return output
