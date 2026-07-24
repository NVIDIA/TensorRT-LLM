# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx
from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch
from tensorrt_llm.logger import logger

try:
    # A missing build raises ImportError; a CuTe/CUTLASS mismatch raises
    # RuntimeError (mirror FlashInfer's own guard) -> Triton fallback.
    # gated_delta_rule: T=1 decode entry (dispatches to the wide_vec fast path
    # when B*HV is large). gated_delta_rule_mtp: T>=1 with batch-scoped
    # intermediate_states_buffer and disable_state_update support, used by the
    # speculative-decoding target-verify path.
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        gated_delta_rule as _fi_gdn_decode_bf16_state_t1
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import \
        gated_delta_rule_mtp as _fi_gdn_decode_bf16_state_mtp
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = True
except (ImportError, RuntimeError):
    _FLASHINFER_GDN_BF16_STATE_AVAILABLE = False

# Max per-sequence token count served by the FlashInfer MTP verify kernel; the
# parity test (test_flashinfer_gdn_verify.py) covers T=1..8 against the Triton
# reference. Longer drafts fall back to the Triton recurrent kernel.
_FI_GDN_MAX_MTP_T = 8


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    total_nh,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    s_h0_0,
    h0_dim0,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    grid_stride_nh = tl.num_programs(2)

    while i_nh < total_nh:
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            all = T
            seq_T = eos - bos
        else:
            bos, eos = i_n * T, i_n * T + T
            all = B * T
            seq_T = T

        # Decode q/k/v/a/b often arrive as views sliced out of larger packed tensors.
        # Use the caller-provided token strides so the kernel can consume those views
        # directly instead of relying on a packed contiguous layout.
        p_q = q + bos * stride_q + i_h * K + o_k
        p_k = k + bos * stride_k + i_h * K + o_k
        p_v = v + bos * stride_v + i_hv * V + o_v
        p_b = b + bos * stride_b + i_hv
        # o is allocated in this wrapper and kept contiguous, so the output
        # pointer arithmetic can use the packed [NK, B, T, HV, V] layout.
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        # Gating computation pointers
        p_A_log = A_log + i_hv
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                tl.device_assert(idx < h0_dim0,
                                 "idx out of bounds in h0_source load")
                # Pool layout [slots, HV, V, K] with K innermost (stride 1).
                # b_h is logically [BK, BV]; element [k, v] lives at
                # offset v*K + k within a (V, K) tile.
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                        o_v[None, :] * K)
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        for _ in range(0, seq_T):
            # Load inputs
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)

            # Compute sigmoid gating
            # Load gating parameters
            b_A_log = tl.load(p_A_log).to(tl.float32)
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

            # Compute beta = sigmoid(b)
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            # Apply L2 normalization if enabled
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)

            b_q = b_q * scale

            # Apply gating to hidden state: h *= exp(g)
            b_h *= tl.exp(b_g)

            # Delta rule: v -= sum(h * k, dim=0)
            b_v -= tl.sum(b_h * b_k[:, None], 0)

            # Apply beta gating: v *= beta
            b_v *= b_beta

            # Update hidden state: h += k[:, None] * v[None, :]
            b_h += b_k[:, None] * b_v[None, :]

            # Compute output: o = sum(h * q, dim=0)
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # Update pointers for next timestep
            p_q += stride_q
            p_k += stride_k
            p_o += HV * V
            p_v += stride_v
            p_b += stride_b
            p_a += stride_a

        # Store final state back to h0_source with bounds checking
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                tl.device_assert(idx < h0_dim0,
                                 "idx out of bounds in h0_source store")
                # Pool layout [slots, HV, V, K] with K innermost (stride 1).
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * V * K + o_k[:, None] +
                        o_v[None, :] * K)
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

        i_nh += grid_stride_nh


def _can_use_flashinfer_gdn_decode(
    initial_state_source: Optional[torch.Tensor],
    K: int,
    V: int,
    T: int,
    N: int,
) -> bool:
    """Check whether FlashInfer GDN bf16-state decode kernel can be used."""
    # Env-var escape hatch for A/B comparison against the Triton fallback.
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    # FlashInfer's GDN decode kernel is built for Hopper (SM90) and datacenter
    # Blackwell (SM100/SM103) only; on consumer Blackwell (SM120) and other archs
    # it aborts at launch -> fall back to the Triton fused-recurrent kernel.
    if not is_flashinfer_gdn_supported_arch():
        return False
    if initial_state_source is None:
        return False
    if initial_state_source.dtype != torch.bfloat16:
        return False
    if K != 128 or V != 128:
        return False
    if N == 0:
        return False
    # Standard decode only: T is the flattened token total, so T == N forces
    # exactly 1 token/sequence (making the [N, 1, ...] reshape valid). Varlen or
    # multi-token batches (T != N) can't be reshaped from T alone -> Triton.
    if T != N:
        return False

    return True


def _flashinfer_gdn_decode(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN standard decode via the FlashInfer CuTe-DSL bf16-state kernel.

    Guarded to ``T_per_seq == 1`` (uses ``gated_delta_rule``); the state pool +
    indices are passed directly, no caller-side gather/scatter.
    """
    N = len(cu_seqlens) - 1
    T_total = q.shape[1]
    T_per_seq = T_total // N
    HV = v.shape[2]
    V = v.shape[3]

    # Reshape from packed varlen [1, N*T, ...] to batched [N, T, ...].
    q_bat = q.view(N, T_per_seq, q.shape[2], q.shape[3])
    k_bat = k.view(N, T_per_seq, k.shape[2], k.shape[3])
    v_bat = v.view(N, T_per_seq, v.shape[2], v.shape[3])
    a_bat = a.view(N, T_per_seq, -1)
    b_bat = b.view(N, T_per_seq, -1)

    output = (output.view(N, T_per_seq, HV, V)
              if output is not None else q.new_empty(N, T_per_seq, HV, V))

    assert T_per_seq == 1, (
        f"_flashinfer_gdn_decode expects standard decode (T_per_seq == 1), got "
        f"{T_per_seq}; _can_use_flashinfer_gdn_decode should keep T == N")
    _fi_gdn_decode_bf16_state_t1(
        A_log=A_log,
        a=a_bat,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q_bat,
        k=k_bat,
        v=v_bat,
        b=b_bat,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices.int(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        output=output,
    )

    # Reshape output from [N, T, HV, V] back to [1, N*T, HV, V].
    return output.reshape(1, T_total, HV, -1)


def _can_use_flashinfer_gdn_verify(
    initial_state_source: Optional[torch.Tensor],
    head_k_dim: int,
    head_v_dim: int,
    draft_token_num: int,
) -> bool:
    """Whether the FlashInfer MTP kernel should serve the speculative verify step.

    Default ON when eligible; set ``TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY=1``
    to force the Triton recurrent verify kernel (``TRTLLM_FLA_DISABLE_FLASHINFER_GDN=1``
    disables all FlashInfer GDN decode paths, including this one). The same
    constraints as the decode path apply (bf16 state pool, K==V==128, supported
    arch, FI MTP API available) plus a per-sequence draft length in
    [1, _FI_GDN_MAX_MTP_T]; longer drafts fall back to Triton.
    """
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN", "0") == "1":
        return False
    if os.environ.get("TRTLLM_FLA_DISABLE_FLASHINFER_GDN_VERIFY", "0") == "1":
        return False
    if not _FLASHINFER_GDN_BF16_STATE_AVAILABLE:
        return False
    if not is_flashinfer_gdn_supported_arch():
        return False
    if initial_state_source is None or initial_state_source.dtype != torch.bfloat16:
        return False
    if head_k_dim != 128 or head_v_dim != 128:
        return False
    if not (1 <= draft_token_num <= _FI_GDN_MAX_MTP_T):
        return False
    return True


def _flashinfer_gdn_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN MTP *verify* via the FlashInfer bf16-state kernel.

    Inputs are batched ``[N, draft_token_num, H, D]``. The kernel gathers the
    initial state from the pool via ``initial_state_indices`` (no host-side
    gather copy), writes the SSM state after each draft token into the
    batch-scoped ``intermediate_states_buffer`` (``[N, draft_token_num, HV, V,
    K]``, matching the Triton verify kernel) and leaves the live state pool
    untouched (``disable_state_update``) so the cache manager selects the
    accepted-position state afterwards. Returns the attention output
    ``[N, draft_token_num, HV, V]``.
    """
    logger.info_once(
        "Using FlashInfer CuTe-DSL kernel for GDN MTP verify "
        "(bf16 state, K=V=128)",
        key="flashinfer_gdn_verify")
    N, T = q.shape[0], q.shape[1]
    HV, V = v.shape[2], v.shape[3]
    output = (output.view(N, T, HV, V) if output is not None else q.new_empty(
        N, T, HV, V))
    # The FI CuTe-DSL kernel asserts 32-byte data alignment on every tensor
    # argument. The int32 index tensor may be a slice of a larger buffer
    # (e.g. state_indices_d = cache_indices[num_prefills:]) whose 4*offset
    # storage offset breaks that; .int() is a no-op for int32, so realign
    # with an explicit copy when needed.
    initial_state_indices = initial_state_indices.int()
    if initial_state_indices.data_ptr() % 32 != 0:
        initial_state_indices = initial_state_indices.clone()
    _fi_gdn_decode_bf16_state_mtp(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        output=output,
    )
    return output


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    When FlashInfer's CuTe-DSL GDN decode kernel is available and the state
    dtype is bfloat16, dispatches to the faster FlashInfer path automatically.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"

    # Dispatch to FlashInfer CuTe-DSL kernel when available and conditions met.
    if (cu_seqlens is not None and _can_use_flashinfer_gdn_decode(
            initial_state_source, K, V, T, N)):
        logger.info_once(
            "Using FlashInfer CuTe-DSL kernel for GDN decode "
            "(bf16 state, K=V=128)",
            key="flashinfer_gdn_decode")
        return _flashinfer_gdn_decode(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            output=output,
        )

    # Fallback: Triton kernel path.
    # Accept native view layouts from forward_decode rather than forcing packed
    # copies through input_guard.
    stride_q = q.stride(1)
    stride_k = k.stride(1)
    stride_v = v.stride(1)
    stride_a = a.stride(-2)
    stride_b = b.stride(-2)
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = output.unsqueeze(0) if output is not None else q.new_empty(NK, *v.shape)
    # (NK, NV, N * HV) is found faster than (N * HV, NV, NK)
    # As max of grid.z is 65535, we cap grid.z and let each Triton program
    # grid-stride across the remaining N * HV tiles.
    grid = (NK, NV, min(N * HV, 65535))

    if initial_state_source is not None:
        s_h0_0, s_h0_1, s_h0_2, s_h0_3 = initial_state_source.stride()
        slot_num = initial_state_source.shape[0]
        # Pool layout is [slots, HV, V, K] with K innermost (stride 1).
        assert s_h0_3 == 1, f"s_h0_3: {s_h0_3} is not 1"
        assert s_h0_2 == K, f"s_h0_2: {s_h0_2} is not {K}"
        assert s_h0_1 == V * K, f"s_h0_1: {s_h0_1} is not {V * K}"
    else:
        s_h0_0 = 0
        slot_num = 0

    # input_guard used to set the active CUDA device and make inputs contiguous.
    # We keep only the device-context part here so Triton launches on q's device
    # without re-packing the decode views.
    with custom_device_ctx(q.device.index):
        fused_sigmoid_gating_delta_rule_update_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T,
            total_nh=N * HV,
            stride_q=stride_q,
            stride_k=stride_k,
            stride_v=stride_v,
            stride_a=stride_a,
            stride_b=stride_b,
            s_h0_0=s_h0_0,
            h0_dim0=slot_num,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    o = o.squeeze(0)
    return o
