# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.utils import custom_device_ctx


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
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * K * V +
                        o_k[:, None] * V + o_v[None, :])
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
                p_h0 = (h0_source + idx * s_h0_0 + i_hv * K * V +
                        o_k[:, None] * V + o_v[None, :])
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

        i_nh += grid_stride_nh


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
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

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

    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    # (NK, NV, N * HV) is found faster than (N * HV, NV, NK)
    # As max of grid.z is 65535, we cap grid.z and let each Triton program
    # grid-stride across the remaining N * HV tiles.
    grid = (NK, NV, min(N * HV, 65535))

    if initial_state_source is not None:
        s_h0_0, s_h0_1, s_h0_2, s_h0_3 = initial_state_source.stride()
        slot_num = initial_state_source.shape[0]
        assert s_h0_3 == 1, f"s_h0_3: {s_h0_3} is not 1"
        assert s_h0_2 == V, f"s_h0_2: {s_h0_2} is not {V}"
        assert s_h0_1 == K * V, f"s_h0_1: {s_h0_1} is not {K * V}"
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
