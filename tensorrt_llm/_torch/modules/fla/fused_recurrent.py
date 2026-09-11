# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/fused_recurrent.py
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.op import exp
from tensorrt_llm._torch.modules.fla.utils import input_guard


@triton.heuristics({
    "USE_INITIAL_STATE":
    lambda args: args["h0_source"] is not None,
    "IS_VARLEN":
    lambda args: args["cu_seqlens"] is not None,
    "CACHE_INTERMEDIATE_STATES":
    lambda args: args["intermediate_states_buffer"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_update_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    intermediate_states_buffer,
    cache_steps,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    IS_BETA_HEADWISE: tl.
    constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,  # whether to disable final state update
    DISABLE_OUTPUT_CALCULATION: tl.
    constexpr,  # whether to disable output calculation
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        # Add bounds checking for idx
        if idx >= 0:  # Assuming negative indices are invalid
            # Pool layout [slots, HV, V, K], K innermost (stride 1).
            p_h0 = (h0_source + idx * HV * V * K + i_hv * V * K + o_k[:, None] +
                    o_v[None, :] * K)
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Prepare intermediate state cache variables if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(h0_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        b_h *= exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        if not DISABLE_OUTPUT_CALCULATION:
            b_o = tl.sum(b_h * b_q[:, None], 0)
            # core attn output
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # store intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                # Match the SSM pool layout [slots, HV, V, K] with K innermost.
                step_offset = step_idx * HV * V * K
                cache_ptr = (intermediate_states_buffer +
                             cache_idx * cache_steps * HV * V * K +
                             step_offset + i_hv * V * K + o_k[:, None] +
                             o_v[None, :] * K)
                tl.store(cache_ptr,
                         b_h.to(cache_ptr.dtype.element_ty),
                         mask=mask_h)

        step_idx += 1

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    # Store final state back to h0_source with bounds checking
    # ssm states (pool layout [slots, HV, V, K], K innermost).
    if not DISABLE_STATE_UPDATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:  # Add bounds checking
            p_h0 = (h0_source + idx * HV * V * K + i_hv * V * K + o_k[:, None] +
                    o_v[None, :] * K)
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_update_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if disable_output_calculation:
        # When output calculation is disabled, allocate minimal tensor
        o = q.new_empty(NK, 1, 1, 1, 1)  # minimal allocation
    else:
        o = output.unsqueeze(0) if output is not None else q.new_empty(
            NK, *v.shape)

    grid = (NK, NV, N * HV)

    fused_recurrent_gated_delta_rule_update_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=0 if cache_steps is None else cache_steps,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        DISABLE_STATE_UPDATE=disable_state_update,
        DISABLE_OUTPUT_CALCULATION=disable_output_calculation,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


class FusedRecurrentUpdateFunction(torch.autograd.Function):

    @staticmethod
    @input_guard(exclude_args=["output"])
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        output: Optional[torch.Tensor] = None,
    ):
        o = fused_recurrent_gated_delta_rule_update_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            disable_state_update=disable_state_update,
            disable_output_calculation=disable_output_calculation,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            output=output,
        )

        return o

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps.")


def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state_source: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        if (initial_state_source is not None
                and initial_state_indices.shape[0] != len(cu_seqlens) - 1):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o = FusedRecurrentUpdateFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        disable_state_update,
        disable_output_calculation,
        intermediate_states_buffer,
        cache_steps,
        output,
    )
    return o
