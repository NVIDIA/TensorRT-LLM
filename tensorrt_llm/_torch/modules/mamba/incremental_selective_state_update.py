# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/ops/mamba_ssm.py
# SPDX-FileCopyrightText: Copyright contributors to the sglang project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import torch
import triton
import triton.language as tl
from einops import rearrange

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID

from .softplus import softplus


@triton.heuristics(
    {"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({
    "HAS_CACHE_BATCH_INDICES":
    lambda args: args["state_batch_indices_ptr"] is not None
})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit()
def _incremental_selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    old_x_ptr,
    old_dt_ptr,
    old_B_ptr,
    prev_num_accepted_tokens_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Matrix dimensions
    T: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_old_x_batch,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    stride_old_dt_batch,
    stride_old_dt_T,
    stride_old_dt_head,
    stride_old_B_batch,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    FAST_FORWARD_REPLAY: tl.constexpr,
    CB_OUTPUT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Resolve batch index for cache (state & old_* tensors)
    if HAS_CACHE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        cache_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b

    # Base pointers for cache-indexed tensors
    state_ptr += cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    old_x_ptr += cache_batch_idx * stride_old_x_batch + pid_h * stride_old_x_head
    old_dt_ptr += cache_batch_idx * stride_old_dt_batch + pid_h * stride_old_dt_head
    old_B_ptr += cache_batch_idx * stride_old_B_batch + (pid_h // nheads_ngroups_ratio) * stride_old_B_group
    prev_num_accepted_tokens_ptr += cache_batch_idx

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim +
                              offs_n[None, :] * stride_state_dstate)
    state_mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr)

    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
        dt_bias = tl.load(dt_bias_ptr).to(tl.float32)
    A_ptr += pid_h * stride_A_head
    A = tl.load(A_ptr).to(tl.float32)

    # ===================================================================
    # Phase 1: Replay old tokens to restore true base state.
    # Compile-time bounded loop (T is constexpr) enables unrolling.
    # ===================================================================
    if FAST_FORWARD_REPLAY:
        # Fast-forward: single reverse pass with no serial state dependency
        # and no redundant dt loads.
        #
        # Process tokens in reverse order.  Maintain `remaining_decay` which
        # accumulates the decay for all tokens after the current one:
        #   state += remaining_decay * dt_t * B_t ⊗ x_t   (add contribution)
        #   remaining_decay *= exp(A * dt_t)                (decay for earlier tokens)
        # After the loop, apply remaining_decay to the initial state:
        #   state *= remaining_decay
        #
        # Same ops as sequential (T loads each of dt/x/B, T exp, T outer products,
        # T+1 state multiplies) but no serial dependency on state between iterations.
        remaining_decay = 1.0
        for t_fwd in range(T):
            t = T - 1 - t_fwd
            if t < prev_num_accepted_tokens:
                old_x_ptrs = old_x_ptr + t * stride_old_x_T + offs_m * stride_old_x_dim
                old_B_ptrs = old_B_ptr + t * stride_old_B_T + offs_n * stride_old_B_dstate
                x = tl.load(old_x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
                dt_val = tl.load(old_dt_ptr + t * stride_old_dt_T).to(tl.float32)
                if HAS_DT_BIAS:
                    dt_val += dt_bias
                if DT_SOFTPLUS:
                    dt_val = softplus(dt_val)
                B = tl.load(old_B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
                state += remaining_decay * dt_val * B[None, :] * x[:, None]
                remaining_decay *= tl.exp(A * dt_val)
        state *= remaining_decay
    else:
        # Sequential replay: straightforward state update with serial dependency.
        for t in range(T):
            if t < prev_num_accepted_tokens:
                old_x_ptrs = old_x_ptr + t * stride_old_x_T + offs_m * stride_old_x_dim
                old_B_ptrs = old_B_ptr + t * stride_old_B_T + offs_n * stride_old_B_dstate
                x = tl.load(old_x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
                dt = tl.load(old_dt_ptr + t * stride_old_dt_T).to(tl.float32)
                if HAS_DT_BIAS:
                    dt += dt_bias
                if DT_SOFTPLUS:
                    dt = softplus(dt)
                dA = tl.exp(A * dt)
                B = tl.load(old_B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
                dB = B * dt
                state = state * dA + dB[None, :] * x[:, None]

    # Write post-replay state back (API contract: state after replay only)
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=state_mask)

    # Set up new input pointers
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    B_ptr += pid_b * stride_B_batch + (pid_h //
                                       nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h //
                                       nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # ===================================================================
    # Phase 2: Process new tokens and compute outputs.
    # ===================================================================
    if CB_OUTPUT:
        # CB formulation: avoid maintaining (M, dstate) state across T iterations.
        #
        #   out_t[m] = exp(A * cumdt_t) * (C_t · state_0)[m]
        #            + Σ_{j≤t} CB[t,j] * decay(j→t) * dt_j * x_j[m]
        #
        # where CB[t,j] = C_t · B_j is a scalar dot product over dstate.
        #
        # Pass 1: Load and cache x, dt, B to old_* buffers + process dt values.
        # Values stay warm in L1 cache for pass 2.
        cumAdt = 0.0
        for t in range(T):
            x_t = tl.load(x_ptr + t * stride_x_T + offs_m * stride_x_dim,
                          mask=offs_m < dim, other=0.0)
            tl.store(old_x_ptr + t * stride_old_x_T + offs_m * stride_old_x_dim,
                     x_t, mask=offs_m < dim)

            dt_raw = tl.load(dt_ptr + t * stride_dt_T)
            tl.store(old_dt_ptr + t * stride_old_dt_T, dt_raw)

            B_t = tl.load(B_ptr + t * stride_B_T + offs_n * stride_B_dstate,
                          mask=offs_n < dstate, other=0.0)
            tl.store(old_B_ptr + t * stride_old_B_T + offs_n * stride_old_B_dstate,
                     B_t, mask=offs_n < dstate)

        # Load D once if needed
        if HAS_D:
            D = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)

        # Pass 2: Compute outputs via CB formulation.
        # Reload B_j/x_j/dt_j from L1 cache (warm from pass 1).
        cumAdt = 0.0
        for t in range(T):
            # Load and process dt_t, accumulate cumulative A*dt
            dt_raw_t = tl.load(dt_ptr + t * stride_dt_T).to(tl.float32)
            dt_proc_t = dt_raw_t
            if HAS_DT_BIAS:
                dt_proc_t = dt_proc_t + dt_bias
            if DT_SOFTPLUS:
                dt_proc_t = softplus(dt_proc_t)
            cumAdt = cumAdt + A * dt_proc_t
            cumAdt_t = cumAdt

            # Load C_t
            C_t = tl.load(C_ptr + t * stride_C_T + offs_n * stride_C_dstate,
                          mask=offs_n < dstate, other=0.0).to(tl.float32)

            # Initial state contribution: C_t · state_0 * exp(cumAdt_t)
            out_t = tl.sum(state * C_t[None, :], axis=1) * tl.exp(cumAdt_t)

            # CB contributions from tokens j = 0..t (reload from L1 cache)
            cumAdt_j = 0.0
            for j in range(T):
                if j <= t:
                    B_j = tl.load(B_ptr + j * stride_B_T + offs_n * stride_B_dstate,
                                  mask=offs_n < dstate, other=0.0).to(tl.float32)
                    x_j = tl.load(x_ptr + j * stride_x_T + offs_m * stride_x_dim,
                                  mask=offs_m < dim, other=0.0).to(tl.float32)
                    dt_raw_j = tl.load(dt_ptr + j * stride_dt_T).to(tl.float32)
                    dt_proc_j = dt_raw_j
                    if HAS_DT_BIAS:
                        dt_proc_j = dt_proc_j + dt_bias
                    if DT_SOFTPLUS:
                        dt_proc_j = softplus(dt_proc_j)
                    cumAdt_j = cumAdt_j + A * dt_proc_j
                    CB_tj = tl.sum(C_t * B_j)
                    decay_jt = tl.exp(cumAdt_t - cumAdt_j)
                    out_t = out_t + CB_tj * decay_jt * dt_proc_j * x_j

            # Apply D and z
            if HAS_D:
                x_t = tl.load(x_ptr + t * stride_x_T + offs_m * stride_x_dim,
                              mask=offs_m < dim, other=0.0).to(tl.float32)
                out_t = out_t + x_t * D
            if HAS_Z:
                z_t = tl.load(z_ptr + t * stride_z_T + offs_m * stride_z_dim,
                              mask=offs_m < dim, other=0.0).to(tl.float32)
                out_t = out_t * z_t * tl.sigmoid(z_t)

            tl.store(out_ptr + t * stride_out_T + offs_m * stride_out_dim,
                     out_t, mask=offs_m < dim)
    else:
        # Sequential state update: straightforward with serial dependency.
        for t in range(T):
            x_ptrs = x_ptr + t * stride_x_T + offs_m * stride_x_dim
            B_ptrs = B_ptr + t * stride_B_T + offs_n * stride_B_dstate
            C_ptrs = C_ptr + t * stride_C_T + offs_n * stride_C_dstate
            out_ptrs = out_ptr + t * stride_out_T + offs_m * stride_out_dim

            x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0)
            tl.store(old_x_ptr + t * stride_old_x_T + offs_m * stride_old_x_dim,
                     x, mask=offs_m < dim)
            x = x.to(tl.float32)

            dt = tl.load(dt_ptr + t * stride_dt_T)
            tl.store(old_dt_ptr + t * stride_old_dt_T, dt)
            dt = dt.to(tl.float32)

            if HAS_DT_BIAS:
                dt += dt_bias
            if DT_SOFTPLUS:
                dt = softplus(dt)
            dA = tl.exp(A * dt)

            B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
            tl.store(old_B_ptr + t * stride_old_B_T + offs_n * stride_old_B_dstate,
                     B, mask=offs_n < dstate)
            B = B.to(tl.float32)

            C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
            if HAS_D:
                D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_Z:
                z = tl.load(z_ptr + t * stride_z_T + offs_m * stride_z_dim,
                            mask=offs_m < dim, other=0.0).to(tl.float32)

            dB = B * dt
            state = state * dA + dB[None, :] * x[:, None]

            out = tl.sum(state * C[None, :], axis=1)
            if HAS_D:
                out += x * D
            if HAS_Z:
                out *= z * tl.sigmoid(z)
            tl.store(out_ptrs, out, mask=offs_m < dim)


def incremental_selective_state_update(
    state: torch.Tensor,
    intermediate_update_inputs: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor | None =None,
    z: torch.Tensor | None =None,
    dt_bias: torch.Tensor | None =None,
    dt_softplus: bool =False,
    state_batch_indices: torch.Tensor | None =None,
    pad_slot_id: int =PAD_SLOT_ID,
    launch_with_pdl=False,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _fast_forward_replay: bool = False,
    _cb_output: bool = False,
):
    """
    Argument:
        state: (batch/cache, nheads, dim, dstate), in-place.
            After the call, contains the state after replaying prev_num_accepted_tokens
            old tokens (i.e. the "true" base state for this step), NOT after the new tokens.
        intermediate_update_inputs: (batch/cache, old_T, nheads*dim + nheads + ngroups*dstate)
            Packed [old_x | old_dt_base | old_B] from the previous step, in-place.
            On return, the first T slots are overwritten with the new step's x, dt, B.
        prev_num_accepted_tokens: (batch/cache,) number of old tokens to replay per slot.
        x: (batch, T, nheads, dim) new token inputs
        dt: (batch, T, nheads, dim) with stride(-1)==0 (tie_hdim)
        A: (nheads, dim, dstate) with stride(-1)==0, stride(-2)==0 (tie_hdim)
        B: (batch, T, ngroups, dstate)
        C: (batch, T, ngroups, dstate)
        out: (batch, T, nheads, dim) preallocated output tensor, in-place updated.
        D: (nheads, dim) optional skip connection
        z: (nheads, dim) optional gating tensor
        dt_bias: (nheads, dim) with stride(-1)==0 (tie_hdim)
        dt_softplus: if True, apply softplus to dt
        state_batch_indices: (batch,) optional indices mapping each batch element to a
            cache slot. If None, batch element i maps to slot i.
            If state_batch_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: state_batch_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at indices 0 and 3.
        pad_slot_id: int sentinel value in state_batch_indices marking padding entries.
        launch_with_pdl: If True, launch with Programmatic Dependent Launch.
            Requires all inputs other than x, B, and C to already be available.
            Allows addressing math and state loading to overlap with the prior kernel.
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    cache_size, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch, )
    assert out.shape == x.shape

    assert intermediate_update_inputs.dim() == 3
    old_T = intermediate_update_inputs.shape[1]
    assert old_T == T
    assert intermediate_update_inputs.shape == (cache_size, old_T, nheads * (dim + 1) + ngroups * dstate)
    old_x, old_dt, old_B = torch.split(intermediate_update_inputs, [nheads*dim, nheads, ngroups*dstate], dim=-1)
    old_x = rearrange(old_x, "b t (h p) -> b t h p", h = nheads)
    old_B = rearrange(old_B, "b t (g n) -> b t g n", g = ngroups)

    assert prev_num_accepted_tokens.shape == (cache_size, )
    assert prev_num_accepted_tokens.stride(0) == 1

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2),
                  z.stride(3)) if z is not None else (0, 0, 0, 0))
    # We don't want autotune since it will overwrite the state.
    # Tuned by hand. For large dstate, fewer warps is critical: the (M, dstate)
    # state matrix lives in registers, and fewer warps means more registers per
    # thread, avoiding costly spills to local memory.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16 else
                               ((16, 4) if dstate <= 32 else
                                ((8, 2) if dstate <= 64 else
                                 ((8, 1) if dstate <= 128 else ((4, 1))))))
    if _block_size_m is not None:
        BLOCK_SIZE_M = _block_size_m
    if _num_warps is not None:
        num_warps = _num_warps
    tie_hdim = (A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0
                and (dt_bias is None or dt_bias.stride(-1) == 0))

    assert tie_hdim

    with torch.cuda.device(x.device.index):
        _incremental_selective_scan_update_kernel[grid](
            state,
            old_x,
            old_dt,
            old_B,
            prev_num_accepted_tokens,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            pad_slot_id,
            T,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            old_x.stride(0),
            old_x.stride(1),
            old_x.stride(2),
            old_x.stride(3),
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else (0,0),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dt_softplus,
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=launch_with_pdl,
            FAST_FORWARD_REPLAY=_fast_forward_replay,
            CB_OUTPUT=_cb_output,
            num_warps=num_warps,
            launch_pdl=launch_with_pdl,
        )
