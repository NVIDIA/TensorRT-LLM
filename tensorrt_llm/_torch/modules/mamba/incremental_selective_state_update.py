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


# ============================================================================
# Precompute kernel: CB_scaled, decay_vec.  Writes new cache (old_B,
# old_dt_proc, old_cumAdt) to the WRITE buffer slot for next step's replay.
# Grid: (batch, nheads).
# ============================================================================

@triton.heuristics(
    {"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.jit()
def _precompute_cb_scaled_kernel(
    # Input pointers
    dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr,
    # Output pointers
    cb_scaled_ptr, decay_vec_ptr,
    # Cache WRITE pointers (write-buffer for next step)
    old_B_ptr, old_dt_proc_ptr, old_cumAdt_ptr,
    # Double-buffer index (per cache slot)
    cache_buf_idx_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr, dstate: tl.constexpr, nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch, stride_dt_T, stride_dt_head,
    stride_dt_bias_head, stride_A_head,
    # B strides
    stride_B_batch, stride_B_T, stride_B_group, stride_B_dstate,
    # C strides
    stride_C_batch, stride_C_T, stride_C_group, stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch, stride_cb_head, stride_cb_t, stride_cb_j,
    # decay_vec strides
    stride_dv_batch, stride_dv_head, stride_dv_t,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache, stride_old_B_dbuf, stride_old_B_T, stride_old_B_group, stride_old_B_dstate,
    # old_dt_proc strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_proc_cache, stride_old_dt_proc_dbuf, stride_old_dt_proc_head, stride_old_dt_proc_T,
    # old_cumAdt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_cumAdt_cache, stride_old_cumAdt_dbuf, stride_old_cumAdt_head, stride_old_cumAdt_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Resolve cache index for writes
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b

    # Launch dependent kernels immediately — the main kernel's replay phase
    # reads from the READ buffer (written by the PREVIOUS step), not from
    # anything this kernel produces.  The main kernel's gdc_wait() gates
    # only the output phase which reads cb_scaled/decay_vec.
    tl.extra.cuda.gdc_launch_dependents()

    # Read buffer index: replay reads from buf_read.  We WRITE to 1 - buf_read.
    buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    buf_write = 1 - buf_read

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < T
    n_mask = offs_n < dstate

    # --- Load and process dt ---
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    dt_all = tl.load(dt_ptr + offs_t * stride_dt_T, mask=t_mask, other=0.0).to(tl.float32)
    dt_proc = dt_all
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
        dt_proc = dt_proc + dt_bias
    if DT_SOFTPLUS:
        dt_proc = softplus(dt_proc)

    # --- Compute cumAdt and decay_vec ---
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    cumAdt = tl.cumsum(A * dt_proc, axis=0)
    decay_vec = tl.exp(cumAdt)

    # --- Store dt_proc, cumAdt to WRITE buffer ---
    old_dt_proc_base = old_dt_proc_ptr + cache_batch_idx * stride_old_dt_proc_cache + buf_write * stride_old_dt_proc_dbuf + pid_h * stride_old_dt_proc_head
    tl.store(old_dt_proc_base + offs_t * stride_old_dt_proc_T, dt_proc, mask=t_mask)

    old_cumAdt_base = old_cumAdt_ptr + cache_batch_idx * stride_old_cumAdt_cache + buf_write * stride_old_cumAdt_dbuf + pid_h * stride_old_cumAdt_head
    tl.store(old_cumAdt_base + offs_t * stride_old_cumAdt_T, cumAdt, mask=t_mask)

    # --- Store decay_vec ---
    dv_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    tl.store(dv_base + offs_t * stride_dv_t, decay_vec, mask=t_mask)

    # --- Precompute decay_matrix and causal_mask (only depend on cumAdt/offs_t) ---
    causal_mask = offs_t[:, None] >= offs_t[None, :]
    decay_matrix = tl.exp(cumAdt[:, None] - cumAdt[None, :])

    # --- Wait for upstream kernel (external PDL) before loading B and C ---
    # Everything above (dt processing, cumAdt, decay_vec, decay_matrix) is
    # independent of the upstream kernel's B/C outputs.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # --- Load C and B, compute CB = C @ B^T ---
    group_idx = pid_h // nheads_ngroups_ratio
    C_ptr += pid_b * stride_C_batch + group_idx * stride_C_group
    B_ptr += pid_b * stride_B_batch + group_idx * stride_B_group

    # C and B are bf16 — load directly without unnecessary fp32 cast
    C_all = tl.load(C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
                    mask=t_mask[:, None] & n_mask[None, :], other=0.0)
    B_all = tl.load(B_ptr + offs_t[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
                    mask=t_mask[:, None] & n_mask[None, :], other=0.0)

    CB = tl.dot(C_all.to(tl.bfloat16), tl.trans(B_all).to(tl.bfloat16))

    # --- Scale CB: decay * dt * causal_mask ---
    CB_scaled = tl.where(causal_mask & t_mask[:, None] & t_mask[None, :],
                         CB * decay_matrix * dt_proc[None, :], 0.0)

    # --- Store CB_scaled ---
    cb_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    tl.store(cb_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
             CB_scaled,
             mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T))

    # --- Store B to WRITE buffer of old_B cache (once per group, not per head) ---
    if pid_h % nheads_ngroups_ratio == 0:
        old_B_base = (old_B_ptr + cache_batch_idx * stride_old_B_cache
                      + buf_write * stride_old_B_dbuf + group_idx * stride_old_B_group)
        tl.store(old_B_base + offs_t[:, None] * stride_old_B_T + offs_n[None, :] * stride_old_B_dstate,
                 B_all, mask=t_mask[:, None] & n_mask[None, :])


# ============================================================================
# Main kernel: tl.dot replay + precomputed CB output.
# Grid: (cdiv(dim, M), batch, nheads).
# ============================================================================

@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({
    "HAS_CACHE_BATCH_INDICES":
    lambda args: args["state_batch_indices_ptr"] is not None
})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.jit()
def _incremental_selective_scan_update_kernel(
    # Pointers
    state_ptr,
    # Cache READ pointers (read-buffer from previous step)
    old_x_ptr, old_B_ptr, old_dt_proc_ptr, old_cumAdt_ptr,
    # Cache WRITE pointer (write-buffer for old_x only; B/dt/cumAdt written by precompute)
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    # New input pointers
    x_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
    # Precomputed pointers
    cb_scaled_ptr, decay_vec_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr, dim: tl.constexpr, dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    # old_x strides: (cache, T, nheads, dim) — single-buffered
    stride_old_x_cache, stride_old_x_T, stride_old_x_head, stride_old_x_dim,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache, stride_old_B_dbuf, stride_old_B_T, stride_old_B_group, stride_old_B_dstate,
    # old_dt_proc strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_proc_cache, stride_old_dt_proc_dbuf, stride_old_dt_proc_head, stride_old_dt_proc_T,
    # old_cumAdt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_cumAdt_cache, stride_old_cumAdt_dbuf, stride_old_cumAdt_head, stride_old_cumAdt_T,
    # x strides
    stride_x_batch, stride_x_T, stride_x_head, stride_x_dim,
    # C strides
    stride_C_batch, stride_C_T, stride_C_group, stride_C_dstate,
    # D strides
    stride_D_head, stride_D_dim,
    # z strides
    stride_z_batch, stride_z_T, stride_z_head, stride_z_dim,
    # out strides
    stride_out_batch, stride_out_T, stride_out_head, stride_out_dim,
    # cb_scaled strides
    stride_cb_batch, stride_cb_head, stride_cb_t, stride_cb_j,
    # decay_vec strides
    stride_dv_batch, stride_dv_head, stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr, HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr, BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b

    # Double-buffer: read from buf_read, write old_x to buf_write
    buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    buf_write = 1 - buf_read

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    # Load state
    state_ptr += cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)

    # ===================================================================
    # Phase 1: Replay via tl.dot fast-forward (reads from READ buffer)
    # ===================================================================
    group_idx = pid_h // nheads_ngroups_ratio

    # Load precomputed dt_proc and cumAdt from READ buffer
    old_dt_proc_base = (old_dt_proc_ptr + cache_batch_idx * stride_old_dt_proc_cache
                + buf_read * stride_old_dt_proc_dbuf + pid_h * stride_old_dt_proc_head)
    old_dt_proc_all = tl.load(old_dt_proc_base + offs_t * stride_old_dt_proc_T,
                              mask=t_mask, other=0.0).to(tl.float32)

    old_cumAdt_base = (old_cumAdt_ptr + cache_batch_idx * stride_old_cumAdt_cache
                + buf_read * stride_old_cumAdt_dbuf + pid_h * stride_old_cumAdt_head)
    old_cumAdt_all = tl.load(old_cumAdt_base + offs_t * stride_old_cumAdt_T,
                             mask=t_mask, other=0.0).to(tl.float32)

    # Load cumAdt at prev_k-1 directly via pointer math (avoids masked reduction)
    prev_k_idx = tl.maximum(prev_num_accepted_tokens - 1, 0)
    total_cumAdt = tl.load(old_cumAdt_base + prev_k_idx * stride_old_cumAdt_T).to(tl.float32)

    # Compute per-token coefficients
    coeff = tl.exp(total_cumAdt - old_cumAdt_all) * old_dt_proc_all
    coeff = tl.where(offs_t < prev_num_accepted_tokens, coeff, 0.0)

    # Load old_x: (BLOCK_SIZE_T, BLOCK_SIZE_M) — single-buffered
    ox_base = old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head
    old_x_all = tl.load(ox_base + offs_t[:, None] * stride_old_x_T + offs_m[None, :] * stride_old_x_dim,
                        mask=t_mask[:, None] & m_mask[None, :], other=0.0).to(tl.float32)

    # Load old_B from READ buffer: (BLOCK_SIZE_T, BLOCK_SIZE_DSTATE)
    oB_base = (old_B_ptr + cache_batch_idx * stride_old_B_cache
               + buf_read * stride_old_B_dbuf + group_idx * stride_old_B_group)
    old_B_all = tl.load(oB_base + offs_t[:, None] * stride_old_B_T + offs_n[None, :] * stride_old_B_dstate,
                        mask=t_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    # Scale B by coefficients
    dB_scaled = coeff[:, None] * old_B_all

    # Apply total decay to initial state FIRST, then add contributions
    total_decay = tl.where(prev_num_accepted_tokens > 0, tl.exp(total_cumAdt), 1.0)
    state *= total_decay

    # tl.dot fast-forward: old_x^T @ dB_scaled → (M, dstate)
    state += tl.dot(tl.trans(old_x_all).to(tl.bfloat16), dB_scaled.to(tl.bfloat16))

    # Write post-replay state
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=state_mask)

    # ===================================================================
    # Phase 2: Output using precomputed CB_scaled and decay_vec
    # ===================================================================
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    C_ptr += pid_b * stride_C_batch + group_idx * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
                    mask=m_mask, other=0.0).to(tl.float32)

    # Load C_all
    C_all = tl.load(C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
                    mask=t_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    # Load x_all and store to old_x (single-buffered, replay already read it)
    x_all = tl.load(x_ptr + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
                    mask=t_mask[:, None] & m_mask[None, :], other=0.0)
    tl.store(ox_base + offs_t[:, None] * stride_old_x_T + offs_m[None, :] * stride_old_x_dim,
             x_all, mask=t_mask[:, None] & m_mask[None, :])
    x_all = x_all.to(tl.float32)

    # Wait for precompute kernel (PDL) before reading its outputs
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Load precomputed CB_scaled and decay_vec
    cb_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    CB_scaled = tl.load(cb_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
                        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
                        other=0.0).to(tl.float32)

    dv_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    decay_vec = tl.load(dv_base + offs_t * stride_dv_t, mask=t_mask, other=0.0).to(tl.float32)

    # init_out = C_all @ state^T * decay_vec
    init_out = tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16)) * decay_vec[:, None]

    # cb_out = CB_scaled @ x_all
    cb_out = tl.dot(CB_scaled.to(tl.bfloat16), x_all.to(tl.bfloat16))

    out_all = init_out + cb_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        for t in range(T):
            z_t = tl.load(z_ptr + t * stride_z_T + offs_m * stride_z_dim,
                          mask=m_mask, other=0.0).to(tl.float32)
            out_t = tl.sum(tl.where((offs_t == t)[:, None], out_all, 0.0), axis=0)
            out_t = out_t * z_t * tl.sigmoid(z_t)
            tl.store(out_ptr + t * stride_out_T + offs_m * stride_out_dim,
                     out_t, mask=m_mask)
    else:
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# ============================================================================
# Python wrapper
# ============================================================================

def incremental_selective_state_update(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt_proc: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    launch_with_pdl=False,
    use_internal_pdl=True,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _num_stages: int | None = None,
    _precompute_num_warps: int | None = None,
    _precompute_num_stages: int | None = None,
):
    """
    Incremental SSM state update with precomputed CB and tl.dot replay.

    Uses double-buffered cache tensors.  cache_buf_idx[slot] indicates which
    buffer (0 or 1) to READ from for replay.  The WRITE buffer is 1 - read.
    Caller must flip cache_buf_idx[slot] after each call.

    Arguments:
        state: (cache, nheads, dim, dstate) in-place.  After the call, contains
            the state after replaying prev_num_accepted_tokens old tokens.
        old_x: (cache, T, nheads, dim) bf16 — old x cache (single-buffered).
        old_B: (cache, 2, T, ngroups, dstate) bf16 — double-buffered old B cache.
        old_dt_proc: (cache, 2, T, nheads) fp32 — double-buffered processed dt.
        old_cumAdt: (cache, 2, T, nheads) fp32 — double-buffered cumulative A*dt.
        cache_buf_idx: (cache,) int32 — which buffer to read (0 or 1).
        prev_num_accepted_tokens: (cache,) int32.
        x: (batch, T, nheads, dim) new token inputs.
        dt: (batch, T, nheads, dim) with stride(-1)==0 (tie_hdim).
        A: (nheads, dim, dstate) with stride(-1)==0, stride(-2)==0 (tie_hdim).
        B: (batch, T, ngroups, dstate).
        C: (batch, T, ngroups, dstate).
        out: (batch, T, nheads, dim) preallocated output.
        D, z, dt_bias: optional, same as before.
        state_batch_indices: (batch,) optional cache slot mapping.
    """
    # --- Unsqueeze inputs to canonical shapes ---
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
    ngroups = B.shape[2]
    assert nheads % ngroups == 0

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    assert old_x.shape == (cache_size, T, nheads, dim)
    assert old_B.shape == (cache_size, 2, T, ngroups, dstate)
    assert old_dt_proc.shape == (cache_size, 2, nheads, T)
    assert old_cumAdt.shape == (cache_size, 2, nheads, T)
    assert cache_buf_idx.shape == (cache_size,)
    assert prev_num_accepted_tokens.shape == (cache_size,)

    tie_hdim = (A.stride(-1) == 0 and A.stride(-2) == 0 and dt.stride(-1) == 0
                and (dt_bias is None or dt_bias.stride(-1) == 0))
    assert tie_hdim

    device = x.device
    BLOCK_SIZE_T = max(triton.next_power_of_2(T), 16)

    # Allocate precomputed intermediates (per-call, not cached)
    cb_scaled = torch.empty(batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_T,
                            device=device, dtype=torch.float32)
    decay_vec = torch.empty(batch, nheads, BLOCK_SIZE_T,
                            device=device, dtype=torch.float32)

    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                 if z is not None else (0, 0, 0, 0))

    # Main kernel tuning: BLOCK_SIZE_M and num_warps.
    # Keyed on total_heads (batch * nheads) and BLOCK_SIZE_T, not batch alone.
    # This ensures equivalent workloads (e.g. TP=1 batch=1 vs TP=8 batch=8,
    # both with 128 total head-instances) get the same tuning.
    #
    # Swept M={4..64}, W={1..4} across batch={1..512}, T={6,16,32},
    # TP={1,4,8} on B200 (Nemotron-3-Super-120B: nheads=128, ngroups=8).
    # Max gap vs per-config optimal: <2% within any single measurement run.
    #
    # BLOCK_SIZE_T splits the heuristic: T<=16 → BLOCK_SIZE_T=16 (small
    # tl.dot tiles, W=1 preferred), T>16 → BLOCK_SIZE_T=32+ (larger tiles,
    # W=2 for warp-cooperative mma).
    total_heads = batch * nheads
    BLOCK_SIZE_T = max(triton.next_power_of_2(T), 16)
    if BLOCK_SIZE_T <= 16:
        if total_heads <= 64:
            BLOCK_SIZE_M, num_warps = 4, 1
        elif total_heads <= 128:
            BLOCK_SIZE_M, num_warps = 8, 1
        elif total_heads <= 256:
            BLOCK_SIZE_M, num_warps = 16, 1
        elif total_heads <= 1024:
            BLOCK_SIZE_M, num_warps = 32, 4
        else:
            BLOCK_SIZE_M, num_warps = 32, 2
    else:  # T > 16
        if total_heads <= 16:
            BLOCK_SIZE_M, num_warps = 4, 2
        elif total_heads <= 64:
            BLOCK_SIZE_M, num_warps = 16, 2
        else:
            BLOCK_SIZE_M, num_warps = 32, 2
    if _block_size_m is not None:
        BLOCK_SIZE_M = _block_size_m
    if _num_warps is not None:
        num_warps = _num_warps

    HAS_CACHE_BATCH_INDICES = state_batch_indices is not None

    with torch.cuda.device(device.index):
        # --- Precompute kernel ---
        _precompute_cb_scaled_kernel[(batch, nheads)](
            dt, dt_bias, A, B, C,
            cb_scaled, decay_vec,
            old_B, old_dt_proc, old_cumAdt,
            cache_buf_idx,
            state_batch_indices,
            pad_slot_id,
            T, dstate, nheads // ngroups,
            # dt strides
            dt.stride(0), dt.stride(1), dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            # B strides
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            # C strides
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            # cb_scaled strides
            cb_scaled.stride(0), cb_scaled.stride(1), cb_scaled.stride(2), cb_scaled.stride(3),
            # decay_vec strides
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            # old_B strides
            old_B.stride(0), old_B.stride(1), old_B.stride(2), old_B.stride(3), old_B.stride(4),
            # old_dt_proc strides
            old_dt_proc.stride(0), old_dt_proc.stride(1), old_dt_proc.stride(2), old_dt_proc.stride(3),
            # old_cumAdt strides
            old_cumAdt.stride(0), old_cumAdt.stride(1), old_cumAdt.stride(2), old_cumAdt.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            num_warps=_precompute_num_warps or 1,
            **({'num_stages': _precompute_num_stages} if _precompute_num_stages else {}),
            launch_pdl=launch_with_pdl,
        )

        # --- Main kernel ---
        grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
        _incremental_selective_scan_update_kernel[grid](
            state,
            old_x, old_B, old_dt_proc, old_cumAdt,
            prev_num_accepted_tokens,
            cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices,
            pad_slot_id,
            T, dim, dstate, nheads // ngroups,
            # state strides
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            # old_x strides (single-buffered: cache, T, nheads, dim)
            old_x.stride(0), old_x.stride(1), old_x.stride(2), old_x.stride(3),
            # old_B strides
            old_B.stride(0), old_B.stride(1), old_B.stride(2), old_B.stride(3), old_B.stride(4),
            # old_dt_proc strides
            old_dt_proc.stride(0), old_dt_proc.stride(1), old_dt_proc.stride(2), old_dt_proc.stride(3),
            # old_cumAdt strides
            old_cumAdt.stride(0), old_cumAdt.stride(1), old_cumAdt.stride(2), old_cumAdt.stride(3),
            # x strides
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            # C strides
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            # D strides
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            # z strides
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            # out strides
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            # cb_scaled strides
            cb_scaled.stride(0), cb_scaled.stride(1), cb_scaled.stride(2), cb_scaled.stride(3),
            # decay_vec strides
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            num_warps=num_warps,
            **({'num_stages': _num_stages} if _num_stages else {}),
            launch_pdl=use_internal_pdl,
        )
