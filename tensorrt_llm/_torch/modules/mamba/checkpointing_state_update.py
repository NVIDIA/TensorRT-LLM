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

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._utils import get_sm_version

from .softplus import softplus


@triton.jit
def _stochastic_round_fp16x2(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    """Stochastic rounding: fp32 pair → fp16x2 using Philox random bits.

    Uses PTX cvt.rs.f16x2.f32 which rounds each fp32 value to fp16 using
    the random bits to break ties, avoiding systematic rounding bias that
    accumulates over many decode steps with fp16 state.

    Adapted from flashinfer (Apache-2.0, vLLM/mamba lineage).
    """
    return tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _stochastic_round_fp8x4_e4m3(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    """Stochastic rounding: fp32 quad → fp8 e4m3 using Philox random bits.

    Uses PTX cvt.rs.satfinite.e4m3x4.f32 which combines stochastic rounding
    and saturating cast in a single op (output is final fp8, no separate
    clamp needed).  The reversed source-register order {$4,$3,$2,$1} is
    load-bearing — PTX packs leftmost source into the high byte but Triton's
    pack=4 is little-endian, so the natural {$1,$2,$3,$4} order would
    silently shuffle every group of 4 contiguous outputs.

    Requires SM_100a+ (Blackwell B200).  Caller must gate at the wrapper
    level — this kernel does not check.

    Adapted from vLLM PR #40012 (Apache-2.0).
    """
    return tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )


# Precompute kernel: CB_scaled, decay_vec.  Writes new cache (old_B,
# old_dt, old_dA_cumsum) to the WRITE buffer slot for next step's replay.
# Grid: (batch, nheads // HEADS_PER_BLOCK).


@triton.jit()
def _replay_precompute_impl(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers (both buffers reachable via stride_*_dbuf).  This
    # kernel writes to either the active (= cache_buf_idx) or inactive
    # (= 1 - cache_buf_idx) buffer depending on WRITE_CHECKPOINT — see
    # comment block at top of kernel body.
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    # Double-buffer index (per cache slot) — selects this step's "active"
    # buffer (= where the historical inputs for this step live).
    cache_buf_idx_ptr,
    # Per-request accepted-tokens count (already-cached old tokens at
    # [0, PNAT) of the active buffer; new tokens this step go after them
    # on no-checkpoint steps).
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    # Slot permutation: maps grid program_id -> original slot index.
    # When USE_PERM=False, pid_b = tl.program_id(0) (today's behavior) and
    # this ptr is unused.  When USE_PERM=True, pid_b = perm[pid_grid] (or
    # perm[B-1-pid_grid] if REVERSE_PERM=True).
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    # Slot permutation flags.  USE_PERM=True gates a slot_perm_ptr load that
    # remaps grid program_id -> original slot index.  REVERSE_PERM=True walks
    # the perm from the tail (B-1-pid_grid).  Used by sorted-dispatch
    # variants of dl/dlgrouped/maindl to cluster early-outs at one end of
    # the grid; ignored by monolithic / dynamic.
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
    # Checkpointing flag — selects target buffer + offset for new-token
    # cache writes.  See "Cache write semantics" block below.
    # Runtime (not constexpr): the only WRITE_CHECKPOINT-dependent code in
    # this body is the write_buf/write_offset selection, which is plain
    # arithmetic — no constexpr-shaped tile or whole-block gate.  Letting
    # it be runtime lets the dynamic dispatch kernel call us once with the
    # per-slot needs_write flag instead of inlining two specializations.
    write_checkpoint,
):
    pid_grid = tl.program_id(axis=0)
    # REVERSE_PERM walks the grid tail-first regardless of USE_PERM —
    # combined with hardcode-sorted prev_tokens (kernel-side USE_PERM=False
    # but PNAT pre-sorted write-first), reverse traversal makes the nowrite
    # half front-load real work.
    pid_grid_eff = (tl.num_programs(axis=0) - 1 - pid_grid) if REVERSE_PERM else pid_grid
    if USE_PERM:
        pid_b = tl.load(slot_perm_ptr + pid_grid_eff)
    else:
        pid_b = pid_grid_eff
    pid_hg = tl.program_id(axis=1)  # head-group index
    first_head = pid_hg * HEADS_PER_BLOCK

    # Resolve cache index for writes
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)


    # --- Cache write semantics ---
    # cache_buf_idx names this step's "active" buffer — the one with the
    # historical inputs at [0, PNAT).  The other buffer is "staging".
    #
    # Where do we write new tokens this step?
    #   WRITE_CHECKPOINT=False (no overflow): append to ACTIVE buffer at
    #       offset [PNAT : PNAT+T).  Caller does NOT flip cache_buf_idx
    #       afterward; PNAT_next = PNAT + accepted.  [0, PNAT) preserved.
    #   WRITE_CHECKPOINT=True (would overflow): write to STAGING buffer at
    #       [0, T).  Caller flips cache_buf_idx afterward; next step's
    #       active = the one we just wrote.  PNAT_next = accepted.  Old
    #       data in the previous active buffer is folded into state via
    #       the replay update and discarded.  This matches today's replay
    #       kernel behavior exactly.
    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if write_checkpoint:
        write_buf = 1 - buf_active
        write_offset = 0
    else:
        write_buf = buf_active
        write_offset = prev_num_accepted_tokens

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < T
    n_mask = offs_n < dstate

    # Causal mask is shared across all heads (depends only on offs_t)
    causal_mask = offs_t[:, None] >= offs_t[None, :]
    valid_mask = causal_mask & t_mask[:, None] & t_mask[None, :]

    # --- Loop 1: compute per-head dt/dA_cumsum/decay BEFORE gdc_wait ---
    # These only depend on dt (from in_proj, not conv1d) and parameters (A, dt_bias).
    # Store to cache; will reload after the wait for CB scaling.
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local

        dt_base = dt_ptr + pid_b * stride_dt_batch + head_idx * stride_dt_head
        dt = tl.load(dt_base + offs_t * stride_dt_T, mask=t_mask, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + head_idx * stride_dt_bias_head).to(tl.float32)
            dt = dt + dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)

        A = tl.load(A_ptr + head_idx * stride_A_head).to(tl.float32)
        dA_cumsum = tl.cumsum(A * dt, axis=0)
        decay_vec = tl.exp(dA_cumsum)

        # Store dt, dA_cumsum to cache at [write_offset : write_offset+T) of
        # write_buf (selected by WRITE_CHECKPOINT — see top of kernel).
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        tl.store(
            old_dt_base + (write_offset + offs_t) * stride_old_dt_T,
            dt,
            mask=t_mask,
        )

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        tl.store(
            old_dA_cumsum_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            dA_cumsum,
            mask=t_mask,
        )

        # decay_vec is per-call scratch (not cached); always write at offs_t.
        decay_vec_base = decay_vec_ptr + pid_b * stride_dv_batch + head_idx * stride_dv_head
        tl.store(decay_vec_base + offs_t * stride_dv_t, decay_vec, mask=t_mask)

    # --- Wait for upstream kernel (external PDL) before loading B and C ---
    # All dt processing above is independent of conv1d outputs.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # --- Load C and B once for the group (shared across HEADS_PER_BLOCK heads) ---
    group_idx = first_head // nheads_ngroups_ratio
    C_base = C_ptr + pid_b * stride_C_batch + group_idx * stride_C_group
    B_base = B_ptr + pid_b * stride_B_batch + group_idx * stride_B_group

    C_all = tl.load(
        C_base + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_all = tl.load(
        B_base + offs_t[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    # Compute raw CB once — shared across all heads in this block
    raw_CB = tl.dot(C_all.to(tl.bfloat16), tl.trans(B_all).to(tl.bfloat16))

    # Store B to cache at [write_offset : write_offset+T) of write_buf.
    if first_head % nheads_ngroups_ratio == 0:
        old_B_base = (
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + write_buf * stride_old_B_dbuf
            + group_idx * stride_old_B_group
        )
        tl.store(
            old_B_base
            + (write_offset + offs_t)[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            B_all,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    # --- Loop 2: reload per-head dA_cumsum/dt from cache, scale CB ---
    # Reload from where loop 1 stored: write_buf at [write_offset, write_offset+T).
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local

        # Reload dt and dA_cumsum from cache (just written in loop 1)
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        dt = tl.load(
            old_dt_base + (write_offset + offs_t) * stride_old_dt_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        dA_cumsum = tl.load(
            old_dA_cumsum_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        # Scale raw_CB with per-head decay and dt
        decay_matrix = tl.exp(dA_cumsum[:, None] - dA_cumsum[None, :])
        CB_scaled = tl.where(valid_mask, raw_CB * decay_matrix * dt[None, :], 0.0)

        cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + head_idx * stride_cb_head
        tl.store(
            cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
            CB_scaled,
            mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        )


# Replay-style precompute kernel.  Thin wrapper around _replay_precompute_impl
# that carries the @triton.heuristics for constexpr derivation; called from
# the Python wrapper on the replay-style path (write or replay-nowrite).
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.jit()
def _checkpointing_precompute_kernel(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    WRITE_CHECKPOINT: tl.constexpr,
    EARLY_OUT: tl.constexpr,
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    # Hoisted PDL signal: fire as the first thing every program does, so
    # main can start its setup regardless of how this program ends (pad,
    # early-out, or full body).  PDL signals are idempotent; main's
    # gdc_wait still gates on prerequisite-kernel completion for
    # correctness.
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    # Per-program early-out gate (option-2 double-launch).  When EARLY_OUT
    # is False the entire block is constexpr-folded out and the wrapper is
    # just an impl call.  When True, this kernel only runs for slots whose
    # (PNAT + T > MAX) status matches WRITE_CHECKPOINT.
    if EARLY_OUT:
        pid_grid_eo = tl.program_id(axis=0)
        pid_grid_eo_eff = (tl.num_programs(axis=0) - 1 - pid_grid_eo) if REVERSE_PERM else pid_grid_eo
        if USE_PERM:
            pid_b_eo = tl.load(slot_perm_ptr + pid_grid_eo_eff)
        else:
            pid_b_eo = pid_grid_eo_eff
        if HAS_CACHE_BATCH_INDICES:
            cbi_eo = tl.load(state_batch_indices_ptr + pid_b_eo).to(tl.int64)
            if cbi_eo == pad_slot_id:
                return
        else:
            cbi_eo = pid_b_eo.to(tl.int64)
        pnat_eo = tl.load(prev_num_accepted_tokens_ptr + cbi_eo)
        if (pnat_eo + T > MAX_REPLAY_BUFFER_LENGTH) != WRITE_CHECKPOINT:
            return
    _replay_precompute_impl(
        dt_ptr,
        dt_bias_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        cb_scaled_ptr,
        decay_vec_ptr,
        old_B_ptr,
        old_dt_ptr,
        old_dA_cumsum_ptr,
        cache_buf_idx_ptr,
        prev_num_accepted_tokens_ptr,
        state_batch_indices_ptr,
        slot_perm_ptr,
        pad_slot_id,
        T,
        dstate,
        nheads_ngroups_ratio,
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
        stride_cb_batch,
        stride_cb_head,
        stride_cb_t,
        stride_cb_j,
        stride_dv_batch,
        stride_dv_head,
        stride_dv_t,
        stride_old_B_cache,
        stride_old_B_dbuf,
        stride_old_B_T,
        stride_old_B_group,
        stride_old_B_dstate,
        stride_old_dt_cache,
        stride_old_dt_dbuf,
        stride_old_dt_head,
        stride_old_dt_T,
        stride_old_dA_cumsum_cache,
        stride_old_dA_cumsum_dbuf,
        stride_old_dA_cumsum_head,
        stride_old_dA_cumsum_T,
        DT_SOFTPLUS,
        HAS_DT_BIAS,
        HAS_CACHE_BATCH_INDICES,
        BLOCK_SIZE_DSTATE,
        BLOCK_SIZE_T,
        LAUNCH_WITH_PDL,
        HEADS_PER_BLOCK,
        USE_PERM,
        REVERSE_PERM,
        WRITE_CHECKPOINT,
    )


# Rectangle precompute kernel: produces a (T, K) CB rectangle that combines
# old-token (B from cache, k ∈ [0, PNAT)) and new-token (B from input, k ∈
# [MAX-T, MAX) at compile-time-static shift) contributions in a single matmul.
# Used only on no-checkpoint steps (nowrite path); pairs with
# `_rectangle_main_kernel`.  K-axis size = max(np2(MAX_REPLAY_BUFFER_LENGTH),
# 16); the static layout is sound because nowrite implies PNAT + T <=
# MAX_REPLAY_BUFFER_LENGTH, so old [0, PNAT) and new [MAX-T, MAX) never
# overlap.  Also folds total_decay into decay_vec at precomp time so main
# can skip materializing a state_prev_decayed (M, dstate) tile.


@triton.jit()
def _rectangle_precompute_impl(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,            # (batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_K) — rectangle
    decay_vec_ptr,            # (batch, nheads, BLOCK_SIZE_T) — total_decay * exp(cumAdt_new[t])
    # Cache pointers (both buffers reachable via stride_*_dbuf).  Nowrite
    # path: read from buf_active at [0, PNAT), write new tokens at
    # [PNAT, PNAT+T) of buf_active (same buffer).
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    # Slot permutation: see _replay_precompute_impl for semantics.
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,  # rectangle K-axis bound
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides (rectangle: (batch, nheads, T, K))
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides: (cache, 2, T_max, ngroups, dstate)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides: (cache, 2, nheads, T_max)
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides: (cache, 2, nheads, T_max)
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    # Slot permutation flags — see _replay_precompute_impl.
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    pid_grid = tl.program_id(axis=0)
    # REVERSE_PERM walks the grid tail-first regardless of USE_PERM —
    # combined with hardcode-sorted prev_tokens (kernel-side USE_PERM=False
    # but PNAT pre-sorted write-first), reverse traversal makes the nowrite
    # half front-load real work.
    pid_grid_eff = (tl.num_programs(axis=0) - 1 - pid_grid) if REVERSE_PERM else pid_grid
    if USE_PERM:
        pid_b = tl.load(slot_perm_ptr + pid_grid_eff)
    else:
        pid_b = pid_grid_eff
    pid_hg = tl.program_id(axis=1)
    first_head = pid_hg * HEADS_PER_BLOCK

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Nowrite-only: write_buf = active, write_offset = PNAT.  No flip after.
    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    write_buf = buf_active
    write_offset = prev_num_accepted_tokens

    # Static rectangle K-axis layout: old at [0, PNAT) (mask is_old_k);
    # new at [MAX-T, MAX) at compile-time shift K_NEW_SHIFT = MAX - T.
    # PNAT + T <= MAX is guaranteed on the nowrite path → no overlap.
    K_NEW_SHIFT: tl.constexpr = MAX_REPLAY_BUFFER_LENGTH - T

    offs_t = tl.arange(0, BLOCK_SIZE_T)  # T-axis (output rows)
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K-axis (rectangle input cols)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < T
    n_mask = offs_n < dstate

    # K-axis masks
    is_old_k = offs_k < prev_num_accepted_tokens
    safe_old_k = tl.where(is_old_k, offs_k, 0)
    k_new_idx = offs_k - K_NEW_SHIFT
    is_new_k = (k_new_idx >= 0) & (k_new_idx < T)
    safe_k_new = tl.where(is_new_k, k_new_idx, 0)

    # Loop 1: per-head dt processing.  dt → dt_processed → dA_cumsum →
    # decay_vec_new (= exp(cumAdt_new)).  Stored to write_buf for next step.
    # decay_vec_full (= total_decay * decay_vec_new) is finalized in loop 2
    # once total_decay is loaded; loop 1 stores raw decay_vec_new to scratch.
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local

        dt_base = dt_ptr + pid_b * stride_dt_batch + head_idx * stride_dt_head
        dt = tl.load(dt_base + offs_t * stride_dt_T, mask=t_mask, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + head_idx * stride_dt_bias_head).to(tl.float32)
            dt = dt + dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)

        A = tl.load(A_ptr + head_idx * stride_A_head).to(tl.float32)
        dA_cumsum = tl.cumsum(A * dt, axis=0)

        # Store dt and dA_cumsum to write_buf at [write_offset, write_offset+T)
        # for next step's replay/rectangle use.
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        tl.store(
            old_dt_base + (write_offset + offs_t) * stride_old_dt_T,
            dt,
            mask=t_mask,
        )

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        tl.store(
            old_dA_cumsum_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            dA_cumsum,
            mask=t_mask,
        )

    # ---- Hoisted: cache-only loads independent of conv1d ----
    # old_B (group-level, BLOCK_K × BLOCK_DSTATE = ~8KB tile) and the
    # decay_vec_full per-head pre-compute (which writes to DRAM and doesn't
    # need cross-gdc_wait variables) are issued BEFORE gdc_wait so their
    # HBM latency overlaps with conv1d.  Per-head factor_dt/exp_diff stay
    # below gdc_wait — they need cross-iteration spans, which Triton can't
    # express without a DRAM round-trip; the per-head LOADS in the post-
    # wait loop are small and cheap, so leave them.
    group_idx = first_head // nheads_ngroups_ratio

    # Group-level: old B from active buffer at [0, PNAT) of the K-axis.
    old_B_read_base = (
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + buf_active * stride_old_B_dbuf
        + group_idx * stride_old_B_group
    )
    old_B_load = tl.load(
        old_B_read_base
        + safe_old_k[:, None] * stride_old_B_T
        + offs_n[None, :] * stride_old_B_dstate,
        mask=is_old_k[:, None] & n_mask[None, :],
        other=0.0,
    )

    # Per-head: pre-compute decay_vec_full (= total_decay * exp(cumAdt_new[t]))
    # and store to scratch.  Also pre-load the rest of the per-head cache
    # data (old_dt, old_dA_cumsum, dt_at_kn, dA_cumsum_at_kn) as L1 warmers
    # — Triton can't span their values across the gdc_wait barrier without a
    # DRAM round-trip, but L1 is per-SM-persistent so the post-wait loop's
    # reload hits the cache cheaply.  Folding the warmer loads into
    # decay_vec_full's `tl.where(False, ...)` keeps Triton from DCE-ing them.
    prev_k_idx = tl.minimum(
        tl.maximum(prev_num_accepted_tokens - 1, 0), MAX_REPLAY_BUFFER_LENGTH - 1
    )
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local
        old_dt_read_base_h = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + buf_active * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        old_dA_cumsum_read_base_h = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        old_dt_write_base_h = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        old_dA_cumsum_write_base_h = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        # Required for decay_vec_full
        total_dA_cumsum_h = tl.load(
            old_dA_cumsum_read_base_h + prev_k_idx * stride_old_dA_cumsum_T
        ).to(tl.float32)
        dA_cumsum_new_h = tl.load(
            old_dA_cumsum_write_base_h + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            mask=t_mask, other=0.0,
        ).to(tl.float32)
        # L1 warmers — same addresses the post-wait loop will reload.
        warm_old_dt = tl.load(
            old_dt_read_base_h + safe_old_k * stride_old_dt_T,
            mask=is_old_k, other=0.0,
        ).to(tl.float32)
        warm_old_dA = tl.load(
            old_dA_cumsum_read_base_h + safe_old_k * stride_old_dA_cumsum_T,
            mask=is_old_k, other=0.0,
        ).to(tl.float32)
        warm_dt_at_kn = tl.load(
            old_dt_write_base_h + (write_offset + safe_k_new) * stride_old_dt_T,
            mask=is_new_k, other=0.0,
        ).to(tl.float32)
        warm_dA_at_kn = tl.load(
            old_dA_cumsum_write_base_h + (write_offset + safe_k_new) * stride_old_dA_cumsum_T,
            mask=is_new_k, other=0.0,
        ).to(tl.float32)

        total_decay_h = tl.where(
            prev_num_accepted_tokens > 0, tl.exp(total_dA_cumsum_h), 1.0
        )
        decay_vec_full_h = total_decay_h * tl.exp(dA_cumsum_new_h)
        # Force the warmer loads to participate via a runtime-False guard.
        # prev_num_accepted_tokens is dynamic, so Triton can't DCE this.
        warm_keep = (
            tl.sum(warm_old_dt) + tl.sum(warm_old_dA)
            + tl.sum(warm_dt_at_kn) + tl.sum(warm_dA_at_kn)
        )
        decay_vec_full_h = tl.where(
            prev_num_accepted_tokens < 0,
            decay_vec_full_h + warm_keep,
            decay_vec_full_h,
        )
        decay_vec_base_h = decay_vec_ptr + pid_b * stride_dv_batch + head_idx * stride_dv_head
        tl.store(
            decay_vec_base_h + offs_t * stride_dv_t, decay_vec_full_h, mask=t_mask
        )

    # ---- gdc_wait: from here on we depend on conv1d's outputs ----
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Conv1d outputs: B and C
    C_base = C_ptr + pid_b * stride_C_batch + group_idx * stride_C_group
    B_new_base = B_ptr + pid_b * stride_B_batch + group_idx * stride_B_group

    C_all = tl.load(
        C_base + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_new_orig = tl.load(
        B_new_base + offs_t[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_new_shifted = tl.load(
        B_new_base + safe_k_new[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
        mask=is_new_k[:, None] & n_mask[None, :],
        other=0.0,
    )
    # Disjoint masks: old at [0, PNAT), new at [K_NEW_SHIFT, K_NEW_SHIFT+T).
    B_combined = old_B_load + B_new_shifted
    raw_rect_CB = tl.dot(C_all.to(tl.bfloat16), tl.trans(B_combined).to(tl.bfloat16))

    # Append new B to cache at [PNAT, PNAT+T) of write_buf (once per group).
    if first_head % nheads_ngroups_ratio == 0:
        old_B_write_base = (
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + write_buf * stride_old_B_dbuf
            + group_idx * stride_old_B_group
        )
        tl.store(
            old_B_write_base
            + (write_offset + offs_t)[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            B_new_orig,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    # Causal mask (BLOCK_SIZE_T × BLOCK_SIZE_K, shared across heads).
    t_idx_2d = offs_t[:, None]
    k_idx_2d = offs_k[None, :]
    is_old_k_2d = k_idx_2d < prev_num_accepted_tokens
    k_new_idx_2d = k_idx_2d - K_NEW_SHIFT
    is_new_causal_2d = (k_new_idx_2d >= 0) & (k_new_idx_2d < T) & (k_new_idx_2d <= t_idx_2d)
    causal_combined = (is_old_k_2d | is_new_causal_2d) & t_mask[:, None]

    # Per-head: factor_dt + exp_diff + rect_CB_scaled.  Loads are post-wait
    # but cache-resident from this kernel's earlier writes (loop 1 + the
    # hoisted decay_vec_full block above), so they hit L1.  Recomputing
    # total_dA_cumsum / dA_cumsum_new here is cheaper than spanning the
    # gdc_wait via a DRAM round-trip.
    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local
        old_dt_read_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + buf_active * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        old_dA_cumsum_read_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        old_dt_write_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + write_buf * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        old_dA_cumsum_write_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + write_buf * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        old_dt_all = tl.load(
            old_dt_read_base + safe_old_k * stride_old_dt_T, mask=is_old_k, other=0.0
        ).to(tl.float32)
        old_dA_cumsum_all = tl.load(
            old_dA_cumsum_read_base + safe_old_k * stride_old_dA_cumsum_T,
            mask=is_old_k, other=0.0,
        ).to(tl.float32)
        total_dA_cumsum = tl.load(
            old_dA_cumsum_read_base + prev_k_idx * stride_old_dA_cumsum_T
        ).to(tl.float32)
        dA_cumsum_new = tl.load(
            old_dA_cumsum_write_base + (write_offset + offs_t) * stride_old_dA_cumsum_T,
            mask=t_mask, other=0.0,
        ).to(tl.float32)
        dt_at_kn = tl.load(
            old_dt_write_base + (write_offset + safe_k_new) * stride_old_dt_T,
            mask=is_new_k, other=0.0,
        ).to(tl.float32)
        dA_cumsum_at_kn = tl.load(
            old_dA_cumsum_write_base + (write_offset + safe_k_new) * stride_old_dA_cumsum_T,
            mask=is_new_k, other=0.0,
        ).to(tl.float32)

        factor_dt = tl.where(is_old_k, old_dt_all, dt_at_kn)
        s_k = tl.where(
            is_old_k, total_dA_cumsum - old_dA_cumsum_all, -dA_cumsum_at_kn
        )
        exp_diff = tl.exp(s_k[None, :] + dA_cumsum_new[:, None])

        rect_CB_scaled = tl.where(
            causal_combined,
            raw_rect_CB * factor_dt[None, :] * exp_diff,
            0.0,
        )
        cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + head_idx * stride_cb_head
        tl.store(
            cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_k[None, :] * stride_cb_j,
            rect_CB_scaled,
            mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_k[None, :] < BLOCK_SIZE_K),
        )


# Rectangle precompute kernel.  Thin wrapper around _rectangle_precompute_impl
# that carries the @triton.heuristics for constexpr derivation; called from
# the Python wrapper on the rectangle nowrite path.
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.jit()
def _rectangle_precompute_kernel(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    EARLY_OUT: tl.constexpr,
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    # Hoisted PDL signal: fire as the first thing every program does, so
    # main can start its setup regardless of how this program ends.
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    # Per-program early-out gate.  Rectangle is nowrite-only, so EARLY_OUT
    # skips slots whose PNAT + T > MAX (slots that would need write).
    if EARLY_OUT:
        pid_grid_eo = tl.program_id(axis=0)
        pid_grid_eo_eff = (tl.num_programs(axis=0) - 1 - pid_grid_eo) if REVERSE_PERM else pid_grid_eo
        if USE_PERM:
            pid_b_eo = tl.load(slot_perm_ptr + pid_grid_eo_eff)
        else:
            pid_b_eo = pid_grid_eo_eff
        if HAS_CACHE_BATCH_INDICES:
            cbi_eo = tl.load(state_batch_indices_ptr + pid_b_eo).to(tl.int64)
            if cbi_eo == pad_slot_id:
                return
        else:
            cbi_eo = pid_b_eo.to(tl.int64)
        pnat_eo = tl.load(prev_num_accepted_tokens_ptr + cbi_eo)
        if pnat_eo + T > MAX_REPLAY_BUFFER_LENGTH:
            return
    _rectangle_precompute_impl(
        dt_ptr,
        dt_bias_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        cb_scaled_ptr,
        decay_vec_ptr,
        old_B_ptr,
        old_dt_ptr,
        old_dA_cumsum_ptr,
        cache_buf_idx_ptr,
        prev_num_accepted_tokens_ptr,
        state_batch_indices_ptr,
        slot_perm_ptr,
        pad_slot_id,
        T,
        MAX_REPLAY_BUFFER_LENGTH,
        dstate,
        nheads_ngroups_ratio,
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
        stride_cb_batch,
        stride_cb_head,
        stride_cb_t,
        stride_cb_j,
        stride_dv_batch,
        stride_dv_head,
        stride_dv_t,
        stride_old_B_cache,
        stride_old_B_dbuf,
        stride_old_B_T,
        stride_old_B_group,
        stride_old_B_dstate,
        stride_old_dt_cache,
        stride_old_dt_dbuf,
        stride_old_dt_head,
        stride_old_dt_T,
        stride_old_dA_cumsum_cache,
        stride_old_dA_cumsum_dbuf,
        stride_old_dA_cumsum_head,
        stride_old_dA_cumsum_T,
        DT_SOFTPLUS,
        HAS_DT_BIAS,
        HAS_CACHE_BATCH_INDICES,
        BLOCK_SIZE_DSTATE,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
        LAUNCH_WITH_PDL,
        HEADS_PER_BLOCK,
        USE_PERM,
        REVERSE_PERM,
    )


# Dynamic precompute kernel.  Single launchable kernel that, per program,
# reads PNAT and dispatches to one of the existing impls:
#
#   if pnat + T > MAX:  replay_precompute_impl(WRITE_CHECKPOINT=True)
#   else if RECTANGLE:  rectangle_precompute_impl
#   else:               replay_precompute_impl(WRITE_CHECKPOINT=False)
#
# RECTANGLE is constexpr (compile-time tuning param); the inner branch
# is folded so only one of the two nowrite paths is emitted per
# specialization.  Reg envelope = max(replay_write, X) where X depends
# on RECTANGLE.  cb_scaled is allocated (T, K) by the wrapper regardless;
# replay paths write to the first T columns, rectangle writes the full K.
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.jit()
def _dynamic_precompute_kernel(
    # Input pointers
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    # Output pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    # Cache pointers
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # dt strides
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    # B strides
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # cb_scaled strides — wrapper allocates (T, K), so stride_cb_t = K
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    # Compile-time pick: rectangle (with replay-write fallback) vs replay-only.
    RECTANGLE: tl.constexpr,
):
    # Hoisted PDL signal: fire as the first thing every program does.
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    pid_b = tl.program_id(axis=0)
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    pnat_local = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    needs_write_runtime = pnat_local + T > MAX_REPLAY_BUFFER_LENGTH
    # write_checkpoint is now runtime in replay precompute, so a single
    # call site handles both write and nowrite for the replay branch.
    # Take rectangle only when RECTANGLE is True AND this slot doesn't
    # need write; everything else funnels into replay.
    if needs_write_runtime or not RECTANGLE:
        _replay_precompute_impl(
            dt_ptr,
            dt_bias_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            cb_scaled_ptr,
            decay_vec_ptr,
            old_B_ptr,
            old_dt_ptr,
            old_dA_cumsum_ptr,
            cache_buf_idx_ptr,
            prev_num_accepted_tokens_ptr,
            state_batch_indices_ptr,
            state_batch_indices_ptr,  # slot_perm_ptr unused (USE_PERM=False)
            pad_slot_id,
            T,
            dstate,
            nheads_ngroups_ratio,
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
            stride_cb_batch,
            stride_cb_head,
            stride_cb_t,
            stride_cb_j,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_t,
            stride_old_B_cache,
            stride_old_B_dbuf,
            stride_old_B_T,
            stride_old_B_group,
            stride_old_B_dstate,
            stride_old_dt_cache,
            stride_old_dt_dbuf,
            stride_old_dt_head,
            stride_old_dt_T,
            stride_old_dA_cumsum_cache,
            stride_old_dA_cumsum_dbuf,
            stride_old_dA_cumsum_head,
            stride_old_dA_cumsum_T,
            DT_SOFTPLUS,
            HAS_DT_BIAS,
            HAS_CACHE_BATCH_INDICES,
            BLOCK_SIZE_DSTATE,
            BLOCK_SIZE_T,
            LAUNCH_WITH_PDL,
            HEADS_PER_BLOCK,
            False,  # USE_PERM
            False,  # REVERSE_PERM
            needs_write_runtime,
        )
    else:
        _rectangle_precompute_impl(
            dt_ptr,
            dt_bias_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            cb_scaled_ptr,
            decay_vec_ptr,
            old_B_ptr,
            old_dt_ptr,
            old_dA_cumsum_ptr,
            cache_buf_idx_ptr,
            prev_num_accepted_tokens_ptr,
            state_batch_indices_ptr,
            state_batch_indices_ptr,  # slot_perm_ptr unused (USE_PERM=False)
            pad_slot_id,
            T,
            MAX_REPLAY_BUFFER_LENGTH,
            dstate,
            nheads_ngroups_ratio,
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
            stride_cb_batch,
            stride_cb_head,
            stride_cb_t,
            stride_cb_j,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_t,
            stride_old_B_cache,
            stride_old_B_dbuf,
            stride_old_B_T,
            stride_old_B_group,
            stride_old_B_dstate,
            stride_old_dt_cache,
            stride_old_dt_dbuf,
            stride_old_dt_head,
            stride_old_dt_T,
            stride_old_dA_cumsum_cache,
            stride_old_dA_cumsum_dbuf,
            stride_old_dA_cumsum_head,
            stride_old_dA_cumsum_T,
            DT_SOFTPLUS,
            HAS_DT_BIAS,
            HAS_CACHE_BATCH_INDICES,
            BLOCK_SIZE_DSTATE,
            BLOCK_SIZE_T,
            BLOCK_SIZE_K,
            LAUNCH_WITH_PDL,
            HEADS_PER_BLOCK,
            False,  # USE_PERM
            False,  # REVERSE_PERM
        )


# Main kernel: tl.dot replay + precomputed CB output.
# Grid: (cdiv(dim, M), batch, nheads).


@triton.jit()
def _replay_main_impl(
    # Pointers
    state_ptr,
    # Per-(cache, head, dim) decode scale, fp32, only consulted when QUANT_MAX>0.
    # Layout (cache, nheads, dim) — broadcast over dstate at load/store.
    state_scales_ptr,
    # Cache READ pointers (read-buffer from previous step)
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    # Cache WRITE pointer (write-buffer for old_x only; B/dt/dA_cumsum written by precompute)
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    # New input pointers
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    # Precomputed pointers
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    # Slot permutation: see _replay_precompute_impl for semantics.
    slot_perm_ptr,
    # Stochastic rounding
    rand_seed_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,  # cache T-axis capacity (= max_window)
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides: (cache, nheads, dim) — only used when QUANT_MAX>0
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides: (cache, T, nheads, dim) — single-buffered
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides: (cache, 2, T, ngroups, dstate)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides: (cache, 2, nheads, T) — T contiguous for coalesced access
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_WINDOW: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    # State quantization: 0.0 means non-quantized (fp16/bf16/fp32); >0 means
    # quantized (int8=127, int16=32767, fp8_e4m3fn=448).  Single in-kernel
    # switch for the dequant-on-load and encode-on-store paths.  Wrapper sets
    # this from state.dtype; kernel-entry static_assert below pins the
    # invariant that it must coincide with int8/int16/float8e4nv state dtype.
    QUANT_MAX: tl.constexpr,
    # Checkpointing flag
    WRITE_CHECKPOINT: tl.constexpr,  # When True: quantize+write post-replay state to HBM (checkpoint step).
                                      # When False: skip state write entirely (non-checkpoint step).
                                      # The rectangle non-checkpoint path is implemented in
                                      # _rectangle_main_kernel (separate kernel pair, picked by
                                      # the wrapper via rectangle_for_nowrite=True).
    # When True: signal PDL dependents at the very top of every program
    # (including pad/early-out programs).  Used by doublelaunch and maindl
    # so the next kernel (the second main, or the second precompute) can
    # start its setup while this main is still computing.  Default False
    # for monolithic / dynamic / the LAST main in dl/maindl chains.
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    # Slot permutation flags — see _replay_precompute_impl.
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    # Hoisted PDL signal: fire as the first thing every program does, so
    # downstream kernels can start setup regardless of how this program
    # ends (pad / early-out / full body).
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    # Compile-time invariant: QUANT_MAX > 0 must coincide with a quantized
    # state dtype (int8 / int16 / float8e4nv) and only those.  Cheap
    # insurance against a wrapper bug that desynchronizes the two.
    tl.static_assert(
        (QUANT_MAX > 0.0)
        == (
            (state_ptr.dtype.element_ty == tl.int8)
            or (state_ptr.dtype.element_ty == tl.int16)
            or (state_ptr.dtype.element_ty == tl.float8e4nv)
        ),
        "QUANT_MAX > 0.0 must coincide with int8 / int16 / float8e4nv state dtype.",
    )

    pid_m = tl.program_id(axis=0)
    pid_grid_b = tl.program_id(axis=1)
    pid_grid_b_eff = (tl.num_programs(axis=1) - 1 - pid_grid_b) if REVERSE_PERM else pid_grid_b
    if USE_PERM:
        pid_b = tl.load(slot_perm_ptr + pid_grid_b_eff)
    else:
        pid_b = pid_grid_b_eff
    pid_h = tl.program_id(axis=2)

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Active buffer (= cache_buf_idx) holds the historical inputs for this
    # step at [0, PNAT).  The replay phase reads from there.  The new-tokens
    # write target depends on WRITE_CHECKPOINT — see Cache write semantics
    # block in the precompute kernel for the full rationale.
    active_buf = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if WRITE_CHECKPOINT:
        write_buf = 1 - active_buf  # noqa: F841 — old_x is single-buffered (no use here)
        write_offset = 0
    else:
        write_buf = active_buf  # noqa: F841 — old_x is single-buffered (no use here)
        write_offset = prev_num_accepted_tokens

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    # Replay axis: separate from offs_t.  Spans [0, BLOCK_SIZE_WINDOW) ⊇
    # [0, MAX_REPLAY_BUFFER_LENGTH); used for old-token loads (mask: offs_window < PNAT).
    offs_window = tl.arange(0, BLOCK_SIZE_WINDOW)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    # Load state
    state_ptr += cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    state_ptrs = (
        state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    # Dequantize on load (per-(head, dim) decode scale, broadcast over dstate).
    # Only consulted when QUANT_MAX>0 — non-quantized paths skip entirely.
    if QUANT_MAX > 0.0:
        state_scales_base = (
            state_scales_ptr
            + cache_batch_idx * stride_state_scales_cache
            + pid_h * stride_state_scales_head
        )
        decode_scale = tl.load(
            state_scales_base + offs_m * stride_state_scales_dim,
            mask=m_mask,
            other=1.0,
        ).to(tl.float32)
        state = state * decode_scale[:, None]

    # Phase 1: Replay via tl.dot fast-forward (reads from active_buf)
    group_idx = pid_h // nheads_ngroups_ratio

    # Old-token mask along the WINDOW (replay) axis.  PNAT ≤ MAX ≤
    # BLOCK_SIZE_WINDOW, so this enables all valid old-cache positions.
    # (Distinct from t_mask = offs_t < T which gates output T-rows only.)
    old_window_mask = offs_window < prev_num_accepted_tokens

    # Load precomputed dt and dA_cumsum from READ buffer at [0, PNAT).
    old_dt_base = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + active_buf * stride_old_dt_dbuf
        + pid_h * stride_old_dt_head
    )
    old_dt_all = tl.load(
        old_dt_base + offs_window * stride_old_dt_T, mask=old_window_mask, other=0.0
    ).to(tl.float32)

    old_dA_cumsum_base = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + active_buf * stride_old_dA_cumsum_dbuf
        + pid_h * stride_old_dA_cumsum_head
    )
    old_dA_cumsum_all = tl.load(
        old_dA_cumsum_base + offs_window * stride_old_dA_cumsum_T,
        mask=old_window_mask, other=0.0,
    ).to(tl.float32)

    # Load dA_cumsum at prev_k-1 directly via pointer math (avoids masked reduction).
    # Clamp to [0, MAX-1] defensively — caller contract gives PNAT ≤ MAX.
    prev_k_idx = tl.minimum(
        tl.maximum(prev_num_accepted_tokens - 1, 0), MAX_REPLAY_BUFFER_LENGTH - 1
    )
    total_dA_cumsum = tl.load(old_dA_cumsum_base + prev_k_idx * stride_old_dA_cumsum_T).to(
        tl.float32
    )

    # Step 0 invariant: PNAT=0 means `state` is already last step's state (not
    # two back).  coeff is all-zero (old_window_mask all-false), total_decay
    # is 1.0, so the replay leaves `state` unchanged — cache contents don't matter.
    coeff = tl.exp(total_dA_cumsum - old_dA_cumsum_all) * old_dt_all

    # Load old_x at [0, PNAT) of the WINDOW axis (single-buffered cache).
    old_x_base = old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head
    old_x_all = tl.load(
        old_x_base + offs_window[:, None] * stride_old_x_T + offs_m[None, :] * stride_old_x_dim,
        mask=old_window_mask[:, None] & m_mask[None, :],
        other=0.0,
    )

    # Load old_B from READ buffer at [0, PNAT) of the WINDOW axis.
    old_B_base = (
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + active_buf * stride_old_B_dbuf
        + group_idx * stride_old_B_group
    )
    old_B_all = tl.load(
        old_B_base + offs_window[:, None] * stride_old_B_T + offs_n[None, :] * stride_old_B_dstate,
        mask=old_window_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Scale B by coefficients
    dB_scaled = coeff[:, None] * old_B_all

    # Apply total decay to initial state FIRST, then add contributions
    total_decay = tl.where(prev_num_accepted_tokens > 0, tl.exp(total_dA_cumsum), 1.0)
    state *= total_decay

    # tl.dot fast-forward: old_x^T @ dB_scaled → (M, dstate)
    state += tl.dot(tl.trans(old_x_all).to(tl.bfloat16), dB_scaled.to(tl.bfloat16))

    # Write post-replay state — only on checkpoint steps.  When
    # WRITE_CHECKPOINT is False, the replay computed `state` is local-only and
    # discarded; skipping the HBM store + Philox path is the main performance
    # win of replay-style checkpointing on the common (non-checkpoint) step.
    if WRITE_CHECKPOINT:
        if USE_RS_ROUNDING:
            # Generate (M, dstate) random tensor for stochastic rounding.
            # Used by fp16 / int8 / int16 / fp8 SR paths below.  Quarter-sized
            # randint4x calls produce 4 u32s each; we interleave them into the
            # full (M, dstate) tensor — 4x fewer PRNG rounds vs per-element.
            rand_seed = tl.load(rand_seed_ptr)
            base_rand = cache_batch_idx * stride_state_batch + pid_h * stride_state_head
            offs_n_q = tl.arange(0, BLOCK_SIZE_DSTATE // 4)
            rand_offsets_q = (
                base_rand
                + offs_m[:, None] * stride_state_dim
                + offs_n_q[None, :] * (stride_state_dstate * 4)
            )  # (M, dstate//4)
            if PHILOX_ROUNDS > 0:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q, PHILOX_ROUNDS)
            else:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q)
            # Interleave 4 quarter-sized tensors → full (M, dstate) random tensor
            r01 = tl.join(r0, r1)  # (M, dstate//4, 2)
            r23 = tl.join(r2, r3)  # (M, dstate//4, 2)
            r0123 = tl.join(r01, r23)  # (M, dstate//4, 2, 2)
            rand = tl.reshape(r0123, (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE))

        if QUANT_MAX > 0.0:
            # Quantized state path: int8 / int16 / fp8_e4m3fn (RN or SR).
            # 1) Per-(head, dim) channel scale via amax over dstate.
            amax = tl.max(tl.abs(state), axis=1)  # (M,)
            encode_scale = tl.where(amax == 0.0, 1.0, QUANT_MAX / amax)  # (M,)
            decode_scale = 1.0 / encode_scale  # (M,)
            # 2) Store decode_scale (1/encode) so reads do a single multiply.
            state_scales_ptrs = (
                state_scales_ptr
                + cache_batch_idx * stride_state_scales_cache
                + pid_h * stride_state_scales_head
                + offs_m * stride_state_scales_dim
            )
            tl.store(state_scales_ptrs, decode_scale, mask=m_mask)
            # 3) Scale state into quant range — into a NEW variable so the
            # downstream output phase still sees the dequantized fp32 state.
            state_q = state * encode_scale[:, None]
            # 4) Round per dtype.  Order matters: handle fp8 SR first (PTX
            # combines round + saturating cast in one op, output is final fp8
            # so we store and finish on that branch).  Other branches share
            # the clamp + cast tail below.
            if USE_RS_ROUNDING and (state_ptrs.dtype.element_ty == tl.float8e4nv):
                # fp8_e4m3fn + SR — PTX cvt.rs.satfinite.e4m3x4.f32.  Output
                # is final fp8 (saturate included); store directly.
                tl.store(
                    state_ptrs,
                    _stochastic_round_fp8x4_e4m3(state_q, rand),
                    mask=state_mask,
                )
            else:
                if USE_RS_ROUNDING:
                    # int8 / int16 + SR — uniform-noise + floor.
                    # (fp8 SR was handled by the early branch above.)
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized SR fall-through expects int8 or int16; "
                        "fp8 SR is handled by the prior branch.",
                    )
                    rand01 = (rand & 0x00FFFFFF).to(tl.float32) * (1.0 / float(1 << 24))
                    state_q = tl.extra.cuda.libdevice.floor(state_q + rand01)
                elif state_ptrs.dtype.element_ty != tl.float8e4nv:
                    # int8 / int16 + RN — explicit round before clamp.
                    # fp8 + RN deliberately skips this — explicit round() would
                    # destroy fp8 sub-integer precision; native cast at store
                    # does RN at the fp8 grid resolution.
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized RN with explicit round() expects int8 or int16.",
                    )
                    state_q = tl.extra.cuda.libdevice.round(state_q)
                # Clamp + cast tail: int8/int16 (RN+SR) and fp8 RN.
                # fp8 RN reaches here without prior round() — .to(float8e4nv)
                # does native RN at the fp8 grid.
                state_q = tl.minimum(tl.maximum(state_q, -QUANT_MAX), QUANT_MAX)
                tl.store(
                    state_ptrs,
                    state_q.to(state_ptrs.dtype.element_ty),
                    mask=state_mask,
                )
        elif USE_RS_ROUNDING:
            # Non-quantized + SR: only fp16 (bf16 has no PTX SR cast; fp32
            # doesn't need rounding).
            tl.static_assert(
                state_ptrs.dtype.element_ty == tl.float16,
                "Non-quantized SR only supports fp16 state.",
            )
            tl.store(state_ptrs, _stochastic_round_fp16x2(state, rand), mask=state_mask)
        else:
            # Non-quantized + RN: fp16 / bf16 / fp32 native cast.
            tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=state_mask)

    # Phase 2: Output using precomputed CB_scaled and decay_vec
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    C_ptr += pid_b * stride_C_batch + group_idx * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=m_mask, other=0.0
        ).to(tl.float32)

    # Wait for precompute kernel (PDL) before reading its outputs.
    # With chained PDL (conv1d → precompute → main), gdc_wait() ensures
    # precompute has completed — which transitively ensures conv1d has
    # completed (precompute waited on conv1d via its own gdc_wait).
    # All loads below (x, C from conv1d; CB_scaled, decay_vec from precompute)
    # are safe after this point.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Load conv1d outputs: C_all and x_all
    C_all = tl.load(
        C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    x_all = tl.load(
        x_ptr + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )
    # Store new x to old_x cache at [write_offset : write_offset+T).
    # old_x is single-buffered: write goes to the active buffer regardless;
    # replay already read positions [0, PNAT) so write_offset = PNAT (no
    # overlap) on no-checkpoint steps.  On checkpoint steps write_offset = 0
    # (cache reset; old data folded into state via replay update).
    tl.store(
        old_x_base
        + (write_offset + offs_t)[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

    # Load precomputed CB_scaled and decay_vec
    cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    CB_scaled = tl.load(
        cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        other=0.0,
    ).to(tl.float32)

    decay_vec_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    decay_vec = tl.load(decay_vec_base + offs_t * stride_dv_t, mask=t_mask, other=0.0).to(
        tl.float32
    )

    # init_out = C_all @ state^T * decay_vec
    init_out = tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16)) * decay_vec[:, None]

    # cb_out = CB_scaled @ x_all
    cb_out = tl.dot(CB_scaled.to(tl.bfloat16), x_all.to(tl.bfloat16))

    out_all = init_out + cb_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        for t in range(T):
            z_t = tl.load(
                z_ptr + t * stride_z_T + offs_m * stride_z_dim, mask=m_mask, other=0.0
            ).to(tl.float32)
            out_t = tl.sum(tl.where((offs_t == t)[:, None], out_all, 0.0), axis=0)
            out_t = out_t * z_t * tl.sigmoid(z_t)
            tl.store(out_ptr + t * stride_out_T + offs_m * stride_out_dim, out_t, mask=m_mask)
    else:
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# Replay-style main kernel.  Thin wrapper around _replay_main_impl that carries
# the @triton.heuristics for constexpr derivation; called from the Python
# wrapper on the replay-style path (write or replay-nowrite).
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None}
)
@triton.heuristics({"USE_RS_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.heuristics(
    {"BLOCK_SIZE_WINDOW": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.jit()
def _checkpointing_main_kernel(
    # Pointers
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    slot_perm_ptr,
    rand_seed_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_WINDOW: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    WRITE_CHECKPOINT: tl.constexpr,
    EARLY_OUT: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    # Hoisted PDL signal: fire as the first thing every program does.
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    # Per-program early-out gate.  Signal-then-skip lets the next kernel
    # in dl/maindl chains start regardless of early-out outcome.
    if EARLY_OUT:
        pid_grid_eo = tl.program_id(axis=1)
        pid_grid_eo_eff = (tl.num_programs(axis=1) - 1 - pid_grid_eo) if REVERSE_PERM else pid_grid_eo
        if USE_PERM:
            pid_b_eo = tl.load(slot_perm_ptr + pid_grid_eo_eff)
        else:
            pid_b_eo = pid_grid_eo_eff
        if HAS_CACHE_BATCH_INDICES:
            cbi_eo = tl.load(state_batch_indices_ptr + pid_b_eo).to(tl.int64)
            if cbi_eo == pad_slot_id:
                return
        else:
            cbi_eo = pid_b_eo.to(tl.int64)
        pnat_eo = tl.load(prev_num_accepted_tokens_ptr + cbi_eo)
        if (pnat_eo + T > MAX_REPLAY_BUFFER_LENGTH) != WRITE_CHECKPOINT:
            return
    _replay_main_impl(
        state_ptr,
        state_scales_ptr,
        old_x_ptr,
        old_B_ptr,
        old_dt_ptr,
        old_dA_cumsum_ptr,
        prev_num_accepted_tokens_ptr,
        cache_buf_idx_ptr,
        x_ptr,
        C_ptr,
        D_ptr,
        z_ptr,
        out_ptr,
        cb_scaled_ptr,
        decay_vec_ptr,
        state_batch_indices_ptr,
        slot_perm_ptr,
        rand_seed_ptr,
        pad_slot_id,
        T,
        MAX_REPLAY_BUFFER_LENGTH,
        dim,
        dstate,
        nheads_ngroups_ratio,
        stride_state_batch,
        stride_state_head,
        stride_state_dim,
        stride_state_dstate,
        stride_state_scales_cache,
        stride_state_scales_head,
        stride_state_scales_dim,
        stride_old_x_cache,
        stride_old_x_T,
        stride_old_x_head,
        stride_old_x_dim,
        stride_old_B_cache,
        stride_old_B_dbuf,
        stride_old_B_T,
        stride_old_B_group,
        stride_old_B_dstate,
        stride_old_dt_cache,
        stride_old_dt_dbuf,
        stride_old_dt_head,
        stride_old_dt_T,
        stride_old_dA_cumsum_cache,
        stride_old_dA_cumsum_dbuf,
        stride_old_dA_cumsum_head,
        stride_old_dA_cumsum_T,
        stride_x_batch,
        stride_x_T,
        stride_x_head,
        stride_x_dim,
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
        stride_cb_batch,
        stride_cb_head,
        stride_cb_t,
        stride_cb_j,
        stride_dv_batch,
        stride_dv_head,
        stride_dv_t,
        BLOCK_SIZE_M,
        HAS_D,
        HAS_Z,
        HAS_CACHE_BATCH_INDICES,
        BLOCK_SIZE_DSTATE,
        BLOCK_SIZE_T,
        BLOCK_SIZE_WINDOW,
        LAUNCH_WITH_PDL,
        USE_RS_ROUNDING,
        PHILOX_ROUNDS,
        QUANT_MAX,
        WRITE_CHECKPOINT,
        LAUNCH_DEPENDENT_KERNELS,
        USE_PERM,
        REVERSE_PERM,
    )


# Rectangle main kernel (nowrite-only): no replay step, no state HBM write,
# no SR codegen.  state_out is computed from state_prev directly using the
# precomp-folded decay_vec_full; token_out is a single rectangle matmul over
# the (T, K) CB rectangle and (K, M) x_combined.  Pairs with
# `_rectangle_precompute_kernel`.


@triton.jit()
def _rectangle_main_impl(
    # Pointers
    state_ptr,
    state_scales_ptr,        # only consulted when QUANT_MAX > 0
    old_x_ptr,
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,           # rectangle (batch, nheads, T, K)
    decay_vec_ptr,           # folded (batch, nheads, T) — total_decay * exp(cumAdt_new[t])
    state_batch_indices_ptr,
    # Slot permutation: see _replay_precompute_impl for semantics.
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides (no quant-store path; state read-only for state_out)
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides: (cache, nheads, dim) — fp32, broadcast over dstate
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides: (cache, T_max, nheads, dim) — single-buffered
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides (rectangle (batch, nheads, T, K))
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    # Slot permutation flags — see _replay_precompute_impl.
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    pid_m = tl.program_id(axis=0)
    pid_grid_b = tl.program_id(axis=1)
    pid_grid_b_eff = (tl.num_programs(axis=1) - 1 - pid_grid_b) if REVERSE_PERM else pid_grid_b
    if USE_PERM:
        pid_b = tl.load(slot_perm_ptr + pid_grid_b_eff)
    else:
        pid_b = pid_grid_b_eff
    pid_h = tl.program_id(axis=2)

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    # Nowrite-only: write_offset = PNAT (new tokens append at [PNAT, PNAT+T)).
    buf_active = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    write_offset = prev_num_accepted_tokens

    # Static rectangle K-axis layout (matches precompute).
    K_NEW_SHIFT: tl.constexpr = MAX_REPLAY_BUFFER_LENGTH - T

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    # K-axis masks
    is_old_k = offs_k < prev_num_accepted_tokens
    safe_old_k = tl.where(is_old_k, offs_k, 0)
    k_new_idx = offs_k - K_NEW_SHIFT
    is_new_k = (k_new_idx >= 0) & (k_new_idx < T)
    safe_k_new = tl.where(is_new_k, k_new_idx, 0)

    # Load state (with dequant on load if quantized).  Read-only — no HBM
    # write on the nowrite path.
    state_ptr += cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    state_ptrs = (
        state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    if QUANT_MAX > 0.0:
        state_scales_base = (
            state_scales_ptr
            + cache_batch_idx * stride_state_scales_cache
            + pid_h * stride_state_scales_head
        )
        decode_scale = tl.load(
            state_scales_base + offs_m * stride_state_scales_dim,
            mask=m_mask, other=1.0,
        ).to(tl.float32)
        state = state * decode_scale[:, None]

    # Group / pointer offset setup
    group_idx = pid_h // nheads_ngroups_ratio
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    C_ptr += pid_b * stride_C_batch + group_idx * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    old_x_base = old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=m_mask, other=0.0
        ).to(tl.float32)

    # Hoist: old_x from cache doesn't depend on conv1d/precompute, so issue
    # the load BEFORE gdc_wait so its HBM latency overlaps with conv1d.
    old_x_load = tl.load(
        old_x_base
        + safe_old_k[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=is_old_k[:, None] & m_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # PDL gate: precompute outputs (cb_scaled, decay_vec_full) become safe
    # after gdc_wait.  conv1d outputs (x, C) also gated by the chained PDL.
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    # Load C and x (conv1d outputs after PDL wait)
    C_all = tl.load(
        C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    x_all = tl.load(
        x_ptr + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )
    # Append new x to cache at [PNAT, PNAT+T) of the active buffer.
    tl.store(
        old_x_base
        + (write_offset + offs_t)[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

    # Build x_combined for the rectangle token_out matmul (STATIC layout).
    # old_x_load is from cache (loaded above gdc_wait); new_x_shifted reads
    # x (conv1d output) at compile-time-shifted [K_NEW_SHIFT, K_NEW_SHIFT+T).
    # Disjoint masks → safe to add.
    new_x_shifted = tl.load(
        x_ptr + safe_k_new[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=is_new_k[:, None] & m_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x_combined = old_x_load + new_x_shifted  # (BLOCK_SIZE_K, BLOCK_SIZE_M)

    # Load precomputed rectangle CB and folded decay_vec.
    cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    CB_scaled = tl.load(
        cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_k[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_k[None, :] < BLOCK_SIZE_K),
        other=0.0,
    ).to(tl.float32)

    decay_vec_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    decay_vec_full = tl.load(
        decay_vec_base + offs_t * stride_dv_t, mask=t_mask, other=0.0
    ).to(tl.float32)

    # state_out: state_prev contribution to output, with decay folded post-matmul.
    # No state_prev_decayed (M, dstate) materialization — state is consumed
    # directly by the matmul, then decay_vec_full multiplies the (T, M) result.
    state_out = (
        tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
        * decay_vec_full[:, None]
    )

    # token_out: combined old + new tokens contribution via the rectangle.
    token_out = tl.dot(CB_scaled.to(tl.bfloat16), x_combined.to(tl.bfloat16))

    out_all = state_out + token_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        for t in range(T):
            z_t = tl.load(
                z_ptr + t * stride_z_T + offs_m * stride_z_dim, mask=m_mask, other=0.0
            ).to(tl.float32)
            out_t = tl.sum(tl.where((offs_t == t)[:, None], out_all, 0.0), axis=0)
            out_t = out_t * z_t * tl.sigmoid(z_t)
            tl.store(out_ptr + t * stride_out_T + offs_m * stride_out_dim, out_t, mask=m_mask)
    else:
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# Rectangle main kernel.  Thin wrapper around _rectangle_main_impl that carries
# the @triton.heuristics for constexpr derivation; called from the Python
# wrapper on the rectangle nowrite path.
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None}
)
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.jit()
def _rectangle_main_kernel(
    # Pointers
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    slot_perm_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    EARLY_OUT: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    USE_PERM: tl.constexpr,
    REVERSE_PERM: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    # Per-program early-out gate.  Rectangle is nowrite-only.
    if EARLY_OUT:
        pid_grid_eo = tl.program_id(axis=1)
        pid_grid_eo_eff = (tl.num_programs(axis=1) - 1 - pid_grid_eo) if REVERSE_PERM else pid_grid_eo
        if USE_PERM:
            pid_b_eo = tl.load(slot_perm_ptr + pid_grid_eo_eff)
        else:
            pid_b_eo = pid_grid_eo_eff
        if HAS_CACHE_BATCH_INDICES:
            cbi_eo = tl.load(state_batch_indices_ptr + pid_b_eo).to(tl.int64)
            if cbi_eo == pad_slot_id:
                return
        else:
            cbi_eo = pid_b_eo.to(tl.int64)
        pnat_eo = tl.load(prev_num_accepted_tokens_ptr + cbi_eo)
        if pnat_eo + T > MAX_REPLAY_BUFFER_LENGTH:
            return
    _rectangle_main_impl(
        state_ptr,
        state_scales_ptr,
        old_x_ptr,
        prev_num_accepted_tokens_ptr,
        cache_buf_idx_ptr,
        x_ptr,
        C_ptr,
        D_ptr,
        z_ptr,
        out_ptr,
        cb_scaled_ptr,
        decay_vec_ptr,
        state_batch_indices_ptr,
        slot_perm_ptr,
        pad_slot_id,
        T,
        MAX_REPLAY_BUFFER_LENGTH,
        dim,
        dstate,
        nheads_ngroups_ratio,
        stride_state_batch,
        stride_state_head,
        stride_state_dim,
        stride_state_dstate,
        stride_state_scales_cache,
        stride_state_scales_head,
        stride_state_scales_dim,
        stride_old_x_cache,
        stride_old_x_T,
        stride_old_x_head,
        stride_old_x_dim,
        stride_x_batch,
        stride_x_T,
        stride_x_head,
        stride_x_dim,
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
        stride_cb_batch,
        stride_cb_head,
        stride_cb_t,
        stride_cb_j,
        stride_dv_batch,
        stride_dv_head,
        stride_dv_t,
        BLOCK_SIZE_M,
        HAS_D,
        HAS_Z,
        HAS_CACHE_BATCH_INDICES,
        BLOCK_SIZE_DSTATE,
        BLOCK_SIZE_T,
        BLOCK_SIZE_K,
        LAUNCH_WITH_PDL,
        QUANT_MAX,
        LAUNCH_DEPENDENT_KERNELS,
        USE_PERM,
        REVERSE_PERM,
    )


# Dynamic main kernel.  Single launchable kernel that, per program, reads
# PNAT and dispatches to one of the existing impls:
#
#   if pnat + T > MAX:          replay_main_impl(WRITE_CHECKPOINT=True)
#   elif RECTANGLE (constexpr): rectangle_main_impl
#   else:                        replay_main_impl(WRITE_CHECKPOINT=False)
#
# Unlike precompute, WRITE_CHECKPOINT stays constexpr in the main impl —
# the body has constexpr-gated state-write code (quant + Philox + HBM
# store) where folding meaningfully shrinks the codegen.  So this kernel
# has TWO replay call sites (one per WRITE_CHECKPOINT specialization)
# both inlined, with a runtime branch picking which runs.  Reg envelope =
# max(replay_write, X) where X = rectangle_nowrite (RECTANGLE=True) or
# replay_nowrite (RECTANGLE=False).  cb_scaled is allocated (T, K) by the
# wrapper regardless.
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"HAS_CACHE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None}
)
@triton.heuristics({"USE_RS_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.heuristics({"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)})
@triton.heuristics(
    {"BLOCK_SIZE_WINDOW": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.heuristics(
    {"BLOCK_SIZE_K": lambda args: max(
        triton.next_power_of_2(args["MAX_REPLAY_BUFFER_LENGTH"]), 16)}
)
@triton.jit()
def _dynamic_main_kernel(
    # Pointers — union of replay-main and rectangle-main pointer args.
    state_ptr,
    state_scales_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    prev_num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    rand_seed_ptr,
    pad_slot_id,
    # Dimensions
    T: tl.constexpr,
    MAX_REPLAY_BUFFER_LENGTH: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # state strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    # state_scales strides
    stride_state_scales_cache,
    stride_state_scales_head,
    stride_state_scales_dim,
    # old_x strides
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    # old_B strides (replay only; passed but unused on rectangle path)
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    # old_dt strides (replay only)
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    # old_dA_cumsum strides (replay only)
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    # x strides
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    # C strides
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    # D strides
    stride_D_head,
    stride_D_dim,
    # z strides
    stride_z_batch,
    stride_z_T,
    stride_z_head,
    stride_z_dim,
    # out strides
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    # cb_scaled strides — wrapper allocates (T, K), so stride_cb_t = K
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    # decay_vec strides
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    # Meta
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_WINDOW: tl.constexpr,  # for replay path
    BLOCK_SIZE_K: tl.constexpr,       # for rectangle path
    LAUNCH_WITH_PDL: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    # Compile-time pick: rectangle (with replay-write fallback) vs replay-only.
    RECTANGLE: tl.constexpr,
    # Default False — dynamic main is normally terminal in its chain.
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    pid_b = tl.program_id(axis=1)
    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == pad_slot_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    pnat_local = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
    if pnat_local + T > MAX_REPLAY_BUFFER_LENGTH:
        # Write slot — replay-style write (WRITE_CHECKPOINT=True constexpr).
        _replay_main_impl(
            state_ptr,
            state_scales_ptr,
            old_x_ptr,
            old_B_ptr,
            old_dt_ptr,
            old_dA_cumsum_ptr,
            prev_num_accepted_tokens_ptr,
            cache_buf_idx_ptr,
            x_ptr,
            C_ptr,
            D_ptr,
            z_ptr,
            out_ptr,
            cb_scaled_ptr,
            decay_vec_ptr,
            state_batch_indices_ptr,
            state_batch_indices_ptr,  # slot_perm_ptr unused (USE_PERM=False)
            rand_seed_ptr,
            pad_slot_id,
            T,
            MAX_REPLAY_BUFFER_LENGTH,
            dim,
            dstate,
            nheads_ngroups_ratio,
            stride_state_batch,
            stride_state_head,
            stride_state_dim,
            stride_state_dstate,
            stride_state_scales_cache,
            stride_state_scales_head,
            stride_state_scales_dim,
            stride_old_x_cache,
            stride_old_x_T,
            stride_old_x_head,
            stride_old_x_dim,
            stride_old_B_cache,
            stride_old_B_dbuf,
            stride_old_B_T,
            stride_old_B_group,
            stride_old_B_dstate,
            stride_old_dt_cache,
            stride_old_dt_dbuf,
            stride_old_dt_head,
            stride_old_dt_T,
            stride_old_dA_cumsum_cache,
            stride_old_dA_cumsum_dbuf,
            stride_old_dA_cumsum_head,
            stride_old_dA_cumsum_T,
            stride_x_batch,
            stride_x_T,
            stride_x_head,
            stride_x_dim,
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
            stride_cb_batch,
            stride_cb_head,
            stride_cb_t,
            stride_cb_j,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_t,
            BLOCK_SIZE_M,
            HAS_D,
            HAS_Z,
            HAS_CACHE_BATCH_INDICES,
            BLOCK_SIZE_DSTATE,
            BLOCK_SIZE_T,
            BLOCK_SIZE_WINDOW,
            LAUNCH_WITH_PDL,
            USE_RS_ROUNDING,
            PHILOX_ROUNDS,
            QUANT_MAX,
            True,  # WRITE_CHECKPOINT (constexpr)
            False,  # LAUNCH_DEPENDENT_KERNELS — already signaled at top
            False,  # USE_PERM
            False,  # REVERSE_PERM
        )
    else:
        if RECTANGLE:
            _rectangle_main_impl(
                state_ptr,
                state_scales_ptr,
                old_x_ptr,
                prev_num_accepted_tokens_ptr,
                cache_buf_idx_ptr,
                x_ptr,
                C_ptr,
                D_ptr,
                z_ptr,
                out_ptr,
                cb_scaled_ptr,
                decay_vec_ptr,
                state_batch_indices_ptr,
                state_batch_indices_ptr,  # slot_perm_ptr unused (USE_PERM=False)
                pad_slot_id,
                T,
                MAX_REPLAY_BUFFER_LENGTH,
                dim,
                dstate,
                nheads_ngroups_ratio,
                stride_state_batch,
                stride_state_head,
                stride_state_dim,
                stride_state_dstate,
                stride_state_scales_cache,
                stride_state_scales_head,
                stride_state_scales_dim,
                stride_old_x_cache,
                stride_old_x_T,
                stride_old_x_head,
                stride_old_x_dim,
                stride_x_batch,
                stride_x_T,
                stride_x_head,
                stride_x_dim,
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
                stride_cb_batch,
                stride_cb_head,
                stride_cb_t,
                stride_cb_j,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_t,
                BLOCK_SIZE_M,
                HAS_D,
                HAS_Z,
                HAS_CACHE_BATCH_INDICES,
                BLOCK_SIZE_DSTATE,
                BLOCK_SIZE_T,
                BLOCK_SIZE_K,
                LAUNCH_WITH_PDL,
                QUANT_MAX,
                False,  # LAUNCH_DEPENDENT_KERNELS — already signaled at top
                False,  # USE_PERM
                False,  # REVERSE_PERM
            )
        else:
            # Replay-style nowrite (WRITE_CHECKPOINT=False constexpr).
            _replay_main_impl(
                state_ptr,
                state_scales_ptr,
                old_x_ptr,
                old_B_ptr,
                old_dt_ptr,
                old_dA_cumsum_ptr,
                prev_num_accepted_tokens_ptr,
                cache_buf_idx_ptr,
                x_ptr,
                C_ptr,
                D_ptr,
                z_ptr,
                out_ptr,
                cb_scaled_ptr,
                decay_vec_ptr,
                state_batch_indices_ptr,
                state_batch_indices_ptr,  # slot_perm_ptr unused (USE_PERM=False)
                rand_seed_ptr,
                pad_slot_id,
                T,
                MAX_REPLAY_BUFFER_LENGTH,
                dim,
                dstate,
                nheads_ngroups_ratio,
                stride_state_batch,
                stride_state_head,
                stride_state_dim,
                stride_state_dstate,
                stride_state_scales_cache,
                stride_state_scales_head,
                stride_state_scales_dim,
                stride_old_x_cache,
                stride_old_x_T,
                stride_old_x_head,
                stride_old_x_dim,
                stride_old_B_cache,
                stride_old_B_dbuf,
                stride_old_B_T,
                stride_old_B_group,
                stride_old_B_dstate,
                stride_old_dt_cache,
                stride_old_dt_dbuf,
                stride_old_dt_head,
                stride_old_dt_T,
                stride_old_dA_cumsum_cache,
                stride_old_dA_cumsum_dbuf,
                stride_old_dA_cumsum_head,
                stride_old_dA_cumsum_T,
                stride_x_batch,
                stride_x_T,
                stride_x_head,
                stride_x_dim,
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
                stride_cb_batch,
                stride_cb_head,
                stride_cb_t,
                stride_cb_j,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_t,
                BLOCK_SIZE_M,
                HAS_D,
                HAS_Z,
                HAS_CACHE_BATCH_INDICES,
                BLOCK_SIZE_DSTATE,
                BLOCK_SIZE_T,
                BLOCK_SIZE_WINDOW,
                LAUNCH_WITH_PDL,
                USE_RS_ROUNDING,
                PHILOX_ROUNDS,
                QUANT_MAX,
                False,  # WRITE_CHECKPOINT (constexpr)
                False,  # LAUNCH_DEPENDENT_KERNELS — already signaled at top
                False,  # USE_PERM
                False,  # REVERSE_PERM
            )


# Python wrapper


_QUANT_MAX_BY_DTYPE = {
    torch.int8: 127.0,
    torch.int16: 32767.0,
    torch.float8_e4m3fn: 448.0,
}


def checkpointing_state_update(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_dA_cumsum: torch.Tensor,
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
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 10,
    state_scales: torch.Tensor | None = None,
    launch_with_pdl=False,
    use_internal_pdl=True,
    write_checkpoint: bool = True,
    rectangle_for_nowrite: bool = False,
    mode: str = "monolithic",
    # Slot permutation: int32 (batch,) tensor mapping grid program_id ->
    # original slot index.  When provided, dl-family kernels (doublelaunch /
    # dlgrouped / maindl) read pid_b through this perm so callers can pre-sort
    # slots (e.g. write-first) to cluster early-outs at one end of the grid.
    # Ignored by monolithic / dynamic.  None => identity (today's behavior).
    slot_perm: torch.Tensor | None = None,
    # When True and slot_perm is provided, the nowrite-side kernels in
    # dlgrouped/doublelaunch traverse the perm in reverse (B-1-pid_grid).
    # Combined with a write-first sort, this front-loads real work in BOTH
    # halves of the dl chain (writes from the head, nowrites from the tail).
    reverse_nowrite: bool = False,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _num_stages: int | None = None,
    _precompute_num_warps: int | None = None,
    _precompute_num_stages: int | None = None,
    _heads_per_block: int | None = None,
    _maxnreg: int | None = None,
    _num_ctas: int | None = None,
):
    """
    Replay SSM state update with precomputed CB and tl.dot fast-forward.

    Two-kernel architecture:
      1. Precompute kernel: computes CB_scaled and decay_vec from B, C, dt, A.
         Writes processed dt/dA_cumsum/B to double-buffered cache for next step.
      2. Main kernel: replays old tokens via tl.dot fast-forward on cached data,
         then computes output using precomputed CB_scaled and new x/C inputs.

    PDL (Programmatic Dependent Launch) chain:
      conv1d → (external PDL) → precompute → (internal PDL) → main
      External PDL: precompute starts while conv1d is running; gdc_wait()
        in precompute blocks until conv1d completes before loading B/C.
      Internal PDL: main starts while precompute is running; main's replay
        phase uses only cached data from the previous step.  gdc_wait() in
        main blocks until precompute completes before loading conv1d outputs
        (x, C) and precompute outputs (CB_scaled, decay_vec).

    Uses double-buffered cache tensors.  cache_buf_idx[slot] indicates which
    buffer (0 or 1) to READ from for replay.  The WRITE buffer is 1 - read.
    Caller must flip cache_buf_idx[slot] after each call.

    Arguments:
        state: (cache, nheads, dim, dstate) in-place.  After the call, contains
            the state after replaying prev_num_accepted_tokens old tokens.
        old_x: (cache, T, nheads, dim) bf16 — old x cache (single-buffered).
        old_B: (cache, 2, T, ngroups, dstate) bf16 — double-buffered old B cache.
        old_dt: (cache, 2, nheads, T) fp32 — double-buffered processed dt.
        old_dA_cumsum: (cache, 2, nheads, T) fp32 — double-buffered cumulative A*dt.
        cache_buf_idx: (cache,) int32 — which buffer to read (0 or 1).
        prev_num_accepted_tokens: (cache,) int32.
        x: (batch, T, nheads, dim) new token inputs.
        dt: (batch, T, nheads, dim) with stride(-1)==0 (tie_hdim).
        A: (nheads, dim, dstate) with stride(-1)==0, stride(-2)==0 (tie_hdim).
        B: (batch, T, ngroups, dstate).
        C: (batch, T, ngroups, dstate).
        out: (batch, T, nheads, dim) preallocated output.
        D: (nheads, dim) optional feed-through parameter.
        z: (batch, T, nheads, dim) optional silu gate.
        dt_bias: (nheads, dim) optional, with stride(-1)==0 (tie_hdim).
        state_batch_indices: (batch,) optional cache slot mapping.
        rand_seed: optional single-element int64 CUDA tensor for Philox PRNG seed.
            When provided, state is stochastically rounded on store.  Supported
            for state.dtype in (fp16, int8, int16, fp8_e4m3fn); other dtypes
            silently use deterministic rounding.  fp16+SR and fp8+SR both
            require sm_100a (Blackwell B200+) — wrapper asserts this loudly.
        philox_rounds: number of Philox PRNG rounds (default 10).
        state_scales: required when state.dtype in (int8, int16, fp8_e4m3fn).
            Shape (cache_size, nheads, dim), fp32.  Per-(head, dim) channel
            decode scale (= 1 / encode_scale).  The kernel writes scales on
            checkpoint steps and reads them on load (broadcast over dstate).
            Ignored for non-quantized state dtypes.
        launch_with_pdl: enable external PDL (conv1d → precompute chain).
            Defaults False; caller opts in when the upstream chain is PDL-safe.
            Ignored on hardware that doesn't support PDL (sm < 90).
        use_internal_pdl: enable internal PDL (precompute → main overlap).
            Defaults True; override for testing only.
            Ignored on hardware that doesn't support PDL (sm < 90).

        _-prefixed kwargs (_block_size_m, _num_warps, _num_stages,
        _precompute_num_warps, _precompute_num_stages, _heads_per_block,
        _maxnreg, _num_ctas) are benchmark-only overrides; production callers
        should leave them None to use the heuristic-tuned defaults.
    """
    # PDL needs sm >= 90.
    if get_sm_version() < 90:
        launch_with_pdl = False
        use_internal_pdl = False

    # Mode selection:
    #   mode="monolithic" (default): today's behavior.  write_checkpoint and
    #       rectangle_for_nowrite together pick a single kernel pair for the
    #       whole batch.  Calls the corresponding kernel pair with EARLY_OUT=False.
    #   mode="dynamic": single kernel pair (_dynamic_*_kernel) that dispatches
    #       per-slot at runtime based on PNAT.  RECTANGLE constexpr (=
    #       rectangle_for_nowrite) picks whether the nowrite path is rectangle
    #       or replay-nowrite.  write_checkpoint is ignored (per-slot from PNAT).
    #   mode="doublelaunch": two kernel pairs launched in sequence, each with
    #       EARLY_OUT=True, partitioning the batch by PNAT-derived mode.
    #       Write half: replay-write.  Nowrite half: rectangle if
    #       rectangle_for_nowrite else replay-nowrite.  write_checkpoint ignored.
    #   mode="dlgrouped": same 4 kernels as doublelaunch, but reordered to
    #       launch both precomputes first, then both mains.  Lets the GPU
    #       run precomp1 || precomp2 in parallel before the mains start.
    #       write_checkpoint ignored.
    #   mode="maindl": shared (dynamic) precompute + doublelaunched main.
    #       One precompute call (_dynamic_precompute_kernel) handles per-slot
    #       dispatch, then two main kernels with EARLY_OUT=True for the write
    #       and nowrite halves.  Strictly fewer kernel launches than
    #       doublelaunch (3 vs 4) at the cost of dispatch precompute's wider
    #       reg envelope.  write_checkpoint ignored.
    assert mode in ("monolithic", "dynamic", "doublelaunch", "dlgrouped", "maindl", "dl_write_only"), (
        f"unknown mode {mode!r}; expected one of "
        "'monolithic', 'dynamic', 'doublelaunch', 'dlgrouped', or 'maindl'"
    )
    use_rectangle = rectangle_for_nowrite and not write_checkpoint

    # --- Hardware support gates ---
    # fp8 e4m3fn (any rounding mode) needs SM 89+ for the fp32↔e4m3 cvt PTX
    # instructions (Ada Lovelace introduced them; Hopper/Blackwell carry them).
    if state.dtype == torch.float8_e4m3fn:
        assert get_sm_version() >= 89, (
            "fp8_e4m3fn state requires SM 89+ (Ada Lovelace / Hopper / Blackwell) "
            f"for fp32↔fp8 cvt PTX instructions; current SM is {get_sm_version()}."
        )

    # PTX cvt.rs.* (stochastic rounding) family lands on Blackwell only.
    # Wrapper fails loud; framework decides fall-back (e.g. drop SR, use RN).
    # int8 / int16 SR uses pure-Triton libdevice.floor + uniform noise — no
    # PTX SR instruction needed, runs anywhere.
    if rand_seed is not None:
        if state.dtype == torch.float16:
            assert get_sm_version() >= 100, (
                "fp16 stochastic rounding (PTX cvt.rs.f16x2.f32) requires "
                f"sm_100a (Blackwell B200+); current SM is {get_sm_version()}."
            )
        elif state.dtype == torch.float8_e4m3fn:
            assert get_sm_version() >= 100, (
                "fp8 stochastic rounding (PTX cvt.rs.satfinite.e4m3x4.f32) "
                f"requires sm_100a (Blackwell B200+); current SM is {get_sm_version()}."
            )

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

    # --- Quantization plumbing ---
    # QUANT_MAX > 0 ⇔ state is int8 / int16 / fp8_e4m3fn.  Kernel-entry
    # static_assert on the Triton side mirrors this invariant.
    quant_max = _QUANT_MAX_BY_DTYPE.get(state.dtype, 0.0)
    is_quantized = quant_max > 0.0
    if is_quantized:
        assert state_scales is not None, (
            f"state.dtype={state.dtype} requires state_scales tensor "
            "(shape (cache_size, nheads, dim), fp32)."
        )
        assert state_scales.shape == (cache_size, nheads, dim), (
            f"state_scales shape mismatch: expected {(cache_size, nheads, dim)}, "
            f"got {state_scales.shape}."
        )
        assert state_scales.dtype == torch.float32, (
            f"state_scales must be fp32, got {state_scales.dtype}."
        )
        assert state_scales.device == state.device

    # Cache T-axis = MAX_WINDOW (the replay buffer capacity).  For the
    # placeholder degenerate case max_window = T (every step is a checkpoint
    # step).  For real replay-style checkpointing, max_window > T and
    # `prev_num_accepted_tokens` can be 0..max_window.
    max_window = old_x.shape[1]
    assert T <= max_window, f"T={T} exceeds cache max_window={max_window}"
    # Replay-style code path uses BLOCK_SIZE_T = max(np2(T), 16) for the
    # combined T-axis (T_new tile size) and reuses it for window loads.  Until
    # the heuristic is generalized to track max_window separately, require
    # max_window to fit within that tile.
    block_size_t = max(triton.next_power_of_2(T), 16)
    assert max_window <= block_size_t, (
        f"max_window={max_window} exceeds BLOCK_SIZE_T={block_size_t} "
        f"derived from T={T}; extend the heuristic to include max_window."
    )

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    assert old_x.shape == (cache_size, max_window, nheads, dim)
    assert old_B.shape == (cache_size, 2, max_window, ngroups, dstate)
    assert old_dt.shape == (cache_size, 2, nheads, max_window)
    assert old_dA_cumsum.shape == (cache_size, 2, nheads, max_window)
    assert cache_buf_idx.shape == (cache_size,)
    assert prev_num_accepted_tokens.shape == (cache_size,)

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )
    assert tie_hdim

    device = x.device
    BLOCK_SIZE_T = max(triton.next_power_of_2(T), 16)
    # Rectangle K-axis bound = window (max_window).  Computed unconditionally
    # so the launch sites can refer to it; only used on the rectangle path.
    BLOCK_SIZE_K = max(triton.next_power_of_2(max_window), 16)

    # Allocate precomputed intermediates (per-call, not cached).  Always
    # allocate (T, K) — the largest layout that any path uses.  Replay-style
    # paths only touch the first T columns; rectangle/dynamic use the full K.
    # The few extra unused columns per row are negligible (~6KB per layer at
    # production sizes) and let the dispatch helpers share one buffer.
    cb_scaled = torch.empty(
        batch, nheads, BLOCK_SIZE_T, BLOCK_SIZE_K, device=device, dtype=torch.float32
    )
    decay_vec = torch.empty(batch, nheads, BLOCK_SIZE_T, device=device, dtype=torch.float32)

    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)
    )

    # Kernel tuning: BLOCK_SIZE_M, num_warps, HEADS_PER_BLOCK, precompute_num_warps.
    # Dtype-aware heuristic from B200 sweeps (batch 1-512, T=6/32, TP=8, conv1d +
    # chained PDL).  Keyed on total_heads, BLOCK_SIZE_T, and state dtype; 16-bit
    # states prefer different tiles from fp32 due to lower bandwidth.  Philox
    # gets its own branch — stochastic rounding shifts compute toward CUDA cores,
    # so small-batch configs want more warps to hide the extra work.
    total_heads = batch * nheads
    heads_per_group = nheads // ngroups
    state_is_16bit = state.dtype in (torch.float16, torch.bfloat16)
    use_philox = rand_seed is not None
    if BLOCK_SIZE_T <= 16:
        if use_philox and state_is_16bit:
            # Philox: more warps at small batch to hide CUDA core work.
            # At large batch, converges to non-Philox fp16 config.
            if total_heads <= 16:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 4, 4, 4, 1
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 8, 1, 2, 1
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 32, 4, 2, 1
        elif state_is_16bit:
            if total_heads <= 16:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 32, 4, 4, 1
            elif total_heads <= 64:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 8, 1, 2, 1
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 16, 2, 2, 1
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    1,
                    min(2, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 32, 4, 2, 1
        else:  # fp32 state (no Philox — fp32 doesn't need stochastic rounding)
            if total_heads <= 32:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 8, 1, 4, 1
            elif total_heads <= 64:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 8, 1, 2, 1
            elif total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 8, 2, 2, 1
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 16, 1, 2, 1
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    2,
                    min(2, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 32, 4, 2, 1
    else:  # T > 16
        if state_is_16bit:
            if total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 16, 2, 4, 1
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    16,
                    1,
                    4,
                    min(2, heads_per_group),
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    1,
                    min(4, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    1,
                    4,
                    min(2, heads_per_group),
                )
        else:  # fp32 state
            if total_heads <= 128:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = 16, 2, 4, 1
            elif total_heads <= 256:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    32,
                    2,
                    4,
                    min(2, heads_per_group),
                )
            elif total_heads <= 512:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    2,
                    min(4, heads_per_group),
                )
            else:
                BLOCK_SIZE_M, num_warps, precompute_num_warps, heads_per_block = (
                    64,
                    2,
                    4,
                    min(2, heads_per_group),
                )
    if _block_size_m is not None:
        BLOCK_SIZE_M = _block_size_m
    if _num_warps is not None:
        num_warps = _num_warps
    if _heads_per_block is not None:
        heads_per_block = _heads_per_block
    if _precompute_num_warps is not None:
        precompute_num_warps = _precompute_num_warps

    HAS_CACHE_BATCH_INDICES = state_batch_indices is not None

    assert nheads % heads_per_block == 0, (
        f"nheads ({nheads}) must be divisible by heads_per_block ({heads_per_block})"
    )
    assert heads_per_block <= heads_per_group, (
        f"heads_per_block ({heads_per_block}) must not cross group boundary ({heads_per_group})"
    )

    # state_scales pointer + strides: real tensor when quantized, otherwise
    # zero-strided dummy (kernel never reads it because QUANT_MAX==0.0).
    if is_quantized:
        state_scales_arg = state_scales
        state_scales_strides = (
            state_scales.stride(0),
            state_scales.stride(1),
            state_scales.stride(2),
        )
    else:
        state_scales_arg = state  # any valid ptr — gated by QUANT_MAX==0
        state_scales_strides = (0, 0, 0)

    # Slot permutation — pointer + USE_PERM gate.  When the caller provides
    # a perm tensor the dl-family launches read pid_b through it; otherwise
    # we pass any valid pointer (state_batch_indices) and USE_PERM=False so
    # the kernel falls back to pid_grid.  Sort-driven dispatch (write-first
    # clustering) is opt-in per call; monolithic / dynamic ignore the flag.
    if slot_perm is not None:
        assert slot_perm.dtype in (torch.int32, torch.int64), (
            f"slot_perm must be int32/int64, got {slot_perm.dtype}"
        )
        assert slot_perm.numel() >= batch, (
            f"slot_perm has {slot_perm.numel()} entries; need >= batch ({batch})"
        )
        slot_perm_arg = slot_perm
        use_perm = True
    else:
        # Any valid ptr — gated by USE_PERM=False at compile time.
        slot_perm_arg = state_batch_indices if state_batch_indices is not None else state
        use_perm = False

    # Grid for main kernels (M tiling × batch × nheads).
    def main_grid(META):
        return (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)

    precomp_grid = (batch, nheads // heads_per_block)
    d_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)

    # ---- Launch helpers (close over locals) -------------------------------
    # Each helper is a thin closure that calls one Triton kernel with the
    # full positional + kwarg argument list.  Mode-dependent constexprs
    # (write_checkpoint, early_out, rectangle) are passed in.

    def launch_replay_precompute(write_checkpoint: bool, early_out: bool,
                                 reverse_perm: bool = False):
        _checkpointing_precompute_kernel[precomp_grid](
            dt, dt_bias, A, B, C,
            cb_scaled, decay_vec,
            old_B, old_dt, old_dA_cumsum,
            cache_buf_idx, prev_num_accepted_tokens,
            state_batch_indices, slot_perm_arg, pad_slot_id,
            T, max_window, dstate, nheads // ngroups,
            dt.stride(0), dt.stride(1), dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            old_B.stride(0), old_B.stride(1), old_B.stride(2),
            old_B.stride(3), old_B.stride(4),
            old_dt.stride(0), old_dt.stride(1),
            old_dt.stride(2), old_dt.stride(3),
            old_dA_cumsum.stride(0), old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2), old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            HEADS_PER_BLOCK=heads_per_block,
            WRITE_CHECKPOINT=write_checkpoint,
            EARLY_OUT=early_out,
            USE_PERM=use_perm,
            REVERSE_PERM=reverse_perm,
            num_warps=precompute_num_warps,
            **({"num_stages": _precompute_num_stages} if _precompute_num_stages else {}),
            launch_pdl=launch_with_pdl,
        )

    def launch_rectangle_precompute(early_out: bool, reverse_perm: bool = False):
        _rectangle_precompute_kernel[precomp_grid](
            dt, dt_bias, A, B, C,
            cb_scaled, decay_vec,
            old_B, old_dt, old_dA_cumsum,
            cache_buf_idx, prev_num_accepted_tokens,
            state_batch_indices, slot_perm_arg, pad_slot_id,
            T, max_window, dstate, nheads // ngroups,
            dt.stride(0), dt.stride(1), dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            old_B.stride(0), old_B.stride(1), old_B.stride(2),
            old_B.stride(3), old_B.stride(4),
            old_dt.stride(0), old_dt.stride(1),
            old_dt.stride(2), old_dt.stride(3),
            old_dA_cumsum.stride(0), old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2), old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            HEADS_PER_BLOCK=heads_per_block,
            EARLY_OUT=early_out,
            USE_PERM=use_perm,
            REVERSE_PERM=reverse_perm,
            num_warps=precompute_num_warps,
            **({"num_stages": _precompute_num_stages} if _precompute_num_stages else {}),
            launch_pdl=launch_with_pdl,
        )

    def launch_dynamic_precompute(rectangle: bool):
        _dynamic_precompute_kernel[precomp_grid](
            dt, dt_bias, A, B, C,
            cb_scaled, decay_vec,
            old_B, old_dt, old_dA_cumsum,
            cache_buf_idx, prev_num_accepted_tokens,
            state_batch_indices, pad_slot_id,
            T, max_window, dstate, nheads // ngroups,
            dt.stride(0), dt.stride(1), dt.stride(2),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            old_B.stride(0), old_B.stride(1), old_B.stride(2),
            old_B.stride(3), old_B.stride(4),
            old_dt.stride(0), old_dt.stride(1),
            old_dt.stride(2), old_dt.stride(3),
            old_dA_cumsum.stride(0), old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2), old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=HAS_CACHE_BATCH_INDICES,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            HEADS_PER_BLOCK=heads_per_block,
            RECTANGLE=rectangle,
            num_warps=precompute_num_warps,
            **({"num_stages": _precompute_num_stages} if _precompute_num_stages else {}),
            launch_pdl=launch_with_pdl,
        )

    def launch_replay_main(write_checkpoint: bool, early_out: bool,
                           launch_dependent_kernels: bool = False,
                           reverse_perm: bool = False):
        _checkpointing_main_kernel[main_grid](
            state, state_scales_arg, old_x,
            old_B, old_dt, old_dA_cumsum,
            prev_num_accepted_tokens, cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices, slot_perm_arg, rand_seed, pad_slot_id,
            T, max_window, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            state_scales_strides[0], state_scales_strides[1], state_scales_strides[2],
            old_x.stride(0), old_x.stride(1), old_x.stride(2), old_x.stride(3),
            old_B.stride(0), old_B.stride(1), old_B.stride(2),
            old_B.stride(3), old_B.stride(4),
            old_dt.stride(0), old_dt.stride(1),
            old_dt.stride(2), old_dt.stride(3),
            old_dA_cumsum.stride(0), old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2), old_dA_cumsum.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            d_strides[0], d_strides[1],
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            PHILOX_ROUNDS=philox_rounds if rand_seed is not None else 0,
            QUANT_MAX=quant_max,
            WRITE_CHECKPOINT=write_checkpoint,
            EARLY_OUT=early_out,
            LAUNCH_DEPENDENT_KERNELS=launch_dependent_kernels and use_internal_pdl,
            USE_PERM=use_perm,
            REVERSE_PERM=reverse_perm,
            num_warps=num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )

    def launch_rectangle_main(early_out: bool,
                              launch_dependent_kernels: bool = False,
                              reverse_perm: bool = False):
        _rectangle_main_kernel[main_grid](
            state, state_scales_arg, old_x,
            prev_num_accepted_tokens, cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices, slot_perm_arg, pad_slot_id,
            T, max_window, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            state_scales_strides[0], state_scales_strides[1], state_scales_strides[2],
            old_x.stride(0), old_x.stride(1), old_x.stride(2), old_x.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            d_strides[0], d_strides[1],
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            QUANT_MAX=quant_max,
            EARLY_OUT=early_out,
            LAUNCH_DEPENDENT_KERNELS=launch_dependent_kernels and use_internal_pdl,
            USE_PERM=use_perm,
            REVERSE_PERM=reverse_perm,
            num_warps=num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )

    def launch_dynamic_main(rectangle: bool,
                            launch_dependent_kernels: bool = False):
        _dynamic_main_kernel[main_grid](
            state, state_scales_arg, old_x,
            old_B, old_dt, old_dA_cumsum,
            prev_num_accepted_tokens, cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices, rand_seed, pad_slot_id,
            T, max_window, dim, dstate, nheads // ngroups,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            state_scales_strides[0], state_scales_strides[1], state_scales_strides[2],
            old_x.stride(0), old_x.stride(1), old_x.stride(2), old_x.stride(3),
            old_B.stride(0), old_B.stride(1), old_B.stride(2),
            old_B.stride(3), old_B.stride(4),
            old_dt.stride(0), old_dt.stride(1),
            old_dt.stride(2), old_dt.stride(3),
            old_dA_cumsum.stride(0), old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2), old_dA_cumsum.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            d_strides[0], d_strides[1],
            z_strides[0], z_strides[1], z_strides[2], z_strides[3],
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cb_scaled.stride(0), cb_scaled.stride(1),
            cb_scaled.stride(2), cb_scaled.stride(3),
            decay_vec.stride(0), decay_vec.stride(1), decay_vec.stride(2),
            BLOCK_SIZE_M,
            LAUNCH_WITH_PDL=use_internal_pdl,
            PHILOX_ROUNDS=philox_rounds if rand_seed is not None else 0,
            QUANT_MAX=quant_max,
            RECTANGLE=rectangle,
            LAUNCH_DEPENDENT_KERNELS=launch_dependent_kernels and use_internal_pdl,
            num_warps=num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )

    # ---- Mode dispatch ----------------------------------------------------
    with torch.cuda.device(device.index):
        if mode == "monolithic":
            if use_rectangle:
                launch_rectangle_precompute(early_out=False)
                launch_rectangle_main(early_out=False)
            else:
                launch_replay_precompute(write_checkpoint=write_checkpoint, early_out=False)
                launch_replay_main(write_checkpoint=write_checkpoint, early_out=False)
        elif mode == "dynamic":
            launch_dynamic_precompute(rectangle=rectangle_for_nowrite)
            launch_dynamic_main(rectangle=rectangle_for_nowrite)
        elif mode == "maindl":
            # Shared dispatch precompute, doublelaunched main.  Precompute
            # runs once with per-slot dispatch (saves the second precomp
            # empty-grid tax of doublelaunch).  Mains stay split with
            # EARLY_OUT so each retains its constexpr-specialized reg
            # envelope.  First main signals PDL dependents so the second
            # main can start its setup while the first is still computing.
            # Dynamic precompute doesn't support sort (no early-out to
            # cluster), so the perm only flows into the two EARLY_OUT mains.
            launch_dynamic_precompute(rectangle=rectangle_for_nowrite)
            launch_replay_main(write_checkpoint=True, early_out=True,
                               launch_dependent_kernels=True)
            if rectangle_for_nowrite:
                launch_rectangle_main(early_out=True, reverse_perm=reverse_nowrite)
            else:
                launch_replay_main(write_checkpoint=False, early_out=True,
                                   reverse_perm=reverse_nowrite)
        elif mode == "dlgrouped":
            # Same 4 kernels as doublelaunch but reordered: both precomputes
            # first, then both mains.  Lets the GPU run precomp1 || precomp2
            # in parallel (they're tiny grids) before the mains start, vs
            # doublelaunch's interleaved precomp1→main1→precomp2→main2.
            # First main signals PDL so the second main's setup overlaps.
            # When slot_perm + reverse_nowrite are set, the nowrite-side
            # walks the perm in reverse so both kernels front-load real work.
            launch_replay_precompute(write_checkpoint=True, early_out=True)
            if rectangle_for_nowrite:
                launch_rectangle_precompute(early_out=True, reverse_perm=reverse_nowrite)
            else:
                launch_replay_precompute(write_checkpoint=False, early_out=True,
                                         reverse_perm=reverse_nowrite)
            launch_replay_main(write_checkpoint=True, early_out=True,
                               launch_dependent_kernels=True)
            if rectangle_for_nowrite:
                launch_rectangle_main(early_out=True, reverse_perm=reverse_nowrite)
            else:
                launch_replay_main(write_checkpoint=False, early_out=True,
                                   reverse_perm=reverse_nowrite)
        elif mode == "dl_write_only":
            # Debug-only: just the write half of doublelaunch.  EARLY_OUT=True
            # means nowrite slots still pay the EO-gate tax (PNAT load + branch),
            # but no nowrite-side kernels run.  Used to isolate "is the sort
            # regression in the write-side kernels?".
            launch_replay_precompute(write_checkpoint=True, early_out=True)
            launch_replay_main(write_checkpoint=True, early_out=True,
                               launch_dependent_kernels=False)
        else:  # mode == "doublelaunch"
            # Write half: always replay-style write.  First main signals
            # PDL dependents so the second precompute can start its setup
            # while the first main is still computing.
            launch_replay_precompute(write_checkpoint=True, early_out=True)
            launch_replay_main(write_checkpoint=True, early_out=True,
                               launch_dependent_kernels=True)
            # Nowrite half: rectangle if asked, else replay-nowrite.
            if rectangle_for_nowrite:
                launch_rectangle_precompute(early_out=True, reverse_perm=reverse_nowrite)
                launch_rectangle_main(early_out=True, reverse_perm=reverse_nowrite)
            else:
                launch_replay_precompute(write_checkpoint=False, early_out=True,
                                         reverse_perm=reverse_nowrite)
                launch_replay_main(write_checkpoint=False, early_out=True,
                                   reverse_perm=reverse_nowrite)
