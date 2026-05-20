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

from .mamba2_metadata import (REPLAY_WORK_CACHE_BUF_IDX,
                              REPLAY_WORK_CACHE_SLOT,
                              REPLAY_WORK_ITEM_WIDTH, REPLAY_WORK_PNAT,
                              REPLAY_WORK_POSITION_IN_DECODE_BATCH)
from .softplus import softplus

_REPLAY_WORK_POSITION_IN_DECODE_BATCH = tl.constexpr(
    REPLAY_WORK_POSITION_IN_DECODE_BATCH)
_REPLAY_WORK_CACHE_SLOT = tl.constexpr(REPLAY_WORK_CACHE_SLOT)
_REPLAY_WORK_PNAT = tl.constexpr(REPLAY_WORK_PNAT)
_REPLAY_WORK_CACHE_BUF_IDX = tl.constexpr(REPLAY_WORK_CACHE_BUF_IDX)
_REPLAY_WORK_ITEM_WIDTH = tl.constexpr(REPLAY_WORK_ITEM_WIDTH)


# Lazy global allocator for Triton TMA tensor descriptors. Required by any
# host- or device-built tensor_descriptor; without it Triton raises at first
# launch.
_TMA_ALLOCATOR_SET = False


def _ensure_tma_allocator() -> None:
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def _alloc_fn(size, alignment, stream):
        # Triton expects an int8 buffer of `size` bytes; alignment is enforced
        # by the allocator returning a buffer satisfying it (PyTorch's
        # cudaMalloc-backed tensors are 256B-aligned, so we're fine).
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_SET = True


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


@triton.jit
def _bitrev32(x: tl.tensor) -> tl.tensor:
    return tl.inline_asm_elementwise(
        asm="brev.b32 $0, $1;",
        constraints="=r,r",
        args=(x,),
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _stochastic_round_int8_packed(
    x: tl.tensor, rand: tl.tensor, offs_n: tl.tensor
) -> tl.tensor:
    """Stochastic rounding for int8 using one random uint32 per 4 values."""
    low = rand & 0x0000FFFF
    high = (rand >> 16) & 0x0000FFFF
    low_rev = _bitrev32(low) >> 16
    high_rev = _bitrev32(high) >> 16
    rand_pos = offs_n & 3
    rand16 = tl.where(
        rand_pos == 0,
        low,
        tl.where(rand_pos == 1, low_rev, tl.where(rand_pos == 2, high, high_rev)),
    )
    rand01 = rand16.to(tl.float32) * (1.0 / float(1 << 16))
    return tl.extra.cuda.libdevice.floor(x + rand01)


@triton.jit
def _stochastic_round_int16_packed(
    x: tl.tensor, rand: tl.tensor, offs_n: tl.tensor
) -> tl.tensor:
    """Stochastic rounding for int16 using one random uint32 per 2 values."""
    rand_bits = tl.where((offs_n & 1) == 0, rand, _bitrev32(rand))
    rand01 = (rand_bits & 0x00FFFFFF).to(tl.float32) * (1.0 / float(1 << 24))
    return tl.extra.cuda.libdevice.floor(x + rand01)


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
    # on no-replay-write steps).
    prev_num_accepted_tokens_ptr,
    state_batch_indices_ptr,
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
    # Checkpoint write flag — selects target buffer + offset for new-token
    # cache writes.  See "Cache write semantics" block below.
    # Runtime (not constexpr): the only WRITE_CHECKPOINT-dependent code in
    # this body is the write_buf/write_offset selection, which is plain
    # arithmetic — no constexpr-shaped tile or whole-block gate.  Letting
    # it be runtime lets the dynamic dispatch kernel call us once with the
    # per-slot needs_write flag instead of inlining two specializations.
    write_checkpoint,
):
    pid_b = tl.program_id(axis=0)
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

    # --- Vectorized pre-wait phase across HEADS_PER_BLOCK heads ---
    # Compute dt, dA_cumsum, decay_vec as (H, T) tiles.  Pre-compute
    # scale_combo = decay_matrix * dt[:, None, :] as an (H, T, T) tile that
    # stays in registers across gdc_wait — eliminates the post-wait reload
    # of dt + dA_cumsum and the per-head loop.
    offs_h = tl.arange(0, HEADS_PER_BLOCK)
    heads_block = first_head + offs_h  # (H,)

    # Load dt (H, T)
    dt_addrs = (
        dt_ptr + pid_b * stride_dt_batch
        + heads_block[:, None] * stride_dt_head
        + offs_t[None, :] * stride_dt_T
    )
    dt = tl.load(dt_addrs, mask=t_mask[None, :], other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + heads_block * stride_dt_bias_head).to(tl.float32)
        dt = dt + dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = softplus(dt)

    A = tl.load(A_ptr + heads_block * stride_A_head).to(tl.float32)  # (H,)
    dA_cumsum = tl.cumsum(A[:, None] * dt, axis=1)  # (H, T)
    decay_vec = tl.exp(dA_cumsum)  # (H, T)

    # Cross-step continuity for old_dA_cumsum: when appending to active_buf at
    # offset PNAT > 0, the previous step left a running cumsum at [0, PNAT)
    # whose tail value lives at active_buf[head, PNAT-1].  Add that tail to
    # this step's per-step-restarted cumsum before storing so the buffer
    # holds one continuous cumsum across N back-to-back nowrites.  Write path
    # (write_buf = 1 - buf_active, write_offset = 0) starts fresh, no prefix.
    # Both branches are on scalar runtime values (write_checkpoint and PNAT),
    # uniform across the block — use scalar if to short-circuit the load.
    if write_checkpoint or prev_num_accepted_tokens == 0:
        prev_total = tl.zeros((HEADS_PER_BLOCK,), dtype=tl.float32)
    else:
        last_cumsum_ptrs = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + heads_block * stride_old_dA_cumsum_head
            + (prev_num_accepted_tokens - 1) * stride_old_dA_cumsum_T
        )
        prev_total = tl.load(last_cumsum_ptrs).to(tl.float32)

    # Store dt, dA_cumsum to cache at [write_offset : write_offset+T) of write_buf.
    old_dt_addrs = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + write_buf * stride_old_dt_dbuf
        + heads_block[:, None] * stride_old_dt_head
        + (write_offset + offs_t)[None, :] * stride_old_dt_T
    )
    tl.store(old_dt_addrs, dt, mask=t_mask[None, :])

    old_dA_cumsum_addrs = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + write_buf * stride_old_dA_cumsum_dbuf
        + heads_block[:, None] * stride_old_dA_cumsum_head
        + (write_offset + offs_t)[None, :] * stride_old_dA_cumsum_T
    )
    tl.store(old_dA_cumsum_addrs, dA_cumsum + prev_total[:, None], mask=t_mask[None, :])

    # decay_vec scratch — always at offs_t.
    decay_vec_addrs = (
        decay_vec_ptr + pid_b * stride_dv_batch
        + heads_block[:, None] * stride_dv_head
        + offs_t[None, :] * stride_dv_t
    )
    tl.store(decay_vec_addrs, decay_vec, mask=t_mask[None, :])

    # scale_combo (H, T, T) = exp(dA_cumsum[h, t1] - dA_cumsum[h, t2]) * dt[h, t2]
    # Stays live across gdc_wait — used post-wait to compute CB_scaled.
    decay_matrix = tl.exp(dA_cumsum[:, :, None] - dA_cumsum[:, None, :])  # (H, T, T)
    scale_combo = decay_matrix * dt[:, None, :]  # (H, T, T)

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

    # --- Vectorized post-wait phase: scale_combo (H, T, T) is still live in
    # registers from pre-wait; multiply by raw_CB (T, T), apply causal mask,
    # store as one (H, T, T) tile. ---
    CB_scaled_block = tl.where(
        valid_mask[None, :, :],
        raw_CB[None, :, :] * scale_combo,
        0.0,
    )  # (H, T, T)
    cb_scaled_addrs = (
        cb_scaled_ptr + pid_b * stride_cb_batch
        + heads_block[:, None, None] * stride_cb_head
        + offs_t[None, :, None] * stride_cb_t
        + offs_t[None, None, :] * stride_cb_j
    )  # (H, T, T)
    cb_store_mask = (
        (offs_t[None, :, None] < BLOCK_SIZE_T)
        & (offs_t[None, None, :] < BLOCK_SIZE_T)
    )
    tl.store(cb_scaled_addrs, CB_scaled_block, mask=cb_store_mask)


# Replay-style precompute kernel.  Thin wrapper around _replay_precompute_impl
# that carries the @triton.heuristics for constexpr derivation; called from
# the Python wrapper on the replay-style path (write or replay-nowrite).
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
    USE_GATHER_FOR_NEW_TOKENS: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
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

    # Rectangle K-axis layout: old at [0, PNAT), new at [PNAT, PNAT+T).
    # PNAT + T <= MAX is guaranteed on the nowrite path → no overlap.

    offs_t = tl.arange(0, BLOCK_SIZE_T)  # T-axis (output rows)
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K-axis (rectangle input cols)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < T
    n_mask = offs_n < dstate

    # K-axis masks. Cache and matmul share rows.
    is_old_k = offs_k < prev_num_accepted_tokens
    safe_old_k = tl.where(is_old_k, offs_k, 0)
    k_new_idx = offs_k - prev_num_accepted_tokens
    is_new_k = (k_new_idx >= 0) & (k_new_idx < T)
    safe_k_new = tl.where(is_new_k, k_new_idx, 0)

    offs_h = tl.arange(0, HEADS_PER_BLOCK)
    heads_block = first_head + offs_h

    # Precompute this step's (H, T) dt and continuous dA_cumsum tiles.
    # Keep them in registers for the rectangle path below; reloading from
    # global memory after storing would create a same-kernel write/read race.
    dt_addrs = (
        dt_ptr
        + pid_b * stride_dt_batch
        + heads_block[:, None] * stride_dt_head
        + offs_t[None, :] * stride_dt_T
    )
    dt_new = tl.load(dt_addrs, mask=t_mask[None, :], other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_heads = tl.load(dt_bias_ptr + heads_block * stride_dt_bias_head).to(tl.float32)
        dt_new = dt_new + dt_bias_heads[:, None]
    if DT_SOFTPLUS:
        dt_new = softplus(dt_new)

    A_heads = tl.load(A_ptr + heads_block * stride_A_head).to(tl.float32)
    dA_cumsum_step = tl.cumsum(A_heads[:, None] * dt_new, axis=1)

    if prev_num_accepted_tokens == 0:
        dA_cumsum_prefix = tl.zeros((HEADS_PER_BLOCK,), dtype=tl.float32)
    else:
        dA_cumsum_prefix_ptrs = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_active * stride_old_dA_cumsum_dbuf
            + heads_block * stride_old_dA_cumsum_head
            + (prev_num_accepted_tokens - 1) * stride_old_dA_cumsum_T
        )
        dA_cumsum_prefix = tl.load(dA_cumsum_prefix_ptrs).to(tl.float32)

    old_dt_write_addrs = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + write_buf * stride_old_dt_dbuf
        + heads_block[:, None] * stride_old_dt_head
        + (write_offset + offs_t)[None, :] * stride_old_dt_T
    )
    tl.store(old_dt_write_addrs, dt_new, mask=t_mask[None, :])

    old_dA_cumsum_write_addrs = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + write_buf * stride_old_dA_cumsum_dbuf
        + heads_block[:, None] * stride_old_dA_cumsum_head
        + (write_offset + offs_t)[None, :] * stride_old_dA_cumsum_T
    )
    tl.store(
        old_dA_cumsum_write_addrs,
        dA_cumsum_step + dA_cumsum_prefix[:, None],
        mask=t_mask[None, :],
    )

    # ---- Work independent of conv1d ----
    # Load historical cache and build combo_block before gdc_wait so this
    # work can overlap the upstream conv1d latency.
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

    # combo_block stays in registers across gdc_wait and is used directly
    # after the wait to compute rect_CB_scaled.

    # Per-head read bases (H,) - broadcast with offs_k for 2D loads.
    old_dt_read_h = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + buf_active * stride_old_dt_dbuf
        + heads_block * stride_old_dt_head
    )
    old_dA_cumsum_read_h = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + buf_active * stride_old_dA_cumsum_dbuf
        + heads_block * stride_old_dA_cumsum_head
    )

    # (H, K) loads at [0, PNAT) — old data from previous step.
    hk_mask = is_old_k[None, :]  # (1, K)
    old_dt_all = tl.load(
        old_dt_read_h[:, None] + safe_old_k[None, :] * stride_old_dt_T,
        mask=hk_mask, other=0.0,
    ).to(tl.float32)
    old_dA_cumsum_all = tl.load(
        old_dA_cumsum_read_h[:, None] + safe_old_k[None, :] * stride_old_dA_cumsum_T,
        mask=hk_mask, other=0.0,
    ).to(tl.float32)
    # Use loop-1 registers for this step's newly appended tokens.  These are
    # exactly the values stored above at [PNAT, PNAT+T); reloading them here
    # would read bytes this same kernel just wrote, which Triton does not
    # guarantee to fence.
    ht_mask = t_mask[None, :]  # (1, T)
    dA_cumsum_new = dA_cumsum_step + dA_cumsum_prefix[:, None]  # (H, T)

    # Production uses matching padded T/K sizes, where tl.gather is the cheap
    # path.  Some tests use a larger padded K than T; Triton rejects that gather
    # axis mismatch, so use a slower one-hot sum fallback for those cases.  The
    # wrapper passes this as an explicit constexpr so fast-path compilations do
    # not carry the fallback branch.
    if USE_GATHER_FOR_NEW_TOKENS:
        new_token_gather_idx = tl.broadcast_to(
            safe_k_new[None, :], (HEADS_PER_BLOCK, BLOCK_SIZE_K)
        )
        dt_at_kn = tl.where(
            is_new_k[None, :],
            tl.gather(dt_new, new_token_gather_idx, axis=1),
            0.0,
        )  # (H, K)
        dA_cumsum_at_kn = tl.where(
            is_new_k[None, :],
            tl.gather(dA_cumsum_new, new_token_gather_idx, axis=1),
            0.0,
        )  # (H, K)
    else:
        new_token_selector = (
            (offs_t[:, None] == (offs_k[None, :] - prev_num_accepted_tokens))
            & is_new_k[None, :]
        )
        dt_at_kn = tl.sum(
            tl.where(new_token_selector[None, :, :], dt_new[:, :, None], 0.0),
            axis=1,
        )  # (H, K)
        dA_cumsum_at_kn = tl.sum(
            tl.where(new_token_selector[None, :, :], dA_cumsum_new[:, :, None], 0.0),
            axis=1,
        )  # (H, K)

    # The write buffer stores continuous cumsum, so decay_vec_full[t] is
    # exp(continuous_cumsum[PNAT+t]) directly.
    decay_vec_full_block = tl.exp(dA_cumsum_new)  # (H, T)
    decay_vec_addrs = (
        decay_vec_ptr
        + pid_b * stride_dv_batch
        + heads_block[:, None] * stride_dv_head
        + offs_t[None, :] * stride_dv_t
    )  # (H, T)
    tl.store(decay_vec_addrs, decay_vec_full_block, mask=ht_mask)

    # combo_block[t, k] = dt[k] * exp(cumsum[t] - cumsum[k]).
    # Old and new K positions share the same formula because both halves use
    # continuous cumsum values.
    factor_dt = tl.where(is_old_k[None, :], old_dt_all, dt_at_kn)  # (H, K)
    s_k = tl.where(
        is_old_k[None, :],
        -old_dA_cumsum_all,
        -dA_cumsum_at_kn,
    )  # (H, K)
    # exp_diff (H, T, K) = exp(s_k (H, 1, K) + dA_cumsum_new (H, T, 1)).
    exp_diff = tl.exp(s_k[:, None, :] + dA_cumsum_new[:, :, None])
    combo_block = factor_dt[:, None, :] * exp_diff  # (H, T, K)

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
    # Disjoint masks: old at [0, PNAT), new at [PNAT, PNAT+T).
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
    # New tokens occupy runtime positions [PNAT, PNAT+T).
    t_idx_2d = offs_t[:, None]
    k_idx_2d = offs_k[None, :]
    is_old_k_2d = k_idx_2d < prev_num_accepted_tokens
    k_new_idx_2d = k_idx_2d - prev_num_accepted_tokens
    is_new_causal_2d = (k_new_idx_2d >= 0) & (k_new_idx_2d < T) & (k_new_idx_2d <= t_idx_2d)
    causal_combined = (is_old_k_2d | is_new_causal_2d) & t_mask[:, None]

    # Post-wait vectorized: combo_block (H, T, K) is still live in registers.
    # rect_CB_scaled = where(causal, raw_rect_CB * combo_block, 0); store as
    # one (H, T, K) tile.
    rect_CB_scaled_block = tl.where(
        causal_combined[None, :, :],
        raw_rect_CB[None, :, :] * combo_block,
        0.0,
    )  # (H, T, K)
    cb_scaled_addrs = (
        cb_scaled_ptr
        + pid_b * stride_cb_batch
        + heads_block[:, None, None] * stride_cb_head
        + offs_t[None, :, None] * stride_cb_t
        + offs_k[None, None, :] * stride_cb_j
    )  # (H, T, K)
    cb_store_mask_3d = (
        (offs_t[None, :, None] < BLOCK_SIZE_T)
        & (offs_k[None, None, :] < BLOCK_SIZE_K)
    )  # (1, T, K) → broadcasts to (H, T, K)
    tl.store(cb_scaled_addrs, rect_CB_scaled_block, mask=cb_store_mask_3d)


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
    RECTANGLE_USE_GATHER: tl.constexpr,
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
            RECTANGLE_USE_GATHER,
        )


# Main kernel: tl.dot replay + precomputed CB output.
# Grid: (cdiv(dim, M), batch, nheads).


@triton.jit()
def _persistent_main_impl(
    # Per-work-unit indices (computed by the persistent wrapper).
    # `pid_b` indexes the row in decode-batch order. Cache metadata is already
    # resolved by the persistent wrapper.
    pid_m,
    pid_b,
    pid_h,
    cache_batch_idx,
    active_buf,
    prev_num_accepted_tokens,
    # Pointers
    state_ptr,
    # state_tma_descriptor: TMA tensor_descriptor over state's flat 2D view, or
    # the same `state_ptr` tensor when neither USE_TMA_LOAD_WRITE/NOWRITE nor
    # USE_TMA_STORE is enabled (kernel ignores it via constexpr).
    state_tma_descriptor,
    state_scales_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    rand_seed_ptr,
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
    # IS_DYNAMIC: when True (persistent_dynamic), is_write is per-slot from
    # PNAT. When False (persistent_main), is_write is constexpr from
    # WRITE_CHECKPOINT. See also WRITE_CHECKPOINT_IS_CONSTEXPR below.
    IS_DYNAMIC: tl.constexpr,
    # WRITE_CHECKPOINT_IS_CONSTEXPR: when True, force is_write to the
    # WRITE_CHECKPOINT constexpr even in persistent_dynamic mode. The rectangle
    # path uses this for the write arm after the outer kernel has already
    # narrowed the runtime branch to write slots. When False,
    # persistent_dynamic keeps one body with a runtime PNAT check.
    WRITE_CHECKPOINT_IS_CONSTEXPR: tl.constexpr = False,
    # TMA flags — picked inside body based on is_write.  When is_write is
    # constexpr (either IS_DYNAMIC=False or WRITE_CHECKPOINT_IS_CONSTEXPR=True), the
    # use_tma_load = USE_TMA_LOAD_WRITE if is_write else USE_TMA_LOAD_NOWRITE
    # ternary constexpr-folds and only one TMA load form survives.
    USE_TMA_LOAD_WRITE: tl.constexpr = False,
    USE_TMA_LOAD_NOWRITE: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
):
    # IS_DYNAMIC: kernel-mode label, used by the outer _persistent_main_kernel
    # to decide slot-range derivation and outer is_w dispatch strategy
    # (constexpr WC for persistent_main; runtime is_w split -> 2 specialized
    # impl calls for persistent_dynamic).  Inside this impl, IS_DYNAMIC is
    # NOT consulted at runtime -- WRITE_CHECKPOINT is the only constexpr that
    # gates the write/nowrite codegen, in BOTH modes.

    # Compile-time invariant: QUANT_MAX > 0 must coincide with a quantized
    # state dtype (int8 / int16 / float8e4nv) and only those.
    tl.static_assert(
        (QUANT_MAX > 0.0)
        == (
            (state_ptr.dtype.element_ty == tl.int8)
            or (state_ptr.dtype.element_ty == tl.int16)
            or (state_ptr.dtype.element_ty == tl.float8e4nv)
        ),
        "QUANT_MAX > 0.0 must coincide with int8 / int16 / float8e4nv state dtype.",
    )

    # Resolve is_write: see WRITE_CHECKPOINT_IS_CONSTEXPR / IS_DYNAMIC docs in the param
    # list above.  Three cases:
    #   - WRITE_CHECKPOINT_IS_CONSTEXPR=True (RECT=1 is_w=True arm callers): use WRITE_CHECKPOINT
    #     constexpr.  Caller knows the slot needs write; inner DCEs nowrite
    #     paths while still constexpr-DCEing the nowrite half.
    #   - IS_DYNAMIC=True (RECT=0 caller, persistent_dynamic): runtime
    #     branch on PNAT. Both write and nowrite codegen live in one body.
    #   - IS_DYNAMIC=False (persistent_main): WRITE_CHECKPOINT constexpr from caller.
    if WRITE_CHECKPOINT_IS_CONSTEXPR:
        is_write: tl.constexpr = WRITE_CHECKPOINT
    elif IS_DYNAMIC:
        is_write = (prev_num_accepted_tokens + T) > MAX_REPLAY_BUFFER_LENGTH
    else:
        is_write = WRITE_CHECKPOINT
    if is_write:
        write_offset = 0
    else:
        write_offset = prev_num_accepted_tokens

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_window = tl.arange(0, BLOCK_SIZE_WINDOW)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    # Load state.  state_tma_descriptor is a host-built tensor_descriptor
    # over a flat (cache*nheads*dim, dstate) view of state when any TMA
    # path is enabled; raw `state_ptr` is the underlying tensor and is
    # always passed.  state_ptrs / state_ptr_raw are the raw-pointer view
    # used for !TMA load and store paths.  offs_y is the flat row index
    # for TMA load/store; computed unconditionally (cheap int math; DCE'd
    # when no TMA path is reachable).
    state_mask = m_mask[:, None] & n_mask[None, :]
    offs_y = (
        cache_batch_idx.to(tl.int32) * (stride_state_batch // stride_state_dim).to(tl.int32)
        + pid_h * dim
        + pid_m * BLOCK_SIZE_M
    )
    state_ptr_raw = state_ptr + cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    state_ptrs = (
        state_ptr_raw + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    # Load state.  Branch on is_write (constexpr = WRITE_CHECKPOINT in BOTH
    # modes after the outer-dispatch refactor), then constexpr-pick TMA-vs-
    # tl.load per side.  Outer `if` DCE's, only the matching side's
    # constexpr-gated load survives -- same compile-time picking for both
    # persistent_main and persistent_dynamic (the latter dispatches at the
    # outer kernel level so each impl instance sees a constexpr WC).
    if is_write:
        if USE_TMA_LOAD_WRITE:
            state = state_tma_descriptor.load([offs_y, 0]).to(tl.float32)
        else:
            state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
    else:
        if USE_TMA_LOAD_NOWRITE:
            state = state_tma_descriptor.load([offs_y, 0]).to(tl.float32)
        else:
            state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)
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

    old_window_mask = offs_window < prev_num_accepted_tokens

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

    prev_k_idx = tl.minimum(
        tl.maximum(prev_num_accepted_tokens - 1, 0), MAX_REPLAY_BUFFER_LENGTH - 1
    )
    total_dA_cumsum = tl.load(old_dA_cumsum_base + prev_k_idx * stride_old_dA_cumsum_T).to(
        tl.float32
    )

    coeff = tl.exp(total_dA_cumsum - old_dA_cumsum_all) * old_dt_all

    old_x_base = old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head
    old_x_all = tl.load(
        old_x_base + offs_window[:, None] * stride_old_x_T + offs_m[None, :] * stride_old_x_dim,
        mask=old_window_mask[:, None] & m_mask[None, :],
        other=0.0,
    )

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

    dB_scaled = coeff[:, None] * old_B_all

    total_decay = tl.where(prev_num_accepted_tokens > 0, tl.exp(total_dA_cumsum), 1.0)
    state *= total_decay

    state += tl.dot(tl.trans(old_x_all).to(tl.bfloat16), dB_scaled.to(tl.bfloat16))

    if is_write:
        if USE_RS_ROUNDING:
            # Generate random tensor for stochastic rounding.  The amount of
            # randomness needed depends on the SR codegen path:
            #   fp16 SR  (cvt.rs.f16x2):            1 b32 per 2 outputs (pack=2)
            #   fp8  SR  (cvt.rs.satfinite.e4m3x4): 1 b32 per 4 outputs (pack=4)
            #   int8 SR  (16b chunks + bitrev16):   1 b32 per 4 outputs
            #   int16 SR (24b + bitrev32):          1 b32 per 2 outputs
            # The PTX cvt.rs.* instructions consume a single 32-bit random
            # and split the bits internally for 2 or 4 conversions.  Generate
            # only what's actually consumed and broadcast to fill the unused
            # slots — saves Philox rounds proportionally.
            if QUANT_MAX > 0.0 and state_ptrs.dtype.element_ty == tl.float8e4nv:
                RAND_DIVISOR: tl.constexpr = 4  # fp8 SR
            elif QUANT_MAX > 0.0 and state_ptrs.dtype.element_ty == tl.int8:
                RAND_DIVISOR: tl.constexpr = 4  # int8 SR
            elif QUANT_MAX > 0.0 and state_ptrs.dtype.element_ty == tl.int16:
                RAND_DIVISOR: tl.constexpr = 2  # int16 SR
            elif QUANT_MAX == 0.0:
                RAND_DIVISOR: tl.constexpr = 2  # fp16 SR (only fp16 supported here)
            else:
                RAND_DIVISOR: tl.constexpr = 1  # unreachable; keeps constexpr initialized

            rand_seed = tl.load(rand_seed_ptr)
            base_rand = cache_batch_idx * stride_state_batch + pid_h * stride_state_head
            # Number of unique randoms per row = dstate / RAND_DIVISOR.
            # randint4x emits 4 randoms per offset, so use that / 4 offsets.
            offs_n_q = tl.arange(0, BLOCK_SIZE_DSTATE // (4 * RAND_DIVISOR))
            rand_offsets_q = (
                base_rand
                + offs_m[:, None] * stride_state_dim
                + offs_n_q[None, :] * (stride_state_dstate * 4 * RAND_DIVISOR)
            )  # (M, dstate / (4*RAND_DIVISOR))
            if PHILOX_ROUNDS > 0:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q, PHILOX_ROUNDS)
            else:
                r0, r1, r2, r3 = tl.randint4x(rand_seed, rand_offsets_q)
            r01 = tl.join(r0, r1)
            r23 = tl.join(r2, r3)
            r0123 = tl.join(r01, r23)
            rand_compact = tl.reshape(
                r0123, (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE // RAND_DIVISOR)
            )
            # Broadcast each unique rand to RAND_DIVISOR adjacent positions.
            # Pack-group (pack=2 fp16 / pack=4 fp8) consumes adjacent positions;
            # the unique rand lands at the asm's read slot; duplicates feed
            # the dead slots.  Triton's broadcast_to is stride-0 in IR.
            if RAND_DIVISOR > 1:
                rand_3d = rand_compact[:, :, None]
                rand_3d = tl.broadcast_to(
                    rand_3d,
                    (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE // RAND_DIVISOR, RAND_DIVISOR),
                )
                rand = tl.reshape(rand_3d, (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE))
            else:
                rand = rand_compact

        if QUANT_MAX > 0.0:
            amax = tl.max(tl.abs(state), axis=1)
            encode_scale = tl.where(amax == 0.0, 1.0, QUANT_MAX / amax)
            decode_scale = 1.0 / encode_scale
            state_scales_ptrs = (
                state_scales_ptr
                + cache_batch_idx * stride_state_scales_cache
                + pid_h * stride_state_scales_head
                + offs_m * stride_state_scales_dim
            )
            tl.store(state_scales_ptrs, decode_scale, mask=m_mask)
            state_q = state * encode_scale[:, None]
            if USE_RS_ROUNDING and (state_ptrs.dtype.element_ty == tl.float8e4nv):
                _state_q_fp8sr = _stochastic_round_fp8x4_e4m3(state_q, rand)
                if USE_TMA_STORE:
                    state_tma_descriptor.store([offs_y, 0], _state_q_fp8sr)
                else:
                    tl.store(state_ptrs, _state_q_fp8sr, mask=state_mask)
            else:
                if USE_RS_ROUNDING:
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized SR fall-through expects int8 or int16; "
                        "fp8 SR is handled by the prior branch.",
                    )
                    if state_ptrs.dtype.element_ty == tl.int8:
                        state_q = _stochastic_round_int8_packed(
                            state_q, rand, offs_n[None, :]
                        )
                    else:
                        state_q = _stochastic_round_int16_packed(
                            state_q, rand, offs_n[None, :]
                        )
                elif state_ptrs.dtype.element_ty != tl.float8e4nv:
                    tl.static_assert(
                        (state_ptrs.dtype.element_ty == tl.int8)
                        or (state_ptrs.dtype.element_ty == tl.int16),
                        "Quantized RN with explicit round() expects int8 or int16.",
                    )
                    state_q = tl.extra.cuda.libdevice.round(state_q)
                state_q = tl.minimum(tl.maximum(state_q, -QUANT_MAX), QUANT_MAX)
                _state_q_cast = state_q.to(state_ptrs.dtype.element_ty)
                if USE_TMA_STORE:
                    state_tma_descriptor.store([offs_y, 0], _state_q_cast)
                else:
                    tl.store(state_ptrs, _state_q_cast, mask=state_mask)
        elif USE_RS_ROUNDING:
            tl.static_assert(
                state_ptrs.dtype.element_ty == tl.float16,
                "Non-quantized SR only supports fp16 state.",
            )
            _state_sr = _stochastic_round_fp16x2(state, rand)
            if USE_TMA_STORE:
                state_tma_descriptor.store([offs_y, 0], _state_sr)
            else:
                tl.store(state_ptrs, _state_sr, mask=state_mask)
        else:
            _state_cast = state.to(state_ptrs.dtype.element_ty)
            if USE_TMA_STORE:
                state_tma_descriptor.store([offs_y, 0], _state_cast)
            else:
                tl.store(state_ptrs, _state_cast, mask=state_mask)

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

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

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
    tl.store(
        old_x_base
        + (write_offset + offs_t)[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

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

    init_out = tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16)) * decay_vec[:, None]
    cb_out = tl.dot(CB_scaled.to(tl.bfloat16), x_all.to(tl.bfloat16))
    out_all = init_out + cb_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        z_all = tl.load(
            z_ptr + offs_t[:, None] * stride_z_T + offs_m[None, :] * stride_z_dim,
            mask=t_mask[:, None] & m_mask[None, :], other=0.0,
        ).to(tl.float32)
        out_all_z = out_all * z_all * tl.sigmoid(z_all)
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all_z, mask=t_mask[:, None] & m_mask[None, :])
    else:
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# `_persistent_rectangle_impl`: rectangle nowrite path for the persistent
# kernel.  Body is a copy of `_rectangle_main_impl` with `pid_m`/`pid_b`/`pid_h`
# lifted to args (same pattern as `_persistent_main_impl` vs `_replay_main_impl`).
# Called only for nowrite slots when the kernel runs with RECTANGLE=True.
# Dropped from the rect impl: LAUNCH_DEPENDENT_KERNELS / REVERSE_PERM
# (kernel-level, signalled once at top); the wrapper resolves replay metadata.
@triton.jit()
def _persistent_rectangle_impl(
    # Per-work-unit indices (computed by the persistent wrapper).
    pid_m,
    pid_b,
    pid_h,
    cache_batch_idx,
    active_buf,
    prev_num_accepted_tokens,
    # Pointers
    state_ptr,
    # state_tma_descriptor: TMA tensor_descriptor (same flat 2D view as
    # replay path).  Used when USE_TMA_LOAD; ignored otherwise.
    state_tma_descriptor,
    state_scales_ptr,        # only consulted when QUANT_MAX > 0
    old_x_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
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
    USE_TMA_LOAD: tl.constexpr = False,
):
    # Nowrite-only: write_offset = PNAT (new tokens append at [PNAT, PNAT+T)).
    write_offset = prev_num_accepted_tokens

    # Rectangle K-axis layout: old at [0, PNAT), new at [PNAT, PNAT+T).

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    # K-axis masks (approach C: PNAT-runtime offset, matches precompute).
    is_old_k = offs_k < prev_num_accepted_tokens
    safe_old_k = tl.where(is_old_k, offs_k, 0)
    k_new_idx = offs_k - prev_num_accepted_tokens
    is_new_k = (k_new_idx >= 0) & (k_new_idx < T)
    safe_k_new = tl.where(is_new_k, k_new_idx, 0)

    # Load state.  Quant scale hoist: defer `* decode_scale` post-matmul.
    if USE_TMA_LOAD:
        offs_y = (
            cache_batch_idx.to(tl.int32) * (stride_state_batch // stride_state_dim).to(tl.int32)
            + pid_h * dim
            + pid_m * BLOCK_SIZE_M
        )
        state = state_tma_descriptor.load([offs_y, 0])
    else:
        state_ptr_local = state_ptr + cache_batch_idx * stride_state_batch + pid_h * stride_state_head
        state_ptrs = (
            state_ptr_local + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )
        state_mask = m_mask[:, None] & n_mask[None, :]
        state = tl.load(state_ptrs, mask=state_mask, other=0.0)
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
    else:
        state = state.to(tl.float32)

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

    # Hoist: old_x doesn't depend on conv1d/precompute; load before gdc_wait.
    old_x_load = tl.load(
        old_x_base
        + safe_old_k[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=is_old_k[:, None] & m_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    C_all = tl.load(
        C_ptr + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    x_K = tl.load(
        x_ptr + safe_k_new[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=is_new_k[:, None] & m_mask[None, :],
        other=0.0,
    )
    tl.store(
        old_x_base
        + offs_k[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_K,
        mask=is_new_k[:, None] & m_mask[None, :],
    )

    x_K_f32 = x_K.to(tl.float32)
    x_combined = old_x_load + x_K_f32

    if HAS_D or HAS_Z:
        sel_tk = (offs_t[:, None] == (offs_k[None, :] - prev_num_accepted_tokens))
        x_all = tl.dot(sel_tk.to(tl.bfloat16), x_K.to(tl.bfloat16))
    else:
        x_all = x_K_f32  # placeholder; unused

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

    state_out = (
        tl.dot(C_all.to(tl.bfloat16), tl.trans(state).to(tl.bfloat16))
        * decay_vec_full[:, None]
    )
    if QUANT_MAX > 0.0:
        state_out = state_out * decode_scale[None, :]

    token_out = tl.dot(CB_scaled.to(tl.bfloat16), x_combined.to(tl.bfloat16))

    out_all = state_out + token_out

    if HAS_D:
        out_all = out_all + x_all * D[None, :]

    if HAS_Z:
        z_all = tl.load(
            z_ptr + offs_t[:, None] * stride_z_T + offs_m[None, :] * stride_z_dim,
            mask=t_mask[:, None] & m_mask[None, :], other=0.0,
        ).to(tl.float32)
        out_all_z = out_all * z_all * tl.sigmoid(z_all)
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all_z, mask=t_mask[:, None] & m_mask[None, :])
    else:
        out_all_ptrs = out_ptr + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
        tl.store(out_all_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


# Persistent main kernel: 1D grid, persistent CTA loop.
# Heuristics mirror those of the replay main kernel.
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
@triton.heuristics(
    {"NUM_PID_M_BLOCKS": lambda args: triton.cdiv(args["dim"], args["BLOCK_SIZE_M"])}
)
@triton.jit()
def _persistent_main_kernel(
    # Pointers
    state_ptr,
    # state_tma_descriptor: TMA tensor_descriptor over state's flat 2D view.
    # Shared across BOTH the replay path (consumed by _persistent_main_impl
    # when USE_TMA_LOAD_*/STORE) AND the rectangle path (consumed by
    # _persistent_rectangle_impl when USE_TMA_LOAD) — same descriptor, same
    # block_shape, just gated by separate constexprs per impl.  Wrapper sets
    # this to a TensorDescriptor when ANY of the three TMA flags is on, else
    # to `state_ptr` (raw); each impl ignores it via its own constexpr when
    # not consuming it.
    state_tma_descriptor,
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
    replay_work_items_ptr,
    rand_seed_ptr,
    pad_slot_id,
    # Persistent-loop work-distribution scalars.  Caller pre-sorts the batch
    # write-first; the kernel uses (n_writes, batch_total, WRITE_CHECKPOINT)
    # to derive its own slot range.  Write half processes [0, n_writes),
    # nowrite half processes [n_writes, batch_total).
    #
    # n_writes_ptr is a device pointer to a (1,) int32 tensor. Reading from
    # device memory keeps the pointer stable across CUDA graph replay while
    # allowing the value to change between iterations.
    # When IS_DYNAMIC=True the value is unused (Triton DCEs the load).
    n_writes_ptr,   # int32 *: device-side count of write-mode slots
    batch_total,    # int32: total slot count
    nheads,         # int32: total head count (== _replay_main_impl's program_id axis 2 count)
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
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    # NUM_PERSISTENT: runtime int (not constexpr).  Used ONLY as the loop
    # stride in `tl.range(pid, total_work, NUM_PERSISTENT, ...)`.  Making it
    # runtime collapses the cta_per_sm tuning dim from the kernel's compile
    # signature: 8 CPS values used to mean 8x recompiles; now they share one
    # compiled kernel.  Work decomposition (pid_m, pid_b_local, pid_h) does
    # NOT depend on NUM_PERSISTENT — it uses constexpr NUM_PID_M_BLOCKS and
    # runtime n_slots_local — so loop unrolling and flatten=/num_stages=/
    # warp_specialize= optimizations on `tl.range` operate independently of
    # the stride value.
    NUM_PERSISTENT,
    NUM_LOOP_STAGES: tl.constexpr,
    NUM_PID_M_BLOCKS: tl.constexpr,
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    IS_DYNAMIC: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr = 16,  # rectangle K-axis (heuristic-derived)
    RECTANGLE: tl.constexpr = False,  # when True, dispatch nowrite slots to _persistent_rectangle_impl
    # 3 TMA toggles per the 3 live paths per-compilation:
    #   USE_TMA_LOAD_WRITE   — SSM state load when is_write
    #   USE_TMA_LOAD_NOWRITE — nowrite-path state load (rect when RECTANGLE,
    #                          else replay-nowrite)
    #   USE_TMA_STORE        — SSM state store (only fires on write
    #                          path; no-op when not is_write)
    # Wrapper picks USE_TMA_LOAD_NOWRITE = _use_tma_rect_load (if rectangle)
    # or _use_tma_replay_nowrite_load (if not).
    USE_TMA_LOAD_WRITE: tl.constexpr = False,
    USE_TMA_LOAD_NOWRITE: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    USE_REPLAY_CACHE_SLOT: tl.constexpr = True,
):
    # PDL signal: fire once at kernel entry (not per work unit).
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    # Load runtime n_writes from device memory.  Read once at kernel entry;
    # used only by the !IS_DYNAMIC slot-range derivation below.  Triton
    # DCEs the load when IS_DYNAMIC=True (n_writes is dead there).
    n_writes = tl.load(n_writes_ptr)

    # Derive this kernel's slot range.  Two modes:
    #   IS_DYNAMIC=False (persistent_main): caller pre-sorts and splits halves;
    #     slot range is [0, n_writes) when WRITE_CHECKPOINT else [n_writes, batch_total)
    #   IS_DYNAMIC=True  (persistent_dynamic): single launch covers full batch;
    #     each work-item dispatches via runtime PNAT check inside the impl.
    if IS_DYNAMIC:
        slot_lo = 0
        slot_hi = batch_total
    else:
        if WRITE_CHECKPOINT:
            slot_lo = 0
            slot_hi = n_writes
        else:
            slot_lo = n_writes
            slot_hi = batch_total
    n_slots_local = slot_hi - slot_lo

    pid = tl.program_id(axis=0)
    total_work = n_slots_local * NUM_PID_M_BLOCKS * nheads

    # Persistent loop.  Decompose tile_id into (pid_h, pid_b_local, pid_m)
    # with pid_m varying fastest (M-tile cache locality on state load), then
    # slot, then head — mirrors the existing 3D grid's axis ordering
    # (axis=0 fastest = pid_m).
    for tile_id in tl.range(
        pid, total_work, NUM_PERSISTENT,
        flatten=FLATTEN, num_stages=NUM_LOOP_STAGES, warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m = tile_id % NUM_PID_M_BLOCKS
        pid_b_local = (tile_id // NUM_PID_M_BLOCKS) % n_slots_local
        pid_h = tile_id // (NUM_PID_M_BLOCKS * n_slots_local)
        work_item_idx = pid_b_local + slot_lo
        if IS_DYNAMIC:
            pid_b = work_item_idx
            if HAS_CACHE_BATCH_INDICES:
                cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
                is_pad = cache_batch_idx == pad_slot_id
            else:
                cache_batch_idx = pid_b.to(tl.int64)
                is_pad = False
            active_buf = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
            pnat = tl.load(prev_num_accepted_tokens_ptr + cache_batch_idx)
        else:
            work_item_base = replay_work_items_ptr + work_item_idx * _REPLAY_WORK_ITEM_WIDTH
            pid_b = tl.load(
                work_item_base + _REPLAY_WORK_POSITION_IN_DECODE_BATCH
            )
            if USE_REPLAY_CACHE_SLOT:
                cache_batch_idx = tl.load(
                    work_item_base + _REPLAY_WORK_CACHE_SLOT
                ).to(tl.int64)
            else:
                cache_batch_idx = work_item_idx.to(tl.int64)
            pnat = tl.load(work_item_base + _REPLAY_WORK_PNAT)
            active_buf = tl.load(
                work_item_base + _REPLAY_WORK_CACHE_BUF_IDX
            ).to(tl.int32)
            is_pad = cache_batch_idx == pad_slot_id

        if not is_pad:
            # Dispatch: when RECTANGLE is set, send nowrite slots to the rectangle
            # impl.  `replay_work_items` carries the cache slot, PNAT and active
            # buffer for persistent_main; persistent_dynamic resolves those once
            # here from the existing tensors.
            if RECTANGLE:
                if IS_DYNAMIC:
                    is_w = (pnat + T) > MAX_REPLAY_BUFFER_LENGTH
                else:
                    is_w = WRITE_CHECKPOINT
                if is_w:
                    # Pass WRITE_CHECKPOINT=True constexpr to specialize this
                    # impl call for the write path.  Under IS_DYNAMIC=True, the
                    # kernel-level WRITE_CHECKPOINT is False (launcher default),
                    # but the OUTER is_w branch we are inside narrows the
                    # runtime path to writes-only, so we override to True here
                    # so the impl's constexpr-gated `if is_write:` blocks DCE
                    # to the write-only codegen.  Under IS_DYNAMIC=False
                    # (persistent_main), the kernel-level WRITE_CHECKPOINT is
                    # itself True for this half (write half launches with
                    # WRITE_CHECKPOINT=True), and the outer is_w = WRITE_CHECKPOINT = True
                    # constexpr-folds; passing literal True here is consistent
                    # and constexpr-equivalent.
                    _persistent_main_impl(
                        pid_m, pid_b, pid_h,
                        cache_batch_idx, active_buf, pnat,
                        state_ptr, state_tma_descriptor, state_scales_ptr,
                        old_x_ptr, old_B_ptr, old_dt_ptr, old_dA_cumsum_ptr,
                        x_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
                        cb_scaled_ptr, decay_vec_ptr,
                        rand_seed_ptr,
                        T, MAX_REPLAY_BUFFER_LENGTH, dim, dstate, nheads_ngroups_ratio,
                        stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
                        stride_state_scales_cache, stride_state_scales_head, stride_state_scales_dim,
                        stride_old_x_cache, stride_old_x_T, stride_old_x_head, stride_old_x_dim,
                        stride_old_B_cache, stride_old_B_dbuf, stride_old_B_T,
                        stride_old_B_group, stride_old_B_dstate,
                        stride_old_dt_cache, stride_old_dt_dbuf, stride_old_dt_head, stride_old_dt_T,
                        stride_old_dA_cumsum_cache, stride_old_dA_cumsum_dbuf,
                        stride_old_dA_cumsum_head, stride_old_dA_cumsum_T,
                        stride_x_batch, stride_x_T, stride_x_head, stride_x_dim,
                        stride_C_batch, stride_C_T, stride_C_group, stride_C_dstate,
                        stride_D_head, stride_D_dim,
                        stride_z_batch, stride_z_T, stride_z_head, stride_z_dim,
                        stride_out_batch, stride_out_T, stride_out_head, stride_out_dim,
                        stride_cb_batch, stride_cb_head, stride_cb_t, stride_cb_j,
                        stride_dv_batch, stride_dv_head, stride_dv_t,
                        BLOCK_SIZE_M, HAS_D, HAS_Z, HAS_CACHE_BATCH_INDICES,
                        BLOCK_SIZE_DSTATE, BLOCK_SIZE_T, BLOCK_SIZE_WINDOW,
                        LAUNCH_WITH_PDL, USE_RS_ROUNDING, PHILOX_ROUNDS, QUANT_MAX,
                        True, IS_DYNAMIC,  # WRITE_CHECKPOINT=True (write arm)
                        True,  # WRITE_CHECKPOINT_IS_CONSTEXPR
                        USE_TMA_LOAD_WRITE, USE_TMA_LOAD_NOWRITE, USE_TMA_STORE,
                    )
                else:
                    _persistent_rectangle_impl(
                        pid_m, pid_b, pid_h,
                        cache_batch_idx, active_buf, pnat,
                        state_ptr, state_tma_descriptor, state_scales_ptr,
                        old_x_ptr,
                        x_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
                        cb_scaled_ptr, decay_vec_ptr,
                        T, MAX_REPLAY_BUFFER_LENGTH, dim, dstate, nheads_ngroups_ratio,
                        stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
                        stride_state_scales_cache, stride_state_scales_head, stride_state_scales_dim,
                        stride_old_x_cache, stride_old_x_T, stride_old_x_head, stride_old_x_dim,
                        stride_x_batch, stride_x_T, stride_x_head, stride_x_dim,
                        stride_C_batch, stride_C_T, stride_C_group, stride_C_dstate,
                        stride_D_head, stride_D_dim,
                        stride_z_batch, stride_z_T, stride_z_head, stride_z_dim,
                        stride_out_batch, stride_out_T, stride_out_head, stride_out_dim,
                        stride_cb_batch, stride_cb_head, stride_cb_t, stride_cb_j,
                        stride_dv_batch, stride_dv_head, stride_dv_t,
                        BLOCK_SIZE_M, HAS_D, HAS_Z, HAS_CACHE_BATCH_INDICES,
                        BLOCK_SIZE_DSTATE, BLOCK_SIZE_T, BLOCK_SIZE_K,
                        LAUNCH_WITH_PDL, QUANT_MAX,
                        USE_TMA_LOAD_NOWRITE,  # rect-load TMA toggle
                    )
            else:
                _persistent_main_impl(
                    pid_m, pid_b, pid_h,
                    cache_batch_idx, active_buf, pnat,
                    state_ptr, state_tma_descriptor, state_scales_ptr,
                    old_x_ptr, old_B_ptr, old_dt_ptr, old_dA_cumsum_ptr,
                    x_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
                    cb_scaled_ptr, decay_vec_ptr,
                    rand_seed_ptr,
                    T, MAX_REPLAY_BUFFER_LENGTH, dim, dstate, nheads_ngroups_ratio,
                    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
                    stride_state_scales_cache, stride_state_scales_head, stride_state_scales_dim,
                    stride_old_x_cache, stride_old_x_T, stride_old_x_head, stride_old_x_dim,
                    stride_old_B_cache, stride_old_B_dbuf, stride_old_B_T,
                    stride_old_B_group, stride_old_B_dstate,
                    stride_old_dt_cache, stride_old_dt_dbuf, stride_old_dt_head, stride_old_dt_T,
                    stride_old_dA_cumsum_cache, stride_old_dA_cumsum_dbuf,
                    stride_old_dA_cumsum_head, stride_old_dA_cumsum_T,
                    stride_x_batch, stride_x_T, stride_x_head, stride_x_dim,
                    stride_C_batch, stride_C_T, stride_C_group, stride_C_dstate,
                    stride_D_head, stride_D_dim,
                    stride_z_batch, stride_z_T, stride_z_head, stride_z_dim,
                    stride_out_batch, stride_out_T, stride_out_head, stride_out_dim,
                    stride_cb_batch, stride_cb_head, stride_cb_t, stride_cb_j,
                    stride_dv_batch, stride_dv_head, stride_dv_t,
                    BLOCK_SIZE_M, HAS_D, HAS_Z, HAS_CACHE_BATCH_INDICES,
                    BLOCK_SIZE_DSTATE, BLOCK_SIZE_T, BLOCK_SIZE_WINDOW,
                    LAUNCH_WITH_PDL, USE_RS_ROUNDING, PHILOX_ROUNDS, QUANT_MAX,
                    WRITE_CHECKPOINT, IS_DYNAMIC,
                    False,  # WRITE_CHECKPOINT_IS_CONSTEXPR
                    USE_TMA_LOAD_WRITE, USE_TMA_LOAD_NOWRITE, USE_TMA_STORE,
                )


# ============================================================================
# Python wrapper
# ============================================================================


_QUANT_MAX_BY_DTYPE = {
    torch.int8: 127.0,
    torch.int16: 32767.0,
    torch.float8_e4m3fn: 448.0,
}


# ---------------------------------------------------------------------------
# Default tunings — looked up by (effective_batch, dtype, sr) when the caller
# leaves mode/knobs as None.
#
# Effective batch = raw_batch × nheads_per_rank.  Our sweep was at TP=8 with
# the standard Mamba2 nheads; at call time we compute it from the input
# tensor shape so callers at other TP / nheads pick up the right cell.
#
# Schema: dict[(dtype_str, sr_str)] → list[(eff_batch_threshold, mode, knobs)]
# sorted by threshold ascending.  Lookup finds the first threshold ≥ eff_b
# (so missing intermediate batches fall up to the next tuned cell).  If
# eff_b exceeds the largest threshold, use the largest entry.
#
# Each `knobs` dict only contains keys for the chosen mode; the wrapper
# unpacks them with the same name as the matching kwargs.  Caller-provided
# kwargs always win over table values.
#
# This table is intentionally NOT parameterized by T or max_window.  Our
# sweep was T=6, max_window=16.  Callers outside that regime silently get
# the same numbers — they may be suboptimal but they're correct.
#
# Source: audit_v2.py --emit-tuning.  Auto-generated from per-cell search
# winners (best of pd / pm by bucket_expected_renorm).  Sweep was TP=8 with
# NHEADS=128 → nheads_per_rank=16; thresholds are in effective_batch units.
# Missing dtype/SR combos (fp16/RN, int8/RN, fp8/*) fall back via the
# _resolve_tuning chain — RN→SR for same dtype, then fp8→int8/SR.
_DEFAULT_TUNING: dict[tuple[str, str], list[tuple[int, str, dict]]] = {
    ("fp32", "RN"): [
        (   16, "persistent_main", {'_block_size_m_nowrite': 16, '_block_size_m_write': 8, '_cta_per_sm_nowrite': 4, '_cta_per_sm_write': 1, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 1, '_num_stages_write': 2, '_num_warps_nowrite': 2, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1, score=6.22us
        (   32, "persistent_main", {'_block_size_m_nowrite': 16, '_block_size_m_write': 16, '_cta_per_sm_nowrite': 9, '_cta_per_sm_write': 7, '_flatten': False, '_heads_per_block': 1, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 4, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=2, score=7.17us
        (   64, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 9, '_cta_per_sm_write': 4, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 3, '_num_stages_nowrite': 1, '_num_stages_write': 2, '_num_warps_nowrite': 4, '_num_warps_write': 4, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=4, score=8.08us
        (  128, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 6, '_cta_per_sm_write': 9, '_flatten': False, '_heads_per_block': 8, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 2, '_num_stages_nowrite': 2, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': True, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': False}),  # raw_batch=8, score=9.00us
        (  256, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 9, '_cta_per_sm_write': 1, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 4, '_num_warps_write': 2, '_precompute_num_warps': 16, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=16, score=10.92us
        (  512, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 4, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 1, '_num_stages_nowrite': 2, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 2, '_precompute_num_warps': 16, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': False}),  # raw_batch=32, score=13.53us
        ( 1024, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 4, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 8, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=64, score=19.50us
        ( 2048, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 8, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 1, '_num_stages_nowrite': 3, '_num_stages_write': 3, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 2, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=128, score=30.28us
        ( 4096, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 4, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 2, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=256, score=50.32us
        ( 8192, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 1, '_num_stages_nowrite': 4, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 1, '_precompute_num_warps': 2, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=512, score=90.99us
        (16384, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 9, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 8, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 1, '_num_stages_nowrite': 1, '_num_stages_write': 3, '_num_warps_nowrite': 2, '_num_warps_write': 1, '_precompute_num_warps': 2, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1024, score=171.69us
    ],
    ("fp16", "SR"): [
        (   16, "persistent_main", {'_block_size_m_nowrite': 8, '_block_size_m_write': 16, '_cta_per_sm_nowrite': 5, '_cta_per_sm_write': 7, '_flatten': False, '_heads_per_block': 1, '_num_loop_stages_nowrite': 4, '_num_loop_stages_write': 3, '_num_stages_nowrite': 1, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 4, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1, score=6.16us
        (   32, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 16, '_cta_per_sm_nowrite': 9, '_cta_per_sm_write': 4, '_flatten': False, '_heads_per_block': 1, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 2, '_num_stages_write': 4, '_num_warps_nowrite': 4, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=2, score=7.01us
        (   64, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 16, '_cta_per_sm_nowrite': 5, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 1, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=4, score=7.95us
        (  128, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 8, '_cta_per_sm_write': 4, '_flatten': False, '_heads_per_block': 1, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 1, '_num_warps_nowrite': 4, '_num_warps_write': 4, '_precompute_num_warps': 4, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=8, score=8.87us
        (  256, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 2, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 1, '_num_stages_nowrite': 1, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 4, '_precompute_num_warps': 16, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': False}),  # raw_batch=16, score=10.28us
        (  512, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 6, '_flatten': False, '_heads_per_block': 8, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 1, '_num_stages_nowrite': 3, '_num_stages_write': 2, '_num_warps_nowrite': 1, '_num_warps_write': 2, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': False}),  # raw_batch=32, score=12.90us
        ( 1024, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 7, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 4, '_num_stages_write': 2, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 8, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=64, score=16.71us
        ( 2048, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 4, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 2, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=128, score=25.71us
        ( 4096, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 1, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=256, score=39.80us
        ( 8192, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 2, '_num_stages_write': 2, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 1, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=512, score=71.34us
        (16384, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 1, '_num_stages_nowrite': 2, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 1, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1024, score=133.51us
    ],
    ("int8", "SR"): [
        (   16, "persistent_main", {'_block_size_m_nowrite': 8, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 5, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 1, '_num_loop_stages_nowrite': 4, '_num_loop_stages_write': 2, '_num_stages_nowrite': 2, '_num_stages_write': 3, '_num_warps_nowrite': 2, '_num_warps_write': 4, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1, score=6.34us
        (   32, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 8, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 4, '_num_stages_write': 2, '_num_warps_nowrite': 4, '_num_warps_write': 4, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=2, score=7.36us
        (   64, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 2, '_cta_per_sm_write': 4, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 2, '_num_stages_write': 3, '_num_warps_nowrite': 4, '_num_warps_write': 4, '_precompute_num_warps': 8, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=4, score=8.40us
        (  128, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 10, '_flatten': False, '_heads_per_block': 2, '_num_loop_stages_nowrite': 2, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 4, '_num_warps_write': 4, '_precompute_num_warps': 16, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=8, score=9.37us
        (  256, "persistent_dynamic", {'_block_size_m': 16, '_cta_per_sm': 8, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages': 1, '_num_stages': 4, '_num_warps': 1, '_precompute_num_warps': 16, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': True, '_use_tma_replay_write_load': False, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': False}),  # raw_batch=16, score=10.02us
        (  512, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 16, '_cta_per_sm_nowrite': 7, '_cta_per_sm_write': 9, '_flatten': False, '_heads_per_block': 8, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 1, '_num_stages_nowrite': 2, '_num_stages_write': 3, '_num_warps_nowrite': 2, '_num_warps_write': 1, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=32, score=13.15us
        ( 1024, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 32, '_cta_per_sm_nowrite': 10, '_cta_per_sm_write': 7, '_flatten': False, '_heads_per_block': 16, '_num_loop_stages_nowrite': 1, '_num_loop_stages_write': 1, '_num_stages_nowrite': 4, '_num_stages_write': 3, '_num_warps_nowrite': 1, '_num_warps_write': 1, '_precompute_num_warps': 8, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': False, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=64, score=17.82us
        ( 2048, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 4, '_cta_per_sm_write': 3, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 4, '_precompute_num_warps': 1, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=128, score=27.01us
        ( 4096, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 4, '_cta_per_sm_write': 3, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 2, '_num_warps_nowrite': 2, '_num_warps_write': 4, '_precompute_num_warps': 1, '_use_tma_rect_load': False, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=256, score=43.23us
        ( 8192, "persistent_main", {'_block_size_m_nowrite': 32, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 8, '_cta_per_sm_write': 6, '_flatten': False, '_heads_per_block': 4, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 2, '_num_stages_nowrite': 3, '_num_stages_write': 1, '_num_warps_nowrite': 1, '_num_warps_write': 4, '_precompute_num_warps': 1, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=512, score=77.01us
        (16384, "persistent_main", {'_block_size_m_nowrite': 64, '_block_size_m_write': 64, '_cta_per_sm_nowrite': 6, '_cta_per_sm_write': 3, '_flatten': False, '_heads_per_block': 8, '_num_loop_stages_nowrite': 3, '_num_loop_stages_write': 2, '_num_stages_nowrite': 1, '_num_stages_write': 4, '_num_warps_nowrite': 2, '_num_warps_write': 4, '_precompute_num_warps': 1, '_use_tma_rect_load': True, '_use_tma_replay_nowrite_load': False, '_use_tma_replay_write_load': True, '_use_tma_replay_write_store': True, '_warp_specialize': False, 'rectangle_for_nowrite': True}),  # raw_batch=1024, score=140.43us
    ],
}


# Knob names that map between the modes' single-value (pd) and split-value
# (pm) namespaces.  Used by `_bridge_tuning_knobs` when caller forces a mode
# different from the table's recommendation.
_PD_TO_PM_SPLIT_MAP = {  # pd unsplit knob → (pm_write_knob, pm_nowrite_knob)
    "_block_size_m": ("_block_size_m_write", "_block_size_m_nowrite"),
    "_num_warps":    ("_num_warps_write", "_num_warps_nowrite"),
    "_num_stages":   ("_num_stages_write", "_num_stages_nowrite"),
    # CPS / LS are persistent-loop knobs; pd uses _cta_per_sm + _num_loop_stages
    # as unsplit, pm uses _cta_per_sm_write/_nowrite + _num_loop_stages_write/_nowrite.
    "_cta_per_sm":      ("_cta_per_sm_write", "_cta_per_sm_nowrite"),
    "_num_loop_stages": ("_num_loop_stages_write", "_num_loop_stages_nowrite"),
}


def _bridge_tuning_knobs(knobs: dict, from_mode: str, to_mode: str) -> dict:
    """Convert a tuning dict between pd ↔ pm knob namespaces.

    pd → pm: copy each unsplit value to both write/nowrite split knobs; drop
    the unsplit form (pm doesn't read it).
    pm → pd: take the nowrite split value as the unsplit knob; drop the
    write/nowrite split forms (pd doesn't read them).
    Shape knobs that exist in both modes (_heads_per_block, _flatten,
    _warp_specialize, TMA flags, rectangle_for_nowrite) carry over unchanged.
    """
    out = dict(knobs)
    if from_mode == "persistent_dynamic" and to_mode == "persistent_main":
        for unsplit, (pm_w, pm_nw) in _PD_TO_PM_SPLIT_MAP.items():
            if unsplit in out:
                out.setdefault(pm_w, out[unsplit])
                out.setdefault(pm_nw, out[unsplit])
                del out[unsplit]
    elif from_mode == "persistent_main" and to_mode == "persistent_dynamic":
        for unsplit, (pm_w, pm_nw) in _PD_TO_PM_SPLIT_MAP.items():
            if pm_nw in out:
                out.setdefault(unsplit, out[pm_nw])
                out.pop(pm_w, None)
                out.pop(pm_nw, None)
    return out


def _resolve_tuning(
    batch: int, nheads_per_rank: int, dt_str: str, sr_str: str,
) -> tuple[str, dict] | None:
    """Look up the default mode + knobs for this (eff_batch, dt, sr) cell.

    Returns (mode, knobs_dict) or None if the table has no entry covering
    this dtype/sr (including the fp8→int8/SR and dtype/RN→dtype/SR fallbacks).
    Returning None lets the wrapper fall back to caller-provided kwargs or
    kernel-side defaults.
    """
    eff_b = batch * max(1, nheads_per_rank)
    # Lookup chain.  Order:
    #   1. Exact (dt, sr).
    #   2. (dt, SR) if RN missing for that dtype.
    #   3. Cross-dtype fallback for dtypes we haven't tuned:
    #        bf16 / int16 → fp16/SR
    #        fp8          → int8/SR
    # Unknown dtype → raise.
    valid_dtypes = {"fp32", "fp16", "bf16", "int8", "int16", "fp8"}
    if dt_str not in valid_dtypes:
        raise ValueError(
            f"replay_selective_state_update: unsupported state dtype {dt_str!r}; "
            f"expected one of {sorted(valid_dtypes)}"
        )
    keys_to_try = [(dt_str, sr_str)]
    if sr_str == "RN":
        keys_to_try.append((dt_str, "SR"))
    if dt_str in ("bf16", "int16"):
        keys_to_try.append(("fp16", "SR"))
    elif dt_str == "fp8":
        keys_to_try.append(("int8", "SR"))
    entries = None
    for k in keys_to_try:
        if k in _DEFAULT_TUNING:
            entries = _DEFAULT_TUNING[k]
            break
    if entries is None:
        return None
    # Find first threshold ≥ eff_b; if none, use largest entry.
    for thresh, mode, knobs in entries:
        if eff_b <= thresh:
            return mode, dict(knobs)
    thresh, mode, knobs = entries[-1]
    return mode, dict(knobs)


def replay_selective_state_update(
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
    # Required persistent-mode plumbing (REQUIRED for both pd and pm; pd
    # ignores both internally but the wrapper still demands them):
    #   n_writes  : (1,) int32 device tensor with the count of write-mode
    #               slots in the batch.  pm uses it to size the two halves;
    #               pd ignores it (per-slot runtime PNAT check).
    n_writes: torch.Tensor,
    replay_work_items: torch.Tensor,
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
    rectangle_for_nowrite: bool | None = None,
    mode: str | None = None,
    _block_size_m: int | None = None,
    _num_warps: int | None = None,
    _num_stages: int | None = None,
    _precompute_num_warps: int | None = None,
    _precompute_num_stages: int | None = None,
    _heads_per_block: int | None = None,
    _maxnreg: int | None = None,
    _num_ctas: int | None = None,
    # Per-main knobs (override shared values for one half of the dl-family /
    # persistent_main launches).  Default None = tied to the shared value
    # (backward compat).  The two main kernels (write vs nowrite) have
    # different per-slot work — write does a state shift + store, nowrite
    # just appends — so the optimum (M, W, S, H) can differ.  Precompute
    # knobs are intentionally NOT split: shared precompute wins (cheaper
    # launch, hotter precompute outputs in L2).  Persistent CPS / LS knobs
    # are also split per-main since the two persistent_main launches have
    # different grid sizes.
    _block_size_m_write: int | None = None,
    _block_size_m_nowrite: int | None = None,
    _num_warps_write: int | None = None,
    _num_warps_nowrite: int | None = None,
    _num_stages_write: int | None = None,
    _num_stages_nowrite: int | None = None,
    # Note: heads_per_block / precompute_num_warps are NOT split — they only
    # affect the precompute kernel, which is shared across write/nowrite.
    # TMA state-tensor toggles — 4 independent paths (see replay design notes
    # item #17 for measured perf profiles).  Each is False=raw load/store, True=
    # use a host-built TMA tensor_descriptor for that path.
    _use_tma_rect_load: bool | None = None,           # rect kernel's state load (nowrite-only)
    _use_tma_replay_write_load: bool | None = None,   # SSM state load when WRITE_CHECKPOINT=True
    _use_tma_replay_write_store: bool | None = None,  # SSM state store when WRITE_CHECKPOINT=True
    _use_tma_replay_nowrite_load: bool | None = None, # SSM state load when WRITE_CHECKPOINT=False
    _use_replay_cache_slot: bool = True,
    # Persistent-mode tuning kwargs (consulted for both pd and pm; pd uses
    # _cta_per_sm / _num_loop_stages, pm uses the _write/_nowrite splits):
    # _cta_per_sm : int — CTAs per SM in the 1D persistent grid.  Internally
    #   expanded to `num_persistent = _cta_per_sm × NUM_SMS`.
    # _num_loop_stages : int — `num_stages` arg on the inner `tl.range(...)`
    #   persistent loop.  Note: this is loop-level, NOT the kernel-arg
    #   `num_stages` (which only pipelines dot-feeding loads).
    # _flatten : bool — `flatten` arg on `tl.range(...)`.
    # _warp_specialize : bool — `warp_specialize` arg on `tl.range(...)`.
    _cta_per_sm: int | None = None,
    _num_loop_stages: int | None = None,
    _flatten: bool | None = None,
    _warp_specialize: bool | None = None,
    # Per-main persistent-specific knobs.  Same rationale as the BLOCK_SIZE_M
    # split above: the two persistent_main launches (write half vs nowrite
    # half) have different grid sizes and per-work-item costs, so they may
    # want different cta_per_sm / num_loop_stages.
    _cta_per_sm_write: int | None = None,
    _cta_per_sm_nowrite: int | None = None,
    _num_loop_stages_write: int | None = None,
    _num_loop_stages_nowrite: int | None = None,
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

    Uses double-buffered cache tensors. cache_buf_idx[slot] indicates which
    buffer (0 or 1) to read from for replay. Checkpoint-write steps write the
    new history to the inactive buffer; no-write steps append to the active
    buffer. The caller must update cache_buf_idx and PNAT with the same
    checkpoint predicate used by the kernel.

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
            for state.dtype in (fp16, int8, int16, fp8_e4m3fn). fp16+SR and
            fp8+SR both require sm_100a (Blackwell B200+) — wrapper asserts
            this loudly.
        philox_rounds: number of Philox PRNG rounds (default 10).
        state_scales: required when state.dtype in (int8, int16, fp8_e4m3fn).
            Shape (cache_size, nheads, dim), fp32.  Per-(head, dim) channel
            decode scale (= 1 / encode_scale).  The kernel writes scales on
            replay-write steps and reads them on load (broadcast over dstate).
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
    #   mode=None (default): look up the table-tuned mode + knobs for this
    #       (effective_batch, dtype, sr) cell.  See `_resolve_tuning` above.
    #   mode="persistent_dynamic": single persistent-CTA kernel covering the
    #       full batch.  Each work-item dispatches via runtime PNAT check
    #       (is_write = (pnat + T) > MAX).  No write/nowrite split.
    #       replay_work_items is ignored.  write_checkpoint is ignored.
    #   mode="persistent_main": persistent-CTA kernel with two launches
    #       (write half + nowrite half).  Caller MUST pre-sort replay_work_items
    #       write-first; the n_writes tensor partitions the persistent loop
    #       into the two halves with the right WRITE_CHECKPOINT constexpr
    #       each time.  RECTANGLE constexpr (= rectangle_for_nowrite) picks
    #       rect vs replay for the nowrite half.  write_checkpoint is ignored.
    # Note: mode-and-knob resolution from the default-tuning table happens
    # below, after we have `batch` and `nheads`.

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
        else:
            assert state.dtype in (torch.int8, torch.int16), (
                "stochastic rounding is supported only for state.dtype in "
                f"(fp16, int8, int16, fp8_e4m3fn), got {state.dtype}."
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

    # --- Default-tuning lookup ---
    # Resolve (mode, knobs) from the table when caller leaves them None.
    # Caller-provided kwargs always win.  If the caller forces a mode that
    # differs from the table's recommendation for this cell, we BRIDGE the
    # table's knobs into the forced mode's knob namespace rather than fall
    # back to (likely-terrible) kernel defaults:
    #   table pd → forced pm: copy each unsplit pd knob (M, W, S, CPS, LS)
    #                         to both write and nowrite split knobs.
    #   table pm → forced pd: take the nowrite split values (Mnw, Wnw, Snw,
    #                         CPSnw, LSnw) as the unsplit knobs.
    # Empty table → no-op (caller passes whatever, mode falls back to pd).
    _dt_str = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.float8_e4m3fn: "fp8",
    }.get(state.dtype, str(state.dtype))
    _sr_str = "SR" if rand_seed is not None else "RN"
    _table_entry = _resolve_tuning(batch, nheads, _dt_str, _sr_str)
    if _table_entry is not None:
        _table_mode, _table_knobs = _table_entry
        if mode is None:
            mode = _table_mode
        if mode != _table_mode:
            # Bridge across modes — see header comment above.
            _table_knobs = _bridge_tuning_knobs(_table_knobs, _table_mode, mode)
        # Fill None-valued kwargs from table.  We can't reliably mutate
        # locals() for re-read, so re-bind each kwarg explicitly.
        if rectangle_for_nowrite is None and "rectangle_for_nowrite" in _table_knobs:
            rectangle_for_nowrite = bool(_table_knobs["rectangle_for_nowrite"])
        _block_size_m = _block_size_m if _block_size_m is not None else _table_knobs.get("_block_size_m")
        _num_warps = _num_warps if _num_warps is not None else _table_knobs.get("_num_warps")
        _num_stages = _num_stages if _num_stages is not None else _table_knobs.get("_num_stages")
        _heads_per_block = _heads_per_block if _heads_per_block is not None else _table_knobs.get("_heads_per_block")
        _precompute_num_warps = _precompute_num_warps if _precompute_num_warps is not None else _table_knobs.get("_precompute_num_warps")
        _precompute_num_stages = _precompute_num_stages if _precompute_num_stages is not None else _table_knobs.get("_precompute_num_stages")
        _block_size_m_write = _block_size_m_write if _block_size_m_write is not None else _table_knobs.get("_block_size_m_write")
        _block_size_m_nowrite = _block_size_m_nowrite if _block_size_m_nowrite is not None else _table_knobs.get("_block_size_m_nowrite")
        _num_warps_write = _num_warps_write if _num_warps_write is not None else _table_knobs.get("_num_warps_write")
        _num_warps_nowrite = _num_warps_nowrite if _num_warps_nowrite is not None else _table_knobs.get("_num_warps_nowrite")
        _num_stages_write = _num_stages_write if _num_stages_write is not None else _table_knobs.get("_num_stages_write")
        _num_stages_nowrite = _num_stages_nowrite if _num_stages_nowrite is not None else _table_knobs.get("_num_stages_nowrite")
        _cta_per_sm = _cta_per_sm if _cta_per_sm is not None else _table_knobs.get("_cta_per_sm")
        _num_loop_stages = _num_loop_stages if _num_loop_stages is not None else _table_knobs.get("_num_loop_stages")
        # persistent_main uses split write/nowrite tuning knobs.
        _num_loop_stages_write = (
            _num_loop_stages_write
            if _num_loop_stages_write is not None
            else _table_knobs.get("_num_loop_stages_write")
        )
        _num_loop_stages_nowrite = (
            _num_loop_stages_nowrite
            if _num_loop_stages_nowrite is not None
            else _table_knobs.get("_num_loop_stages_nowrite")
        )
        _cta_per_sm_write = (
            _cta_per_sm_write
            if _cta_per_sm_write is not None
            else _table_knobs.get("_cta_per_sm_write")
        )
        _cta_per_sm_nowrite = (
            _cta_per_sm_nowrite
            if _cta_per_sm_nowrite is not None
            else _table_knobs.get("_cta_per_sm_nowrite")
        )
        _flatten = _flatten if _flatten is not None else _table_knobs.get("_flatten")
        _warp_specialize = _warp_specialize if _warp_specialize is not None else _table_knobs.get("_warp_specialize")
        if _use_tma_rect_load is None:
            _use_tma_rect_load = bool(_table_knobs.get("_use_tma_rect_load", False))
        if _use_tma_replay_write_load is None:
            _use_tma_replay_write_load = bool(_table_knobs.get("_use_tma_replay_write_load", False))
        if _use_tma_replay_write_store is None:
            _use_tma_replay_write_store = bool(_table_knobs.get("_use_tma_replay_write_store", False))
        if _use_tma_replay_nowrite_load is None:
            _use_tma_replay_nowrite_load = bool(_table_knobs.get("_use_tma_replay_nowrite_load", False))
    # Final defaults if neither caller nor table set them (empty table case).
    if mode is None:
        mode = "persistent_dynamic"
    if rectangle_for_nowrite is None:
        rectangle_for_nowrite = False
    _use_tma_rect_load = bool(_use_tma_rect_load)
    _use_tma_replay_write_load = bool(_use_tma_replay_write_load)
    _use_tma_replay_write_store = bool(_use_tma_replay_write_store)
    _use_tma_replay_nowrite_load = bool(_use_tma_replay_nowrite_load)
    assert mode in ("persistent_dynamic", "persistent_main"), (
        f"unknown mode {mode!r}; expected 'persistent_dynamic' or 'persistent_main'"
    )
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
    # placeholder degenerate case max_window = T (every step writes replay
    # state).  For real replay-style history, max_window > T and
    # `prev_num_accepted_tokens` can be 0..max_window.  Window-axis kernel
    # tiles (BLOCK_SIZE_WINDOW, BLOCK_SIZE_K) are derived independently from
    # MAX_REPLAY_BUFFER_LENGTH so max_window can exceed BLOCK_SIZE_T freely.
    max_window = old_x.shape[1]
    assert T <= max_window, f"T={T} exceeds cache max_window={max_window}"

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
    # If this differs from BLOCK_SIZE_T, rectangle precompute uses a slower
    # one-hot fallback because tl.gather requires matching padded axis sizes.
    # Production uses matching padded T/K; mismatches are for tests/debugging.
    BLOCK_SIZE_K = max(triton.next_power_of_2(max_window), 16)
    rectangle_use_gather = BLOCK_SIZE_T == BLOCK_SIZE_K

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
        # Cap at heads_per_group: HEADS_PER_BLOCK divides the kernel's head
        # axis, so a table value larger than the model's heads-per-group
        # would overshoot.  Protects callers running smaller models than
        # the one we tuned against.
        heads_per_block = min(_heads_per_block, heads_per_group)
    if _precompute_num_warps is not None:
        precompute_num_warps = _precompute_num_warps

    # Per-main knob resolution: each _*_{write,nowrite} arg, if not None,
    # overrides the corresponding shared value for ONE main launch only.
    # Default (None) = tied to shared value (current behavior).
    BLOCK_SIZE_M_WRITE = _block_size_m_write if _block_size_m_write is not None else BLOCK_SIZE_M
    BLOCK_SIZE_M_NOWRITE = _block_size_m_nowrite if _block_size_m_nowrite is not None else BLOCK_SIZE_M
    NUM_WARPS_WRITE = _num_warps_write if _num_warps_write is not None else num_warps
    NUM_WARPS_NOWRITE = _num_warps_nowrite if _num_warps_nowrite is not None else num_warps
    NUM_STAGES_WRITE = _num_stages_write if _num_stages_write is not None else _num_stages
    NUM_STAGES_NOWRITE = _num_stages_nowrite if _num_stages_nowrite is not None else _num_stages
    # Persistent-only per-main:
    CTA_PER_SM_WRITE = _cta_per_sm_write if _cta_per_sm_write is not None else _cta_per_sm
    CTA_PER_SM_NOWRITE = _cta_per_sm_nowrite if _cta_per_sm_nowrite is not None else _cta_per_sm
    NUM_LOOP_STAGES_WRITE = _num_loop_stages_write if _num_loop_stages_write is not None else _num_loop_stages
    NUM_LOOP_STAGES_NOWRITE = _num_loop_stages_nowrite if _num_loop_stages_nowrite is not None else _num_loop_stages

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

    # Per-path TMA descriptors for state — write-side and nowrite-side.  Each
    # kernel launch consumes the descriptor whose block_shape[0] matches its
    # BLOCK_SIZE_M constexpr.  With M-split (Mw != Mnw) the two sides need
    # distinct descriptors; otherwise the descriptor's block_shape[0] would
    # mismatch the kernel's BLOCK_SIZE_M and downstream tl.dot / arithmetic
    # on the loaded tile fails shape inference at compile time
    # ("Cannot make_shape_compatible: incompatible dimensions").  When Mw ==
    # Mnw (tied, the common case) the two descriptors are the same object.
    # Same memory (state's flat 2D view, shape (cache*nheads*dim, dstate))
    # and same dstate block_shape — only block_shape[0] differs.
    # When no TMA flag is on, both variables hold the raw `state` tensor as a
    # dummy; kernels never reference it because their constexprs are all
    # False (Triton DCEs the dead branches).
    # `triton.set_allocator()` must run before any descriptor-using launch.
    if (_use_tma_rect_load or _use_tma_replay_write_load
            or _use_tma_replay_write_store or _use_tma_replay_nowrite_load):
        from triton.tools.tensor_descriptor import TensorDescriptor
        _ensure_tma_allocator()
        assert state.is_contiguous(), "TMA state requires contiguous state"
        assert state.stride(-1) == 1, "TMA state requires inner stride 1"
        _state_flat = state.view(-1, state.shape[-1])
        _dstate_pow2 = triton.next_power_of_2(dstate)
        state_tma_descriptor_write = TensorDescriptor.from_tensor(
            _state_flat, block_shape=[BLOCK_SIZE_M_WRITE, _dstate_pow2],
        )
        if BLOCK_SIZE_M_NOWRITE == BLOCK_SIZE_M_WRITE:
            state_tma_descriptor_nowrite = state_tma_descriptor_write
        else:
            state_tma_descriptor_nowrite = TensorDescriptor.from_tensor(
                _state_flat, block_shape=[BLOCK_SIZE_M_NOWRITE, _dstate_pow2],
            )
    else:
        state_tma_descriptor_write = state   # dummy; all consuming constexprs False
        state_tma_descriptor_nowrite = state  # dummy; all consuming constexprs False

    # Work items are sorted write-first for persistent_main. Each row carries
    # decode-batch position, cache slot, PNAT, and active cache buffer index.
    assert isinstance(replay_work_items, torch.Tensor), (
        "replay_work_items must be a torch.Tensor, got "
        f"{type(replay_work_items).__name__}"
    )
    assert replay_work_items.device == device, (
        f"replay_work_items must be on device {device}, got "
        f"{replay_work_items.device}"
    )
    assert replay_work_items.dtype == torch.int32, (
        f"replay_work_items must be int32, got {replay_work_items.dtype}"
    )
    assert replay_work_items.shape == (batch, REPLAY_WORK_ITEM_WIDTH), (
        "replay_work_items must have shape "
        f"(batch={batch}, {REPLAY_WORK_ITEM_WIDTH}), got "
        f"{tuple(replay_work_items.shape)}"
    )
    assert replay_work_items.is_contiguous(), "replay_work_items must be contiguous"
    assert isinstance(n_writes, torch.Tensor), (
        f"n_writes must be a torch.Tensor, got {type(n_writes).__name__}"
    )
    assert n_writes.device == device, (
        f"n_writes must be on device {device}, got {n_writes.device}"
    )
    assert n_writes.dtype == torch.int32, (
        f"n_writes must be int32, got {n_writes.dtype}"
    )
    assert n_writes.shape == (1,), (
        f"n_writes must have shape (1,), got {tuple(n_writes.shape)}"
    )
    replay_work_items_arg = replay_work_items

    precomp_grid = (batch, nheads // heads_per_block)
    d_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)

    # ---- Launch helpers (close over locals) -------------------------------
    # Each helper is a thin closure that calls one Triton kernel with the
    # full positional + kwarg argument list.  Mode-dependent constexprs
    # (write_checkpoint, early_out, rectangle) are passed in.

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
            RECTANGLE_USE_GATHER=rectangle_use_gather,
            num_warps=precompute_num_warps,
            **({"num_stages": _precompute_num_stages} if _precompute_num_stages else {}),
            launch_pdl=launch_with_pdl,
        )

    # ---- launch_persistent_main ------------------------------------------
    # Persistent-CTA main kernel.  Single launch covers `n_slots` slots
    # starting at `slot_offset`.  Caller invokes twice: once for the write
    # half (slot_offset=0, n_slots=n_writes, write_checkpoint=True) and
    # once for the nowrite half (slot_offset=n_writes,
    # n_slots=batch-n_writes, write_checkpoint=False).  Hard-sort
    # contract: caller has pre-sorted slots so [0, n_writes) are writes
    # and [n_writes, batch) are nowrites.

    # Resolve persistent-mode bench knobs.  Defaults: cta_per_sm = 1
    # (one CTA per SM, matches upstream `_p_matmul_ogs.py`); num_loop_stages
    # = 2 (matches in-tree `swiglu` precedent for non-dot persistent loops);
    # flatten = True (canonical Triton 3.6 idiom); warp_specialize = False.
    _num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    cta_per_sm_arg = _cta_per_sm if _cta_per_sm else 1
    num_persistent_arg = cta_per_sm_arg * _num_sms
    num_loop_stages_arg = _num_loop_stages if _num_loop_stages else 2
    flatten_arg = True if _flatten is None else bool(_flatten)
    warp_specialize_arg = False if _warp_specialize is None else bool(_warp_specialize)
    # Per-launch work-item count.  At small batch, total_work may be < the
    # full persistent grid; capping `grid` at `min(NUM_PERSISTENT, total_work)`
    # avoids launching empty CTAs that pay setup cost for no work.  Correctness:
    # the kernel's `tl.range(pid, total_work, NUM_PERSISTENT)` ensures each
    # tile_id is covered exactly once across all live pids in [0, grid) when
    # grid <= NUM_PERSISTENT (each CTA does 1 tile; loop step >= total_work
    # exits immediately) AND when grid == NUM_PERSISTENT (each CTA loops over
    # multiple tiles).  NUM_PERSISTENT is now a runtime int (see kernel def
    # docstring at _persistent_main_kernel) so changing cta_per_sm does NOT
    # trigger a new Triton compile — same kernel binary, different loop step.
    # (Named UPPERCASE for historical Triton-style consistency only; not
    # constexpr.)
    _num_pid_m = (dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    def launch_persistent_main(write_checkpoint: bool,
                               *,
                               launch_dependent_kernels: bool = False,
                               rectangle: bool = False):
        # `n_writes` is the (1,) int32 device tensor with the write count.
        # Both halves always launch; an empty half has a zero-length slot
        # range and the persistent loop does no work.
        _bsm = BLOCK_SIZE_M_WRITE if write_checkpoint else BLOCK_SIZE_M_NOWRITE
        _nw = NUM_WARPS_WRITE if write_checkpoint else NUM_WARPS_NOWRITE
        _ns = NUM_STAGES_WRITE if write_checkpoint else NUM_STAGES_NOWRITE
        _cps = CTA_PER_SM_WRITE if write_checkpoint else CTA_PER_SM_NOWRITE
        _cps = _cps if _cps else 1
        _nls = NUM_LOOP_STAGES_WRITE if write_checkpoint else NUM_LOOP_STAGES_NOWRITE
        _nls = _nls if _nls else 2
        _num_persistent = _cps * _num_sms
        _num_pid_m_local = (dim + _bsm - 1) // _bsm
        # Grid sizing: cap at min(full persistent grid, upper-bound total work).
        # We use `batch` as the upper bound on slots-per-half — overcounting
        # by a few CTAs is fine since the kernel derives the exact slot range
        # from n_writes at runtime.
        _total_work_launch = max(1, batch * _num_pid_m_local * nheads)
        grid = (min(_num_persistent, _total_work_launch),)
        # Per-path TMA descriptor — block_shape[0] must match _bsm.
        _desc = (state_tma_descriptor_write if write_checkpoint
                 else state_tma_descriptor_nowrite)
        _persistent_main_kernel[grid](
            state, _desc, state_scales_arg, old_x,
            old_B, old_dt, old_dA_cumsum,
            prev_num_accepted_tokens, cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices, replay_work_items_arg, rand_seed, pad_slot_id,
            n_writes, batch, nheads,
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
            _bsm,
            LAUNCH_WITH_PDL=use_internal_pdl,
            PHILOX_ROUNDS=philox_rounds if rand_seed is not None else 0,
            QUANT_MAX=quant_max,
            WRITE_CHECKPOINT=write_checkpoint,
            LAUNCH_DEPENDENT_KERNELS=launch_dependent_kernels and use_internal_pdl,
            NUM_PERSISTENT=_num_persistent,
            NUM_LOOP_STAGES=_nls,
            FLATTEN=flatten_arg,
            WARP_SPECIALIZE=warp_specialize_arg,
            IS_DYNAMIC=False,
            RECTANGLE=rectangle,
            # 3 TMA flags.  IS_DYNAMIC=False: WC fixed per launch; impl
            # constexpr-folds the LOAD pick.  When WRITE_CHECKPOINT=True (write half),
            # NOWRITE_LOAD is dummy False; when WRITE_CHECKPOINT=False, WRITE_LOAD/STORE
            # dummy False.  NOWRITE_LOAD picks rect-load (RECTANGLE) or
            # replay-nowrite-load.
            USE_TMA_LOAD_WRITE=bool(_use_tma_replay_write_load and write_checkpoint),
            USE_TMA_LOAD_NOWRITE=bool(
                (_use_tma_rect_load if rectangle else _use_tma_replay_nowrite_load)
                and not write_checkpoint
            ),
            USE_TMA_STORE=bool(_use_tma_replay_write_store and write_checkpoint),
            USE_REPLAY_CACHE_SLOT=bool(_use_replay_cache_slot),
            num_warps=_nw,
            **({"num_stages": _ns} if _ns else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )

    def launch_persistent_dynamic_main(n_writes_tensor: torch.Tensor,
                                       launch_dependent_kernels: bool = False,
                                       rectangle: bool = False):
        # Single-launch persistent kernel covering the whole batch with
        # runtime per-slot WRITE_CHECKPOINT branch.  No half-split, no
        # n_writes needed (the kernel ignores n_writes_tensor when
        # IS_DYNAMIC=True; Triton DCEs the load).  is_write is computed
        # at runtime per work-item from the loaded PNAT.
        # We still pass the same tensor as persistent_main so the kernel
        # signature is uniform.
        # Grid sizing: cap at total_work (= batch * num_pid_m * nheads) for
        # the dynamic case (full-batch coverage); see launch_persistent_main
        # comment for correctness rationale.
        _total_work_launch = max(1, batch * _num_pid_m * nheads)
        grid = (min(num_persistent_arg, _total_work_launch),)
        # Persistent-dynamic kernel uses a single BLOCK_SIZE_M (same as the
        # wrapper's BLOCK_SIZE_M == BLOCK_SIZE_M_WRITE tied convention), so
        # the write-side descriptor matches.  Both write and nowrite slots
        # in this kernel share that BSM.
        _persistent_main_kernel[grid](
            state, state_tma_descriptor_write, state_scales_arg, old_x,
            old_B, old_dt, old_dA_cumsum,
            prev_num_accepted_tokens, cache_buf_idx,
            x, C, D, z, out,
            cb_scaled, decay_vec,
            state_batch_indices, replay_work_items_arg, rand_seed, pad_slot_id,
            n_writes_tensor, batch, nheads,
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
            WRITE_CHECKPOINT=False,
            LAUNCH_DEPENDENT_KERNELS=launch_dependent_kernels and use_internal_pdl,
            NUM_PERSISTENT=num_persistent_arg,
            NUM_LOOP_STAGES=num_loop_stages_arg,
            FLATTEN=flatten_arg,
            WARP_SPECIALIZE=warp_specialize_arg,
            IS_DYNAMIC=True,
            RECTANGLE=rectangle,
            # 3 TMA flags.  IS_DYNAMIC=True: is_write is runtime per slot;
            # impl's load TMA picks per-slot (constexpr ternary becomes a
            # runtime branch — both load forms emitted, ~negligible cost).
            # NOWRITE_LOAD picks rect-load when RECTANGLE, else
            # replay-nowrite-load.  STORE only fires on runtime is_write.
            USE_TMA_LOAD_WRITE=bool(_use_tma_replay_write_load),
            USE_TMA_LOAD_NOWRITE=bool(
                _use_tma_rect_load if rectangle else _use_tma_replay_nowrite_load
            ),
            USE_TMA_STORE=bool(_use_tma_replay_write_store),
            USE_REPLAY_CACHE_SLOT=bool(_use_replay_cache_slot),
            num_warps=num_warps,
            **({"num_stages": _num_stages} if _num_stages else {}),
            **({"num_ctas": _num_ctas} if _num_ctas else {}),
            **({"maxnreg": _maxnreg} if _maxnreg else {}),
            launch_pdl=use_internal_pdl,
        )

    # ---- Mode dispatch ----------------------------------------------------
    with torch.cuda.device(device.index):
        if mode == "persistent_dynamic":
            # Single-launch persistent kernel covering the full batch.  Each
            # work-item dispatches via runtime PNAT check.  Kernel ignores
            # n_writes (Triton DCEs the load) when IS_DYNAMIC=True; we still
            # pass the wrapper-provided tensor as required by the signature.
            launch_dynamic_precompute(rectangle=rectangle_for_nowrite)
            launch_persistent_dynamic_main(
                n_writes,
                launch_dependent_kernels=False,
                rectangle=rectangle_for_nowrite,
            )
        elif mode == "persistent_main":
            # Persistent-CTA main kernel.  One shared dynamic_precompute
            # (per-slot dispatch via PNAT) feeds two persistent_main
            # launches (write half + nowrite half).  Both halves ALWAYS
            # launch; the kernel's runtime check iterates only the slots
            # belonging to its half (write: [0, n_writes), nowrite:
            # [n_writes, batch)).
            #
            # Caller-provided contract: `n_writes` is a (1,) int32 device
            # tensor (the kernel reads it at runtime, after the precompute);
            # `replay_work_items` is a (batch, 4) int32 device tensor
            # pre-sorted write-first.
            launch_dynamic_precompute(rectangle=rectangle_for_nowrite)
            launch_persistent_main(
                write_checkpoint=True,
                launch_dependent_kernels=True,
                rectangle=False,  # write always replay-style
            )
            launch_persistent_main(
                write_checkpoint=False,
                launch_dependent_kernels=False,
                rectangle=rectangle_for_nowrite,
            )
        else:
            raise ValueError(
                f"mode={mode!r} is not supported.  Supported modes: "
                f"'persistent_dynamic', 'persistent_main'."
            )
