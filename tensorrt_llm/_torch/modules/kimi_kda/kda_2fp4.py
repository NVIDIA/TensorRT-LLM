# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 checkpoint replay for non-speculative KDA decoding.

The V2 checkpoint is one packed byte array per head: primary E2M1 values,
residual E2M1 values, their E4M3 block scales, and a shared FP32 outer scale.
Only checkpoint materialization/commit reads the residual. K/U histories
are BF16 and cumulative per-key G is FP32. Batch and pool sizes are runtime
parameters; the CuTe DSL replay/commit kernels have no batch-size autotuning.
Prefill checkpoint packing and materialization use Triton helpers.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
HISTORY_SIZE = 16
_VALUES_BYTES = tl.constexpr(HEAD_DIM * HEAD_DIM // 2)
_SCALES_BYTES = tl.constexpr(HEAD_DIM * HEAD_DIM // 16)
_PRIMARY_SCALE = tl.constexpr(2 * _VALUES_BYTES.value)
_RESIDUAL_SCALE = tl.constexpr(_PRIMARY_SCALE.value + _SCALES_BYTES.value)
_OUTER_SCALE = tl.constexpr(_RESIDUAL_SCALE.value + _SCALES_BYTES.value)
# Align each head so its FP32 outer scale and subsequent head remain aligned.
STATE_BYTES_PER_HEAD = triton.cdiv(_OUTER_SCALE.value + 4, 16) * 16
_STATE_BYTES = tl.constexpr(STATE_BYTES_PER_HEAD)


@dataclass(frozen=True)
class ReplayHistory:
    """One layer's live-request histories, independent of V2 snapshot slots."""

    k: torch.Tensor  # [live_capacity, H, 16, 128], BF16
    u: torch.Tensor  # [live_capacity, H, 16, 128], BF16
    g: torch.Tensor  # [live_capacity, H, 16, 128], FP32
    length: torch.Tensor  # [live_capacity], int32
    slot_map: torch.Tensor  # [V2 slot capacity], int32; maps to live rows


@triton.jit
def _expand_scale(scales, outer, ROWS: tl.constexpr):
    return tl.reshape(
        tl.broadcast_to(scales.to(tl.float32)[:, :, None] * outer, (ROWS, 8, 16)),
        (ROWS, 128),
    )


@triton.jit
def _dequantize(values, scales, outer, ROWS: tl.constexpr):
    codes = tl.reshape(tl.join(values & 15, values >> 4), (ROWS, 128))
    magnitude_code = codes & 7
    magnitude = tl.where(
        magnitude_code <= 4,
        magnitude_code.to(tl.float32) * 0.5,
        tl.where(magnitude_code == 5, 3.0, tl.where(magnitude_code == 6, 4.0, 6.0)),
    )
    return tl.where((codes & 8) != 0, -magnitude, magnitude) * _expand_scale(scales, outer, ROWS)


@triton.jit
def _quantize(values, outer, ROWS: tl.constexpr):
    grouped = tl.reshape(values, (ROWS, 8, 16))
    block_amax = tl.max(tl.abs(grouped), 2)
    safe_outer = tl.where(outer == 0.0, 1.0, outer)
    # E4M3 conversion must not overflow to NaN at the maximum block scale.
    scales = tl.minimum(448.0, block_amax / (6.0 * safe_outer)).to(tl.float8e4nv)
    expanded = _expand_scale(scales, outer, ROWS)
    scaled = tl.where(expanded == 0.0, 0.0, values / expanded)
    lo, hi = tl.split(tl.reshape(scaled, (ROWS, 64, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u16.u8 $0, t; }",
        constraints="=h,f,f",
        args=(lo, hi),
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    return packed, scales


@triton.jit
def _load_checkpoint(base, row, ROWS: tl.constexpr, RESIDUAL: tl.constexpr):
    col = tl.arange(0, 64)
    group = tl.arange(0, 8)
    packed = tl.load(base + row[:, None] * 64 + col[None, :])
    scales = tl.load(base + _PRIMARY_SCALE + row[:, None] * 8 + group[None, :]).to(
        tl.float8e4nv, bitcast=True
    )
    outer = tl.load((base + _OUTER_SCALE).to(tl.pointer_type(tl.float32)))
    state = _dequantize(packed, scales, outer, ROWS)
    if RESIDUAL:
        residual = tl.load(base + _VALUES_BYTES + row[:, None] * 64 + col[None, :])
        residual_scales = tl.load(base + _RESIDUAL_SCALE + row[:, None] * 8 + group[None, :]).to(
            tl.float8e4nv, bitcast=True
        )
        state += _dequantize(residual, residual_scales, outer, ROWS)
    return state


@triton.jit
def _store_checkpoint(base, state):
    row = tl.arange(0, 128)
    col = tl.arange(0, 64)
    group = tl.arange(0, 8)
    outer = tl.max(tl.max(tl.abs(state), 1), 0) / (6.0 * 448.0)
    primary, scales = _quantize(state, outer, 128)
    residual, residual_scales = _quantize(
        state - _dequantize(primary, scales, outer, 128), outer, 128
    )
    offsets = row[:, None] * 64 + col[None, :]
    scale_offsets = row[:, None] * 8 + group[None, :]
    tl.store(base + offsets, primary)
    tl.store(base + _VALUES_BYTES + offsets, residual)
    tl.store(base + _PRIMARY_SCALE + scale_offsets, scales.to(tl.uint8, bitcast=True))
    tl.store(base + _RESIDUAL_SCALE + scale_offsets, residual_scales.to(tl.uint8, bitcast=True))
    tl.store((base + _OUTER_SCALE).to(tl.pointer_type(tl.float32)), outer)
    padding = tl.arange(0, 16)
    tl.store(base + _OUTER_SCALE + 4 + padding, 0, mask=padding < 12)


@triton.jit
def _materialize_kernel(pool, slots, initial, dense, slot_stride, H: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + batch).to(tl.int64)
    row = tl.arange(0, 128)
    col = tl.arange(0, 128)
    state = tl.full((128, 128), 0, tl.float32)
    if tl.load(initial + batch):
        state = _load_checkpoint(pool + slot * slot_stride + head * _STATE_BYTES, row, 128, True)
    tl.store(dense + (batch * H + head) * 16384 + row[:, None] * 128 + col[None, :], state)


@triton.jit
def _pack_kernel(pool, slots, dense, slot_stride, H: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + batch).to(tl.int64)
    row = tl.arange(0, 128)
    col = tl.arange(0, 128)
    state = tl.load(dense + (batch * H + head) * 16384 + row[:, None] * 128 + col[None, :])
    _store_checkpoint(pool + slot * slot_stride + head * _STATE_BYTES, state)


@triton.jit(do_not_specialize_on_alignment=["slots"])
def _snapshot_lengths_kernel(slots, slot_map, lengths, old_lengths):
    batch = tl.program_id(0)
    slot = tl.load(slots + batch)
    live = tl.load(slot_map + slot)
    length = tl.load(lengths + live, mask=live >= 0, other=0)
    tl.store(old_lengths + batch, length)


def materialize_checkpoint(
    pool: torch.Tensor, slots: torch.Tensor, has_initial_state: torch.Tensor
) -> torch.Tensor:
    """Materialize addressed packed checkpoints as temporary [B,H,128,128] FP32 rows."""
    _validate_checkpoint(pool)
    dense = torch.empty(
        (slots.numel(), pool.shape[1], 128, 128), dtype=torch.float32, device=pool.device
    )
    _materialize_kernel[(slots.numel(), pool.shape[1])](
        pool, slots, has_initial_state, dense, pool.stride(0), pool.shape[1], num_warps=8
    )
    return dense


def store_checkpoint(pool: torch.Tensor, slots: torch.Tensor, dense: torch.Tensor) -> None:
    """Pack temporary dense FP32 state rows into V2, including the commit-only residual."""
    _validate_checkpoint(pool)
    if dense.shape != (slots.numel(), pool.shape[1], 128, 128) or dense.dtype != torch.float32:
        raise ValueError("Expected FP32 KDA state rows [B,H,128,128].")
    _pack_kernel[(slots.numel(), pool.shape[1])](
        pool, slots, dense.contiguous(), pool.stride(0), pool.shape[1], num_warps=8
    )


def commit_history(
    pool: torch.Tensor,
    slots: torch.Tensor,
    history: ReplayHistory,
    *,
    full_only: bool = False,
    dummy: torch.Tensor | None = None,
    old_lengths: torch.Tensor | None = None,
) -> None:
    """Commit and advance full histories, or flush before suspension/export.

    Normal decode supplies immutable [B] int32 lengths recorded by replay.
    Standalone partial flushes snapshot the addressed lengths before the
    commit kernel can advance/reset the live counters.
    """
    from ...custom_ops.cute_dsl_kda_2fp4_commit import commit

    _validate_checkpoint(pool)
    if full_only and dummy is None:
        raise ValueError("Full rollover requires the replay batch's dummy mask.")
    if slots.numel() == 0:
        return
    slots = slots.to(torch.int32)
    if old_lengths is None:
        old_lengths = torch.empty_like(slots)
        _snapshot_lengths_kernel[(slots.numel(),)](
            slots, history.slot_map, history.length, old_lengths
        )
    if dummy is None:
        dummy = torch.zeros(slots.numel(), dtype=torch.bool, device=slots.device)
    commit(
        pool,
        slots,
        history.slot_map,
        history.k,
        history.u,
        history.g,
        history.length,
        old_lengths,
        dummy,
        full_only,
    )


@torch.no_grad()
def replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    history: ReplayHistory,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused one-token Conv4, replay, and gated output norm.

    Raw BF16 inputs are x [B,3*H*128], gate [B,H*128], beta [B,H],
    z [B,H,128], Conv4 weights [3*H*128,4], and norm weights [128].
    Conv state [S,3*H*128,3] and packed checkpoints may have independent
    slot strides. Projection row strides are preserved. The FP32 A_log
    and dt_bias contain H and H*128 values, respectively. The optional
    BF16 output [B,H,128] is written directly, including BCG output views.
    """
    from ...custom_ops.cute_dsl_kda_2fp4_replay import fused_replay

    _validate_checkpoint(pool)
    batch, heads = x.shape[0], pool.shape[1]
    width = heads * HEAD_DIM
    if (
        x.shape != (batch, 3 * width)
        or gate.shape != (batch, width)
        or beta.shape != (batch, heads)
        or z.shape != (batch, heads, HEAD_DIM)
        or conv_weight.shape != (3 * width, 4)
        or norm_weight.shape != (HEAD_DIM,)
        or slots.numel() != batch
        or dummy is None
        or dummy.numel() != batch
    ):
        raise ValueError("KDA replay requires one raw Conv4 input and output gate per batch row.")
    if any(tensor.dtype != torch.bfloat16 for tensor in (x, gate, beta, z, norm_weight)):
        raise ValueError("KDA replay requires BF16 activations and output norm weights.")
    if output is None:
        output = torch.empty((batch, heads, HEAD_DIM), dtype=x.dtype, device=x.device)
    if (
        output.shape != (batch, heads, HEAD_DIM)
        or output.dtype != torch.bfloat16
        or output.stride(-1) != 1
        or output.stride(-2) != HEAD_DIM
        or output.stride(0) != heads * HEAD_DIM
    ):
        raise ValueError("KDA replay output requires dense BF16 [B,H,128].")
    if batch == 0:
        return output
    slots = slots.to(torch.int32)
    old_lengths = torch.empty_like(slots)
    fused_replay(
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool.view(pool.shape[0], heads, STATE_BYTES_PER_HEAD),
        slots,
        dummy,
        history.slot_map,
        history.k,
        history.u,
        history.g,
        history.length,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        lower_bound,
        norm_eps,
        output,
        old_lengths,
    )
    commit_history(pool, slots, history, full_only=True, dummy=dummy, old_lengths=old_lengths)
    return output


def _validate_checkpoint(pool: torch.Tensor) -> None:
    if (
        pool.dtype != torch.uint8
        or pool.ndim not in (3, 4)
        or pool.stride(1) != STATE_BYTES_PER_HEAD
        or pool.stride(-1) != 1
        or pool.shape[-1] != STATE_BYTES_PER_HEAD
        or (pool.ndim == 4 and pool.shape[-2] != 1)
    ):
        raise ValueError("KDA 2FP4 checkpoint requires V2 byte payloads with dense heads.")
