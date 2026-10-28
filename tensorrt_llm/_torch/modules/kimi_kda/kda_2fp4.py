# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 checkpoint replay for KDA decoding and DSpark verification.

The V2 checkpoint is one packed byte array per head: primary E2M1 values,
residual E2M1 values, their E4M3 block scales, and a shared FP32 outer scale.
Only checkpoint materialization/commit reads the residual. K/U histories
are BF16 and cumulative per-key G is FP32. Batch and pool sizes are runtime
parameters; no kernel specializes on batch size. Prefill checkpoint helpers
and the all-layer DSpark post-sampling commit use Triton.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

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


def validate_speculative_config(spec_config: "DecodingBaseConfig | None") -> None:
    """Accept ordinary decoding or linear DSpark verification up to MTP7."""
    if spec_config is None:
        return
    if not spec_config.spec_dec_mode.is_dspark():
        raise ValueError("KDA 2FP4 speculative decoding currently supports DSpark only")
    if not 2 <= spec_config.tokens_per_gen_step <= 8:
        raise ValueError("KDA 2FP4 DSpark requires 1..7 draft tokens")
    if getattr(spec_config, "use_dynamic_tree", False):
        raise ValueError("KDA 2FP4 DSpark requires a linear draft chain")


@dataclass(frozen=True)
class ReplayHistory:
    """One layer's live-request histories, independent of V2 snapshot slots."""

    k: torch.Tensor  # [live_capacity, H, history_capacity, 128], BF16
    u: torch.Tensor  # [live_capacity, H, history_capacity, 128], BF16
    g: torch.Tensor  # [live_capacity, H, history_capacity, 128], FP32
    length: torch.Tensor  # [live_capacity], int32
    slot_map: torch.Tensor  # [V2 slot capacity], int32; maps to live rows
    candidate_x: torch.Tensor | None = None  # [B_capacity, T_capacity, 3*H*128], BF16
    old_lengths: torch.Tensor | None = None  # [B_capacity], immutable verification snapshot


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

    if history.candidate_x is not None:
        return verify(
            x,
            conv_state,
            conv_weight,
            gate,
            beta,
            z,
            norm_weight,
            pool,
            slots,
            dummy,
            history,
            A_log,
            dt_bias,
            lower_bound,
            norm_eps,
            output,
        )
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


@torch.no_grad()
def verify(
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
    """Verify a linear [B,T] candidate block without advancing either checkpoint.

    Inputs are batch-major flattened [B*T,...] BF16 projection views; state indices
    and the dummy mask have B entries. Only the sampler's accepted prefix
    may subsequently advance Conv, history lengths, or the packed state.
    """
    from ...custom_ops.cute_dsl_kda_2fp4_replay import fused_replay

    _validate_checkpoint(pool)
    batch, heads = slots.numel(), pool.shape[1]
    if batch == 0:
        return x.new_empty((0, heads, HEAD_DIM)) if output is None else output
    tokens, remainder = divmod(x.shape[0], batch)
    width = heads * HEAD_DIM
    if (
        remainder
        or not 1 <= tokens <= 8
        or history.candidate_x is None
        or history.old_lengths is None
        or history.candidate_x.shape[0] < batch
        or history.candidate_x.shape[1] < tokens
        or history.k.shape[2] < HISTORY_SIZE + tokens
        or x.shape != (batch * tokens, 3 * width)
        or gate.shape != (batch * tokens, width)
        or beta.shape != (batch * tokens, heads)
        or z.shape != (batch * tokens, heads, HEAD_DIM)
        or conv_weight.shape != (3 * width, 4)
        or norm_weight.shape != (HEAD_DIM,)
        or dummy is None
        or dummy.shape != (batch,)
    ):
        raise ValueError(
            "2FP4 verification requires 1..8 tokens and sufficiently sized replay buffers"
        )
    if any(t.dtype != torch.bfloat16 for t in (x, gate, beta, z, norm_weight, conv_weight)):
        raise TypeError("2FP4 verification requires BF16 activations and Conv/norm weights")
    if output is None:
        output = x.new_empty((batch * tokens, heads, HEAD_DIM))
    if (
        output.shape != (batch * tokens, heads, HEAD_DIM)
        or output.dtype != torch.bfloat16
        or not output.is_contiguous()
    ):
        raise ValueError("2FP4 verification output must be contiguous [B*T,H,128]")
    fused_replay(
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool.view(pool.shape[0], heads, STATE_BYTES_PER_HEAD),
        slots.to(torch.int32),
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
        history.old_lengths[:batch],
        candidate_x=history.candidate_x[:batch, :tokens],
        tokens_per_step=tokens,
    )
    return output


@triton.jit
def _commit_verified_kernel(
    descriptors,
    slots,
    slot_map,
    accepted,
    dummy,
    H: tl.constexpr,
    CAP: tl.constexpr,
    TMAX: tl.constexpr,
    WINDOW: tl.constexpr,
):
    batch, head, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    is_dummy = tl.load(dummy + batch)
    slot = tl.load(slots + batch)
    live = tl.load(slot_map + slot, mask=~is_dummy, other=-1)
    if live >= 0:
        base = descriptors + layer * 10
        pool = tl.load(base).to(tl.pointer_type(tl.uint8))
        pool_stride = tl.load(base + 1)
        conv = tl.load(base + 2).to(tl.pointer_type(tl.bfloat16))
        conv_stride = tl.load(base + 3)
        hk = tl.load(base + 4).to(tl.pointer_type(tl.bfloat16))
        hu = tl.load(base + 5).to(tl.pointer_type(tl.bfloat16))
        hg = tl.load(base + 6).to(tl.pointer_type(tl.float32))
        lengths = tl.load(base + 7).to(tl.pointer_type(tl.int32))
        old_lengths = tl.load(base + 8).to(tl.pointer_type(tl.int32))
        candidates = tl.load(base + 9).to(tl.pointer_type(tl.bfloat16))
        count = tl.load(accepted + batch)
        tl.device_assert((count >= 0) & (count <= TMAX), "invalid accepted prefix")
        old = tl.load(old_lengths + batch)
        n = old + count
        # Each head owns its three disjoint Q/K/V channel sections.
        channel = tl.arange(0, 512)
        section, key = channel // 128, channel % 128
        absolute_channel = (section * H + head) * 128 + key
        window = tl.arange(0, 4)
        source = count + window
        from_old = source < 3
        old_values = tl.load(
            conv
            + slot.to(tl.int64) * conv_stride
            + absolute_channel[:, None] * 3
            + source[None, :],
            mask=(channel[:, None] < 384) & (window[None, :] < 3) & from_old[None, :] & (count > 0),
            other=0,
        )
        new_values = tl.load(
            candidates
            + batch.to(tl.int64) * TMAX * (3 * H * 128)
            + (source[None, :] - 3) * (3 * H * 128)
            + absolute_channel[:, None],
            mask=(channel[:, None] < 384)
            & (window[None, :] < 3)
            & ~from_old[None, :]
            & (count > 0),
            other=0,
        )
        tl.store(
            conv
            + slot.to(tl.int64) * conv_stride
            + absolute_channel[:, None] * 3
            + window[None, :],
            tl.where(from_old[None, :], old_values, new_values),
            mask=(channel[:, None] < 384) & (window[None, :] < 3) & (count > 0),
        )
        if n >= WINDOW:
            row = tl.arange(0, 128)
            col = tl.arange(0, 128)
            hist = tl.arange(0, 32)
            state_base = pool + slot.to(tl.int64) * pool_stride + head * _STATE_BYTES
            state = _load_checkpoint(state_base, row, 128, True)
            hist_base = (live.to(tl.int64) * H + head) * CAP * 128
            end_g = tl.load(hg + hist_base + (n - 1) * 128 + col)
            kval = tl.load(
                hk + hist_base + hist[:, None] * 128 + col[None, :], mask=hist[:, None] < n, other=0
            ).to(tl.float32)
            gval = tl.load(
                hg + hist_base + hist[:, None] * 128 + col[None, :], mask=hist[:, None] < n, other=0
            )
            adjusted = tl.where(hist[:, None] < n, kval * tl.exp(end_g[None, :] - gval), 0).to(
                tl.bfloat16
            )
            uval = tl.load(
                hu + hist_base + hist[:, None] * 128 + row[None, :], mask=hist[:, None] < n, other=0
            )
            updated = state * tl.exp(end_g)[None, :] + tl.dot(tl.trans(uval), adjusted)
            _store_checkpoint(state_base, updated)
        if head == 0:
            tl.store(lengths + live, tl.where(n >= WINDOW, 0, n))


def commit_verified(
    descriptors: torch.Tensor,
    slots: torch.Tensor,
    slot_map: torch.Tensor,
    accepted_tokens: torch.Tensor,
    dummy: torch.Tensor,
    heads: int,
    capacity: int,
    max_tokens: int,
) -> None:
    """Commit accepted Conv and recurrent histories across all local layers.

    Each descriptor row contains checkpoint pointer/slot stride, Conv
    pointer/slot stride, K/U/G/length pointers, an immutable old-length
    snapshot pointer, and the compact candidate-Conv pointer (all int64).
    Accepted counts include the golden token. The candidate buffer keeps
    its configured max_tokens stride even when verification uses fewer tokens.
    """
    if slots.numel():
        _commit_verified_kernel[(slots.numel(), heads, descriptors.shape[0])](
            descriptors,
            slots,
            slot_map,
            accepted_tokens.to(torch.int32),
            dummy,
            heads,
            capacity,
            max_tokens,
            HISTORY_SIZE,
            num_warps=8,
        )


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
