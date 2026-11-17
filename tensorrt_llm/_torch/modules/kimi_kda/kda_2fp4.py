# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 checkpoint replay for KDA decoding and DSpark verification.

Checkpoint
    Each head's 128x128 FP32 recurrent state S is stored as one packed byte
    array (``STATE_BYTES_PER_HEAD``): a primary term P and a residual term R,
    both E2M1 values with E4M3 block-16 scales, and one FP32 outer scale
    ``outer = amax(|S|) / (6 * 448)`` shared by both terms. 6 is the largest
    E2M1 magnitude and 448 the largest E4M3 value, so ``S / outer`` fits the
    product of the two ranges and every block scale ``amax(block) / (6 * outer)``
    is a representable E4M3 value. P quantizes S; R quantizes S - dequant(P)
    with the same outer scale, so the residual's block scales encode its
    smaller magnitude relative to S. S ~= dequant(P) + dequant(R). Replay
    projections read only P; materialization and commits read both terms.

History
    Each request owns a replay slot holding, per head, up to the history
    window of rows of normalized K and delta-rule update U (BF16) and the
    cumulative per-key log decay G (FP32). G_i is the sum of the per-token
    log decays g_1..g_i since the checkpoint, so the state after i tokens is
    S_i = S * exp(G_i) + sum_{j<=i} u_j (x) (k_j * exp(G_i - G_j)), with the
    exponentials applied per key channel. Replay reconstructs every token's
    state from S and the history without materializing it.

Commit
    When the history reaches its window (and, with DSpark, for the accepted
    prefix), the checkpoint advances to
    S_new = S * exp(G_last) + sum_i u_i (x) (k_i * exp(G_last - G_i))
    over the committed rows, and P/R are requantized with a new outer scale.

Batch and pool sizes are runtime parameters; no kernel specializes on batch
size. Decode replay, DSpark verification and the all-layer post-sampling
commit use CuTe DSL; prefill checkpoint helpers use Triton.
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
    """One layer's histories, addressed by replay slot.

    Replay slots may be strided (KV-cache pool pages); each slot's K/U/G
    payload is dense and 16-byte aligned, and ``length`` may be strided.
    """

    k: torch.Tensor  # [replay_slots, H, history_capacity, 128], BF16
    u: torch.Tensor  # [replay_slots, H, history_capacity, 128], BF16
    g: torch.Tensor  # [replay_slots, H, history_capacity, 128], FP32
    length: torch.Tensor  # [replay_slots], int32
    candidate_x: torch.Tensor | None = None  # [B_capacity, T_capacity, 3*H*128], BF16
    old_lengths: torch.Tensor | None = None  # [B_capacity], immutable verification snapshot
    arrivals: torch.Tensor | None = None  # [B_capacity], int32 zeros; decode row arrival counters


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
    replay_indices: torch.Tensor,
    dummy: torch.Tensor,
    history: ReplayHistory,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused one-token Conv4, replay, gated output norm, and full-window commit.

    Raw BF16 inputs are x [B,3*H*128], gate [B,H*128], beta [B,H],
    z [B,H,128], Conv4 weights [3*H*128,4], and norm weights [128].
    Conv state [S,3*H*128,3] and packed checkpoints may have independent
    slot strides; both are addressed by ``slots``, and the histories by
    ``replay_indices`` (int32, one replay slot per batch row). Projection row strides are preserved. The FP32 A_log
    and dt_bias contain H and H*128 values, respectively. The optional
    BF16 output [B,H,128] is written directly, including BCG output views.
    ``history.arrivals`` holds persistent int32 zero counters for every row.
    The kernel prefetches this layer's checkpoint, history and Conv rows
    before its PDL wait, so it must not directly follow another replay of
    the same layer in a stream.
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
            replay_indices,
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
        or replay_indices.shape != (batch,)
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
    arrivals = history.arrivals
    if arrivals is None or arrivals.numel() < batch or arrivals.dtype != torch.int32:
        raise ValueError("KDA replay requires persistent int32 arrival counters for every row.")
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
        replay_indices.to(torch.int32),
        history.k,
        history.u,
        history.g,
        history.length,
        A_log.reshape(-1),
        dt_bias.reshape(-1),
        lower_bound,
        norm_eps,
        output,
        arrivals,
    )
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
    replay_indices: torch.Tensor,
    dummy: torch.Tensor,
    history: ReplayHistory,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a linear [B,T] candidate block without advancing either checkpoint.

    Inputs are batch-major flattened [B*T,...] BF16 projection views; state
    indices, replay indices and the dummy mask have B entries. Only the sampler's accepted prefix
    may subsequently advance Conv, history lengths, or the packed state.
    """
    from ...custom_ops.cute_dsl_kda_2fp4_mtp_replay import fused_replay

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
        or replay_indices.shape != (batch,)
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
        replay_indices.to(torch.int32),
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


def commit_descriptor(
    state_pool: torch.Tensor, conv_pool: torch.Tensor, history: ReplayHistory
) -> tuple[int, ...]:
    """One layer's int64 row for ``commit_verified``.

    Rows of all local layers stack into the contiguous [L, F] int64 tensor
    ``commit_verified`` takes. ``history`` needs ``candidate_x`` and
    ``old_lengths``.
    """
    _validate_checkpoint(state_pool)
    if history.candidate_x is None or history.old_lengths is None:
        raise ValueError("2FP4 commit descriptors require DSpark candidate buffers")
    from ...custom_ops.cute_dsl_kda_2fp4_mtp_replay_packed import _check_histories

    _check_histories(
        history.k,
        history.u,
        history.g,
        history.length,
        state_pool.shape[1],
        history.k.shape[2],
    )
    return (
        state_pool.data_ptr(),
        state_pool.stride(0) * state_pool.element_size(),
        conv_pool.data_ptr(),
        conv_pool.stride(0),
        history.k.data_ptr(),
        history.u.data_ptr(),
        history.g.data_ptr(),
        history.length.data_ptr(),
        history.old_lengths.data_ptr(),
        history.candidate_x.data_ptr(),
        history.k.stride(0) * history.k.element_size(),
        history.u.stride(0) * history.u.element_size(),
        history.g.stride(0) * history.g.element_size(),
        history.length.stride(0) * history.length.element_size(),
    )


def commit_verified(
    descriptors: torch.Tensor,
    slots: torch.Tensor,
    replay_slots: torch.Tensor,
    accepted_tokens: torch.Tensor,
    dummy: torch.Tensor,
    heads: int,
    capacity: int,
    max_tokens: int,
) -> None:
    """Commit accepted Conv and recurrent histories across all local layers.

    ``descriptors`` stacks one ``commit_descriptor`` row per local layer.
    ``slots`` address the checkpoints and Conv windows, ``replay_slots`` the
    histories; both have one entry per batch row. Accepted counts include the
    golden token. The candidate buffer keeps its configured max_tokens stride
    even when verification uses fewer tokens.
    """
    from ...custom_ops.cute_dsl_kda_2fp4_mtp_commit import commit

    if slots.numel():
        commit(
            descriptors,
            slots.to(torch.int32),
            replay_slots.to(torch.int32),
            accepted_tokens.to(torch.int32),
            dummy,
            heads,
            capacity,
            max_tokens,
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
