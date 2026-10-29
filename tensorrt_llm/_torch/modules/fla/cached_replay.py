# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.modules.fla.op import exp
from tensorrt_llm._torch.modules.fla.utils import input_guard
from tensorrt_llm._utils import get_sm_version

_SMALL_GRID_HEAD_TILES = 512
_BV16_MAX_HEAD_TILES = 64
_BV32_MAX_HEAD_TILES = 128
_RATIO2_FINE_MAPPING_MAX_HEAD_TILES = 256
_EIGHT_WARP_COMMIT_HEAD_TILES = 1024
_PIPELINED_COMMIT_HEAD_TILES = 2048
_TWO_STAGE_REPLAY_HEAD_TILES = 4096
_L2_STREAMING_HEAD_TILES = 8192

CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE = 16


def _default_cached_replay_block_v(
    use_tuned_bf16_mapping: bool,
    head_tiles: int,
    value_dim: int,
) -> int:
    """Select the replay value tile without changing its cache layout."""
    if use_tuned_bf16_mapping:
        if head_tiles <= _BV16_MAX_HEAD_TILES:
            return 16
        if head_tiles <= _BV32_MAX_HEAD_TILES:
            return 32
        if head_tiles <= _SMALL_GRID_HEAD_TILES:
            return 64
    return triton.next_power_of_2(value_dim)


def _supports_fine_grained_replay_tiling(
    num_key_heads: int,
    num_value_heads: int,
    head_tiles: int,
) -> bool:
    """Return whether the measured fine-grained value tiling applies."""
    return num_value_heads == 4 * num_key_heads or (
        num_value_heads == 2 * num_key_heads and head_tiles <= _RATIO2_FINE_MAPPING_MAX_HEAD_TILES
    )


@triton.jit
def _gdc_wait_with_memory_clobber():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _cached_replay_kernel(
    q,
    k,
    v,
    packed_qkv,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    replay_indices,
    old_u,
    old_k,
    old_G,
    cache_buf_idx,
    pnat,
    scale,
    pool_stride_slot,
    old_u_stride_slot,
    old_k_stride_slot,
    old_G_stride_slot,
    pnat_stride,
    cache_buf_idx_stride,
    g_stride_row,
    beta_stride_row,
    A_log,
    dt_bias,
    T: tl.constexpr,
    HIST: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    STATE_FP32: tl.constexpr,
    FUSED_GATING: tl.constexpr,
    USE_L2_STATE_CACHE: tl.constexpr,
    USE_L2_STREAMING_INPUTS: tl.constexpr,
    USE_L2_SHARED_INPUTS: tl.constexpr,
    USE_PACKED_QKV: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    ENABLE_STATE_COMMIT: tl.constexpr,
):
    """Replay new tokens from cached causal cached updates.

    old_u/old_k/old_G contain the prefix-invariant cached update vectors,
    normalized keys, and cumulative log-decay from the current checkpoint.
    Only the T new cached updates are solved on each verify step. The
    checkpoint is addressed by the state slot from ``h0_indices``; the history
    and its bookkeeping by the replay slot from ``replay_indices``.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    o_hist = tl.arange(0, BH)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_t = o_t < T
    mask_hist_capacity = o_hist < HIST

    slot = tl.load(h0_indices + i_n)
    if slot >= 0:
        replay_slot = tl.load(replay_indices + i_n).to(tl.int64)
        b_pnat = tl.load(pnat + replay_slot * pnat_stride)
        b_buf = tl.load(cache_buf_idx + replay_slot * cache_buf_idx_stride)
        is_write = (b_pnat + T) > HIST
        w_buf = tl.where(is_write, 1 - b_buf, b_buf).to(tl.int64)
        w_off = tl.where(is_write, 0, b_pnat)
        is_hist = (o_hist < b_pnat) & mask_hist_capacity

        k_slot = old_k + replay_slot * old_k_stride_slot
        u_slot = old_u + replay_slot * old_u_stride_slot
        G_slot = old_G + replay_slot * old_G_stride_slot
        hk_base = k_slot + b_buf.to(tl.int64) * HIST * H * K + i_h * K
        hu_base = u_slot + b_buf.to(tl.int64) * HIST * HV * V + i_hv * V
        hG_base = G_slot + (b_buf.to(tl.int64) * HV + i_hv) * HIST
        wk_base = k_slot + w_buf * HIST * H * K + i_h * K
        wu_base = u_slot + w_buf * HIST * HV * V + i_hv * V
        wG_base = G_slot + (w_buf * HV + i_hv) * HIST

        if USE_L2_SHARED_INPUTS:
            b_kh = tl.load(
                hk_base + o_hist[:, None] * H * K + o_k[None, :],
                mask=is_hist[:, None] & mask_k[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_kh = tl.load(
                hk_base + o_hist[:, None] * H * K + o_k[None, :],
                mask=is_hist[:, None] & mask_k[None, :],
                other=0,
            )
        if USE_L2_STREAMING_INPUTS:
            b_uh = tl.load(
                hu_base + o_hist[:, None] * HV * V + o_v[None, :],
                mask=is_hist[:, None] & mask_v[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_uh = tl.load(
                hu_base + o_hist[:, None] * HV * V + o_v[None, :],
                mask=is_hist[:, None] & mask_v[None, :],
                other=0,
            )
        b_Gh = tl.load(hG_base + o_hist, mask=is_hist, other=0.0).to(tl.float32)
        g_start = tl.load(
            hG_base + b_pnat - 1,
            mask=b_pnat > 0,
            other=0.0,
        ).to(tl.float32)

        if LAUNCH_WITH_PDL:
            _gdc_wait_with_memory_clobber()

        if USE_PACKED_QKV:
            qkv_row = (i_n * T + o_t[:, None]) * (2 * H * K + HV * V)
            p_k = packed_qkv + qkv_row + H * K + i_h * K + o_k[None, :]
            p_q = packed_qkv + qkv_row + i_h * K + o_k[None, :]
            p_v = packed_qkv + qkv_row + 2 * H * K + i_hv * V + o_v[None, :]
        else:
            p_k = k + ((i_n * T + o_t[:, None]) * H + i_h) * K + o_k[None, :]
            p_q = q + ((i_n * T + o_t[:, None]) * H + i_h) * K + o_k[None, :]
            p_v = v + ((i_n * T + o_t[:, None]) * HV + i_hv) * V + o_v[None, :]

        if USE_L2_SHARED_INPUTS:
            b_k = tl.load(
                p_k,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0,
                cache_modifier=".cg",
            )
            b_q = tl.load(
                p_q,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_k = tl.load(
                p_k,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0,
            )
            b_q = tl.load(
                p_q,
                mask=mask_t[:, None] & mask_k[None, :],
                other=0,
            )
        if USE_L2_STREAMING_INPUTS:
            b_v = tl.load(
                p_v,
                mask=mask_t[:, None] & mask_v[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_v = tl.load(
                p_v,
                mask=mask_t[:, None] & mask_v[None, :],
                other=0,
            )
        b_g = tl.load(
            g + (i_n * T + o_t) * g_stride_row + i_hv,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta + (i_n * T + o_t) * beta_stride_row + i_hv,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        if FUSED_GATING:
            g_A_exp = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
            g_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
            x = b_g + g_dt_bias
            softplus = tl.where(
                x <= 20.0,
                0.6931471805599453 * tldevice.fast_log2f(1.0 + tldevice.fast_expf(x)),
                x,
            )
            b_g = tl.where(mask_t, -g_A_exp * softplus, 0.0)
            b_beta = tl.where(
                mask_t,
                tldevice.fast_dividef(1.0, 1.0 + tldevice.fast_expf(-b_beta)),
                0.0,
            )

        if USE_QK_L2NORM_IN_KERNEL:
            b_kf = b_k.to(tl.float32)
            b_qf = b_q.to(tl.float32)
            inv_k = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + 1e-6)
            inv_q = scale / tl.sqrt(tl.sum(b_qf * b_qf, 1) + 1e-6)
            b_kn = (b_kf * inv_k[:, None]).to(b_k.dtype)
            b_qn = (b_qf * inv_q[:, None]).to(b_q.dtype)
        else:
            b_kn = b_k
            b_qn = (b_q.to(tl.float32) * scale).to(b_q.dtype)

        b_G_local = tl.cumsum(b_g, 0)
        b_G = g_start + b_G_local

        p_h0 = (
            h0_source
            + slot.to(tl.int64) * pool_stride_slot
            + i_hv * V * K
            + o_k[:, None]
            + o_v[None, :] * K
        )
        if STATE_FP32:
            if USE_L2_STATE_CACHE:
                b_h = tl.load(
                    p_h0,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0,
                    cache_modifier=".cg",
                )
            else:
                b_h = tl.load(
                    p_h0,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0,
                )
            b_h_hi = b_h.to(b_kn.dtype)
            b_h_lo = (b_h - b_h_hi.to(tl.float32)).to(b_kn.dtype)
            b_kh0 = tl.dot(b_kn, b_h_hi) + tl.dot(b_kn, b_h_lo)
            b_qh0 = tl.dot(b_qn, b_h_hi) + tl.dot(b_qn, b_h_lo)
        else:
            if USE_L2_STATE_CACHE:
                b_h = tl.load(
                    p_h0,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0,
                    cache_modifier=".cg",
                ).to(b_kn.dtype)
            else:
                b_h = tl.load(
                    p_h0,
                    mask=mask_k[:, None] & mask_v[None, :],
                    other=0,
                ).to(b_kn.dtype)
            b_kh0 = tl.dot(b_kn, b_h)
            b_qh0 = tl.dot(b_qn, b_h)

        hist_decay = exp(b_G[:, None] - b_Gh[None, :])
        b_kk_hist = tl.dot(b_kn, tl.trans(b_kh))
        b_qk_hist = tl.dot(b_qn, tl.trans(b_kh))
        b_k_hist_coeff = tl.where(is_hist[None, :], b_kk_hist * hist_decay, 0.0)
        b_q_hist_coeff = tl.where(is_hist[None, :], b_qk_hist * hist_decay, 0.0)
        b_k_hist = tl.dot(b_k_hist_coeff.to(b_uh.dtype), b_uh)
        b_q_hist = tl.dot(b_q_hist_coeff.to(b_uh.dtype), b_uh)

        b_rhs = b_beta[:, None] * (b_v.to(tl.float32) - exp(b_G)[:, None] * b_kh0 - b_k_hist)
        lower = o_t[:, None] > o_t[None, :]
        b_kk_new = tl.dot(b_kn, tl.trans(b_kn))
        new_decay = exp(b_G[:, None] - b_G[None, :])
        b_A = tl.where(
            lower,
            b_beta[:, None] * b_kk_new * new_decay,
            0.0,
        )

        # T is at most eight for the cached replay path.  Forward substitution
        # avoids constructing/inverting the full HIST+T triangular system.
        b_U = tl.zeros([BT, BV], dtype=tl.float32)
        for row in range(T):
            row_mask = o_t == row
            rhs_row = tl.sum(tl.where(row_mask[:, None], b_rhs, 0.0), axis=0)
            a_row = tl.sum(tl.where(row_mask[:, None], b_A, 0.0), axis=0)
            correction = tl.sum(a_row[:, None] * b_U, axis=0)
            u_row = rhs_row - correction
            b_U += tl.where(row_mask[:, None], u_row[None, :], 0.0)

        incl = o_t[:, None] >= o_t[None, :]
        b_qk_new = tl.dot(b_qn, tl.trans(b_kn))
        b_q_new_coeff = tl.where(incl, b_qk_new * new_decay, 0.0)
        # BT is only four for MTP draft-3 verification, while tl.dot requires
        # a reduction dimension of at least 16 on this architecture.  Keep
        # this genuinely small operation scalar instead of padding it into a
        # mostly-empty tensor-core GEMM.
        b_q_new = tl.zeros([BT, BV], dtype=tl.float32)
        for row in range(T):
            row_mask = o_t == row
            coeff_row = tl.sum(tl.where(row_mask[:, None], b_q_new_coeff, 0.0), axis=0)
            q_new_row = tl.sum(coeff_row[:, None] * b_U, axis=0)
            b_q_new += tl.where(row_mask[:, None], q_new_row[None, :], 0.0)
        b_o = exp(b_G)[:, None] * b_qh0 + b_q_hist + b_q_new
        p_o = o + ((i_n * T + o_t[:, None]) * HV + i_hv) * V + o_v[None, :]
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_t[:, None] & mask_v[None, :],
        )

        write_pos = w_off + o_t
        tl.store(
            wu_base + write_pos[:, None] * HV * V + o_v[None, :],
            b_U.to(wu_base.dtype.element_ty),
            mask=mask_t[:, None] & mask_v[None, :],
        )
        if i_v == 0:
            stored_G = tl.where(is_write, b_G_local, b_G)
            tl.store(wG_base + write_pos, stored_G, mask=mask_t)
            if i_hv % (HV // H) == 0:
                tl.store(
                    wk_base + write_pos[:, None] * H * K + o_k[None, :],
                    b_kn.to(wk_base.dtype.element_ty),
                    mask=mask_t[:, None] & mask_k[None, :],
                )

        if ENABLE_STATE_COMMIT:
            if is_write:
                commit_decay = exp(g_start - b_Gh)
                b_Uc = b_uh.to(tl.float32) * tl.where(is_hist, commit_decay, 0.0)[:, None]
                b_Uc_hi = b_Uc.to(b_kh.dtype)
                b_Uc_lo = (b_Uc - b_Uc_hi.to(tl.float32)).to(b_kh.dtype)
                b_hc = (
                    b_h.to(tl.float32) * exp(g_start)
                    + tl.dot(tl.trans(b_kh), b_Uc_hi)
                    + tl.dot(tl.trans(b_kh), b_Uc_lo)
                )
                tl.store(
                    p_h0,
                    b_hc.to(p_h0.dtype.element_ty),
                    mask=mask_k[:, None] & mask_v[None, :],
                )


@triton.jit
def _cached_replay_layered_commit_kernel(
    h0_source,
    h0_descriptors,
    old_u,
    old_k,
    old_G,
    history_descriptors,
    replay_work_items,
    n_writes,
    work_item_stride,
    pool_stride_layer,
    pool_stride_slot,
    old_u_stride_layer,
    old_k_stride_layer,
    old_G_stride_layer,
    old_u_stride_slot,
    old_k_stride_slot,
    old_G_stride_slot,
    HIST: tl.constexpr,
    BH: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NUM_PERSISTENT: tl.constexpr,
    USE_L2_STATE_CACHE: tl.constexpr,
    USE_L2_STREAMING_INPUTS: tl.constexpr,
    USE_L2_SHARED_INPUTS: tl.constexpr,
    USE_H0_DESCRIPTORS: tl.constexpr,
    USE_HISTORY_DESCRIPTORS: tl.constexpr,
    REPLAY_SLOT_FIELD: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """Advance every local GDN layer from one cached-history snapshot.

    A work item addresses the checkpoint by its state slot (field 1) and the
    history by its replay slot (field ``REPLAY_SLOT_FIELD``, or the state slot
    when that is negative). Each layer's history is either at a fixed layer
    stride from ``old_*`` or, with ``USE_HISTORY_DESCRIPTORS``, at the pointers
    and slot strides of its ``history_descriptors`` row.
    """
    pid = tl.program_id(0)
    total_work = tl.load(n_writes) * NUM_LAYERS * HV * NV
    for tile_id in tl.range(
        pid,
        total_work,
        NUM_PERSISTENT,
        num_stages=PIPE_STAGES,
        flatten=True,
    ):
        i_v = tile_id % NV
        tile_id = tile_id // NV
        i_hv = tile_id % HV
        tile_id = tile_id // HV
        layer = tile_id % NUM_LAYERS
        work_idx = tile_id // NUM_LAYERS
        i_h = i_hv // (HV // H)

        work_base = replay_work_items + work_idx * work_item_stride
        slot = tl.load(work_base + 1).to(tl.int64)
        b_pnat = tl.load(work_base + 2)
        b_buf = tl.load(work_base + 3).to(tl.int64)
        if REPLAY_SLOT_FIELD >= 0:
            replay_slot = tl.load(work_base + REPLAY_SLOT_FIELD).to(tl.int64)
        else:
            replay_slot = slot
        layer_i64 = layer.to(tl.int64)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        o_hist = tl.arange(0, BH)
        mask_k = o_k < K
        mask_v = o_v < V
        is_hist = (o_hist < b_pnat) & (o_hist < HIST)

        if USE_HISTORY_DESCRIPTORS:
            descriptor = history_descriptors + layer_i64 * 6
            layer_u = tl.load(descriptor).to(tl.pointer_type(old_u.type.element_ty))
            layer_k = tl.load(descriptor + 1).to(tl.pointer_type(old_k.type.element_ty))
            layer_G = tl.load(descriptor + 2).to(tl.pointer_type(old_G.type.element_ty))
            u_slot = layer_u + replay_slot * tl.load(descriptor + 3)
            k_slot = layer_k + replay_slot * tl.load(descriptor + 4)
            G_slot = layer_G + replay_slot * tl.load(descriptor + 5)
        else:
            u_slot = old_u + layer_i64 * old_u_stride_layer + replay_slot * old_u_stride_slot
            k_slot = old_k + layer_i64 * old_k_stride_layer + replay_slot * old_k_stride_slot
            G_slot = old_G + layer_i64 * old_G_stride_layer + replay_slot * old_G_stride_slot
        hk_base = k_slot + b_buf * HIST * H * K + i_h * K
        hu_base = u_slot + b_buf * HIST * HV * V + i_hv * V
        hG_base = G_slot + (b_buf * HV + i_hv) * HIST

        if USE_L2_SHARED_INPUTS:
            b_kh = tl.load(
                hk_base + o_hist[:, None] * H * K + o_k[None, :],
                mask=is_hist[:, None] & mask_k[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_kh = tl.load(
                hk_base + o_hist[:, None] * H * K + o_k[None, :],
                mask=is_hist[:, None] & mask_k[None, :],
                other=0,
            )
        if USE_L2_STREAMING_INPUTS:
            b_uh = tl.load(
                hu_base + o_hist[:, None] * HV * V + o_v[None, :],
                mask=is_hist[:, None] & mask_v[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_uh = tl.load(
                hu_base + o_hist[:, None] * HV * V + o_v[None, :],
                mask=is_hist[:, None] & mask_v[None, :],
                other=0,
            )
        b_Gh = tl.load(hG_base + o_hist, mask=is_hist, other=0.0).to(tl.float32)
        g_start = tl.load(hG_base + b_pnat - 1).to(tl.float32)

        if USE_H0_DESCRIPTORS:
            descriptor = h0_descriptors + layer_i64 * 2
            layer_h0 = tl.load(descriptor).to(tl.pointer_type(h0_source.type.element_ty))
            layer_slot_stride = tl.load(descriptor + 1)
            p_h0 = (
                layer_h0 + slot * layer_slot_stride + i_hv * V * K + o_k[:, None] + o_v[None, :] * K
            )
        else:
            p_h0 = (
                h0_source
                + layer_i64 * pool_stride_layer
                + slot * pool_stride_slot
                + i_hv * V * K
                + o_k[:, None]
                + o_v[None, :] * K
            )
        if USE_L2_STATE_CACHE:
            b_h = tl.load(
                p_h0,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0,
                cache_modifier=".cg",
            )
        else:
            b_h = tl.load(
                p_h0,
                mask=mask_k[:, None] & mask_v[None, :],
                other=0,
            )
        commit_decay = exp(g_start - b_Gh)
        b_Uc = b_uh.to(tl.float32) * tl.where(is_hist, commit_decay, 0.0)[:, None]
        b_Uc_hi = b_Uc.to(b_kh.dtype)
        b_Uc_lo = (b_Uc - b_Uc_hi.to(tl.float32)).to(b_kh.dtype)
        b_hc = (
            b_h.to(tl.float32) * exp(g_start)
            + tl.dot(tl.trans(b_kh), b_Uc_hi)
            + tl.dot(tl.trans(b_kh), b_Uc_lo)
        )
        tl.store(
            p_h0,
            b_hc.to(p_h0.dtype.element_ty),
            mask=mask_k[:, None] & mask_v[None, :],
        )


def _assert_dense_from(tensor: torch.Tensor, dim: int, name: str) -> None:
    """Dimensions ``dim`` onward must be dense; the ones before may have any stride."""
    expected = []
    step = 1
    for size in reversed(tensor.shape[dim:]):
        expected.append(step)
        step *= size
    actual = tensor.stride()[dim:]
    assert all(
        size == 1 or stride == want
        for size, stride, want in zip(tensor.shape[dim:], actual, reversed(expected))
    ), f"{name} must be dense from dimension {dim}; got strides {tensor.stride()}"


# Columns of a history descriptor row: per-layer base pointers of old_u, old_k
# and old_G, then their slot strides in elements.
_HISTORY_DESCRIPTOR_WIDTH = 6


def build_gdn_history_descriptors(
    old_u: list[torch.Tensor], old_k: list[torch.Tensor], old_G: list[torch.Tensor]
) -> torch.Tensor:
    """Describe per-layer ``[slots, 2, ...]`` history views for the all-layer commit.

    Use it when the layers' views are not evenly spaced in memory. Every view
    must be dense after its slot dimension.
    """
    assert len(old_u) == len(old_k) == len(old_G) > 0
    rows = []
    for u, k, G in zip(old_u, old_k, old_G, strict=True):
        for name, tensor in (("old_u", u), ("old_k", k), ("old_G", G)):
            _assert_dense_from(tensor, 1, name)
        rows.append(
            (u.data_ptr(), k.data_ptr(), G.data_ptr(), u.stride(0), k.stride(0), G.stride(0))
        )
    return torch.tensor(rows, dtype=torch.int64, device=old_u[0].device)


def commit_gdn_cached_replay_history_layers(
    *,
    ssm_states: torch.Tensor,
    ssm_state_descriptors: Optional[torch.Tensor] = None,
    ssm_state_num_layers: Optional[int] = None,
    ssm_state_layer_stride: Optional[int] = None,
    ssm_state_slot_stride: Optional[int] = None,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    replay_work_items: torch.Tensor,
    n_writes: torch.Tensor,
    history_size: int,
    history_descriptors: Optional[torch.Tensor] = None,
    persistent_waves: int = 2,
    commit_block_v: Optional[int] = None,
    commit_num_warps: Optional[int] = None,
    commit_pipeline_stages: Optional[int] = None,
) -> None:
    """Advance all local layer checkpoints from cached replay histories.

    The history is either stacked ``[layers, slots, 2, ...]`` tensors, with any
    layer and slot strides, or, with ``history_descriptors`` from
    :func:`build_gdn_history_descriptors`, the first layer's ``[slots, 2, ...]``
    views (for dtype and shape) and one descriptor row per layer. Work items of
    width ``REPLAY_WORK_ITEM_WIDTH`` address the history by their replay slot;
    narrower rows by their state slot.
    """
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import REPLAY_WORK_REPLAY_SLOT

    use_state_descriptors = ssm_state_descriptors is not None
    state_stride_args = (
        ssm_state_num_layers,
        ssm_state_layer_stride,
        ssm_state_slot_stride,
    )
    use_state_strides = any(value is not None for value in state_stride_args)
    assert not (use_state_descriptors and use_state_strides)
    if use_state_descriptors or use_state_strides:
        assert ssm_states.ndim == 4
        _, HV, V, K = ssm_states.shape
        if use_state_descriptors:
            assert ssm_state_descriptors is not None
            assert ssm_state_descriptors.ndim == 2 and ssm_state_descriptors.shape[1] == 2
            assert ssm_state_descriptors.dtype == torch.int64
            assert ssm_state_descriptors.device == ssm_states.device
            assert ssm_state_descriptors.is_contiguous()
            num_layers = ssm_state_descriptors.shape[0]
            pool_stride_layer = 0
            pool_stride_slot = 0
        else:
            assert all(value is not None for value in state_stride_args)
            assert ssm_state_num_layers is not None
            assert ssm_state_layer_stride is not None
            assert ssm_state_slot_stride is not None
            num_layers = ssm_state_num_layers
            pool_stride_layer = ssm_state_layer_stride
            pool_stride_slot = ssm_state_slot_stride
            assert num_layers > 0 and pool_stride_slot > 0
            assert num_layers == 1 or pool_stride_layer != 0
    else:
        assert ssm_states.ndim == 5
        num_layers, _, HV, V, K = ssm_states.shape
        pool_stride_layer = ssm_states.stride(0)
        pool_stride_slot = ssm_states.stride(1)
    use_history_descriptors = history_descriptors is not None
    if use_history_descriptors:
        assert old_u.ndim == 5 and old_k.ndim == 5 and old_G.ndim == 4
        assert history_descriptors.shape == (num_layers, _HISTORY_DESCRIPTOR_WIDTH)
        assert history_descriptors.dtype == torch.int64
        assert history_descriptors.device == ssm_states.device
        assert history_descriptors.is_contiguous()
        history_dense_dim = 1
        layer_strides = (0, 0, 0)
        slot_strides = (0, 0, 0)
    else:
        assert old_u.ndim == 6 and old_k.ndim == 6 and old_G.ndim == 5
        assert old_u.shape[0] == num_layers
        assert old_k.shape[0] == num_layers
        assert old_G.shape[0] == num_layers
        history_dense_dim = 2
        layer_strides = (old_u.stride(0), old_k.stride(0), old_G.stride(0))
        slot_strides = (old_u.stride(1), old_k.stride(1), old_G.stride(1))
    H = old_k.shape[-2]
    assert old_u.shape[-4:] == (2, history_size, HV, V)
    assert old_k.shape[-4:] == (2, history_size, H, K)
    assert old_G.shape[-3:] == (2, HV, history_size)
    for name, tensor in (("old_u", old_u), ("old_k", old_k), ("old_G", old_G)):
        _assert_dense_from(tensor, history_dense_dim, name)
    # Rows start with (position, state slot, PNAT, active buffer); full-width
    # rows add the replay slot.
    assert replay_work_items.ndim == 2 and replay_work_items.shape[1] >= 4
    assert replay_work_items.stride(1) == 1
    replay_slot_field = (
        REPLAY_WORK_REPLAY_SLOT if replay_work_items.shape[1] > REPLAY_WORK_REPLAY_SLOT else -1
    )
    assert replay_work_items.dtype == torch.int32
    assert n_writes.dtype == torch.int32 and n_writes.numel() == 1
    assert persistent_waves > 0

    N = replay_work_items.shape[0]
    if N == 0 or num_layers == 0:
        return
    BK = triton.next_power_of_2(K)
    BH = triton.next_power_of_2(history_size)
    assert BK == K and BH <= 16
    per_layer_head_tiles = N * HV
    use_tuned_bf16_mapping = (
        history_size <= 16
        and HV == 4 * H
        and K == 128
        and V == 128
        and ssm_states.dtype == torch.bfloat16
    )
    use_small_grid_mapping = (
        use_tuned_bf16_mapping and per_layer_head_tiles <= _SMALL_GRID_HEAD_TILES
    )
    if commit_block_v is None:
        commit_block_v = 64 if use_small_grid_mapping else triton.next_power_of_2(V)
    if commit_num_warps is None:
        commit_num_warps = (
            8
            if use_tuned_bf16_mapping
            and per_layer_head_tiles >= _EIGHT_WARP_COMMIT_HEAD_TILES
            and commit_block_v == 128
            else 2
            if use_tuned_bf16_mapping
            else 4
        )
    use_large_workload_mapping = (
        per_layer_head_tiles >= _L2_STREAMING_HEAD_TILES
        if use_tuned_bf16_mapping
        else N >= 128 and ssm_states.dtype == torch.bfloat16
    )
    if commit_pipeline_stages is None:
        commit_pipeline_stages = (
            5
            if use_tuned_bf16_mapping and per_layer_head_tiles >= _PIPELINED_COMMIT_HEAD_TILES
            else 5
            if not use_tuned_bf16_mapping and use_large_workload_mapping
            else 1
        )
    assert triton.next_power_of_2(commit_block_v) == commit_block_v
    assert commit_block_v <= V and commit_pipeline_stages > 0

    commit_nv = triton.cdiv(V, commit_block_v)
    num_sms = torch.cuda.get_device_properties(ssm_states.device).multi_processor_count
    total_tiles = N * num_layers * HV * commit_nv
    num_persistent = min(num_sms * persistent_waves, total_tiles)
    _cached_replay_layered_commit_kernel[(num_persistent,)](
        h0_source=ssm_states,
        h0_descriptors=(ssm_state_descriptors if ssm_state_descriptors is not None else n_writes),
        old_u=old_u,
        old_k=old_k,
        old_G=old_G,
        history_descriptors=(history_descriptors if use_history_descriptors else n_writes),
        replay_work_items=replay_work_items,
        n_writes=n_writes,
        work_item_stride=replay_work_items.stride(0),
        pool_stride_layer=pool_stride_layer,
        pool_stride_slot=pool_stride_slot,
        old_u_stride_layer=layer_strides[0],
        old_k_stride_layer=layer_strides[1],
        old_G_stride_layer=layer_strides[2],
        old_u_stride_slot=slot_strides[0],
        old_k_stride_slot=slot_strides[1],
        old_G_stride_slot=slot_strides[2],
        HIST=history_size,
        BH=BH,
        NUM_LAYERS=num_layers,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=commit_block_v,
        NV=commit_nv,
        NUM_PERSISTENT=num_persistent,
        USE_L2_STATE_CACHE=use_large_workload_mapping,
        USE_L2_STREAMING_INPUTS=use_large_workload_mapping,
        USE_L2_SHARED_INPUTS=use_large_workload_mapping,
        USE_H0_DESCRIPTORS=use_state_descriptors,
        USE_HISTORY_DESCRIPTORS=use_history_descriptors,
        REPLAY_SLOT_FIELD=replay_slot_field,
        PIPE_STAGES=commit_pipeline_stages,
        num_warps=commit_num_warps,
        num_stages=commit_pipeline_stages,
    )


@input_guard(
    exclude_args=[
        "q",
        "k",
        "v",
        "g",
        "beta",
        "packed_qkv",
        "ssm_states",
        "old_u",
        "old_k",
        "old_G",
        "old_beta",
        "cache_buf_idx",
        "prev_num_accepted_tokens",
        "replay_work_items",
        "n_writes",
        "output",
    ]
)
def fused_recurrent_gated_delta_rule_cached_replay_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    old_beta: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    history_size: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
    replay_work_items: Optional[torch.Tensor] = None,
    n_writes: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
    num_warps: Optional[int] = None,
    use_l2_state_cache: Optional[bool] = None,
    use_l2_streaming_inputs: Optional[bool] = None,
    use_l2_shared_inputs: Optional[bool] = None,
    main_num_stages: Optional[int] = None,
    packed_qkv: Optional[torch.Tensor] = None,
    use_all_layer_commit: bool = False,
    output: Optional[torch.Tensor] = None,
    replay_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN replay using cached causal updates rather than raw history.

    ``state_indices`` address the checkpoint in ``ssm_states``. The history
    (``old_u``, ``old_k``, ``old_G``: ``[slots, 2, ...]``, dense after the slot
    dimension) and its per-slot ``prev_num_accepted_tokens`` and
    ``cache_buf_idx`` (any element stride) are addressed by ``replay_indices``,
    which default to ``state_indices``.
    """
    del old_beta  # Signature-compatible placeholder; cached updates embed beta.
    N, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    assert q.shape == k.shape
    assert v.shape[:2] == (N, T)
    assert g.shape == (N, T, HV)
    assert beta.shape == (N, T, HV)
    assert g.stride(2) == 1, "g head dimension must be contiguous"
    assert beta.stride(2) == 1, "beta head dimension must be contiguous"
    assert g.stride(0) == T * g.stride(1), "g token rows must have a uniform stride"
    assert beta.stride(0) == T * beta.stride(1), "beta token rows must have a uniform stride"
    use_packed_qkv = packed_qkv is not None
    if use_packed_qkv:
        qkv_width = 2 * H * K + HV * V
        assert packed_qkv is not None
        assert packed_qkv.shape == (N * T, qkv_width)
        assert packed_qkv.dtype == q.dtype
        assert packed_qkv.device == q.device
        assert packed_qkv.is_contiguous()
    else:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        packed_qkv = q
    BK = triton.next_power_of_2(K)
    # The complete workload mapping is retained for the 4:1 shape; the
    # 2:1 fine tiling is enabled only through the measured low-batch range.
    use_production_bf16_shape = (
        T == 4
        and history_size <= 16
        and K == 128
        and V == 128
        and ssm_states.dtype == torch.bfloat16
    )
    head_tiles = N * HV
    use_tuned_bf16_mapping = use_production_bf16_shape and HV == 4 * H
    use_fine_grained_mapping = use_production_bf16_shape and _supports_fine_grained_replay_tiling(
        H, HV, head_tiles
    )
    use_small_grid_mapping = use_fine_grained_mapping and head_tiles <= _SMALL_GRID_HEAD_TILES
    if block_v is None:
        block_v = _default_cached_replay_block_v(use_fine_grained_mapping, head_tiles, V)
    if num_warps is None:
        num_warps = (
            2 if use_fine_grained_mapping and (use_small_grid_mapping or launch_with_pdl) else 4
        )
    BV = block_v
    use_large_workload_mapping = (
        head_tiles >= _L2_STREAMING_HEAD_TILES
        if use_tuned_bf16_mapping
        else N >= 128 and ssm_states.dtype == torch.bfloat16
    )
    if use_l2_state_cache is None:
        use_l2_state_cache = use_large_workload_mapping
    if use_l2_streaming_inputs is None:
        use_l2_streaming_inputs = use_large_workload_mapping
    if use_l2_shared_inputs is None:
        use_l2_shared_inputs = use_large_workload_mapping
    if main_num_stages is None:
        if use_tuned_bf16_mapping:
            main_num_stages = 2 if head_tiles >= _TWO_STAGE_REPLAY_HEAD_TILES else 1
        else:
            main_num_stages = 3 if use_large_workload_mapping else 1
    BT = triton.next_power_of_2(T)
    BH = triton.next_power_of_2(history_size)
    assert BK == K and triton.next_power_of_2(BV) == BV and BV <= V
    assert main_num_stages > 0
    assert T <= 8
    assert history_size >= T
    assert BH <= 16
    if scale is None:
        scale = K**-0.5

    fused_gating = A_log is not None
    if fused_gating:
        assert dt_bias is not None
        assert A_log.numel() == HV and dt_bias.numel() == HV
    else:
        A_log = q
        dt_bias = q

    if launch_with_pdl and get_sm_version() < 90:
        launch_with_pdl = False

    s_h0_0, s_h0_1, s_h0_2, s_h0_3 = ssm_states.stride()
    assert s_h0_3 == 1 and s_h0_2 == K and s_h0_1 == V * K
    assert old_u.shape[1:] == (2, history_size, HV, V)
    assert old_k.shape[1:] == (2, history_size, H, K)
    assert old_G.shape[1:] == (2, HV, history_size)
    for name, tensor in (("old_u", old_u), ("old_k", old_k), ("old_G", old_G)):
        _assert_dense_from(tensor, 1, name)
    if replay_indices is None:
        replay_indices = state_indices
    else:
        assert replay_indices.shape[0] >= N and replay_indices.stride(0) == 1
    assert prev_num_accepted_tokens.ndim == 1 and cache_buf_idx.ndim == 1
    if (replay_work_items is None) != (n_writes is None):
        raise ValueError("replay_work_items and n_writes must either both be set or both be None")
    if replay_work_items is not None:
        assert replay_work_items.shape[0] == N and replay_work_items.shape[1] >= 4
        assert replay_work_items.dtype == torch.int32
        assert replay_work_items.is_contiguous()
        assert n_writes is not None and n_writes.dtype == torch.int32
        assert n_writes.numel() == 1
        if not use_all_layer_commit:
            raise ValueError("Partitioned replay requires the all-layer commit")
    elif use_all_layer_commit:
        raise ValueError("use_all_layer_commit requires replay work items")

    if output is None:
        output = q.new_empty(N, T, HV, V)
    else:
        assert output.is_contiguous(), "output must be contiguous"
    NV = triton.cdiv(V, BV)
    grid = (NV, N * HV)

    def launch(
        *,
        enable_state_commit: bool,
        use_pdl: bool,
    ):
        _cached_replay_kernel[grid](
            q=q,
            k=k,
            v=v,
            packed_qkv=packed_qkv,
            g=g,
            beta=beta,
            o=output,
            h0_source=ssm_states,
            h0_indices=state_indices,
            replay_indices=replay_indices,
            old_u=old_u,
            old_k=old_k,
            old_G=old_G,
            cache_buf_idx=cache_buf_idx,
            pnat=prev_num_accepted_tokens,
            scale=scale,
            pool_stride_slot=s_h0_0,
            old_u_stride_slot=old_u.stride(0),
            old_k_stride_slot=old_k.stride(0),
            old_G_stride_slot=old_G.stride(0),
            pnat_stride=prev_num_accepted_tokens.stride(0),
            cache_buf_idx_stride=cache_buf_idx.stride(0),
            g_stride_row=g.stride(1),
            beta_stride_row=beta.stride(1),
            A_log=A_log,
            dt_bias=dt_bias,
            T=T,
            HIST=history_size,
            BT=BT,
            BH=BH,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            STATE_FP32=ssm_states.dtype == torch.float32,
            FUSED_GATING=fused_gating,
            USE_L2_STATE_CACHE=use_l2_state_cache,
            USE_L2_STREAMING_INPUTS=use_l2_streaming_inputs,
            USE_L2_SHARED_INPUTS=use_l2_shared_inputs,
            USE_PACKED_QKV=use_packed_qkv,
            LAUNCH_WITH_PDL=use_pdl,
            ENABLE_STATE_COMMIT=enable_state_commit,
            num_warps=num_warps,
            num_stages=main_num_stages,
            launch_pdl=use_pdl,
        )

    if replay_work_items is None:
        launch(
            enable_state_commit=True,
            use_pdl=launch_with_pdl,
        )
    else:
        # Large batches keep replay free of the checkpoint-state expression.
        # The cache manager advances every local GDN layer in one launch after
        # all layers have populated their history caches.
        launch(
            enable_state_commit=False,
            use_pdl=launch_with_pdl,
        )
    return output
