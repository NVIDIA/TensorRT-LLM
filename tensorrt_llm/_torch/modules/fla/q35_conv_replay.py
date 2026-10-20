# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fused GDN target-verify kernel: causal conv1d update + cached replay in one launch (Qwen3.5 low latency).

Replaces the pair `causal_conv1d_update` (spec-verify mode with intermediate conv windows) -> cached replay
(`fused_recurrent_gated_delta_rule_cached_replay_update`) on the MTP target-verify path. Each program owns one
(request, value head, value tile) like the replay kernel; it computes the depthwise conv + SiLU of the query/key
channels of its key head and of its value channels directly from the in_proj output, writes the per-token
intermediate conv windows (each channel by exactly one program), and then runs the cached replay on the conv
output kept in registers.

Arithmetic matches the two-kernel path (same bf16 products / fp32 accumulation order in the conv, same replay
math), so outputs are bitwise identical. Differences from the two-kernel path:
  * the persistent conv state is NOT rewritten in place (the two-kernel path writes the window after all draft
    tokens, which update_mamba_states() always overwrites with the accepted window before anything reads it);
  * every input produced in the previous step (SSM state, history, conv state, weights) is fetched before
    griddepcontrol.wait; in_proj / ba outputs only after it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.modules.fla.cached_replay import Q35_GDN_REPLAY_MAX_T
from tensorrt_llm._torch.modules.fla.op import exp
from tensorrt_llm._utils import get_sm_version


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
def _conv_taps(
    x,
    x_stride_tok,
    cs_base,
    cs_stride_dim,
    cs_stride_tok,
    row0,
    ch,
    o_t,
    mask_t,
    mask_c,
    T: tl.constexpr,
    J: tl.constexpr,
):
    """Window element J (0 = oldest) for every token t: state[t + J] if t + J < 3 else x[t + J - 3]."""
    src = o_t + J
    from_state = src < 3
    from_x = (src >= 3) & mask_t
    xs = tl.load(
        x + (row0 + src - 3)[:, None] * x_stride_tok + ch[None, :],
        mask=from_x[:, None] & mask_c[None, :],
        other=0.0,
    )
    ss = tl.load(
        cs_base + ch[None, :] * cs_stride_dim + src[:, None] * cs_stride_tok,
        mask=from_state[:, None] & mask_c[None, :],
        other=0.0,
    )
    return tl.where(from_state[:, None], ss, xs)


@triton.jit
def _q35_conv_replay_kernel(
    x,
    x_stride_tok,
    conv_w,
    w_stride_dim,
    w_stride_width,
    conv_state,
    cs_stride_slot,
    cs_stride_dim,
    cs_stride_tok,
    inter_conv,
    ic_stride_row,
    ic_stride_step,
    ic_stride_dim,
    ic_stride_win,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    old_u,
    old_k,
    old_G,
    cache_buf_idx,
    pnat,
    scale,
    pool_stride_slot,
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
    LAUNCH_WITH_PDL: tl.constexpr,
    ENABLE_STATE_COMMIT: tl.constexpr,
):
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
        # ---------------- pre-wait: everything produced in a previous step ----------------
        b_pnat = tl.load(pnat + slot)
        b_buf = tl.load(cache_buf_idx + slot)
        is_write = (b_pnat + T) > HIST
        w_buf = tl.where(is_write, 1 - b_buf, b_buf)
        w_off = tl.where(is_write, 0, b_pnat)
        is_hist = (o_hist < b_pnat) & mask_hist_capacity

        hk_base = old_k + (slot.to(tl.int64) * 2 + b_buf) * HIST * H * K + i_h * K
        hu_base = old_u + (slot.to(tl.int64) * 2 + b_buf) * HIST * HV * V + i_hv * V
        hG_base = old_G + ((slot.to(tl.int64) * 2 + b_buf) * HV + i_hv) * HIST
        wk_base = old_k + (slot.to(tl.int64) * 2 + w_buf) * HIST * H * K + i_h * K
        wu_base = old_u + (slot.to(tl.int64) * 2 + w_buf) * HIST * HV * V + i_hv * V
        wG_base = old_G + ((slot.to(tl.int64) * 2 + w_buf) * HV + i_hv) * HIST

        b_kh = tl.load(
            hk_base + o_hist[:, None] * H * K + o_k[None, :],
            mask=is_hist[:, None] & mask_k[None, :],
            other=0,
        )
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

        p_h0 = (
            h0_source
            + slot.to(tl.int64) * pool_stride_slot
            + i_hv * V * K
            + o_k[:, None]
            + o_v[None, :] * K
        )
        b_h_raw = tl.load(
            p_h0,
            mask=mask_k[:, None] & mask_v[None, :],
            other=0,
        )

        # conv weights / state of this program's channels: q and k of key head i_h, v of (i_hv, i_v tile)
        ch_q = i_h * K + o_k
        ch_k = H * K + i_h * K + o_k
        ch_v = 2 * H * K + i_hv * V + o_v
        wq0 = tl.load(conv_w + ch_q * w_stride_dim + 0 * w_stride_width, mask=mask_k, other=0.0)
        wq1 = tl.load(conv_w + ch_q * w_stride_dim + 1 * w_stride_width, mask=mask_k, other=0.0)
        wq2 = tl.load(conv_w + ch_q * w_stride_dim + 2 * w_stride_width, mask=mask_k, other=0.0)
        wq3 = tl.load(conv_w + ch_q * w_stride_dim + 3 * w_stride_width, mask=mask_k, other=0.0)
        wk0 = tl.load(conv_w + ch_k * w_stride_dim + 0 * w_stride_width, mask=mask_k, other=0.0)
        wk1 = tl.load(conv_w + ch_k * w_stride_dim + 1 * w_stride_width, mask=mask_k, other=0.0)
        wk2 = tl.load(conv_w + ch_k * w_stride_dim + 2 * w_stride_width, mask=mask_k, other=0.0)
        wk3 = tl.load(conv_w + ch_k * w_stride_dim + 3 * w_stride_width, mask=mask_k, other=0.0)
        wv0 = tl.load(conv_w + ch_v * w_stride_dim + 0 * w_stride_width, mask=mask_v, other=0.0)
        wv1 = tl.load(conv_w + ch_v * w_stride_dim + 1 * w_stride_width, mask=mask_v, other=0.0)
        wv2 = tl.load(conv_w + ch_v * w_stride_dim + 2 * w_stride_width, mask=mask_v, other=0.0)
        wv3 = tl.load(conv_w + ch_v * w_stride_dim + 3 * w_stride_width, mask=mask_v, other=0.0)
        cs_base = conv_state + slot.to(tl.int64) * cs_stride_slot
        g_A_exp = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
        g_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

        if LAUNCH_WITH_PDL:
            _gdc_wait_with_memory_clobber()

        # ---------------- post-wait: in_proj (x) and ba (g / beta) outputs ----------------
        row0 = i_n * T
        # Window taps J = 0..3 (0 = oldest); same bf16 products / fp32 accumulation order as the conv kernel.
        q0 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_q, o_t, mask_t, mask_k, T, 0)
        q1 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_q, o_t, mask_t, mask_k, T, 1)
        q2 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_q, o_t, mask_t, mask_k, T, 2)
        q3 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_q, o_t, mask_t, mask_k, T, 3)
        k0 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_k, o_t, mask_t, mask_k, T, 0)
        k1 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_k, o_t, mask_t, mask_k, T, 1)
        k2 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_k, o_t, mask_t, mask_k, T, 2)
        k3 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_k, o_t, mask_t, mask_k, T, 3)
        v0 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_v, o_t, mask_t, mask_v, T, 0)
        v1 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_v, o_t, mask_t, mask_v, T, 1)
        v2 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_v, o_t, mask_t, mask_v, T, 2)
        v3 = _conv_taps(x, x_stride_tok, cs_base, cs_stride_dim, cs_stride_tok, row0, ch_v, o_t, mask_t, mask_v, T, 3)

        acc_q = tl.zeros([BT, BK], dtype=tl.float32)
        acc_q += q0 * wq0[None, :]
        acc_q += q1 * wq1[None, :]
        acc_q += q2 * wq2[None, :]
        acc_q += q3 * wq3[None, :]
        acc_k = tl.zeros([BT, BK], dtype=tl.float32)
        acc_k += k0 * wk0[None, :]
        acc_k += k1 * wk1[None, :]
        acc_k += k2 * wk2[None, :]
        acc_k += k3 * wk3[None, :]
        acc_v = tl.zeros([BT, BV], dtype=tl.float32)
        acc_v += v0 * wv0[None, :]
        acc_v += v1 * wv1[None, :]
        acc_v += v2 * wv2[None, :]
        acc_v += v3 * wv3[None, :]
        acc_q = acc_q / (1 + tl.exp(-acc_q))
        acc_k = acc_k / (1 + tl.exp(-acc_k))
        acc_v = acc_v / (1 + tl.exp(-acc_v))
        b_q = acc_q.to(x.dtype.element_ty)
        b_k = acc_k.to(x.dtype.element_ty)
        b_v = acc_v.to(x.dtype.element_ty)

        # Intermediate conv windows after token t: [win[t+1], win[t+2], win[t+3]] = taps 1..3.
        ic_base = inter_conv + i_n.to(tl.int64) * ic_stride_row
        ic_t = o_t[:, None] * ic_stride_step
        if (i_hv % (HV // H) == 0) & (i_v == 0):
            mq = mask_t[:, None] & mask_k[None, :]
            pq = ic_base + ic_t + ch_q[None, :] * ic_stride_dim
            tl.store(pq + 0 * ic_stride_win, q1, mask=mq)
            tl.store(pq + 1 * ic_stride_win, q2, mask=mq)
            tl.store(pq + 2 * ic_stride_win, q3, mask=mq)
            pk = ic_base + ic_t + ch_k[None, :] * ic_stride_dim
            tl.store(pk + 0 * ic_stride_win, k1, mask=mq)
            tl.store(pk + 1 * ic_stride_win, k2, mask=mq)
            tl.store(pk + 2 * ic_stride_win, k3, mask=mq)
        mv = mask_t[:, None] & mask_v[None, :]
        pv = ic_base + ic_t + ch_v[None, :] * ic_stride_dim
        tl.store(pv + 0 * ic_stride_win, v1, mask=mv)
        tl.store(pv + 1 * ic_stride_win, v2, mask=mv)
        tl.store(pv + 2 * ic_stride_win, v3, mask=mv)

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
        x_gate = b_g + g_dt_bias
        softplus = tl.where(
            x_gate <= 20.0,
            0.6931471805599453 * tldevice.fast_log2f(1.0 + tldevice.fast_expf(x_gate)),
            x_gate,
        )
        b_g = tl.where(mask_t, -g_A_exp * softplus, 0.0)
        b_beta = tl.where(
            mask_t,
            tldevice.fast_dividef(1.0, 1.0 + tldevice.fast_expf(-b_beta)),
            0.0,
        )

        # ---------------- cached replay (identical to _cached_replay_kernel) ----------------
        if USE_QK_L2NORM_IN_KERNEL:
            b_kf = b_k.to(tl.float32)
            b_qf = b_q.to(tl.float32)
            inv_k = 1.0 / (tl.sqrt(tl.sum(b_kf * b_kf, 1)) + 1e-6)
            inv_q = scale / (tl.sqrt(tl.sum(b_qf * b_qf, 1)) + 1e-6)
            b_kn = (b_kf * inv_k[:, None]).to(b_k.dtype)
            b_qn = (b_qf * inv_q[:, None]).to(b_q.dtype)
        else:
            b_kn = b_k
            b_qn = (b_q.to(tl.float32) * scale).to(b_q.dtype)

        b_G_local = tl.cumsum(b_g, 0)
        b_G = g_start + b_G_local

        if STATE_FP32:
            b_h = b_h_raw
            b_h_hi = b_h.to(b_kn.dtype)
            b_h_lo = (b_h - b_h_hi.to(tl.float32)).to(b_kn.dtype)
            b_kh0 = tl.dot(b_kn, b_h_hi) + tl.dot(b_kn, b_h_lo)
            b_qh0 = tl.dot(b_qn, b_h_hi) + tl.dot(b_qn, b_h_lo)
        else:
            b_h = b_h_raw.to(b_kn.dtype)
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


def q35_conv_replay_supported(
    num_decodes: int,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_width: int,
    history_size: int,
    ssm_dtype: torch.dtype,
) -> bool:
    """Shapes the fused kernel handles (the Qwen3.5 TP decode shapes); other shapes keep the two-kernel path."""
    return (
        2 <= draft_token_num <= Q35_GDN_REPLAY_MAX_T
        and history_size <= 16
        and history_size >= draft_token_num
        and head_k_dim == 128
        and head_v_dim == 128
        and conv_width == 4
        and num_v_heads % num_k_heads == 0
        and num_decodes * num_v_heads <= 64
        and ssm_dtype in (torch.bfloat16, torch.float32)
    )


def q35_conv_replay_update(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_states: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    history_size: int,
    num_decodes: int,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    output: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
    block_v: int = 16,
    num_warps: int = 2,
) -> torch.Tensor:
    """Fused conv1d update + cached replay for the MTP target-verify step.

    mixed_qkv: [N*T, >= conv_dim] in_proj rows (pre-conv q|k|v channels first); a, b: [N*T, HV] row-strided views.
    conv_states: [slots, conv_dim, width - 1] (dim stride width - 1, token stride 1); intermediate_conv_window:
    [>= N, T, conv_dim, width - 1]; state_indices: [N] slots (conv and SSM). Returns o [N, T, HV, V].
    """
    N, T = num_decodes, draft_token_num
    H, HV, K, V = num_k_heads, num_v_heads, head_k_dim, head_v_dim
    conv_dim = 2 * H * K + HV * V
    assert mixed_qkv.shape[0] == N * T and mixed_qkv.shape[1] >= conv_dim and mixed_qkv.stride(1) == 1
    assert conv_weight.shape[0] == conv_dim and conv_weight.shape[1] == 4
    assert conv_states.shape[1] == conv_dim and conv_states.shape[2] == 3
    assert intermediate_conv_window.shape[1] == T and intermediate_conv_window.shape[2] == conv_dim
    assert a.shape == (N * T, HV) and b.shape == (N * T, HV) and a.stride(1) == 1 and b.stride(1) == 1
    s_h0_0, s_h0_1, s_h0_2, s_h0_3 = ssm_states.stride()
    assert s_h0_3 == 1 and s_h0_2 == K and s_h0_1 == V * K
    for name, tensor in (("old_u", old_u), ("old_k", old_k), ("old_G", old_G)):
        assert tensor.is_contiguous(), f"{name} must be contiguous"
    assert T <= Q35_GDN_REPLAY_MAX_T and history_size >= T and history_size <= 16
    if launch_with_pdl and get_sm_version() < 90:
        launch_with_pdl = False
    if output is None:
        output = mixed_qkv.new_empty(N, T, HV, V)
    else:
        assert output.is_contiguous()
    BV = block_v
    grid = (triton.cdiv(V, BV), N * HV)
    _q35_conv_replay_kernel[grid](
        mixed_qkv,
        mixed_qkv.stride(0),
        conv_weight,
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_states,
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        intermediate_conv_window,
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        intermediate_conv_window.stride(3),
        a,
        b,
        output,
        ssm_states,
        state_indices,
        old_u,
        old_k,
        old_G,
        cache_buf_idx,
        prev_num_accepted_tokens,
        K**-0.5,
        s_h0_0,
        a.stride(0),
        b.stride(0),
        A_log,
        dt_bias,
        T=T,
        HIST=history_size,
        BT=triton.next_power_of_2(T),
        BH=triton.next_power_of_2(history_size),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=True,
        STATE_FP32=ssm_states.dtype == torch.float32,
        LAUNCH_WITH_PDL=launch_with_pdl,
        ENABLE_STATE_COMMIT=True,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_with_pdl,
    )
    return output
