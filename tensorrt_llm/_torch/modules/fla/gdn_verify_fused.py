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
"""Fused GDN target-verify core: cached replay + gated RMSNorm (+ FP8) of an MTP verify step in one launch.

Replaces the cached replay -> token-major gated RMSNorm (+ FP8) launches between the convolution and out_proj on
the MTP target-verify path. With PDL every program reads its inputs only after griddepcontrol.wait. Outputs equal
the two-kernel chain up to rare rounding-level differences of the replay output.

The last program of each (request, value head) to finish (an atomic counter reaching NV - 1) reloads the full rows
from L2, writes the norm output and resets the counter for the next call.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.modules.fla.op import exp, gdc_wait_with_memory_clobber
from tensorrt_llm._utils import get_sm_version

GDN_VERIFY_FUSED_MAX_T = 16


@triton.jit
def _row(tile, o_t, r: tl.constexpr):
    """Row r of a [BT, X] tile as an [X] vector (exact: one nonzero term per column)."""
    return tl.sum(tl.where(o_t[:, None] == r, tile, 0.0), axis=0)


@triton.jit
def _elem(vec, o_t, s: tl.constexpr):
    """Element s of a [BT] vector as a scalar (exact)."""
    return tl.sum(tl.where(o_t == s, vec, 0.0), axis=0)


@triton.jit
def _tree_sum(t, BT: tl.constexpr):
    """Sum of the BT row terms of a tuple, grouped like a row reduction whose rows sit on lanes 4r (BT=8: butterfly
    xor 16 / 8 / 4) or, for BT=16, on lanes 4r and the second register (in-thread pair first)."""
    if BT == 8:
        return ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]))
    else:
        p0 = t[0] + t[8]
        p1 = t[1] + t[9]
        p2 = t[2] + t[10]
        p3 = t[3] + t[11]
        p4 = t[4] + t[12]
        p5 = t[5] + t[13]
        p6 = t[6] + t[14]
        p7 = t[7] + t[15]
        return ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))


@triton.jit
def _gdn_verify_fused_kernel(
    # producer outputs (read after griddepcontrol.wait): convolved q|k|v rows, gating inputs, gate
    x,
    x_stride_tok,
    g,
    beta,
    g_stride_row,
    beta_stride_row,
    z,
    z_stride_tok,
    # replay
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
    A_log,
    dt_bias,
    # gated RMSNorm
    norm_w,
    fp8_scale,
    y,
    y_stride_tok,
    counters,
    eps,
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
    NV: tl.constexpr,
    STATE_FP32: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    G = HV // H
    i_h = i_hv // G

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    o_hist = tl.arange(0, BH)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_t = o_t < T
    mask_hist_capacity = o_hist < HIST

    if LAUNCH_WITH_PDL:
        # Dependents are released at entry; every input is read after the dependency wait.
        tl.extra.cuda.gdc_launch_dependents()
        gdc_wait_with_memory_clobber()

    slot = tl.load(h0_indices + i_n)
    if slot >= 0:
        # ---------------- produced in an earlier step, or static ----------------
        b_pnat = tl.load(pnat + slot)
        b_buf = tl.load(cache_buf_idx + slot)
        is_write = (b_pnat + T) > HIST
        w_buf = tl.where(is_write, 1 - b_buf, b_buf)
        w_off = tl.where(is_write, 0, b_pnat)
        is_hist = (o_hist < b_pnat) & mask_hist_capacity

        wk_base = old_k + (slot.to(tl.int64) * 2 + w_buf) * HIST * H * K + i_h * K
        wu_base = old_u + (slot.to(tl.int64) * 2 + w_buf) * HIST * HV * V + i_hv * V
        wG_base = old_G + ((slot.to(tl.int64) * 2 + w_buf) * HV + i_hv) * HIST

        # Both history buffers, addressed by the slot only (no dependent round trip on PNAT / active buffer); select
        # and mask afterwards.
        hb0 = slot.to(tl.int64) * 2
        m_kh = mask_hist_capacity[:, None] & mask_k[None, :]
        m_uh = mask_hist_capacity[:, None] & mask_v[None, :]
        kh_0 = tl.load(
            old_k + hb0 * HIST * H * K + i_h * K + o_hist[:, None] * H * K + o_k[None, :],
            mask=m_kh,
            other=0,
        )
        kh_1 = tl.load(
            old_k + (hb0 + 1) * HIST * H * K + i_h * K + o_hist[:, None] * H * K + o_k[None, :],
            mask=m_kh,
            other=0,
        )
        uh_0 = tl.load(
            old_u + hb0 * HIST * HV * V + i_hv * V + o_hist[:, None] * HV * V + o_v[None, :],
            mask=m_uh,
            other=0,
        )
        uh_1 = tl.load(
            old_u + (hb0 + 1) * HIST * HV * V + i_hv * V + o_hist[:, None] * HV * V + o_v[None, :],
            mask=m_uh,
            other=0,
        )
        gh_0 = tl.load(
            old_G + (hb0 * HV + i_hv) * HIST + o_hist, mask=mask_hist_capacity, other=0.0
        )
        gh_1 = tl.load(
            old_G + ((hb0 + 1) * HV + i_hv) * HIST + o_hist, mask=mask_hist_capacity, other=0.0
        )
        use0 = b_buf == 0
        b_kh = tl.where(is_hist[:, None], tl.where(use0, kh_0, kh_1), 0)
        b_uh = tl.where(is_hist[:, None], tl.where(use0, uh_0, uh_1), 0)
        gh_sel = tl.where(use0, gh_0, gh_1).to(tl.float32)
        b_Gh = tl.where(is_hist, gh_sel, 0.0)
        g_start = tl.sum(tl.where((o_hist == b_pnat - 1) & (b_pnat > 0), gh_sel, 0.0), axis=0)
        p_h0 = (
            h0_source
            + slot.to(tl.int64) * pool_stride_slot
            + i_hv * V * K
            + o_k[:, None]
            + o_v[None, :] * K
        )
        b_h_pre = tl.load(
            p_h0,
            mask=mask_k[:, None] & mask_v[None, :],
            other=0,
        )

        g_A_exp = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
        g_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
        o_full = tl.arange(0, V)
        b_nw = tl.load(norm_w + o_full).to(tl.float32)

        # ---------------- producer outputs ----------------
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
        b_G_local = tl.cumsum(b_g, 0)
        b_G = g_start + b_G_local
        # Convolved q|k|v rows (packed rows of width x_stride_tok).
        qkv_row = (i_n * T + o_t[:, None]) * x_stride_tok
        b_q = tl.load(
            x + qkv_row + i_h * K + o_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=0,
        )
        b_k = tl.load(
            x + qkv_row + H * K + i_h * K + o_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=0,
        )
        b_v = tl.load(
            x + qkv_row + 2 * H * K + i_hv * V + o_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0,
        )

        # ---------------- cached replay (the math of _cached_replay_kernel) ----------------
        b_kf = b_k.to(tl.float32)
        b_qf = b_q.to(tl.float32)
        inv_k = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + 1e-6)
        inv_q = scale / tl.sqrt(tl.sum(b_qf * b_qf, 1) + 1e-6)
        b_kn = (b_kf * inv_k[:, None]).to(b_k.dtype)
        b_qn = (b_qf * inv_q[:, None]).to(b_q.dtype)

        if STATE_FP32:
            b_h = b_h_pre
            b_h_hi = b_h.to(b_kn.dtype)
            b_h_lo = (b_h - b_h_hi.to(tl.float32)).to(b_kn.dtype)
            b_kh0 = tl.dot(b_kn, b_h_hi) + tl.dot(b_kn, b_h_lo)
            b_qh0 = tl.dot(b_qn, b_h_hi) + tl.dot(b_qn, b_h_lo)
        else:
            b_h = b_h_pre.to(b_kn.dtype)
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

        # Forward substitution on a tuple of row vectors: the rows of rhs and the scalars A[r, s] are extracted up
        # front (independent work), so the serial chain is FMAs only.
        zero_v = _row(b_rhs, o_t, BT)  # +0.0 (no row matches), in the row-vector layout
        u_rows = ()
        for r in tl.static_range(BT):
            if r < T:
                a_row = _row(b_A, o_t, r)
                terms = ()
                for s_i in tl.static_range(BT):
                    if s_i < r:
                        terms = terms + (_elem(a_row, o_t, s_i) * u_rows[s_i],)
                    else:
                        terms = terms + (zero_v,)
                u_rows = u_rows + (_row(b_rhs, o_t, r) - _tree_sum(terms, BT),)
            else:
                u_rows = u_rows + (zero_v,)
        b_U = tl.where(o_t[:, None] == BT, b_rhs, 0.0)
        for r in tl.static_range(T):
            b_U = tl.where(o_t[:, None] == r, u_rows[r][None, :], b_U)
        incl = o_t[:, None] >= o_t[None, :]
        b_qk_new = tl.dot(b_qn, tl.trans(b_kn))
        b_q_new_coeff = tl.where(incl, b_qk_new * new_decay, 0.0)
        b_q_new = tl.where(o_t[:, None] == BT, b_rhs, 0.0)
        for r in tl.static_range(T):
            c_row = _row(b_q_new_coeff, o_t, r)
            terms = ()
            for s_i in tl.static_range(BT):
                if s_i <= r:
                    terms = terms + (_elem(c_row, o_t, s_i) * u_rows[s_i],)
                else:
                    terms = terms + (zero_v,)
            b_q_new = tl.where(o_t[:, None] == r, _tree_sum(terms, BT)[None, :], b_q_new)
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
            if i_hv % G == 0:
                tl.store(
                    wk_base + write_pos[:, None] * H * K + o_k[None, :],
                    b_kn.to(wk_base.dtype.element_ty),
                    mask=mask_t[:, None] & mask_k[None, :],
                )

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

        # Last of the NV programs of (request, value head): gated RMSNorm over the full 128-wide rows.
        # Every thread's o stores are ordered before thread 0's release (bar.sync, then a gpu-scope acq_rel
        # atomic); the last program's loads follow its acquire (and bypass L1).
        tl.debug_barrier()
        cnt = tl.atomic_add(counters + i_nh, 1, sem="acq_rel", scope="gpu")
        if cnt == NV - 1:
            p_on = o + ((i_n * T + o_t[:, None]) * HV + i_hv) * V + o_full[None, :]
            b_x = tl.load(p_on, mask=mask_t[:, None], other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            var = tl.sum(b_x * b_x, axis=1) / V
            rstd = 1.0 / tl.sqrt(var + eps)
            b_y = b_x * rstd[:, None] * b_nw[None, :]
            b_z = tl.load(
                z + (i_n * T + o_t[:, None]) * z_stride_tok + i_hv * V + o_full[None, :],
                mask=mask_t[:, None],
                other=0.0,
            ).to(tl.float32)
            b_y *= b_z * tl.sigmoid(b_z)
            if OUTPUT_FP8:
                # As the norm kernel: round to the replay output dtype, then the static quantization.
                b_y = b_y.to(o.dtype.element_ty).to(tl.float32)
                b_y *= tldevice.rcp_rn(tl.load(fp8_scale).to(tl.float32))
            tl.store(
                y + (i_n * T + o_t[:, None]) * y_stride_tok + i_hv * V + o_full[None, :],
                b_y.to(y.dtype.element_ty),
                mask=mask_t[:, None],
            )
            tl.atomic_xchg(counters + i_nh, 0, sem="relaxed", scope="gpu")
    else:
        if i_v == 0:
            # Padded request (no slot): defined zeros for out_proj instead of an uninitialized buffer.
            o_zv = tl.arange(0, V)
            tl.store(
                y + (i_n * T + o_t[:, None]) * y_stride_tok + i_hv * V + o_zv[None, :],
                tl.zeros([BT, V], dtype=tl.float32).to(y.dtype.element_ty),
                mask=mask_t[:, None],
            )


def gdn_verify_fused_supported(
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    history_size: int,
    ssm_dtype: torch.dtype,
) -> bool:
    """Shapes the fused kernel handles; others keep the separate kernels."""
    return (
        2 <= draft_token_num <= GDN_VERIFY_FUSED_MAX_T
        and draft_token_num <= history_size <= 16
        and head_k_dim == 128
        and head_v_dim == 128
        and num_v_heads % num_k_heads == 0
        and ssm_dtype in (torch.bfloat16, torch.float32)
    )


def gdn_verify_fused_update(
    mixed_qkv: torch.Tensor,
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
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    fp8_scale: Optional[torch.Tensor],
    counters: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
    num_warps: int = 4,
) -> torch.Tensor:
    """Cached replay + gated RMSNorm (+ FP8) of the MTP target-verify step on convolved q|k|v rows.

    mixed_qkv: [N*T, >= conv_dim] convolved q|k|v rows (q|k|v channels first, unit column stride). a, b: [N*T, HV]
    row-strided views. state_indices: [N] SSM slots. z: [N*T, HV, V] token-major gate. counters: int32 [>= N*HV],
    zero-initialized. output: scratch for the replay output o [N, T, HV, V].

    Returns the gated RMSNorm output [N*T, HV*V], FP8 E4M3 when ``fp8_scale`` is given, else the input dtype.
    """
    N, T = num_decodes, draft_token_num
    H, HV, K, V = num_k_heads, num_v_heads, head_k_dim, head_v_dim
    conv_dim = 2 * H * K + HV * V
    assert (
        mixed_qkv.shape[0] == N * T and mixed_qkv.shape[1] >= conv_dim and mixed_qkv.stride(1) == 1
    )
    assert (
        a.shape == (N * T, HV) and b.shape == (N * T, HV) and a.stride(1) == 1 and b.stride(1) == 1
    )
    s_h0_0, s_h0_1, s_h0_2, s_h0_3 = ssm_states.stride()
    assert s_h0_3 == 1 and s_h0_2 == K and s_h0_1 == V * K
    for name, tensor in (("old_u", old_u), ("old_k", old_k), ("old_G", old_G)):
        assert tensor.is_contiguous(), f"{name} must be contiguous"
    assert 2 <= T <= GDN_VERIFY_FUSED_MAX_T and T <= history_size <= 16
    assert K == 128 and V == 128 and HV % H == 0
    assert z.shape == (N * T, HV, V) and z.stride(2) == 1 and z.stride(1) == V
    assert norm_weight.numel() == V and norm_weight.is_contiguous()
    assert counters.dtype == torch.int32 and counters.numel() >= N * HV
    if launch_with_pdl and get_sm_version() < 90:
        launch_with_pdl = False
    if output is None:
        output = mixed_qkv.new_empty(N, T, HV, V)
    else:
        assert output.is_contiguous() and output.numel() == N * T * HV * V
    out_dtype = torch.float8_e4m3fn if fp8_scale is not None else mixed_qkv.dtype
    norm_out = torch.empty(N * T, HV * V, dtype=out_dtype, device=mixed_qkv.device)
    # _tree_sum groups 8 or 16 rows; shorter steps pad to 8 rows (rows >= T are masked).
    BT = max(8, triton.next_power_of_2(T))
    BV = 16 if BT <= 8 else 32
    NV = triton.cdiv(V, BV)
    grid = (NV, N * HV)
    _gdn_verify_fused_kernel[grid](
        mixed_qkv,
        mixed_qkv.stride(0),
        a,
        b,
        a.stride(0),
        b.stride(0),
        z,
        z.stride(0),
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
        A_log,
        dt_bias,
        norm_weight,
        fp8_scale if fp8_scale is not None else A_log,
        norm_out,
        norm_out.stride(0),
        counters,
        norm_eps,
        T=T,
        HIST=history_size,
        BT=BT,
        BH=triton.next_power_of_2(history_size),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        NV=NV,
        STATE_FP32=ssm_states.dtype == torch.float32,
        LAUNCH_WITH_PDL=launch_with_pdl,
        OUTPUT_FP8=fp8_scale is not None,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=launch_with_pdl,
    )
    return norm_out
