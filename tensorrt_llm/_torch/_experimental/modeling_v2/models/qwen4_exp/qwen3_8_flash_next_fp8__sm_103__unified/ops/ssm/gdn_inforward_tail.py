# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tails of an in-forward Mamba snapshot split in one launch: each split request's few tokens of
gated delta rule from its state at P (its live pool slot) into its staging row, read in place.

Per value head `i` (key head `i // (HV // H)`), fp32 state `h` `[V, K]` stored in bf16, `g` the decay:

    h *= g[t];  u = (v[t] - h k[t]) * beta[t];  h += u k[t]^T;  o[t] = h (K^-0.5 q[t])

token by token, so it matches FlashInfer's chunked form to rounding, not bitwise. q and k arrive
L2-normalized.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# The tail's token, sequence, slot and stage-row lists are scalar-read views.
@triton.jit(do_not_specialize_on_alignment=["tok_ptr", "cu_ptr", "slot_ptr", "row_ptr"])
def _gdn_tail_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    tok_ptr,
    cu_ptr,
    pool_ptr,
    slot_ptr,
    stage_ptr,
    row_ptr,
    out_ptr,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_gt,
    stride_bt,
    stride_ps,
    stride_ph,
    stride_pv,
    stride_ss,
    stride_sh,
    stride_sv,
    stride_ot,
    stride_oh,
    scale,
    GROUP: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    """One program: sequence `j`, value head `h`, value rows `[vb * BV, (vb + 1) * BV)` of the state."""
    j = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    kh = h // GROUP
    rv = vb * BV + tl.arange(0, BV)
    rk = tl.arange(0, K)
    slot = tl.load(slot_ptr + j).to(tl.int64)
    src = pool_ptr + slot * stride_ps + h * stride_ph + rv[:, None] * stride_pv + rk[None, :]
    state = tl.load(src).to(tl.float32)
    t0 = tl.load(cu_ptr + j)
    t1 = tl.load(cu_ptr + j + 1)
    for t in range(t0, t1):
        row = tl.load(tok_ptr + t).to(tl.int64)
        kv = tl.load(k_ptr + row * stride_kt + kh * stride_kh + rk).to(tl.float32)
        qv = tl.load(q_ptr + row * stride_qt + kh * stride_qh + rk).to(tl.float32) * scale
        vv = tl.load(v_ptr + row * stride_vt + h * stride_vh + rv).to(tl.float32)
        gt = tl.load(g_ptr + row * stride_gt + h).to(tl.float32)
        bt = tl.load(beta_ptr + row * stride_bt + h).to(tl.float32)
        state = state * gt
        u = (vv - tl.sum(state * kv[None, :], axis=1)) * bt
        state = state + u[:, None] * kv[None, :]
        o = tl.sum(state * qv[None, :], axis=1)
        tl.store(out_ptr + row * stride_ot + h * stride_oh + rv, o.to(out_ptr.dtype.element_ty))
    srow = tl.load(row_ptr + j).to(tl.int64)
    dst = stage_ptr + srow * stride_ss + h * stride_sh + rv[:, None] * stride_sv + rk[None, :]
    tl.store(dst, state.to(stage_ptr.dtype.element_ty))


def gdn_inforward_tail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    tail_tok: torch.Tensor,
    cu_tail: torch.Tensor,
    pool: torch.Tensor,
    tail_slots: torch.Tensor,
    stage: torch.Tensor,
    stage_rows: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Tail `j` covers rows `tail_tok[cu_tail[j] : cu_tail[j + 1]]` of q / k `[1, T, H, K]`, v, g / beta
    (fp32) and starts from `pool[tail_slots[j]]` (any slot stride); its final state goes to
    `stage[stage_rows[j]]` and its outputs to rows `tail_tok` of `out` `[1, T, HV, V]`."""
    num = tail_slots.numel()
    if num == 0:
        return
    _, _, heads, kdim = q.shape
    _, _, vheads, vdim = v.shape
    assert kdim == 128 and vdim % 32 == 0, (kdim, vdim)
    assert pool.stride(3) == 1 and stage.stride(3) == 1 and q.stride(3) == 1 and k.stride(3) == 1
    assert v.stride(3) == 1 and out.stride(3) == 1 and g.stride(2) == 1 and beta.stride(2) == 1
    bv = 32
    _gdn_tail_kernel[(num, vheads, vdim // bv)](
        q, k, v, g, beta, tail_tok, cu_tail, pool, tail_slots, stage, stage_rows, out,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        g.stride(1), beta.stride(1),
        pool.stride(0), pool.stride(1), pool.stride(2),
        stage.stride(0), stage.stride(1), stage.stride(2),
        out.stride(1), out.stride(2),
        kdim**-0.5,
        GROUP=vheads // heads, K=kdim, BV=bv, num_warps=4,
    )  # fmt: skip


def gdn_inforward_tail_reference(
    q, k, v, g, beta, tail_tok, cu_tail, pool, tail_slots, stage, stage_rows, out
) -> None:
    """The same recurrence in float64 on the host side of torch (tests)."""
    heads, vheads = q.shape[2], v.shape[2]
    group = vheads // heads
    scale = q.shape[3] ** -0.5
    cu = cu_tail.tolist()
    toks = tail_tok.tolist()
    for j, slot in enumerate(tail_slots.tolist()):
        state = pool[slot].double()
        for t in range(cu[j], cu[j + 1]):
            r = toks[t]
            kk = k[0, r].double().repeat_interleave(group, 0)  # [HV, K]
            qq = q[0, r].double().repeat_interleave(group, 0) * scale
            state = state * g[0, r].double()[:, None, None]
            u = (v[0, r].double() - torch.einsum("hvk,hk->hv", state, kk)) * beta[0, r].double()[
                :, None
            ]
            state = state + u[:, :, None] * kk[:, None, :]
            out[0, r] = torch.einsum("hvk,hk->hv", state, qq).to(out.dtype)
        stage[int(stage_rows[j])] = state.to(stage.dtype)


@triton.jit(do_not_specialize_on_alignment=["slot_ptr", "row_ptr", "win_ptr"])
def _conv_swap_kernel(
    conv_ptr,
    stage_ptr,
    x_ptr,
    slot_ptr,
    row_ptr,
    win_ptr,
    C,
    stride_cs,
    stride_cc,
    stride_ss,
    stride_sc,
    stride_xt,
    W: tl.constexpr,
    BW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program: split request `j`, channels `[cb * BLOCK, (cb + 1) * BLOCK)`."""
    j = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK + tl.arange(0, BLOCK)
    w = tl.arange(0, BW)
    m = (c[:, None] < C) & (w[None, :] < W)
    slot = tl.load(slot_ptr + j).to(tl.int64)
    srow = tl.load(row_ptr + j).to(tl.int64)
    live = conv_ptr + slot * stride_cs + c[:, None] * stride_cc + w[None, :]
    tl.store(
        stage_ptr + srow * stride_ss + c[:, None] * stride_sc + w[None, :],
        tl.load(live, mask=m),
        mask=m,
    )
    rows = tl.load(win_ptr + j * W + w, mask=w < W, other=0).to(tl.int64)
    tl.store(live, tl.load(x_ptr + rows[None, :] * stride_xt + c[:, None], mask=m), mask=m)


def gdn_inforward_conv_swap(
    conv: torch.Tensor,
    stage: torch.Tensor,
    x: torch.Tensor,
    tail_slots: torch.Tensor,
    stage_rows: torch.Tensor,
    conv_win: torch.Tensor,
) -> None:
    """For split request `j`: `stage[stage_rows[j]] = conv[tail_slots[j]]` (the window ending at the prompt end the
    conv pass left), then `conv[tail_slots[j]][:, w] = x[conv_win[j * W + w]]` (the raw rows before P), with `conv`
    `[S, C, W]` (any slot stride, rows dense), `stage` `[rows, C, W]` and `x` the `[T, C]` projection columns."""
    num = tail_slots.numel()
    if num == 0:
        return
    _, chans, width = conv.shape
    assert (
        conv.stride(2) == 1
        and conv.stride(1) == width
        and stage.stride(2) == 1
        and x.stride(1) == 1
    )
    assert x.dtype == conv.dtype == stage.dtype
    block = 256
    _conv_swap_kernel[(num, triton.cdiv(chans, block))](
        conv, stage, x, tail_slots, stage_rows, conv_win, chans,
        conv.stride(0), conv.stride(1), stage.stride(0), stage.stride(1), x.stride(0),
        W=width, BW=triton.next_power_of_2(width), BLOCK=block, num_warps=4,
    )  # fmt: skip
