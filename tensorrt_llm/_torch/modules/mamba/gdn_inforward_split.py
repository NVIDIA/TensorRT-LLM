# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data movement of the GDN in-forward Mamba snapshot split as four Triton launches.

Each kernel does the copies of several ``index_select`` / ``index_copy_`` calls of ``_inforward_split_conv_states``
and ``_inforward_split_prefill_core`` (gdn_mixer.py) at the point in the layer where their inputs are ready. They only
copy (no arithmetic, no dtype change), so every value is bit-identical to the index ops.

Index tensors are ``InforwardSnapshotSplit`` fields (int64 views of one persistent device buffer); the launch sizes
depend only on host-known counts, so the kernels capture into CUDA graphs.
"""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .mamba2_metadata import InforwardSnapshotSplit

# Elements per program along a row.
_BLOCK = 4096


@triton.jit
def _split_conv_state_kernel(
    cs_ptr,
    stage_ptr,
    x_ptr,
    tail_slots_ptr,
    stage_rows_ptr,
    conv_win_ptr,
    conv_dim,
    stride_cs_b,
    stride_cs_c,
    stride_cs_l,
    stride_st_b,
    stride_st_c,
    stride_st_l,
    stride_x_tok,
    W: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    """Split request j: stage its live conv state (the window that ends at the prompt end), then load the window that
    ends at P, i.e. the W raw inputs before P (rows conv_win[j * W : (j + 1) * W] of x, oldest first)."""
    j = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = cols < conv_dim
    slot = tl.load(tail_slots_ptr + j)
    row = tl.load(stage_rows_ptr + j)
    cs = cs_ptr + slot * stride_cs_b + cols * stride_cs_c
    st = stage_ptr + row * stride_st_b + cols * stride_st_c
    for w in tl.static_range(W):
        tl.store(st + w * stride_st_l, tl.load(cs + w * stride_cs_l, mask=m), mask=m)
        tok = tl.load(conv_win_ptr + j * W + w)
        tl.store(cs + w * stride_cs_l, tl.load(x_ptr + tok * stride_x_tok + cols, mask=m), mask=m)


# The per-batch counts (n_live here, k in _tail_scatter_kernel) only pick a row's branch: not specialized, so one
# compiled variant serves every batch (Triton would otherwise compile 1, multiples of 16 and other counts apart).
@triton.jit(do_not_specialize=["n_live"])
def _split_state_rows_kernel(
    final_ptr,
    ssm_ptr,
    tail_init_ptr,
    live_ptr,
    main_rows_ptr,
    split_rows_ptr,
    n_live,
    stride_final,
    stride_ssm,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Rows r < n_live: ssm[live[r]] = final[main_rows[r]]; rows n_live + j: tail_init[j] = final[split_rows[j]]."""
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < ROW
    if r < n_live:
        src = tl.load(main_rows_ptr + r)
        dst = ssm_ptr + tl.load(live_ptr + r) * stride_ssm
    else:
        j = r - n_live
        src = tl.load(split_rows_ptr + j)
        dst = tail_init_ptr + j.to(tl.int64) * ROW
    tl.store(dst + cols, tl.load(final_ptr + src * stride_final + cols, mask=m), mask=m)


@triton.jit
def _tail_token_gather_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    qt_ptr,
    kt_ptr,
    vt_ptr,
    gt_ptr,
    bt_ptr,
    tok_ptr,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    stride_beta,
    QK_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    G_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Tail token t: q / k / v / g / beta rows of token tail_tok[t] into row t of the compact tail tensors."""
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(tok_ptr + t)
    t64 = t.to(tl.int64)
    mv = cols < V_ROW
    tl.store(vt_ptr + t64 * V_ROW + cols, tl.load(v_ptr + tok * stride_v + cols, mask=mv), mask=mv)
    mq = cols < QK_ROW
    tl.store(qt_ptr + t64 * QK_ROW + cols, tl.load(q_ptr + tok * stride_q + cols, mask=mq), mask=mq)
    tl.store(kt_ptr + t64 * QK_ROW + cols, tl.load(k_ptr + tok * stride_k + cols, mask=mq), mask=mq)
    mg = cols < G_ROW
    tl.store(gt_ptr + t64 * G_ROW + cols, tl.load(g_ptr + tok * stride_g + cols, mask=mg), mask=mg)
    tl.store(
        bt_ptr + t64 * G_ROW + cols, tl.load(beta_ptr + tok * stride_beta + cols, mask=mg), mask=mg
    )


@triton.jit(do_not_specialize=["k"])
def _tail_scatter_kernel(
    tail_final_ptr,
    stage_ptr,
    tail_out_ptr,
    out_ptr,
    stage_rows_ptr,
    tok_ptr,
    k,
    stride_stage,
    stride_out,
    STATE_ROW: tl.constexpr,
    OUT_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Rows r < k: stage[stage_rows[r]] = tail_final[r]; rows k + t: out[tail_tok[t]] = tail_out[t]."""
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if r < k:
        m = cols < STATE_ROW
        dst = stage_ptr + tl.load(stage_rows_ptr + r) * stride_stage
        tl.store(
            dst + cols, tl.load(tail_final_ptr + r.to(tl.int64) * STATE_ROW + cols, mask=m), mask=m
        )
    else:
        t = r - k
        m = cols < OUT_ROW
        dst = out_ptr + tl.load(tok_ptr + t) * stride_out
        tl.store(
            dst + cols, tl.load(tail_out_ptr + t.to(tl.int64) * OUT_ROW + cols, mask=m), mask=m
        )


def _rows_contiguous(x: torch.Tensor, first: int) -> bool:
    """Dims first.. of x are packed (each row of them is one contiguous run)."""
    expected = 1
    for d in range(x.dim() - 1, first - 1, -1):
        if x.shape[d] != 1 and x.stride(d) != expected:
            return False
        expected *= x.shape[d]
    return True


def split_conv_states_ok(
    conv_states: torch.Tensor, mixed_qkv: torch.Tensor, stage_conv: torch.Tensor
) -> bool:
    """Layout / dtype conditions of split_conv_states (else keep the index ops)."""
    return (
        conv_states.dim() == 3
        and stage_conv.dim() == 3
        and mixed_qkv.dim() == 2
        and conv_states.dtype == stage_conv.dtype == mixed_qkv.dtype
        and conv_states.shape[1] == stage_conv.shape[1] == mixed_qkv.shape[1]
        and conv_states.shape[2] == stage_conv.shape[2]
        and mixed_qkv.stride(1) == 1
    )


def split_conv_states(
    conv_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    split: "InforwardSnapshotSplit",
    stage_conv: torch.Tensor,
) -> None:
    """``_inforward_split_conv_states`` with the window read from ``mixed_qkv`` itself: stage the split requests'
    conv states, then load the raw ``d_conv - 1`` inputs before P. ``mixed_qkv`` must still hold the raw inputs
    (the fused prefill conv does not write it)."""
    if split.k == 0:
        return
    width = conv_states.shape[2]
    conv_dim = conv_states.shape[1]
    block_c = 1024
    _split_conv_state_kernel[(split.k, triton.cdiv(conv_dim, block_c))](
        conv_states,
        stage_conv,
        mixed_qkv,
        split.tail_slots,
        split.stage_rows,
        split.conv_win,
        conv_dim,
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        stage_conv.stride(0),
        stage_conv.stride(1),
        stage_conv.stride(2),
        mixed_qkv.stride(0),
        W=width,
        BLOCK_C=block_c,
        num_warps=4,
    )


def split_core_ok(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    final: torch.Tensor,
    ssm_states: torch.Tensor,
    stage_ssm: torch.Tensor,
    core_out: torch.Tensor,
) -> bool:
    """Layout / dtype conditions of the three core-side copies (else keep the index ops)."""
    qkv_ok = all(
        x.dim() == 4 and x.shape[0] == 1 and _rows_contiguous(x, 2) for x in (query, key, value)
    )
    gb_ok = all(x.dim() == 3 and x.shape[0] == 1 and x.stride(2) == 1 for x in (g, beta))
    state_ok = (
        final.dim() == 4
        and ssm_states.dim() == 4
        and stage_ssm.dim() == 4
        and final.is_contiguous()
        and _rows_contiguous(ssm_states, 1)
        and _rows_contiguous(stage_ssm, 1)
        and final.dtype == ssm_states.dtype == stage_ssm.dtype
        and final.shape[1:] == ssm_states.shape[1:] == stage_ssm.shape[1:]
    )
    out_ok = core_out.dim() == 4 and core_out.shape[0] == 1 and _rows_contiguous(core_out, 2)
    return (
        qkv_ok
        and gb_ok
        and state_ok
        and out_ok
        and query.shape == key.shape
        and g.shape == beta.shape
        and g.dtype == beta.dtype
        and query.dtype == key.dtype
    )


def split_state_rows(
    final: torch.Tensor, ssm_states: torch.Tensor, split: "InforwardSnapshotSplit"
) -> torch.Tensor:
    """``ssm_states.index_copy_(0, live, final.index_select(0, main_rows))`` and
    ``final.index_select(0, split_rows)`` (returned: the tails' initial states) in one launch."""
    row = final[0].numel()
    tail_init = torch.empty((split.k, *final.shape[1:]), dtype=final.dtype, device=final.device)
    n_live = split.live.shape[0]
    rows = n_live + split.k
    if rows > 0:
        _split_state_rows_kernel[(rows, triton.cdiv(row, _BLOCK))](
            final,
            ssm_states,
            tail_init,
            split.live,
            split.main_rows,
            split.split_rows,
            n_live,
            final.stride(0),
            ssm_states.stride(0),
            ROW=row,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return tail_init


def gather_tail_tokens(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    split: "InforwardSnapshotSplit",
) -> list[torch.Tensor]:
    """``x.index_select(1, tail_tok)`` for query, key, value, g and beta in one launch (contiguous outputs)."""
    nt = split.tail_tok.shape[0]
    outs = [
        torch.empty((1, nt, *x.shape[2:]), dtype=x.dtype, device=x.device)
        for x in (query, key, value, g, beta)
    ]
    if nt > 0:
        qk_row = query.shape[2] * query.shape[3]
        v_row = value.shape[2] * value.shape[3]
        g_row = g.shape[2]
        _tail_token_gather_kernel[(nt, triton.cdiv(max(qk_row, v_row, g_row), _BLOCK))](
            query,
            key,
            value,
            g,
            beta,
            *outs,
            split.tail_tok,
            query.stride(1),
            key.stride(1),
            value.stride(1),
            g.stride(1),
            beta.stride(1),
            QK_ROW=qk_row,
            V_ROW=v_row,
            G_ROW=g_row,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return outs


def scatter_tail(
    tail_final: torch.Tensor,
    stage_ssm: torch.Tensor,
    tail_out: torch.Tensor,
    core_out: torch.Tensor,
    split: "InforwardSnapshotSplit",
) -> bool:
    """``stage_ssm.index_copy_(0, stage_rows, tail_final)`` and ``core_out[0].index_copy_(0, tail_tok,
    tail_out[0])`` in one launch. Returns False (nothing written) when the tensors do not match its conditions."""
    nt = split.tail_tok.shape[0]
    if not (
        tail_final.dtype == stage_ssm.dtype
        and tail_out.dtype == core_out.dtype
        and tail_final.is_contiguous()
        and tail_out.is_contiguous()
        and tail_final.shape[0] == split.k
        and tail_final.shape[1:] == stage_ssm.shape[1:]
        and _rows_contiguous(stage_ssm, 1)
        and tail_out.dim() == 4
        and tail_out.shape[1] == nt
        and core_out.dim() == 4
        and core_out.shape[0] == 1
        and _rows_contiguous(core_out, 2)
        and tail_out.shape[2:] == core_out.shape[2:]
    ):
        return False
    state_row = stage_ssm[0].numel()
    out_row = core_out.shape[2] * core_out.shape[3]
    rows = split.k + nt
    if rows > 0:
        _tail_scatter_kernel[(rows, triton.cdiv(max(state_row, out_row), _BLOCK))](
            tail_final,
            stage_ssm,
            tail_out,
            core_out,
            split.stage_rows,
            split.tail_tok,
            split.k,
            stage_ssm.stride(0),
            core_out.stride(1),
            STATE_ROW=state_row,
            OUT_ROW=out_row,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return True
