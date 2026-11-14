# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN prefill causal conv1d + post-conv in one pass over the token-major in_proj output.

Replaces extract_transpose_prefill_slice + causal_conv1d_fn + fused_gdn_post_conv for a context-only batch with
two Triton launches: conv + SiLU + q / k L2 norm + gates, then the new conv states. g, beta and the conv states equal
the stock pipeline's bit for bit; q / k / v differ in the SiLU (the stock CUDA conv takes tanh.approx above -7, this
kernel the exact form) and in the FP32 summation order of the q / k L2 norms; the optional linear-space gate
alpha = exp(g) is libdevice expf (torch.exp's value), up to a few tens of FP32 ulp (relative 3e-6) from the
stock Triton ``tl.exp`` of ``fused_gdn_post_conv(g_linear=True)``.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

# Conv kernel: tokens per program, channels per program (8 consecutive channels per thread), rows per pipelined
# load and its pipeline depth.
_CHUNK = 64
_BLOCK_C = 1024
_NUM_WARPS = 4
_STEP = 4
_NUM_STAGES = 3


# PTX of v * __frcp_rn(1 + __expf(-v)) (causal_conv1d_fwd_kernel's SiLU) for operand $i, result $o. The fast form
# is rcp.rn's own fast path (rcp.approx + one Newton step) without its range checks: exact for v > -87, where
# ex2's flushed denormals round to the same 1 + e.
_SILU_FAST = (
    "mul.rn.f32 t, $i, 0fBFB8AA3B; ex2.approx.ftz.f32 e, t; add.rn.f32 d, e, 0f3F800000; "
    "rcp.approx.ftz.f32 r, d; fma.rn.f32 q, d, r, 0fBF800000; sub.ftz.f32 q, 0f80000000, q; "
    "fma.rn.f32 r, r, q, r; mul.rn.f32 $o, $i, r;"
)
_SILU_EXACT = (
    "mul.rn.f32 t, $i, 0fBFB8AA3B; ex2.approx.f32 e, t; add.rn.f32 d, e, 0f3F800000; rcp.rn.f32 r, d; "
    "mul.rn.f32 $o, $i, r;"
)


def _silu8_asm() -> str:
    """SiLU of a thread's 8 values ($8..$15 -> $0..$7): the fast form unless some value of the warp is <= -87."""
    each = lambda form: "\n".join(  # noqa: E731
        form.replace("$i", f"${8 + k}").replace("$o", f"${k}") for k in range(8)
    )
    return (
        "{\n.reg .pred fast, all_fast;\n.reg .f32 m, t, e, d, r, q;\nmin.f32 m, $8, $9;\n"
        + "\n".join(f"min.f32 m, m, ${k};" for k in range(10, 16))
        + "\nsetp.gt.f32 fast, m, 0fC2AE0000;\nvote.sync.all.pred all_fast, fast, 0xffffffff;\n"
        + "@!all_fast bra SLOW;\n"
        + each(_SILU_FAST)
        + "\nbra DONE;\nSLOW:\n"
        + each(_SILU_EXACT)
        + "\nDONE:\n}"
    )


_SILU8_ASM = tl.constexpr(_silu8_asm())


@triton.jit
def _silu(v, BLOCK_C: tl.constexpr):
    """_SILU8_ASM over v [BLOCK_C] (8 consecutive values per thread)."""
    c0, c1 = tl.split(tl.reshape(v, (BLOCK_C // 8, 2, 2, 2)))  # value bits 2, 1, 0
    c00, c01 = tl.split(c0)
    c10, c11 = tl.split(c1)
    e0, e4 = tl.split(c00)
    e2, e6 = tl.split(c01)
    e1, e5 = tl.split(c10)
    e3, e7 = tl.split(c11)
    y0, y1, y2, y3, y4, y5, y6, y7 = tl.inline_asm_elementwise(
        _SILU8_ASM,
        "=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f",
        [e0, e1, e2, e3, e4, e5, e6, e7],
        dtype=(tl.float32,) * 8,
        is_pure=False,
        pack=1,
    )
    y = tl.join(
        tl.join(tl.join(y0, y4), tl.join(y2, y6)), tl.join(tl.join(y1, y5), tl.join(y3, y7))
    )
    return tl.reshape(y, (BLOCK_C,))


@triton.jit
def _inv_norm(s, eps):
    """1 / sqrt(s + eps) as Triton's fp32 1.0 / tl.sqrt(s + eps)."""
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 t, r;
        add.f32 t, $1, $2;
        sqrt.approx.ftz.f32 r, t;
        div.full.f32 $0, 0f3F800000, r;
        }""",
        "=f,f,f",
        [s, eps],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _l2_normalize(o, eps, BLOCK_C: tl.constexpr):
    """o [BLOCK_C] (128-channel heads, BF16) L2-normalized per head in FP32. A head is 16 runs of 8 channels (one
    thread each): a run's squares add in channel order, then the runs' sums in a butterfly across the 16 lanes."""
    f = tl.reshape(o.to(tl.float32), (BLOCK_C // 128, 16, 8))
    c0, c1 = tl.split(tl.reshape(f, (BLOCK_C // 128, 16, 2, 2, 2)))  # channel bits 2, 1, 0
    c00, c01 = tl.split(c0)
    c10, c11 = tl.split(c1)
    e0, e4 = tl.split(c00)
    e2, e6 = tl.split(c01)
    e1, e5 = tl.split(c10)
    e3, e7 = tl.split(c11)
    s = e0 * e0
    s = tl.fma(e1, e1, s)
    s = tl.fma(e2, e2, s)
    s = tl.fma(e3, e3, s)
    s = tl.fma(e4, e4, s)
    s = tl.fma(e5, e5, s)
    s = tl.fma(e6, e6, s)
    s = tl.fma(e7, e7, s)
    s = tl.sum(s, axis=1)
    inv = _inv_norm(s, tl.full(s.shape, eps, tl.float32))
    return tl.reshape(f * inv[:, None, None], (BLOCK_C,)).to(o.dtype)


@triton.jit
def _seq_state(seq, qsl_ptr, cidx_ptr, hinit_ptr, num_seqs, pad_slot):
    """(next sequence start, slot, has_initial_state, conv_ok) of sequence seq (or of the padding past the last
    one)."""
    in_seq = seq < num_seqs
    next_start = tl.load(qsl_ptr + seq + 1, mask=in_seq, other=2147483647)
    slot = tl.load(cidx_ptr + seq, mask=in_seq, other=pad_slot)
    hinit = tl.load(hinit_ptr + seq, mask=in_seq, other=0) != 0
    return next_start, slot, hinit, in_seq & (slot != pad_slot)


@triton.jit
def _window_tap(x_rows, p, start, stride_x, cs_row, cs_l, conv_ok, hinit, cols):
    """Raw input at p as the conv of a later row of the sequence starting at start sees it."""
    from_x = (cols >= 0) & conv_ok & (p >= start)
    from_state = (cols >= 0) & conv_ok & (p < start) & hinit
    xv = tl.load(x_rows + p.to(tl.int64) * stride_x, mask=from_x, other=0.0)
    sv = tl.load(cs_row + (p - start + 3) * cs_l, mask=from_state, other=0.0)
    return tl.where(from_x, xv, sv).to(tl.float32)


@triton.jit
def _tile_rows(tile, BLOCK_C: tl.constexpr):
    """The 4 rows of tile [4, BLOCK_C] as a tuple."""
    h0, h1 = tl.split(tl.reshape(tl.permute(tile, (1, 0)), (BLOCK_C, 2, 2)))
    r0, r2 = tl.split(h0)
    r1, r3 = tl.split(h1)
    return (r0, r1, r2, r3)


@triton.jit
def _conv_post_chunk(
    x_ptr,
    stride_x,
    w_ptr,
    bias_ptr,
    cs_ptr,
    cs_b,
    cs_c,
    cs_l,
    qsl_ptr,
    cidx_ptr,
    hinit_ptr,
    out_ptrs,
    out_row,
    cols,
    t0,
    num_tokens,
    num_seqs,
    pad_slot,
    eps,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    QK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    STEP: tl.constexpr,
):
    """_gdn_conv_post_kernel's conv for one program: STEP-row tile loads pipelined NUM_STAGES deep."""
    w_row = w_ptr + cols.to(tl.int64) * 4
    w0 = tl.load(w_row).to(tl.float32)
    w1 = tl.load(w_row + 1).to(tl.float32)
    w2 = tl.load(w_row + 2).to(tl.float32)
    w3 = tl.load(w_row + 3).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_C,), tl.float32)
    t_end = tl.minimum(t0 + CHUNK, num_tokens)

    # Sequence of t0: the largest s in [0, num_seqs] with qsl[s] <= t0.
    lo = num_seqs * 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(qsl_ptr + mid) <= t0:
            lo = mid
        else:
            hi = mid - 1
    seq = lo
    start = tl.load(qsl_ptr + tl.minimum(seq, num_seqs - 1))
    next_start, slot, hinit, conv_ok = _seq_state(
        seq, qsl_ptr, cidx_ptr, hinit_ptr, num_seqs, pad_slot
    )
    cs_row = cs_ptr + slot.to(tl.int64) * cs_b + cols.to(tl.int64) * cs_c
    x_rows = x_ptr + cols
    # Window: the raw inputs at t0 - 3, t0 - 2, t0 - 1 as the conv of row t0 sees them.
    wa = _window_tap(x_rows, t0 - 3, start, stride_x, cs_row, cs_l, conv_ok, hinit, cols)
    wb = _window_tap(x_rows, t0 - 2, start, stride_x, cs_row, cs_l, conv_ok, hinit, cols)
    wc = _window_tap(x_rows, t0 - 1, start, stride_x, cs_row, cs_l, conv_ok, hinit, cols)

    for tb in tl.range(t0, t_end, STEP, num_stages=NUM_STAGES):
        r = tb + tl.arange(0, STEP)
        tile = tl.load(
            x_ptr + r.to(tl.int64)[:, None] * stride_x + cols[None, :],
            mask=(r < t_end)[:, None],
            other=0.0,
        )
        cur = _tile_rows(tile, BLOCK_C)
        for u in tl.static_range(STEP):
            t = tb + u
            if t == next_start:
                # A new sequence (or the padding past the last one) starts at t.
                seq += 1
                while (seq < num_seqs) & (
                    tl.load(qsl_ptr + seq + 1, mask=seq < num_seqs, other=0) <= t
                ):
                    seq += 1
                next_start, slot, hinit, conv_ok = _seq_state(
                    seq, qsl_ptr, cidx_ptr, hinit_ptr, num_seqs, pad_slot
                )
                cs_row = cs_ptr + slot.to(tl.int64) * cs_b + cols.to(tl.int64) * cs_c
                take = (cols >= 0) & conv_ok & hinit
                wa = tl.load(cs_row, mask=take, other=0.0).to(tl.float32)
                wb = tl.load(cs_row + cs_l, mask=take, other=0.0).to(tl.float32)
                wc = tl.load(cs_row + 2 * cs_l, mask=take, other=0.0).to(tl.float32)
            x = cur[u]
            xf = x.to(tl.float32)
            acc = tl.fma(w0, wa, bias)
            acc = tl.fma(w1, wb, acc)
            acc = tl.fma(w2, wc, acc)
            acc = tl.fma(w3, xf, acc)
            if SILU:
                acc = _silu(acc, BLOCK_C)
            o = tl.where(conv_ok, acc.to(x.dtype), x)
            if QK:
                o = _l2_normalize(o, eps, BLOCK_C)
            tl.store(out_ptrs + t.to(tl.int64) * out_row, o, mask=(cols >= 0) & (t < t_end))
            wa = wb
            wb = wc
            wc = xf


@triton.jit(do_not_specialize=["num_tokens", "num_seqs", "pad_slot"])
def _gdn_conv_post_kernel(
    x_ptr,
    stride_x,
    w_ptr,
    bias_ptr,
    cs_ptr,
    cs_b,
    cs_c,
    cs_l,
    qsl_ptr,
    cidx_ptr,
    hinit_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_tokens,
    num_seqs,
    pad_slot,
    eps,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    alpha_ptr,
    stride_a_tok,
    stride_a_head,
    stride_b_tok,
    stride_b_head,
    softplus_threshold,
    NK_D: tl.constexpr,
    NV_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    STEP: tl.constexpr,
    GATE_T: tl.constexpr,
    WITH_ALPHA: tl.constexpr,
):
    """Conv1d (width 4) + SiLU of CHUNK tokens x BLOCK_C channels (all q / k or all v), sliding a window of each
    channel's last 3 raw inputs; q / k heads are then L2-normalized, v programs also write their heads' gates. At a
    sequence start the window is the slot's old conv state (has_initial_state) or zeros. Rows outside a sequence or
    of a padded slot keep their raw input. Only reads conv states."""
    t0 = tl.program_id(0) * CHUNK
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    if tl.program_id(1) < 2 * NK_D // BLOCK_C:
        _conv_post_chunk(
            x_ptr,
            stride_x,
            w_ptr,
            bias_ptr,
            cs_ptr,
            cs_b,
            cs_c,
            cs_l,
            qsl_ptr,
            cidx_ptr,
            hinit_ptr,
            tl.where(cols < NK_D, q_ptr + cols, k_ptr + (cols - NK_D)),
            NK_D,
            cols,
            t0,
            num_tokens,
            num_seqs,
            pad_slot,
            eps,
            HAS_BIAS,
            SILU,
            True,
            CHUNK,
            BLOCK_C,
            NUM_STAGES,
            STEP,
        )
    else:
        _conv_post_chunk(
            x_ptr,
            stride_x,
            w_ptr,
            bias_ptr,
            cs_ptr,
            cs_b,
            cs_c,
            cs_l,
            qsl_ptr,
            cidx_ptr,
            hinit_ptr,
            v_ptr + (cols - 2 * NK_D),
            NV_D,
            cols,
            t0,
            num_tokens,
            num_seqs,
            pad_slot,
            eps,
            HAS_BIAS,
            SILU,
            False,
            CHUNK,
            BLOCK_C,
            NUM_STAGES,
            STEP,
        )
        _gates(
            a_ptr,
            b_ptr,
            A_log_ptr,
            dt_bias_ptr,
            g_ptr,
            beta_ptr,
            alpha_ptr,
            t0,
            num_tokens,
            (tl.program_id(1) * BLOCK_C - 2 * NK_D) // 128,
            stride_a_tok,
            stride_a_head,
            stride_b_tok,
            stride_b_head,
            softplus_threshold,
            NV_D // 128,
            BLOCK_C // 128,
            CHUNK,
            GATE_T,
            WITH_ALPHA,
        )


@triton.jit
def _gates(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    alpha_ptr,
    t0,
    num_tokens,
    h0,
    stride_a_tok,
    stride_a_head,
    stride_b_tok,
    stride_b_head,
    softplus_threshold,
    NUM_V_HEADS: tl.constexpr,
    HEADS: tl.constexpr,
    CHUNK: tl.constexpr,
    GATE_T: tl.constexpr,
    WITH_ALPHA: tl.constexpr,
):
    """g, beta FP32 of tokens [t0, t0 + CHUNK) x value heads [h0, h0 + HEADS) with _fused_gdn_post_conv_kernel's
    expressions; with WITH_ALPHA also alpha = exp(g), FlashInfer's linear-space gate, as libdevice expf (bit-identical
    to torch.exp; the stock kernel's tl.exp differs by up to a few tens of ulp)."""
    t = t0 + tl.arange(0, GATE_T)
    h = h0 + tl.arange(0, HEADS)
    t64 = t.to(tl.int64)
    m = ((t < num_tokens) & (t < t0 + CHUNK))[:, None]
    a_values = tl.load(
        a_ptr + t64[:, None] * stride_a_tok + h[None, :] * stride_a_head, mask=m, other=0.0
    ).to(tl.float32)
    b_values = tl.load(
        b_ptr + t64[:, None] * stride_b_tok + h[None, :] * stride_b_head, mask=m, other=0.0
    ).to(tl.float32)
    A_log = tl.load(A_log_ptr + h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h).to(tl.float32)
    gate_input = a_values + dt_bias[None, :]
    softplus = tl.where(
        gate_input <= softplus_threshold, tl.log(1.0 + tl.exp(gate_input)), gate_input
    )
    g_values = -tl.exp(A_log)[None, :] * softplus
    beta_values = tl.sigmoid(b_values)
    off = t64[:, None] * NUM_V_HEADS + h[None, :]
    tl.store(g_ptr + off, g_values, mask=m)
    tl.store(beta_ptr + off, beta_values, mask=m)
    if WITH_ALPHA:
        tl.store(alpha_ptr + off, tldevice.exp(g_values), mask=m)


@triton.jit
def _state_value(
    x_ptr, base, cols, cmask, start, end, hinit, w, stride_x_tok, stride_cs_l, KW: tl.constexpr
):
    p = end - (KW - 1) + w
    from_x = p >= start
    xv = tl.load(x_ptr + p.to(tl.int64) * stride_x_tok + cols, mask=cmask & from_x, other=0.0)
    sv = tl.load(
        base + (p - start + (KW - 1)) * stride_cs_l, mask=cmask & (from_x == 0) & hinit, other=0.0
    )
    return tl.where(from_x, xv, sv)


@triton.jit
def _gdn_conv_state_update_kernel(
    x_ptr,
    cs_ptr,
    qsl_ptr,
    cidx_ptr,
    hinit_ptr,
    pad_slot_id,
    conv_dim,
    stride_x_tok,
    stride_cs_b,
    stride_cs_c,
    stride_cs_l,
    KW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """New conv state of each sequence = its last KW - 1 inputs (the old state shifted by the length, or zeros,
    for short sequences), as causal_conv1d_fwd writes it back (empty sequences included). KW == 4."""
    s = tl.program_id(0)
    slot = tl.load(cidx_ptr + s)
    if slot != pad_slot_id:
        start = tl.load(qsl_ptr + s)
        end = tl.load(qsl_ptr + s + 1)
        hinit = tl.load(hinit_ptr + s) != 0
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = cols < conv_dim
        base = cs_ptr + slot.to(tl.int64) * stride_cs_b + cols * stride_cs_c
        v0 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 0, stride_x_tok, stride_cs_l, KW
        )
        v1 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 1, stride_x_tok, stride_cs_l, KW
        )
        v2 = _state_value(
            x_ptr, base, cols, cmask, start, end, hinit, 2, stride_x_tok, stride_cs_l, KW
        )
        tl.store(base, v0, mask=cmask)
        tl.store(base + stride_cs_l, v1, mask=cmask)
        tl.store(base + 2 * stride_cs_l, v2, mask=cmask)


def gdn_prefill_conv_post(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: str = "silu",
    l2_norm_eps: float = 1e-6,
    softplus_threshold: float = 20.0,
    pad_slot_id: int = -1,
    alpha_out: Optional[torch.Tensor] = None,
    out: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k (L2-normed), v [1, T, H, D] in mixed_qkv's dtype and g, beta [1, T, HV] FP32, with conv_states updated
    in place; g, beta and the conv states equal extract_transpose_prefill_slice + causal_conv1d_fn +
    fused_gdn_post_conv, q / k / v up to the SiLU form and the q / k L2 norms' FP32 summation order (module
    docstring). mixed_qkv: [T, conv_dim] BF16 token-major (gdn_prefill_conv_eligible).
    alpha_out: optional [1, T, HV] FP32 tensor that receives exp(g), bit-identical to torch.exp(g).
    out: optional contiguous (q, k, v, g, beta) of the shapes / dtypes above that the kernel writes and that are
    returned (e.g. views of a scratch shared by the layers of one iteration); same launches, same values."""
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")
    T, conv_dim = mixed_qkv.shape
    if not (
        gdn_prefill_conv_supported(
            head_k_dim,
            head_v_dim,
            conv_weight.reshape(conv_dim, -1).shape[1],
            activation,
            conv_dim,
            num_k_heads,
        )
        and conv_dim == (2 * num_k_heads + num_v_heads) * head_k_dim
        and gdn_prefill_conv_eligible(
            mixed_qkv, conv_weight, conv_states, query_start_loc, cache_indices, has_initial_state
        )
    ):
        raise ValueError("shapes / layouts not supported by gdn_prefill_conv_post")
    num_seqs = query_start_loc.shape[0] - 1
    D = head_k_dim
    dev = mixed_qkv.device
    if out is None:
        q = torch.empty((1, T, num_k_heads, D), dtype=mixed_qkv.dtype, device=dev)
        k = torch.empty_like(q)
        v = torch.empty((1, T, num_v_heads, D), dtype=mixed_qkv.dtype, device=dev)
        g = torch.empty((1, T, num_v_heads), dtype=torch.float32, device=dev)
        beta = torch.empty((1, T, num_v_heads), dtype=torch.float32, device=dev)
    else:
        q, k, v, g, beta = out
        shapes = (
            ((1, T, num_k_heads, D),) * 2 + ((1, T, num_v_heads, D),) + ((1, T, num_v_heads),) * 2
        )
        dtypes = (mixed_qkv.dtype,) * 3 + (torch.float32,) * 2
        if any(
            t.shape != shape or t.dtype != dtype or not t.is_contiguous()
            for t, shape, dtype in zip(out, shapes, dtypes)
        ):
            raise ValueError(
                "out must be contiguous (q, k, v, g, beta) of [1, T, Hk, D] x 2, [1, T, HV, D] in mixed_qkv's "
                "dtype and [1, T, HV] FP32 x 2"
            )
    if alpha_out is not None and (
        alpha_out.shape != (1, T, num_v_heads)
        or alpha_out.dtype != torch.float32
        or not alpha_out.is_contiguous()
    ):
        raise ValueError("alpha_out must be a contiguous [1, T, num_v_heads] FP32 tensor")
    if num_seqs == 0:
        return q, k, v, g, beta
    if T > 0:
        _gdn_conv_post_kernel[(triton.cdiv(T, _CHUNK), conv_dim // _BLOCK_C)](
            mixed_qkv,
            mixed_qkv.stride(0),
            conv_weight,
            conv_bias,
            conv_states,
            conv_states.stride(0),
            conv_states.stride(1),
            conv_states.stride(2),
            query_start_loc,
            cache_indices,
            has_initial_state,
            q,
            k,
            v,
            T,
            num_seqs,
            pad_slot_id,
            l2_norm_eps,
            a,
            b,
            A_log,
            dt_bias,
            g,
            beta,
            alpha_out if alpha_out is not None else g,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            softplus_threshold,
            NK_D=num_k_heads * D,
            NV_D=num_v_heads * D,
            HAS_BIAS=conv_bias is not None,
            SILU=activation is not None,
            CHUNK=_CHUNK,
            BLOCK_C=_BLOCK_C,
            NUM_STAGES=_NUM_STAGES,
            STEP=_STEP,
            GATE_T=triton.next_power_of_2(_CHUNK),
            WITH_ALPHA=alpha_out is not None,
            num_warps=_NUM_WARPS,
        )
    block_c = 1024
    _gdn_conv_state_update_kernel[(num_seqs, triton.cdiv(conv_dim, block_c))](
        mixed_qkv,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        pad_slot_id,
        conv_dim,
        mixed_qkv.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        KW=4,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return q, k, v, g, beta


def gdn_prefill_conv_supported(
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    activation: Optional[str],
    conv_dim: int,
    num_k_heads: int,
) -> bool:
    """Shapes the fused prefill conv handles: 128-wide heads, conv width 4, SiLU or none, q / k and v channels in
    whole _BLOCK_C-channel programs."""
    return (
        head_k_dim == head_v_dim == 128
        and conv_kernel_size == 4
        and activation in (None, "silu", "swish")
        and conv_dim % _BLOCK_C == 0
        and (2 * num_k_heads * head_k_dim) % _BLOCK_C == 0
    )


def gdn_prefill_conv_eligible(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> bool:
    """Run-time layout / dtype conditions of gdn_prefill_conv_post (else the caller keeps the stock kernels)."""
    return (
        mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.dim() == 2
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) % 8 == 0
        and mixed_qkv.data_ptr() % 16 == 0
        and conv_weight.dtype == torch.bfloat16
        and conv_weight.is_contiguous()
        and conv_states.dtype == torch.bfloat16
        and conv_states.dim() == 3
        and conv_states.shape[2] == 3
        and conv_states.shape[1] == mixed_qkv.shape[1]
        and query_start_loc.dtype == torch.int32
        and cache_indices.dtype == torch.int32
        and has_initial_state.dtype == torch.bool
    )
