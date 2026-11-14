# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gated-DeltaNet prefill front end in one pass: causal conv1d + SiLU over the
context rows, the q/k/v split, the q/k L2 norm and both gates, plus the final conv
states.

One Triton kernel reads the token-major in_proj_qkvz column view and writes contiguous
q, k, v, fp32 g (log space) and beta, replacing the transpose + C++
``causal_conv1d_fwd`` + ``fused_gdn_post_conv`` chain bit for bit. Each program owns
one (sequence-aligned 16-token tile, 128-channel head) block, so a head's L2 norm is
done in registers; ``REDUCE`` replays the order in which ``fused_gdn_post_conv``'s
layout sums the squared norm for that token count. The conv arithmetic is
``_gdn_conv_taps``'.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..quantization._nvfp4_triton import _mul_rn
from ._gdn_conv_taps import _final_state_tap, _first_tile_tap, _fma_rn, _silu_ref

#: This checkpoint's Gated DeltaNet at tp1: 16 q/k heads, 48 v heads, head dims 128, a
#: width-4 conv over the 10240 q|k|v channels.
H, HV, HEAD, WIDTH = 16, 48, 128, 4
CONV_DIM = (2 * H + HV) * HEAD
#: Constants the checkpoint's GDN fixes: the q/k L2 norm's epsilon and the decay gate's
#: softplus threshold; a slot of -1 marks padding.
L2_NORM_EPS, SOFTPLUS_THRESHOLD, PAD_SLOT_ID = 1e-6, 20.0, -1
_BLOCK_T, _NUM_WARPS = 16, 4


@triton.jit
def _drop_bit(x, BT: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """x[BT, M]; add the two halves selected by index bit N (lo + hi) -> [BT, M // 2]."""
    y = tl.reshape(x, [BT, M // (2 * N), 2, N])
    y = tl.permute(y, (0, 1, 3, 2))
    lo, hi = tl.split(y)
    return tl.reshape(lo + hi, [BT, M // 2])


@triton.jit
def _split_bit(x, BT: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """x[BT, M] -> (bit N clear, bit N set), each [BT, M // 2]."""
    y = tl.reshape(x, [BT, M // (2 * N), 2, N])
    y = tl.permute(y, (0, 1, 3, 2))
    lo, hi = tl.split(y)
    return tl.reshape(lo, [BT, M // 2]), tl.reshape(hi, [BT, M // 2])


@triton.jit
def _warps4(s, BT: tl.constexpr):
    """s[BT, 4] indexed by warp id -> [BT]."""
    s = _drop_bit(s, BT, 4, 2)
    t = _drop_bit(s, BT, 2, 1)
    return tl.reshape(t, [BT])


@triton.jit
def _sumsq(y, BT: tl.constexpr, D: tl.constexpr, REDUCE: tl.constexpr):
    if REDUCE == 4:
        # Single-token layout of _fused_gdn_post_conv_kernel (sizePerThread [1, 8],
        # threadsPerWarp [2, 16]): column c = l*8 + e. Per thread the squares
        # (not fused into the adds) are summed ((e0+e2)+(e4+e6)) + ((e1+e3)+(e5+e7)),
        # then lanes xor 8, 4, 2, 1.
        sq = _mul_rn(y, y)
        s = _drop_bit(sq, BT, 128, 2)  # e bit 1
        s = _drop_bit(s, BT, 64, 2)  # e bit 2
        s = _drop_bit(s, BT, 32, 1)  # e bit 0 -> [BT, 16], index l
        s = _drop_bit(s, BT, 16, 8)
        s = _drop_bit(s, BT, 8, 4)
        s = _drop_bit(s, BT, 4, 2)
        s = _drop_bit(s, BT, 2, 1)
        return tl.reshape(s, [BT])
    if REDUCE == 5:
        # Pure-prefill layout for T % 16 != 0 (sizePerThread [1, 1],
        # threadsPerWarp [16, 2], warpsPerCTA [1, 4]): column c = rep*8 + w*2 + l
        # (rep = c bits 3..6). Per thread (PTX of the post-conv kernel): f32x2 pairs of
        # fma(r, r, rn(r'^2)) for reps (r, r + 2), then ((f0 + f4) + (f8 + f12))
        # per parity, even + odd; lanes xor 16 (l); then the 4 warp partials.
        A, B = _split_bit(y, BT, 128, 16)  # rep bit 1
        f = _fma_rn(A, A, _mul_rn(B, B))  # [BT, 64]
        f = _drop_bit(f, BT, 64, 16)  # rep bit 2
        f = _drop_bit(f, BT, 32, 16)  # rep bit 3
        f = _drop_bit(f, BT, 16, 8)  # rep bit 0 (parity) -> [BT, 8], j = w*2 + l
        s = _drop_bit(f, BT, 8, 1)  # lane bit -> [BT, 4], index w
        return _warps4(s, BT)
    sq = y * y
    # REDUCE == 1: layout sizePerThread [8,1], threadsPerWarp [2,16], warpsPerCTA [1,4]:
    # column c = r*64 + w*16 + l. thread: sq[r0] + sq[r1]; lanes: xor 8,4,2,1;
    # then the 4 warp partials.
    s = _drop_bit(sq, BT, 128, 64)  # [BT, 64], j = w*16 + l
    s = _drop_bit(s, BT, 64, 8)
    s = _drop_bit(s, BT, 32, 4)
    s = _drop_bit(s, BT, 16, 2)
    s = _drop_bit(s, BT, 8, 1)  # [BT, 4], j = w
    return _warps4(s, BT)


@triton.jit
def _gdn_prefill_onepass_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    qsl_ptr,
    cache_idx_ptr,
    has_init_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    num_seqs,
    stride_x_t,
    stride_w_c,
    stride_w_k,
    stride_s_b,
    stride_s_c,
    stride_s_k,
    a_stride_t,
    a_stride_h,
    b_stride_t,
    b_stride_h,
    pad_slot_id,
    l2_norm_eps,
    softplus_threshold,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REDUCE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    ii = tl.arange(0, BLOCK_T)
    cc = tl.arange(0, HEAD_DIM)
    c_first = pid_h * HEAD_DIM

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < num_seqs
    starts = tl.load(qsl_ptr + offs_n, mask=n_mask, other=0)
    ends = tl.load(qsl_ptr + offs_n + 1, mask=n_mask, other=0)
    ntiles = tl.where(n_mask, tl.maximum((ends - starts + BLOCK_T - 1) // BLOCK_T, 1), 0)
    cum = tl.cumsum(ntiles, 0)
    # Spare tiles (seq == num_seqs) clamp to the last sequence with a tile
    # index past its end: every token is masked and tile != 0.
    seq = tl.minimum(tl.sum((cum <= pid_t).to(tl.int32), 0), num_seqs - 1)
    sel = offs_n == seq
    seq_start = tl.sum(tl.where(sel, starts, 0), 0)
    seq_end = tl.sum(tl.where(sel, ends, 0), 0)
    tile = pid_t - tl.sum(tl.where(sel, cum - ntiles, 0), 0)

    slot = tl.load(cache_idx_ptr + seq)
    is_pad = slot == pad_slot_id
    has_init = tl.load(has_init_ptr + seq) != 0
    has_init = has_init & (~is_pad)

    t_first = seq_start + tile * BLOCK_T
    t_mask = (t_first + ii) < seq_end
    c_mask = cc < HEAD_DIM
    mask = t_mask[:, None] & c_mask[None, :]

    x_base = x_ptr + t_first.to(tl.int64) * stride_x_t + c_first
    x_offs = ii[:, None] * stride_x_t + cc[None, :]
    s_base = state_ptr + slot.to(tl.int64) * stride_s_b + c_first.to(tl.int64) * stride_s_c
    s_offs_c = cc * stride_s_c

    x0_raw = tl.load(x_base + x_offs, mask=mask, other=0.0)
    if tile == 0:
        x1 = _first_tile_tap(
            x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t, stride_s_k, 1, WIDTH
        )
        x2 = _first_tile_tap(
            x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t, stride_s_k, 2, WIDTH
        )
        x3 = _first_tile_tap(
            x_base, x_offs, s_base, s_offs_c, ii, mask, has_init, stride_x_t, stride_s_k, 3, WIDTH
        )
    else:
        x1 = tl.load(x_base + x_offs - stride_x_t, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + x_offs - 2 * stride_x_t, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_base + x_offs - 3 * stride_x_t, mask=mask, other=0.0).to(tl.float32)

    w_base = w_ptr + c_first * stride_w_c
    w0 = tl.load(w_base + cc * stride_w_c + 0 * stride_w_k).to(tl.float32)
    w1 = tl.load(w_base + cc * stride_w_c + 1 * stride_w_k).to(tl.float32)
    w2 = tl.load(w_base + cc * stride_w_c + 2 * stride_w_k).to(tl.float32)
    w3 = tl.load(w_base + cc * stride_w_c + 3 * stride_w_k).to(tl.float32)
    acc = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
    acc = _fma_rn(tl.broadcast_to(w0[None, :], [BLOCK_T, HEAD_DIM]), x3, acc)
    acc = _fma_rn(tl.broadcast_to(w1[None, :], [BLOCK_T, HEAD_DIM]), x2, acc)
    acc = _fma_rn(tl.broadcast_to(w2[None, :], [BLOCK_T, HEAD_DIM]), x1, acc)
    acc = _fma_rn(tl.broadcast_to(w3[None, :], [BLOCK_T, HEAD_DIM]), x0_raw.to(tl.float32), acc)
    acc = _silu_ref(acc)
    res = acc.to(x_ptr.dtype.element_ty)
    # A padded sequence is skipped by the C++ conv, which leaves the input.
    res = tl.where(is_pad, x0_raw, res)
    rows = t_first + ii
    row_mask = t_mask

    if (tile == 0) & (~is_pad):
        offs_c = c_first + cc
        s0 = _final_state_tap(
            x_ptr,
            s_base,
            s_offs_c,
            offs_c,
            c_mask,
            seq_start,
            seq_end,
            has_init,
            stride_x_t,
            stride_s_k,
            0,
            WIDTH,
        )
        s1 = _final_state_tap(
            x_ptr,
            s_base,
            s_offs_c,
            offs_c,
            c_mask,
            seq_start,
            seq_end,
            has_init,
            stride_x_t,
            stride_s_k,
            1,
            WIDTH,
        )
        s2 = _final_state_tap(
            x_ptr,
            s_base,
            s_offs_c,
            offs_c,
            c_mask,
            seq_start,
            seq_end,
            has_init,
            stride_x_t,
            stride_s_k,
            2,
            WIDTH,
        )
        # All reads of this sequence's initial state for these channels
        # happen in this CTA; finish them before overwriting the state.
        tl.debug_barrier()
        tl.store(s_base + s_offs_c + 0 * stride_s_k, s0, mask=c_mask)
        tl.store(s_base + s_offs_c + 1 * stride_s_k, s1, mask=c_mask)
        tl.store(s_base + s_offs_c + 2 * stride_s_k, s2, mask=c_mask)

    rows64 = rows.to(tl.int64)
    st_mask = row_mask[:, None]
    if pid_h < 2 * NUM_K_HEADS:
        y = res.to(tl.float32)
        inv = 1.0 / tl.sqrt(_sumsq(y, BLOCK_T, HEAD_DIM, REDUCE) + l2_norm_eps)
        y *= inv[:, None]
        if pid_h < NUM_K_HEADS:
            out_ptr = q_ptr
            hh = pid_h
        else:
            out_ptr = k_ptr
            hh = pid_h - NUM_K_HEADS
        offs = rows64[:, None] * (NUM_K_HEADS * HEAD_DIM) + hh * HEAD_DIM + cc[None, :]
        tl.store(out_ptr + offs, y, mask=st_mask)
    else:
        vh = pid_h - 2 * NUM_K_HEADS
        offs = rows64[:, None] * (NUM_V_HEADS * HEAD_DIM) + vh * HEAD_DIM + cc[None, :]
        tl.store(v_ptr + offs, res, mask=st_mask)

        a_values = tl.load(
            a_ptr + rows64 * a_stride_t + vh * a_stride_h, mask=row_mask, other=0.0
        ).to(tl.float32)
        b_values = tl.load(
            b_ptr + rows64 * b_stride_t + vh * b_stride_h, mask=row_mask, other=0.0
        ).to(tl.float32)
        A_log = tl.load(A_log_ptr + vh).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + vh).to(tl.float32)
        gate_input = a_values + dt_bias
        softplus = tl.where(
            gate_input <= softplus_threshold,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        g_values = -tl.exp(A_log) * softplus
        beta_values = tl.sigmoid(b_values)
        gate_offs = rows64 * NUM_V_HEADS + vh
        tl.store(g_ptr + gate_offs, g_values, mask=row_mask)
        tl.store(beta_ptr + gate_offs, beta_values, mask=row_mask)


def reduce_layout(num_prefill_tokens: int) -> int:
    """The layout Triton picks for ``fused_gdn_post_conv``'s q/k tile over a
    channel-major ``[conv_dim, P]`` input (token stride 1, channel stride P: Triton
    specializes that stride as == 1 or divisible by 16, and the load vectorization
    decides the blocked layout of the [16, 128] tile)."""
    if num_prefill_tokens == 1:
        return 4  # sizePerThread [1, 8], threadsPerWarp [2, 16], warpsPerCTA [4, 1]
    if num_prefill_tokens % 16 == 0:
        return 1  # sizePerThread [8, 1], threadsPerWarp [2, 16], warpsPerCTA [1, 4]
    return 5  # sizePerThread [1, 1], threadsPerWarp [16, 2], warpsPerCTA [1, 4]


def _launch(x, weight, conv_states, query_start_loc, cache_indices, has_initial_state,
            a, b, A_log, dt_bias, num_k_heads, num_v_heads):  # fmt: skip
    num_prefill_tokens, dim = x.shape
    width = weight.shape[-1]
    weight2d = weight.view(dim, width) if weight.dim() != 2 else weight
    dev = x.device
    q = torch.empty((1, num_prefill_tokens, num_k_heads, HEAD), dtype=x.dtype, device=dev)
    k = torch.empty_like(q)
    v = torch.empty((1, num_prefill_tokens, num_v_heads, HEAD), dtype=x.dtype, device=dev)
    g = torch.empty((1, num_prefill_tokens, num_v_heads), dtype=torch.float32, device=dev)
    beta = torch.empty((1, num_prefill_tokens, num_v_heads), dtype=torch.float32, device=dev)
    num_seqs = query_start_loc.shape[0] - 1
    if num_prefill_tokens == 0 or num_seqs <= 0:
        return q, k, v, g, beta
    block_n = max(16, triton.next_power_of_2(num_seqs))
    grid_t = triton.cdiv(num_prefill_tokens, _BLOCK_T) + num_seqs
    _gdn_prefill_onepass_kernel[(grid_t, 2 * num_k_heads + num_v_heads)](
        x,
        weight2d,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        num_seqs,
        x.stride(0),
        weight2d.stride(0),
        weight2d.stride(1),
        conv_states.stride(-3),
        conv_states.stride(-2),
        conv_states.stride(-1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        PAD_SLOT_ID,
        L2_NORM_EPS,
        SOFTPLUS_THRESHOLD,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_DIM=HEAD,
        WIDTH=width,
        BLOCK_N=block_n,
        BLOCK_T=_BLOCK_T,
        REDUCE=reduce_layout(num_prefill_tokens),
        num_warps=_NUM_WARPS,
    )
    return q, k, v, g, beta


def _cell(why: str, seqlens: tuple[int, ...], **spec) -> Cell:
    return Cell(why=why, spec=dict(seqlens=seqlens, **spec))


class GdnPrefillOnepass(OpWrapper):
    """The GDN front end over P context tokens of N sequences. Sequence `n` (rows
    `[qsl[n], qsl[n+1])`, slot `s = cache_indices[n]`), channel `c`, token `t`:

        window = [conv_states[s, c, :] if has_initial_state[n] else 0, x[seq, c]]
        y[t, c] = bf16(silu(sum_j weight[c, j] * window[t + j]))         (fp32, j = 0..3)
        q, k = bf16(y_h / sqrt(sum(y_h^2) + 1e-6)) per 128-channel head  (channels 0..4095)
        v    = y                                                         (channels 4096..)
        g    = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)  (fp32, per v head)
        conv_states[s] = the sequence's last 3 inputs (shifted in from the old state
                         for a sequence shorter than 3)

    Returns `(q, k, v, g, beta)`: q, k `[1, P, H, 128]`, v `[1, P, HV, 128]` (x's dtype),
    g, beta `[1, P, HV]` fp32, all new and contiguous; `conv_states` is advanced in place.

    Fusion boundary: conv, SiLU, split, q/k norm, both gates, the conv-state advance.
    The projections producing `x` and `a`/`b`, slot allocation and the recurrence are
    the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: seqlens; init (per-sequence has_initial_state, default all True); pool
    # ("compact" | "coalesced": the cache manager's view, slot stride 48 layers' conv
    # states). x is the [P, 10240] column view of the [P, 16384] in_proj_qkvz output, a/b
    # the column halves of the [P, 96] in_proj_ba output.
    CELLS: tuple[Cell, ...] = (
        _cell("1 x 2048 context: P % 16 == 0 (REDUCE 1)", (2048,)),
        _cell("1 x 1 context: P == 1 (REDUCE 4)", (1,)),
        _cell(
            "ragged [1000, 37, 2] (REDUCE 5): a fresh sequence, one shorter than the conv window",
            (1000, 37, 2),
            init=(True, False, True),
        ),
        _cell("4 x 2048 contexts over the coalesced conv pool", (2048,) * 4, pool="coalesced"),
    )

    note = """
    Metadata consumed: `query_start_loc` (`[N + 1]` int32 or int64, the context prefix of
    the step's: `query_start_loc[:N + 1]` or `query_start_loc_long[:N + 1]` -- only token
    indices are computed from it), `cache_indices` (int32 `state_indices_p`),
    `has_initial_state` (bool, the context prefix). The head counts are read off the
    operands: HV = `len(A_log)`, H = `(x.shape[1] / 128 - HV) / 2`.

    Call site (modeling): every step with context rows, on the context rows only (decode
    riders of a mixed step go through `causal_conv1d_update` and `gdn_decode_t1`). x the
    `[P, 10240]` column view of in_proj_qkvz's output (unit channel stride, any row
    stride); weight the conv weight `[10240, 4]` (or `[10240, 1, 4]`); conv_states the
    layer's whole bf16 conv pool `[S, 10240, 3]` (any slot stride); a, b `[P, 48]` views
    of in_proj_ba's output; A_log, dt_bias `[48]`. The outputs feed
    `flashinfer_chunk_gated_delta_rule_q8` unchanged. Run the slot reset before it: a
    fresh slot's conv state must not be read (`has_initial_state` False reads zeros
    instead).

    Launch: grid `(cdiv(P, 16) + N, 2 * H + HV)`, 16-token tiles that never straddle a
    sequence, 4 warps. `REDUCE` from P alone: 4 at P == 1, 1 when P % 16 == 0, else 5.

    Numerics: the conv is the C++ kernel's fma.rn chain from +0 and non-ftz SiLU, rounded
    once to bf16; q/k normalized as `y * (1 / sqrt(sum y^2 + 1e-6))` (the prefill form),
    the sum in `fused_gdn_post_conv`'s order for this P. Bitwise equal to
    `extract_transpose_prefill_slice` + `causal_conv1d_fn` + `fused_gdn_post_conv` --
    for the Triton build whose layout choice `REDUCE` replays. The gates use the same
    expressions as `fused_gdn_post_conv` (`log(1 + exp(x))` below 20, approximate
    exp / log).

    A sequence whose slot is -1 passes x through as v / q / k input and touches no
    state (the C++ conv's behaviour); not certified, contexts always carry a slot.

    Not checked by the op (is_valid covers it): x's unit channel stride, operand
    lengths, slot ids distinct and in range (two sequences sharing a slot race on its
    conv state).

    Hard limits
      head dims 128, conv width 4 (the kernel's taps and reduce layouts)
    """

    def __call__(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_v_heads = A_log.shape[0]
        num_k_heads = (x.shape[1] // HEAD - num_v_heads) // 2
        return _launch(
            x, weight, conv_states, query_start_loc, cache_indices, has_initial_state,
            a, b, A_log, dt_bias, num_k_heads, num_v_heads,
        )  # fmt: skip

    def reference(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """`(q, k, v, g, beta, conv_states_after)` in float64, each rounded once to the
        op's output dtype (the conv output once to bf16 before the norm, as the op stores
        it). Mutates nothing."""
        dt = torch.float64
        num_tokens, dim = x.shape
        num_v_heads = A_log.shape[0]
        num_k_heads = (dim // HEAD - num_v_heads) // 2
        width = weight.shape[-1]
        w = weight.reshape(dim, width).to(dt)
        conv_after = conv_states.clone()
        y = torch.empty(num_tokens, dim, dtype=x.dtype, device=x.device)
        bounds = query_start_loc.tolist()
        for n, (slot, init) in enumerate(zip(cache_indices.tolist(), has_initial_state.tolist())):
            start, end = bounds[n], bounds[n + 1]
            if slot == PAD_SLOT_ID:
                y[start:end] = x[start:end]
                continue
            xs = x[start:end].to(dt)
            state_len = conv_states.shape[2]
            halo = (
                conv_states[slot].to(dt).t()
                if init
                else torch.zeros(state_len, dim, dtype=dt, device=x.device)
            )
            window = torch.cat([halo, xs])  # [state_len + L, dim]
            acc = sum(
                w[:, j]
                * window[state_len - (width - 1) + j : state_len - (width - 1) + j + end - start]
                for j in range(width)
            )
            y[start:end] = (acc * torch.sigmoid(acc)).to(x.dtype)
            conv_after[slot] = window[-state_len:].t().to(conv_states.dtype)
        key = num_k_heads * HEAD
        yq = y[:, : 2 * key].to(dt).reshape(num_tokens, 2 * num_k_heads, HEAD)
        yq = yq / torch.sqrt(yq.pow(2).sum(-1, keepdim=True) + L2_NORM_EPS)
        q = yq[:, :num_k_heads].to(x.dtype).unsqueeze(0)
        k = yq[:, num_k_heads:].to(x.dtype).unsqueeze(0)
        v = y[:, 2 * key :].reshape(1, num_tokens, num_v_heads, HEAD)
        gate = a.to(dt) + dt_bias.to(dt)
        softplus = torch.where(gate <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(gate)), gate)
        g = (-torch.exp(A_log.to(dt)) * softplus).float().unsqueeze(0)
        beta = torch.sigmoid(b.to(dt)).float().unsqueeze(0)
        return q, k, v.contiguous(), g, beta, conv_after

    def is_valid(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> None:
        num_tokens, dim = x.shape
        num_v_heads = A_log.shape[0]
        num_k_heads = (dim // HEAD - num_v_heads) // 2
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1, "bf16 x, unit channel stride"
        assert dim == (2 * num_k_heads + num_v_heads) * HEAD and num_k_heads > 0, (
            "x holds q | k | v at head dim 128 (H = (dim / 128 - HV) / 2, HV = len(A_log))"
        )
        assert weight.numel() == dim * WIDTH and weight.shape[-1] == WIDTH, "conv weight [dim, 4]"
        assert conv_states.dim() == 3 and conv_states.shape[1:] == (dim, WIDTH - 1), (
            "conv pool [S, dim, 3]"
        )
        assert conv_states.dtype == x.dtype, "the final conv states are stored in x's dtype"
        num_seqs = query_start_loc.shape[0] - 1
        bounds = query_start_loc.tolist()
        assert bounds[0] == 0 and bounds[-1] == num_tokens, "query_start_loc covers x's rows"
        for name, t in (("cache_indices", cache_indices), ("has_initial_state", has_initial_state)):
            assert t.shape == (num_seqs,) and t.is_contiguous(), f"{name} contiguous [N]"
        slots = cache_indices.tolist()
        live = [s for s in slots if s != PAD_SLOT_ID]
        assert len(set(live)) == len(live), "two sequences sharing a conv slot race on it"
        assert all(0 <= s < conv_states.shape[0] for s in live), "slots in [0, S)"
        for name, t in (("a", a), ("b", b)):
            assert t.shape == (num_tokens, num_v_heads), f"{name} [P, HV]"
        for name, t in (("A_log", A_log), ("dt_bias", dt_bias)):
            assert t.shape == (num_v_heads,) and t.is_contiguous(), f"{name} contiguous [HV]"


gdn_prefill_onepass = GdnPrefillOnepass()
