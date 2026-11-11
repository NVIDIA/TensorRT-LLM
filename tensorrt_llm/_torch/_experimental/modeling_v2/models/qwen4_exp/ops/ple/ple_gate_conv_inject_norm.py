# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE block tail for decode-shaped batches: the key and query grouped norms,
the per-stream sigmoid gate, the conv-input grouped norm, the dilated
depthwise short conv against the paged conv state (published back), the
residual add into the hyper-connection stream, and the next site's grouped
norm -- three Triton launches behind one call.

Called by the qwen4_exp target at every layout; uses the row-to-sequence
search of ``ple_ngram_embed``. Its ``reference`` is that chain in float64,
rounded to the activation dtype after each step: three grouped norms, the
sigmoid-gated value, the dilated depthwise conv over the carried window, two
adds and the next site's grouped norm. No ``torch.ops.trtllm`` op or FlashInfer
callable takes a dilation.

The three launches split where the parallelism changes shape. The norms and
the gate are row reductions, one program per (row, stream). The conv and the
state publish are per channel, and each sequence's conv state is 184 KB of
`[channels, 9]` rows: one program per (sequence, block of channels), so the
state moves as contiguous spans across many SMs instead of strided through
four. The residual add and the next site's norm are row reductions again.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.jit_variants import (
    compile_only,
    first_time,
    mark_warmed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ngram_token_history import (
    DEP4_SLOTS,
    TEP4_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_ngram_embed import (
    PER_SEQ_MAX_ROWS,
    _seq_of_rows,
)
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber


@triton.jit
def _rms(x, w, HIDDEN: tl.constexpr, eps):
    """Gemma RMS norm in fp32: `x * rsqrt(mean(x^2) + eps) * (1 + w)`."""
    return x * tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps) * (1.0 + w)


@triton.jit
def _injected_stream(
    hc_ptr,
    row,
    stride_hc,
    ch,
    cols,
    m,
    stream,
    block_ptr,
    inj_ptr,
    sh_ptr,
    gate_ptr,
    stride_block,
    stride_inj,
    stride_sh,
    stride_gate,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
):
    """One stream of the residual, fp32. With `HAS_INJECT` the previous MoE block's injection
    comes first, rounded as `qwen4_exp_hc_inject_norm` rounds it."""
    x = tl.load(hc_ptr + row * stride_hc + ch, mask=m, other=0.0).to(tl.float32)
    if HAS_INJECT:
        b = tl.load(block_ptr + row * stride_block + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_SHARED:
            sg = tl.sigmoid(tl.load(gate_ptr + row * stride_gate).to(tl.float32))
            sh = tl.load(sh_ptr + row * stride_sh + cols, mask=m, other=0.0)
            b = (b + sg * sh.to(tl.float32)).to(block_ptr.dtype.element_ty).to(tl.float32)
        g = tl.load(inj_ptr + row * stride_inj + stream).to(tl.float32)
        x = (x + b * g).to(hc_ptr.dtype.element_ty).to(tl.float32)
    return x


@triton.jit
def _ple_gate_kernel(
    hc_ptr,
    key_ptr,
    value_ptr,
    nk_ptr,
    nq_ptr,
    nc_ptr,
    gated_ptr,
    xc_ptr,
    stride_hc,
    stride_key,
    stride_value,
    stride_out,
    eps,
    clamp_min,
    sqrt_d,
    block_ptr,
    inj_ptr,
    sh_ptr,
    gate_ptr,
    stride_block,
    stride_inj,
    stride_sh,
    stride_gate,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Per (row, stream): the key and query grouped norms, the gate, the gated
    value and the conv-input grouped norm. Writes `gated` and `xc`. With `PDL` the norm
    weights are read before the wait, everything else after it."""
    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    m = cols < HIDDEN
    ch = stream * HIDDEN + cols
    wk = tl.load(nk_ptr + ch, mask=m, other=0.0).to(tl.float32)
    wq = tl.load(nq_ptr + ch, mask=m, other=0.0).to(tl.float32)
    wc = tl.load(nc_ptr + ch, mask=m, other=0.0).to(tl.float32)
    if PDL:
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    x = _injected_stream(hc_ptr, row, stride_hc, ch, cols, m, stream, block_ptr, inj_ptr, sh_ptr,
                         gate_ptr, stride_block, stride_inj, stride_sh, stride_gate, HAS_INJECT,
                         HAS_SHARED)  # fmt: skip
    kp = tl.load(key_ptr + row * stride_key + ch, mask=m, other=0.0).to(tl.float32)
    v = tl.load(value_ptr + row * stride_value + cols, mask=m, other=0.0).to(tl.float32)
    kn = _rms(kp, wk, HIDDEN, eps).to(tl.bfloat16).to(tl.float32)
    qn = _rms(x, wq, HIDDEN, eps).to(tl.bfloat16).to(tl.float32)
    g = tl.sum(kn * qn, axis=0) / sqrt_d
    sg = tl.sqrt(tl.maximum(tl.abs(g), clamp_min))
    sg = tl.where(g > 0, sg, tl.where(g < 0, -sg, 0.0))
    gated = ((1.0 / (1.0 + tl.exp(-sg))) * v).to(tl.bfloat16)
    xc = _rms(gated.to(tl.float32), wc, HIDDEN, eps).to(tl.bfloat16)
    tl.store(gated_ptr + row * stride_out + ch, gated, mask=m)
    tl.store(xc_ptr + row * stride_out + ch, xc, mask=m)


# The per-sequence metadata may begin at any row of a mixed step's tensors.
@triton.jit(
    do_not_specialize=["publish_end"],
    do_not_specialize_on_alignment=["slots_ptr", "carry_ptr", "qsl_ptr"],
)
def _ple_conv_kernel(
    xc_ptr,
    gated_ptr,
    ple_ptr,
    cw_ptr,
    state_ptr,
    slots_ptr,
    carry_ptr,
    qsl_ptr,
    stride_out,
    stride_state,
    pad_slot,
    publish_end,
    KSIZE: tl.constexpr,
    DIL: tl.constexpr,
    STATE: tl.constexpr,
    HEAD: tl.constexpr,
    MAX_P: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_CARRY: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Per (sequence, block of ROWS channels): the dilated depthwise conv of
    each of the sequence's rows against `[state | conv inputs]`, its SiLU added
    to the gated value (`ple`), and the state advanced in place for the
    sequences below `publish_end` (the others' states are left as they were).

    A channel's state is `STATE` contiguous columns. The publish moves it as a
    `[ROWS, HEAD]` tile of the first HEAD (a power of two) columns plus, when
    `STATE == HEAD + 1`, one `[ROWS]` vector for the last, so no lane is spent
    on padding and every index is a plain 2-D offset. Every load is issued
    before the first store."""
    if PDL:
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    s = tl.program_id(0)
    ch0 = tl.program_id(1) * ROWS
    begin = tl.load(qsl_ptr + s).to(tl.int64)
    length = (tl.load(qsl_ptr + s + 1).to(tl.int64) - begin).to(tl.int32)
    slot = tl.load(slots_ptr + s).to(tl.int64)
    live = slot != pad_slot
    # A sequence without carry reads a zero state (it still publishes).
    if HAS_CARRY:
        seeded = live & (tl.load(carry_ptr + s) != 0)
    else:
        seeded = live
    slot = tl.where(live, slot, 0)
    block = state_ptr + slot * stride_state + ch0 * STATE
    row0 = begin * stride_out + ch0
    xrows = xc_ptr + row0
    r = tl.arange(0, ROWS)
    i = tl.arange(0, HEAD)

    # The published state, `window[length : length + STATE]` per channel: the
    # surviving state columns, then this sequence's newest conv inputs.
    src = length + i
    keep = tl.load(
        block + r[:, None] * STATE + src[None, :], mask=seeded & (src < STATE)[None, :], other=0.0
    )
    fresh = tl.load(
        xrows + (src - STATE)[None, :] * stride_out + r[:, None],
        mask=live & (src >= STATE)[None, :],
        other=0.0,
    )
    head = tl.where((src < STATE)[None, :], keep, fresh)
    if STATE > HEAD:
        src_t = length + HEAD
        keep_t = tl.load(block + r * STATE + src_t, mask=seeded & (src_t < STATE), other=0.0)
        fresh_t = tl.load(
            xrows + (src_t - STATE) * stride_out + r, mask=live & (src_t >= STATE), other=0.0
        )
        tail = tl.where(src_t < STATE, keep_t, fresh_t)

    ch = ch0 + r
    for j in tl.static_range(MAX_P):
        valid = j < length
        gated = tl.load(gated_ptr + row0 + j * stride_out + r, mask=valid, other=0.0)
        acc = tl.zeros([ROWS], dtype=tl.float32)
        # Tap k of row j reads window column j + k * DIL of `[state | xc rows]`.
        for k in tl.static_range(KSIZE):
            q = j + k * DIL
            wtap = tl.load(cw_ptr + ch * KSIZE + k).to(tl.float32)
            if q < STATE:
                tap = tl.load(block + r * STATE + q, mask=seeded, other=0.0)
            else:
                tap = tl.load(xrows + (q - STATE) * stride_out + r, mask=valid, other=0.0)
            acc += wtap * tap.to(tl.float32)
        conv = acc.to(tl.bfloat16).to(tl.float32)
        y = (conv / (1.0 + tl.exp(-conv))).to(tl.bfloat16).to(tl.float32)
        out = (gated.to(tl.float32) + y).to(tl.bfloat16)
        tl.store(ple_ptr + row0 + j * stride_out + r, out, mask=valid)

    # Every read of these state rows -- `keep` and the taps -- lands before
    # the first store over them.
    tl.debug_barrier()
    publish = live & (s < publish_end)
    tl.store(block + r[:, None] * STATE + i[None, :], head, mask=publish)
    if STATE > HEAD:
        tl.store(block + r * STATE + HEAD, tail, mask=publish)


# The packed row and sequence counts follow the step.
@triton.jit(
    do_not_specialize=["num_rows", "num_seqs"],
    do_not_specialize_on_alignment=["slots_ptr", "carry_ptr", "qsl_ptr"],
)
def _ple_conv_rows_kernel(
    xc_ptr,
    gated_ptr,
    ple_ptr,
    cw_ptr,
    state_ptr,
    slots_ptr,
    carry_ptr,
    qsl_ptr,
    num_rows,
    num_seqs,
    stride_out,
    stride_state,
    pad_slot,
    KSIZE: tl.constexpr,
    DIL: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ROWS: tl.constexpr,
    LOG_S: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    """Context-shaped batches: per (BLOCK_T packed rows, ROWS channels), each
    row's dilated conv over `[state | conv inputs]` of its own sequence, its
    SiLU added to the gated value. Reads the state only; the publish is the
    per-sequence kernel with no rows, launched after."""
    t = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_rows
    ch = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    s = _seq_of_rows(qsl_ptr, t, num_seqs, LOG_S)
    begin = tl.load(qsl_ptr + s, mask=tmask, other=0).to(tl.int64)
    tpos = t - begin
    slot = tl.load(slots_ptr + s, mask=tmask, other=pad_slot).to(tl.int64)
    seeded = slot != pad_slot
    if HAS_CARRY:
        seeded = seeded & (tl.load(carry_ptr + s, mask=tmask, other=0) != 0)
    slot = tl.where(seeded, slot, 0)
    m2 = tmask[:, None]
    offs = t[:, None] * stride_out + ch[None, :]
    gated = tl.load(gated_ptr + offs, mask=m2, other=0.0)
    acc = tl.zeros([BLOCK_T, ROWS], dtype=tl.float32)
    for k in tl.static_range(KSIZE):
        wtap = tl.load(cw_ptr + ch * KSIZE + k).to(tl.float32)
        q = tpos + k * DIL
        from_state = q < STATE
        tap_s = tl.load(
            state_ptr + slot[:, None] * stride_state + ch[None, :] * STATE + q[:, None],
            mask=m2 & (from_state & seeded)[:, None],
            other=0.0,
        )
        tap_x = tl.load(
            xc_ptr + offs + (k * DIL - STATE) * stride_out,
            mask=m2 & (~from_state)[:, None],
            other=0.0,
        )
        tap = tl.where(from_state[:, None], tap_s, tap_x)
        acc += wtap[None, :] * tap.to(tl.float32)
    conv = acc.to(tl.bfloat16).to(tl.float32)
    y = (conv / (1.0 + tl.exp(-conv))).to(tl.bfloat16).to(tl.float32)
    out = (gated.to(tl.float32) + y).to(tl.bfloat16)
    tl.store(ple_ptr + offs, out, mask=m2)


@triton.jit
def _ple_residual_norm_kernel(
    hc_ptr,
    ple_ptr,
    nx_ptr,
    hc_out_ptr,
    normed_ptr,
    stride_hc,
    stride_ple,
    stride_hc_out,
    stride_normed,
    eps,
    block_ptr,
    inj_ptr,
    sh_ptr,
    gate_ptr,
    stride_block,
    stride_inj,
    stride_sh,
    stride_gate,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Per (row, stream): the PLE output added into the stream, then the next
    site's grouped norm of the result. With `PDL` the norm weight is read before the wait."""
    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    m = cols < HIDDEN
    ch = stream * HIDDEN + cols
    wx = tl.load(nx_ptr + ch, mask=m, other=0.0).to(tl.float32)
    if PDL:
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    x = _injected_stream(hc_ptr, row, stride_hc, ch, cols, m, stream, block_ptr, inj_ptr, sh_ptr,
                         gate_ptr, stride_block, stride_inj, stride_sh, stride_gate, HAS_INJECT,
                         HAS_SHARED)  # fmt: skip
    ple = tl.load(ple_ptr + row * stride_ple + ch, mask=m, other=0.0).to(tl.float32)
    hn = (x + ple).to(tl.bfloat16)
    tl.store(hc_out_ptr + row * stride_hc_out + ch, hn, mask=m)
    nrm = _rms(hn.to(tl.float32), wx, HIDDEN, eps)
    tl.store(normed_ptr + row * stride_normed + ch, nrm.to(tl.bfloat16), mask=m)


def _inject_args(hc: torch.Tensor, inject_block: Optional[tuple]) -> tuple:
    """`(block, inject, shared, gate)`, their row strides, `HAS_INJECT`, `HAS_SHARED` for the gate
    and residual kernels (unused operands point at `hc`)."""
    if inject_block is None:
        return (hc, hc, hc, hc, 0, 0, 0, 0, False, False)
    block_out, inject, shared, gate_logits = inject_block
    has_shared = shared is not None
    return (
        block_out,
        inject,
        shared if has_shared else hc,
        gate_logits if has_shared else hc,
        block_out.stride(0),
        inject.stride(0),
        shared.stride(0) if has_shared else 0,
        gate_logits.stride(0) if has_shared else 0,
        True,
        has_shared,
    )


def _state_head(state_len: int) -> int:
    """The largest power of two not above `state_len`: the conv kernel's tile
    width; the one column past it, if any, moves as a vector."""
    return 1 << (state_len.bit_length() - 1)


#: Channels per conv program.
_CONV_ROWS = 128
_CONV_WARPS = 4
#: Context-shaped conv tiles: packed rows x channels per program.
_ROWS_BLOCK_T = 32
_ROWS_CHANNELS = 128
_ROWS_WARPS = 4
_MIN_LOG_S = 8

#: This checkpoint's PLE tail geometry: 4 streams of 2560, conv kernel 4 at
#: dilation 3 (a 9-column state), the gate floor `_PLE_CLAMP_MIN`.
CKPT_HC_COUNT = 4
CKPT_HIDDEN = 2560
CKPT_DILATION = 3
CKPT_CLAMP_MIN = 1e-6
#: The PLE-conv slot count of a full-size tp1 engine run (the n-gram pool's slot space). The
#: op addresses a slot by its index alone, so a pool of another size takes the same path.
ENGINE_SLOTS = 461
#: The default target's per-rank PLE-conv pools: the conv state and the n-gram context are
#: bound over one slot space (the stock manager asserts their slot counts equal), so the
#: counts are the n-gram pool's (`TEP4_SLOTS`, `DEP4_SLOTS`). Every rank holds the whole
#: `[hc_count * hidden, 9]` conv state at every member. A tp4 rank holds the same per-rank
#: PLE and GDN state as a tep4 rank, so tp4 is bounded by tep4's.
TP4_SLOTS = TEP4_SLOTS


def _spec(lengths, **spec) -> dict:
    return dict(lengths=tuple(lengths), **spec)


class PleGateConvInjectNorm(OpWrapper):
    """With `C = hc_count`, `D = value.shape[-1]`, every grouped norm a gemma
    RMS norm over each `D`-wide stream (`n(x, w) = x * rsqrt(mean(x^2) + eps)
    * (1 + w)`), and each step rounded to the activation dtype as `reference`
    rounds it -- for row `t` of sequence `s`, stream `c`::

        kn   = round(n(key[t, c], key_norm[c]))
        qn   = round(n(hc[t, c], query_norm[c]))
        g    = sum(kn * qn) / sqrt(D);  sg = sign(g) * sqrt(max(|g|, clamp_min))
        gt   = round(sigmoid(sg) * value[t])
        xc   = round(n(gt, conv_norm[c]))
        conv = round(sum_k conv_weight[c, k] * window[c, t' + k * dilation])
        ple  = round(gt + round(silu(conv)))
        hc_out[t, c] = round(hc[t, c] + ple)
        normed[t, c] = round(n(hc_out[t, c], next_norm[c]))

    where `window` is `[conv_states[slot] | xc of the sequence's rows]` and
    `t'` the row's offset in its sequence; the state part is zero for a
    sequence whose `has_initial_states` flag is false (None: all carry). Then
    `conv_states[slot]` becomes `window[:, L : L + state_len]`, the raw
    conv-input tail. Every sequence
    carries its state (a decode batch); a `pad_slot_id` sequence reads a zero
    state and publishes nothing.

    `max_rows > PER_SEQ_MAX_ROWS` (context-shaped steps) takes a fourth
    launch: the conv over packed (row tile, channel block) programs, the
    per-sequence kernel then only publishing the state.

    Returns `(hc_out, normed)`, `[T, C*D]` in `hc`'s dtype: fresh, or the
    two row-strided buffers of `out` (neither overlapping `hc`) written in place.
    `conv_states` is advanced in place. When `conv_input` is given (a
    contiguous `[T, C*D]` buffer in `hc`'s dtype) the conv input `xc` is
    written there, for a caller that has to rebuild the conv state at another
    row of the sequence later (the speculative rollback of a verify step).

    Fusion boundary: everything from the key/query norms to the next site's
    norm. The n-gram embedding and the key / value projections are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: lengths (rows per sequence), pool_slots (default num_seqs + 2),
    # max_rows (max(lengths)), carries (None -> the argument is None),
    # conv_input (True -> a staged conv-input buffer is passed). The
    # checkpoint's 4 x 2560 bf16 geometry throughout.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode 128 x 1 row at the 461-slot pool",
            spec=_spec((1,) * 128, pool_slots=ENGINE_SLOTS),
        ),
        Cell(
            why="MTP3 verify: 64 sequences x 4 rows",
            spec=_spec((4,) * 64, pool_slots=ENGINE_SLOTS),
        ),
        # Context-shaped (prefill / mixed) steps: the packed-row conv.
        Cell(
            why="prefill 8192 rows, fresh",
            spec=_spec((8192,), carries=(False,), pool_slots=ENGINE_SLOTS),
        ),
        Cell(
            why="mixed step: 2 contexts + 64 decode rows",
            spec=_spec(
                (3001, 5000) + (1,) * 64,
                carries=(False, True) + (True,) * 64,
                pool_slots=ENGINE_SLOTS,
            ),
        ),
        # The default target's members, each at the per-rank conv pool its rank binds, as
        # the forward calls it: decode and verify with `has_initial_states` None and
        # `max_rows` the rows per sequence (a verify step also stages the conv
        # input), prefill with one flag per sequence and `max_rows` the step's rows.
        Cell(
            why=f"tep4 decode: 128 x 1 row, {TEP4_SLOTS} slots",
            spec=_spec((1,) * 128, pool_slots=TEP4_SLOTS),
        ),
        Cell(
            why=f"dep4 decode: a rank's 32 x 1 row, {DEP4_SLOTS} slots",
            spec=_spec((1,) * 32, pool_slots=DEP4_SLOTS),
        ),
        Cell(
            why=f"tep4 MTP1 verify: 128 sequences x 2 rows, {TEP4_SLOTS} slots",
            spec=_spec((2,) * 128, pool_slots=TEP4_SLOTS, conv_input=True),
        ),
        Cell(
            why=f"tep4 MTP7 verify: 128 sequences x 8 rows (packed-row conv), {TEP4_SLOTS} slots",
            spec=_spec((8,) * 128, pool_slots=TEP4_SLOTS, conv_input=True),
        ),
        Cell(
            why=f"tep4 prefill: one fresh 8192-token context, {TEP4_SLOTS} slots",
            spec=_spec((8192,), carries=(False,), pool_slots=TEP4_SLOTS),
        ),
        Cell(
            why=f"envelope: 8192 rows over 128 sequences (1 context + 127 decode), {TEP4_SLOTS} slots",
            spec=_spec(
                (8192 - 127,) + (1,) * 127,
                carries=(False,) + (True,) * 127,
                max_rows=8192,
                pool_slots=TEP4_SLOTS,
            ),
        ),
    )

    note = """
    Metadata consumed: none. Call site: the decode target's PLE block, once
    per forward: `hc` the `[T, 10240]` stream, `key` the key projection
    `[T, 10240]` and `value` the value projection `[T, 2560]` (both cuBLAS,
    bf16), the three PLE norm weights and the next site's attention-side norm
    weight as contiguous `[10240]`, the `[10240, 1, 4]` conv weight, the whole
    `[slots, 10240, 9]` conv pool, int32 slots, int64 `query_start_loc_long`
    (the rebuild under MTP verify), `max_rows = gen_p`.

    Preconditions (data; not guarded): at most `max_rows` rows per sequence;
    distinct live slots; `query_start_loc[-1] == T`. Operands 2-D with unit
    column stride; weights contiguous. Certified at the checkpoint's 4 x 2560
    bf16 geometry with every slot live (the target never passes `pad_slot_id`),
    at a 128 x 1-row decode, a 64 x 4-row MTP3 verify, an 8192-row fresh
    prefill and a mixed 2-context + 64-decode step, all on the 461-slot pool;
    and at the default target's per-rank pools (`TEP4_SLOTS` 1874, `DEP4_SLOTS` 850,
    slots spread over the whole pool with both ends taken): a 128 x 1 tep4 and
    a 32 x 1 dep4 decode, 128 x 2 (MTP1) and 128 x 8 (MTP7, the packed-row conv
    with no carry flags) verifies staging `conv_input`, an 8192-row fresh
    prefill, and the envelope's largest step, 8192 rows over 128 sequences (one
    8065-row context, 127 carrying decode rows). A tp4 rank binds the same
    conv state as a tep4 rank, so the tep4 cells cover it by bound.
    Not certified: sequences shorter than `max_rows` on the per-sequence path
    (a verify step gives every sequence `max_rows` rows); verify widths 3, 5,
    6 and 7 rows at the per-rank pools (4 rows only at 64 x 4 over 461 slots);
    the MTP member's own pool (max_seq_len 16384; the verify cells run at the
    tep4 count); a dep4 prefill or verify step; a chunked continuation at the
    per-rank pools (a carrying context chunk is certified only by the 461-slot
    mixed cell). Every pool count is the stock engine's with the n-gram table
    host-resident.

    Index range: slots and rows are widened to int64 before every offset, so
    neither the pool (2^31 elements at 23301 slots of `[10240, 9]`) nor the
    rows (2^31 elements at 209715 rows of 10240) is near a 32-bit limit at
    the envelope (1874 slots, 8192 rows); nothing is indexed in fp32.

    Source: the four Triton kernels in this file (three launches a call, the
    packed-row conv a fourth on context-shaped steps), JIT-compiled for the
    running device; every cell ran the JIT path on sm_103 (GB300).

    Numerics: every rounding point of the composition is kept, so the only
    differences are fp32 reduction order (the three norms' sums of squares
    and the gate's dot) and fp32 `exp` -- each can move a value across a
    rounding midpoint by one ulp, and that ulp propagates down the chain.

    CUDA-graph capturable: no host sync, outputs and scratch (`gated` / `ple`,
    `xc`) allocated per call. Three launches: gate (rows x streams), conv
    (sequences x channel blocks of `_CONV_ROWS`), residual + norm (rows x
    streams).
    """

    def __call__(
        self,
        hc: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_norm: torch.Tensor,
        query_norm: torch.Tensor,
        conv_norm: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        next_norm: torch.Tensor,
        eps: float,
        hc_count: int,
        clamp_min: float,
        dilation: int,
        pad_slot_id: int,
        max_rows: int,
        has_initial_states: Optional[torch.Tensor] = None,
        conv_input: Optional[torch.Tensor] = None,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inject_block: Optional[tuple] = None,
        pdl: bool = False,
        publish_end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = hc.shape
        hidden = width // hc_count
        ksize = conv_weight.shape[-1]
        state_len = conv_states.shape[-1]
        if out is None:
            hc_out = torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
            normed = torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
        else:
            hc_out, normed = out
        num_seqs = state_indices.shape[0]
        if num_seqs == 0 or rows == 0:
            return hc_out, normed
        gated = torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
        xc = (
            conv_input
            if conv_input is not None
            else torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
        )
        block = triton.next_power_of_2(hidden)
        warps = 8 if block >= 2048 else 4
        inj = _inject_args(hc, inject_block)
        _ple_gate_kernel[(rows, hc_count)](
            hc,
            key,
            value,
            key_norm,
            query_norm,
            conv_norm,
            gated,
            xc,
            hc.stride(0),
            key.stride(0),
            value.stride(0),
            gated.stride(0),
            eps,
            clamp_min,
            math.sqrt(hidden),
            *inj[:-2],
            HIDDEN=hidden,
            BLOCK=block,
            HAS_INJECT=inj[-2],
            HAS_SHARED=inj[-1],
            PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
        # `ple` reuses `gated`'s storage row for row: each element is read by
        # the program that then writes it.
        conv_rows = min(_CONV_ROWS, width)
        has_carry = has_initial_states is not None
        carry = has_initial_states if has_carry else state_indices
        per_seq = max_rows <= PER_SEQ_MAX_ROWS
        tile_ch = min(_ROWS_CHANNELS, width)
        log_s = max(_MIN_LOG_S, (num_seqs - 1).bit_length())
        rows_args = (
            xc,
            gated,
            gated,
            conv_weight,
            conv_states,
            state_indices,
            carry,
            query_start_loc,
            rows,
            num_seqs,
            gated.stride(0),
            conv_states.stride(0),
            pad_slot_id,
        )
        rows_constexprs = dict(
            KSIZE=ksize,
            DIL=dilation,
            STATE=state_len,
            BLOCK_T=_ROWS_BLOCK_T,
            ROWS=tile_ch,
            LOG_S=log_s,
            HAS_CARRY=has_carry,
            num_warps=_ROWS_WARPS,
        )
        conv_args = (
            xc,
            gated,
            gated,
            conv_weight,
            conv_states,
            state_indices,
            carry,
            query_start_loc,
            gated.stride(0),
            conv_states.stride(0),
            pad_slot_id,
            num_seqs if publish_end is None else int(publish_end),
        )
        conv_constexprs = dict(
            KSIZE=ksize,
            DIL=dilation,
            STATE=state_len,
            HEAD=_state_head(state_len),
            MAX_P=max_rows if per_seq else 0,
            ROWS=conv_rows,
            HAS_CARRY=has_carry,
            PDL=pdl,
            num_warps=_CONV_WARPS,
            launch_pdl=pdl,
        )
        # `MAX_P` follows the step's longest sequence (0 past PER_SEQ_MAX_ROWS): every value, and
        # the context-shaped rows kernel, compile at a geometry's first call.
        warm_key = (
            xc.dtype,
            conv_states.dtype,
            carry.dtype,
            state_indices.dtype,
            query_start_loc.dtype,
            gated.stride(0),
            conv_states.stride(0),
            pad_slot_id,
            ksize,
            dilation,
            state_len,
            conv_rows,
            has_carry,
            log_s,
            pdl,
        )
        if first_time("ple_gate_conv_inject_norm", warm_key):
            for max_p in range(PER_SEQ_MAX_ROWS + 1):
                compile_only(_ple_conv_kernel, conv_args, **{**conv_constexprs, "MAX_P": max_p})
            compile_only(_ple_conv_rows_kernel, rows_args, **rows_constexprs)
            mark_warmed("ple_gate_conv_inject_norm", warm_key)
        if not per_seq:
            _ple_conv_rows_kernel[(triton.cdiv(rows, _ROWS_BLOCK_T), width // tile_ch)](
                *rows_args, **rows_constexprs
            )
        # Per sequence: the rows (decode) and the state publish. On the
        # context-shaped path it carries no rows and only publishes, after
        # every program above has read the state.
        _ple_conv_kernel[(num_seqs, width // conv_rows)](*conv_args, **conv_constexprs)
        _ple_residual_norm_kernel[(rows, hc_count)](
            hc,
            gated,
            next_norm,
            hc_out,
            normed,
            hc.stride(0),
            gated.stride(0),
            hc_out.stride(0),
            normed.stride(0),
            eps,
            *inj[:-2],
            HIDDEN=hidden,
            BLOCK=block,
            HAS_INJECT=inj[-2],
            HAS_SHARED=inj[-1],
            PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
        return hc_out, normed

    def reference(
        self,
        hc: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_norm: torch.Tensor,
        query_norm: torch.Tensor,
        conv_norm: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        next_norm: torch.Tensor,
        eps: float,
        hc_count: int,
        clamp_min: float,
        dilation: int,
        pad_slot_id: int,
        max_rows: int,
        has_initial_states: Optional[torch.Tensor] = None,
        conv_input: Optional[torch.Tensor] = None,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inject_block: Optional[tuple] = None,
        pdl: bool = False,
        publish_end: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(hc_out, normed, conv_states)` as the call leaves them: float64
        throughout, rounded to the activation dtype exactly where the op (and
        the unfused composition) rounds; the conv one sequence at a time as
        an explicit tap sum. Mutates nothing."""
        dtype = hc.dtype
        rows, width = hc.shape
        hidden = width // hc_count

        def rnd(x):
            return x.to(dtype).double()

        if inject_block is not None:
            # The previous MoE block injected first (`qwen4_exp_hc_inject_norm`'s rounding).
            block_out, inject, shared, gate_logits = inject_block
            b = block_out.double()
            if shared is not None:
                b = rnd(b + torch.sigmoid(gate_logits.reshape(rows, 1).double()) * shared.double())
            hc = hc.double().view(rows, hc_count, hidden) + b.unsqueeze(
                1
            ) * inject.double().unsqueeze(-1)
            hc = hc.reshape(rows, width).to(dtype)

        def norm(x, w):
            w = w.double().view(hc_count, hidden)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)

        streams = hc.double().view(rows, hc_count, hidden)
        kn = rnd(norm(key.double().view(rows, hc_count, hidden), key_norm))
        qn = rnd(norm(streams, query_norm))
        g = (kn * qn).sum(-1, keepdim=True) / math.sqrt(hidden)
        sg = g.abs().clamp_min(clamp_min).sqrt() * g.sign()
        gated = rnd(torch.sigmoid(sg) * value.double().unsqueeze(-2))
        xc = rnd(norm(gated, conv_norm)).reshape(rows, width)

        ksize = conv_weight.shape[-1]
        state_len = conv_states.shape[-1]
        w = conv_weight.double().reshape(width, ksize)
        conv = torch.zeros(rows, width, dtype=torch.float64, device=hc.device)
        pool = conv_states.clone()
        starts = query_start_loc.tolist()
        carries = (
            [True] * state_indices.shape[0]
            if has_initial_states is None
            else [bool(c) for c in has_initial_states.tolist()]
        )
        for s, slot in enumerate(state_indices.tolist()):
            begin, end = starts[s], starts[s + 1]
            live = slot != pad_slot_id
            prefix = (
                conv_states[slot].double()
                if live and carries[s]
                else torch.zeros(width, state_len, dtype=torch.float64, device=hc.device)
            )
            window = torch.cat([prefix, xc[begin:end].t()], dim=1)
            n = end - begin
            acc = torch.zeros(width, n, dtype=torch.float64, device=hc.device)
            for k in range(ksize):
                acc += w[:, k : k + 1] * window[:, k * dilation : k * dilation + n]
            conv[begin:end] = rnd(acc.t())
            if live and (publish_end is None or s < publish_end):
                pool[slot] = window[:, end - begin : end - begin + state_len].to(dtype)
        ple = rnd(gated.reshape(rows, width) + rnd(torch.nn.functional.silu(conv)))
        hc_out = (hc.double() + ple).to(dtype)
        normed = norm(hc_out.double().view(rows, hc_count, hidden), next_norm)
        return hc_out, normed.reshape(rows, width).to(dtype), pool

    def is_valid(
        self,
        hc: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_norm: torch.Tensor,
        query_norm: torch.Tensor,
        conv_norm: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        next_norm: torch.Tensor,
        eps: float,
        hc_count: int,
        clamp_min: float,
        dilation: int,
        pad_slot_id: int,
        max_rows: int,
        has_initial_states: Optional[torch.Tensor] = None,
        conv_input: Optional[torch.Tensor] = None,
        out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inject_block: Optional[tuple] = None,
        pdl: bool = False,
        publish_end: Optional[int] = None,
    ) -> None:
        rows, width = hc.shape
        if inject_block is not None:
            block_out, inject, shared, gate_logits = inject_block
            hidden_ = width // hc_count
            assert block_out.shape == (rows, hidden_) and block_out.stride(1) == 1, tuple(
                block_out.shape
            )
            assert inject.shape == (rows, hc_count) and inject.stride(1) == 1, tuple(inject.shape)
            assert block_out.dtype == inject.dtype == hc.dtype
            if shared is not None:
                assert shared.shape == (rows, hidden_) and shared.stride(1) == 1, tuple(
                    shared.shape
                )
                assert shared.dtype == hc.dtype
                assert gate_logits is not None and gate_logits.numel() == rows
        assert width % hc_count == 0, f"width {width} not divisible by hc_count {hc_count}"
        hidden = width // hc_count
        for name, t in (("hc", hc), ("key", key), ("value", value)):
            assert t.dim() == 2 and t.stride(1) == 1, f"{name} must be 2-D, unit column stride"
        assert key.shape == hc.shape, f"key {tuple(key.shape)} must match hc {tuple(hc.shape)}"
        assert value.shape == (rows, hidden), f"value must be [{rows}, {hidden}]"
        for name, t in (
            ("key_norm", key_norm),
            ("query_norm", query_norm),
            ("conv_norm", conv_norm),
            ("next_norm", next_norm),
        ):
            assert t.shape == (width,) and t.is_contiguous(), f"{name} must be contiguous [{width}]"
        assert conv_weight.dim() == 3 and conv_weight.shape[:2] == (width, 1), (
            "conv_weight [W, 1, K]"
        )
        assert conv_weight.is_contiguous(), "conv_weight contiguous"
        ksize = conv_weight.shape[-1]
        assert conv_states.dim() == 3 and conv_states.shape[1:] == (
            width,
            (ksize - 1) * dilation,
        ), f"conv_states must be [slots, {width}, {(ksize - 1) * dilation}]"
        assert conv_states.stride(2) == 1 and conv_states.stride(1) == conv_states.shape[2], (
            "conv_states rows must be dense"
        )
        assert conv_states.dtype == hc.dtype, "conv state dtype must match the activation"
        assert query_start_loc.shape[0] == state_indices.shape[0] + 1, (
            "query_start_loc must be [num_seqs + 1]"
        )
        state_len = conv_states.shape[-1]
        assert state_len - _state_head(state_len) <= 1, (
            f"a {state_len}-column conv state is not a power of two plus at most one"
        )
        assert width % min(_CONV_ROWS, width) == 0, (
            f"width {width} must be a multiple of the conv block ({_CONV_ROWS} channels)"
        )
        assert max_rows >= 1
        assert has_initial_states is None or has_initial_states.shape == state_indices.shape, (
            "has_initial_states must be one flag per sequence"
        )
        assert conv_input is None or (
            conv_input.shape == hc.shape
            and conv_input.dtype == hc.dtype
            and conv_input.is_contiguous()
        ), "conv_input must be a contiguous buffer shaped and typed as hc"
        for t in out or ():
            assert t.shape == hc.shape and t.dtype == hc.dtype and t.stride(1) == 1, (
                "out buffers must be shaped and typed as hc, unit column stride"
            )


ple_gate_conv_inject_norm = PleGateConvInjectNorm()
