# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDN verify step's conv without its window snapshots and its pool advance (`conv_lean`),
and the promotion that builds the accepted window from the step's raw rows (`promote_x`).

Stock pair (`causal_conv1d_triton_update` with snapshots, then the runtime's
`promote_conv_and_advance_replay_triton`): the conv of a verify step reads a request's window
(3 columns) and its T new rows, writes the T outputs, T window snapshots of 3 columns each and
the advanced pool window; after the step one launch copies the accepted snapshot of every layer
into the pool and advances the replay bookkeeping. At T = 4 the snapshots and the advance are
15 column stores a channel besides the 4 outputs.

Lean pair: `conv_lean` writes the T outputs and the T raw input rows (the q|k|v columns of the
input projection, `xsave [layers, max_batch, T, C]`) and leaves the pool as it was. After the
step `promote_x` writes each request's window from what the snapshots were made of -- with k
accepted tokens the new window is elements k .. k+2 of `[old0, old1, old2, x0, ..., x(T-1)]`,
the values snapshot k-1 holds -- and advances the bookkeeping of the request's replay slot with
the stock promotion's statements (layer 0's first tile).

Values: the conv output is the stock kernel's statements in the stock order (bf16 products,
fp32 accumulation from zero in tap order, `acc / (1 + exp(-acc))`, one rounding), so it is
bitwise the stock launch's; the promoted window holds the bf16 values the accepted snapshot
holds; the bookkeeping is the stock arithmetic. Width 4 (3 pool columns) only.

What differs from the stock pair, by construction:
- between the conv and the promotion the pool holds the pre-step window (stock: the window
  after all T rows). The next reader of the conv pool is the next step's conv, behind the
  promotion.
- the snapshot buffer is not written; only the stock promotion reads it.
- a row on the pad slot: both launches skip it (no output row, no saved row, no window, no
  bookkeeping).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.archs import SERVED_ARCHS
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

#: The conv's kernel width (3 pool columns + the new row).
WIDTH = 4
#: The conv's launch shape: channels a program and warps (GB300 micro, best of nine at 24
#: requests a rank).
BLOCK_N, NUM_WARPS = 512, 16
#: The promotion's channels a program.
PROMOTE_BLOCK_N = 2048
#: An attention-DP rank's GDN conv width (all 16 k / 48 v heads) and its input-projection row
#: stride (`[q | k | v | z | a | b]`).
DP_DIM, DP_X_STRIDE = 10240, 16480
#: GDN layers a rank's stacked conv pool and `xsave` hold.
GDN_LAYERS = 36


@triton.jit
def _conv_lean_kernel(
    x_ptr,  # [rows, >= C] bf16, row stride `stride_x_row`: the in-proj output, q|k|v first
    w_ptr,  # [C, 4]
    state_ptr,  # one layer's pool view [slots, C, 3]
    slot_ptr,  # [n] int: the pool slot of batch row b
    xrow_ptr,  # [n] int: the row of `xs` batch row b writes (`intermediate_state_indices`)
    o_ptr,  # [rows, C] contiguous
    xs_ptr,  # one layer's [max_batch, T, C] contiguous
    stride_x_row,
    stride_state_slot,
    pad_slot_id: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    WAIT: tl.constexpr,
    RELEASE: tl.constexpr,
):
    if RELEASE:
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0)
    slot = tl.load(slot_ptr + b).to(tl.int64)
    if slot == pad_slot_id:
        # Not a sequence: no output row, no saved row (the stock kernel's rule).
        return
    xrow = tl.load(xrow_ptr + b).to(tl.int64)
    feats = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    sbase = state_ptr + slot * stride_state_slot + feats * 3
    col0 = tl.load(sbase)
    col1 = tl.load(sbase + 1)
    col2 = tl.load(sbase + 2)
    wbase = w_ptr + feats * 4
    w0 = tl.load(wbase)
    w1 = tl.load(wbase + 1)
    w2 = tl.load(wbase + 2)
    w3 = tl.load(wbase + 3)
    if WAIT:
        # A PDL dependent of the producer of x: the slot, the window and the taps (data of
        # earlier steps and parameters, which the producer does not write) are read above.
        gdc_wait_with_memory_clobber()
    # All T rows in flight before the first store (the stores may alias for ptxas).
    xs = ()
    for t in tl.static_range(T):
        xs = xs + (tl.load(x_ptr + (b * T + t) * stride_x_row + feats),)
    for t in tl.static_range(T):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        acc += col0 * w0
        acc += col1 * w1
        acc += col2 * w2
        acc += xs[t] * w3
        col0 = col1
        col1 = col2
        col2 = xs[t]
        acc = acc / (1 + tl.exp(-acc))
        tl.store(o_ptr + (b * T + t) * C + feats, acc)
        tl.store(xs_ptr + (xrow * T + t) * C + feats, xs[t])


def _conv_lean(x, weight, conv_state, slots, xrows, xsave, seqlen, pad_slot_id, block_n,
               num_warps, wait, release) -> torch.Tensor:  # fmt: skip
    rows = x.shape[0]
    n = rows // seqlen
    c = weight.shape[0]
    out = torch.empty((rows, c), dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    _conv_lean_kernel[(n, c // block_n)](
        x, weight, conv_state, slots, xrows, out, xsave, x.stride(0), conv_state.stride(0),
        pad_slot_id=pad_slot_id, C=c, T=seqlen, BLOCK=block_n, WAIT=wait, RELEASE=release,
        num_warps=num_warps, **({"launch_pdl": True} if wait else {}),
    )  # fmt: skip
    return out


def _lean_cell(why: str, batch: int, **kw) -> Cell:
    """An attention-DP rank's verify call: 10240 channels of x at row stride 16480, 4 rows a
    request, the layer's view of a 36-layer conv page, `xsave` a layer's view."""
    spec = dict(dim=DP_DIM, x_row_stride=DP_X_STRIDE, batch=batch, seqlen=4, pool_slots=96)
    spec.update(kw)
    return Cell(why=why, spec=spec)


class GdnConvLean(OpWrapper):
    """The verify step's multi-token conv over a request's window and its T new rows, out of
    place, with no window snapshots and no pool advance; the raw rows are saved instead.

    Batch row `b`, slot `s = slots[b]`: `window = [conv_state[s, :, 0:3], x[b*T : b*T+T, :C]^T]`,
    `out[b*T + t, c] = silu(sum_j bf16(weight[c, j] * window[c, t + j]))` -- each product rounded
    to bf16, accumulated in fp32 from zero in tap order, SiLU as `acc / (1 + exp(-acc))`, rounded
    once -- and `xsave[xrows[b], t, :] = x[b*T + t, :C]`. The pool is read, never written.

    Fusion boundary: the window read, the conv, SiLU and the raw-row store. The projection, the
    q/k/v split and the promotion (`promote_x`) are the caller's.

    Returns a fresh `[rows, C]` token-major tensor. A row whose slot equals `pad_slot_id` is not
    written (its output rows are allocator garbage, its `xsave` row is untouched).

    `wait`: the launch is a programmatic dependent of the producer of `x` (it reads the slots,
    the window and the taps before `griddepcontrol.wait`, `x` after it). `release`: it signals
    its own dependents at its start (they read the output after their own wait).
    """

    ARCHS = SERVED_ARCHS

    # spec keys: dim, x_row_stride, batch, seqlen, pool_slots, pool_layers (default 36: the
    # layer is the last of a page this deep), wait / release (default True), pad (a batch row
    # on the pad slot).
    CELLS: tuple[Cell, ...] = (
        _lean_cell("dep rank verify, 5 requests x 4 rows (20 rows, the lean floor)", 5),
        _lean_cell("dep rank verify, 12 requests x 4 rows", 12),
        _lean_cell("dep rank verify, 64 requests x 4 rows (256 rows, the lean ceiling)", 64),
        _lean_cell("dep rank verify, 2 requests x 4 rows (DP_LOWROWS=2)", 2),
        _lean_cell("no PDL attributes (eager form)", 8, wait=False, release=False),
        _lean_cell("a batch row on the pad slot is skipped", 6, pad=3),
    )

    note = """
    Metadata consumed: none.

    Call site: an attention-DP rank's generation-only verify step (`_gdn_verify` under
    `DP_GDN_CONV_LEAN_ENV`): `x` the `[n * 4, 10240]` q|k|v column view of the `[n * 4, 16480]`
    input projection, the layer's conv view of the stacked pool, `slots` the step's state
    indices, `xrows` the manager's `intermediate_state_indices`, `xsave` the layer's view of the
    target's `[36, max_batch, 4, 10240]` buffer. Certified at 2..64 requests of 4 rows.

    Bitwise the stock conv's output (`causal_conv1d_triton_update`, the same statements in the
    same order); the GPU test also holds `conv_lean` + `promote_x` against the stock conv with
    snapshots + `promote_conv_and_advance_replay_triton`, window and bookkeeping bitwise.

    Not guarded (content): duplicate slots or `xsave` rows race; out-of-range slot values are an
    illegal access.

    Hard limits: the channel count must be a multiple of `block_n` (no channel mask); `x`'s
    column stride 1; the pool view `[slots, C, 3]` with strides `(s, 3, 1)`.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        slots: torch.Tensor,
        xrows: torch.Tensor,
        xsave: torch.Tensor,
        seqlen: int,
        pad_slot_id: int,
        block_n: int = BLOCK_N,
        num_warps: int = NUM_WARPS,
        wait: bool = True,
        release: bool = True,
    ) -> torch.Tensor:
        return _conv_lean(
            x, weight, conv_state, slots, xrows, xsave, seqlen, pad_slot_id, block_n, num_warps,
            wait, release,
        )  # fmt: skip

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        slots: torch.Tensor,
        xrows: torch.Tensor,
        xsave: torch.Tensor,
        seqlen: int,
        pad_slot_id: int,
        block_n: int = BLOCK_N,
        num_warps: int = NUM_WARPS,
        wait: bool = True,
        release: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, xsave)` as the call leaves them; mutates nothing. Rows the op leaves unwritten
        are NaN in `out`. `block_n`, `num_warps`, `wait` and `release` are ignored. Products in
        the operand dtype, as the kernel forms them."""
        c = weight.shape[0]
        rows = x.shape[0]
        n = rows // seqlen
        out = torch.full((rows, c), float("nan"), dtype=x.dtype, device=x.device)
        saved = xsave.clone()
        for b, (slot, row) in enumerate(zip(slots[:n].tolist(), xrows[:n].tolist())):
            if slot == pad_slot_id:
                continue
            new = x[b * seqlen : (b + 1) * seqlen, :c]
            full = torch.cat([conv_state[slot], new.t()], dim=1)
            acc = torch.zeros(c, seqlen, dtype=torch.float32, device=x.device)
            for j in range(WIDTH):
                acc = acc + (weight[:, j].unsqueeze(1) * full[:, j : j + seqlen]).float()
            acc = acc / (1.0 + torch.exp(-acc))
            out[b * seqlen : (b + 1) * seqlen] = acc.t().to(x.dtype)
            saved[row] = new
        return out, saved

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        slots: torch.Tensor,
        xrows: torch.Tensor,
        xsave: torch.Tensor,
        seqlen: int,
        pad_slot_id: int,
        block_n: int = BLOCK_N,
        num_warps: int = NUM_WARPS,
        wait: bool = True,
        release: bool = True,
    ) -> None:
        c = weight.shape[0]
        rows = x.shape[0]
        assert rows % seqlen == 0 and min(slots.shape[0], xrows.shape[0]) >= rows // seqlen, (
            f"{rows} rows do not divide into requests of {seqlen}, or fewer slots / xsave rows "
            "than requests -- the kernel reads past the index tensors"
        )
        assert x.dtype == weight.dtype == conv_state.dtype == xsave.dtype == torch.bfloat16
        # The tap loads are unmasked and hard-coded to 4 taps.
        assert weight.shape[1] == WIDTH and weight.is_contiguous(), (
            f"weight must be a contiguous [C, {WIDTH}]; got {tuple(weight.shape)} -- another "
            "width is read as a width-4 conv"
        )
        assert tuple(conv_state.shape[1:]) == (c, WIDTH - 1) and conv_state.stride()[1:] == (
            WIDTH - 1,
            1,
        ), f"conv_state must be [slots, {c}, 3] with strides (s, 3, 1)"
        # No channel mask: a remainder would read and write past the row.
        assert c % block_n == 0 and x.stride(1) == 1 and x.shape[1] >= c, (
            f"{c} channels must be a multiple of block_n {block_n}, x's columns unit-strided and "
            "at least C wide"
        )
        assert xsave.is_contiguous() and tuple(xsave.shape[1:]) == (seqlen, c), (
            f"xsave must be a contiguous [max_batch, {seqlen}, {c}] layer view; got "
            f"{tuple(xsave.shape)} -- another layout is written at the wrong offsets"
        )


gdn_conv_lean = GdnConvLean()


def conv_lean(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    xrows: torch.Tensor,
    xsave: torch.Tensor,
    seqlen: int,
    *,
    pad_slot_id: int = -1,
    block_n: int = BLOCK_N,
    num_warps: int = NUM_WARPS,
    wait: bool = True,
    release: bool = True,
) -> torch.Tensor:
    """`gdn_conv_lean` with keyword defaults (the target's call)."""
    return gdn_conv_lean(
        x, weight, conv_state, slots, xrows, xsave, seqlen, pad_slot_id, block_n, num_warps,
        wait, release,
    )  # fmt: skip


@triton.jit
def _promote_x_kernel(
    xs_ptr,  # [layers, max_batch, T, C] contiguous
    pool_ptr,  # [layers, slots, C, 3] view (layer and slot strided)
    acc_ptr,  # [n] int: accepted tokens, 1..T
    src_ptr,  # [n] int: the row of `xs` of batch row g (`source_state_indices`)
    slot_ptr,  # [n] int: the pool slot of batch row g
    ridx_ptr,  # [n] int: the replay slot of batch row g (the bookkeeping's row)
    dummy_ptr,  # [n] bool: batch row g is a dummy request
    pnat_ptr,  # [replay slots] int32, element stride `pnat_stride`
    cbi_ptr,  # [replay slots] int32, element stride `cbi_stride`
    num_gens,
    pnat_stride,
    cbi_stride,
    xs_s_layer,
    pool_s_layer,
    pool_s_slot,
    pad_slot_id: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    STEP: tl.constexpr,
    HIST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    slot = tl.load(slot_ptr + g).to(tl.int64)
    if slot == pad_slot_id:
        return
    k = tl.load(acc_ptr + g).to(tl.int32)
    row = tl.load(src_ptr + g).to(tl.int64)
    # The replay bookkeeping is batch-scoped: layer 0's first tile owns it (the statements of
    # `promote_conv_and_advance_replay_triton`).
    if (layer == 0) & (tile == 0):
        is_dummy = tl.load(dummy_ptr + g)
        rslot = tl.load(ridx_ptr + g).to(tl.int64)
        pnat = tl.load(pnat_ptr + rslot * pnat_stride)
        cbi = tl.load(cbi_ptr + rslot * cbi_stride)
        wrote = pnat + STEP > HIST
        tl.store(pnat_ptr + rslot * pnat_stride, tl.where(wrote, k, pnat + k), mask=~is_dummy)
        tl.store(cbi_ptr + rslot * cbi_stride, tl.where(wrote, 1 - cbi, cbi), mask=~is_dummy)
    feats = tile * BLOCK + tl.arange(0, BLOCK)
    live = feats < C
    pbase = pool_ptr + layer * pool_s_layer + slot * pool_s_slot + feats * 3
    xbase = xs_ptr + layer * xs_s_layer + (row * T) * C + feats
    # New window = elements k .. k+2 of [old0, old1, old2, x0 .. x(T-1)]: element e is pool
    # column e when e < 3, else raw row e - 3 (element k+2 is always a raw row). Masked loads:
    # the side not taken is not read. All loads precede the stores.
    e0 = k
    e1 = k + 1
    p0 = tl.load(pbase + tl.minimum(e0, 2), mask=live & (e0 < 3), other=0.0)
    x0 = tl.load(xbase + tl.maximum(e0 - 3, 0).to(tl.int64) * C, mask=live & (e0 >= 3), other=0.0)
    p1 = tl.load(pbase + tl.minimum(e1, 2), mask=live & (e1 < 3), other=0.0)
    x1 = tl.load(xbase + tl.maximum(e1 - 3, 0).to(tl.int64) * C, mask=live & (e1 >= 3), other=0.0)
    x2 = tl.load(xbase + (k - 1).to(tl.int64) * C, mask=live)
    tl.store(pbase, tl.where(e0 < 3, p0, x0), mask=live)
    tl.store(pbase + 1, tl.where(e1 < 3, p1, x1), mask=live)
    tl.store(pbase + 2, x2, mask=live)


def _promote_x(pool_stack, xsave, src_rows, slots, accepted, dummy, pnat, cbi, replay_slots,
               step_width, history, pad_slot_id, block_n) -> None:  # fmt: skip
    layers, _, c, _ = pool_stack.shape
    n = slots.shape[0]
    if n == 0:
        return
    ridx = slots if replay_slots is None else replay_slots
    _promote_x_kernel[(layers * n, triton.cdiv(c, block_n))](
        xsave, pool_stack, accepted, src_rows, slots, ridx, dummy, pnat, cbi, n,
        pnat.stride(0), cbi.stride(0), xsave.stride(0), pool_stack.stride(0),
        pool_stack.stride(1), pad_slot_id=pad_slot_id, C=c, T=xsave.shape[2], STEP=step_width,
        HIST=history, BLOCK=block_n, num_warps=4,
    )  # fmt: skip


class GdnPromoteX(OpWrapper):
    """The lean verify step's promotion, one launch over every layer: batch row `g` read its T
    raw rows into `xsave[:, src_rows[g]]`, owns pool slot `slots[g]` and replay slot
    `replay_slots[g]` (None: `slots[g]`) and accepted `k = accepted[g]` (1..T) tokens.

    `pool_stack[l, slots[g], :, j]` becomes element `k + j` of `[pool[l, slots[g], :, 0:3],
    xsave[l, src_rows[g], 0:T, :]^T]` for j = 0, 1, 2 -- the window the stock conv's snapshot
    `k - 1` holds. The bookkeeping of `replay_slots[g]` advances as the stock promotion's does:
    `wrote = pnat + step_width > history`; `pnat <- wrote ? k : pnat + k`, `cbi <- wrote ? 1 - cbi
    : cbi`; not for a row whose `dummy[g]` is set. A row on `pad_slot_id` is skipped.

    Fusion boundary: the window promotion of every layer and the bookkeeping. In place; returns
    None.
    """

    ARCHS = SERVED_ARCHS

    # spec keys: dim, layers, batch, seqlen, pool_slots, history (default 16), replay (separate
    # replay slots, default False), accepted (a tuple, else a spread over 1..T), dummy (a batch
    # row that is a dummy request), pad (a batch row on the pad slot).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep rank, 36 layers x 12 requests, every accepted count 1..4",
            spec=dict(dim=DP_DIM, layers=GDN_LAYERS, batch=12, seqlen=4, pool_slots=96),
        ),
        Cell(
            why="separate replay slots (the V2 manager's replay region)",
            spec=dict(dim=DP_DIM, layers=4, batch=8, seqlen=4, pool_slots=40, replay=True),
        ),
        Cell(
            why="a dummy request keeps its bookkeeping; its window is promoted",
            spec=dict(dim=DP_DIM, layers=2, batch=6, seqlen=4, pool_slots=24, dummy=2),
        ),
        Cell(
            why="checkpoint wrap: pnat + step width past the history",
            spec=dict(dim=DP_DIM, layers=2, batch=8, seqlen=4, pool_slots=24, history=4),
        ),
        Cell(
            why="a batch row on the pad slot is skipped",
            spec=dict(dim=DP_DIM, layers=2, batch=5, seqlen=4, pool_slots=24, pad=1),
        ),
    )

    note = """
    Metadata consumed: none (the replay bookkeeping is passed in).

    Call site: the wrapped `GDNReplayState._promote_conv_states_and_advance_replay` of a lean
    step (`DP_GDN_CONV_LEAN_ENV`): `pool_stack` the state's `_stacked_conv`, `xsave` the target's
    `[36, max_batch, T, C]` buffer, the batch's `source_state_indices`,
    `destination_state_indices`, `num_accepted_tokens`, `is_dummy_request`, the replay
    metadata's `prev_num_accepted_tokens` / `cache_buf_idx` (int32, any element stride) and the
    promotion's `replay_indices`.

    Not guarded (content): `accepted[g]` must be in 1..T (0 reads `xsave` row -1); duplicate
    slots race.
    """

    def __call__(
        self,
        pool_stack: torch.Tensor,
        xsave: torch.Tensor,
        src_rows: torch.Tensor,
        slots: torch.Tensor,
        accepted: torch.Tensor,
        dummy: torch.Tensor,
        pnat: torch.Tensor,
        cbi: torch.Tensor,
        replay_slots: torch.Tensor | None,
        step_width: int,
        history: int,
        pad_slot_id: int,
        block_n: int = PROMOTE_BLOCK_N,
    ) -> None:
        _promote_x(
            pool_stack, xsave, src_rows, slots, accepted, dummy, pnat, cbi, replay_slots,
            step_width, history, pad_slot_id, block_n,
        )  # fmt: skip

    def reference(
        self,
        pool_stack: torch.Tensor,
        xsave: torch.Tensor,
        src_rows: torch.Tensor,
        slots: torch.Tensor,
        accepted: torch.Tensor,
        dummy: torch.Tensor,
        pnat: torch.Tensor,
        cbi: torch.Tensor,
        replay_slots: torch.Tensor | None,
        step_width: int,
        history: int,
        pad_slot_id: int,
        block_n: int = PROMOTE_BLOCK_N,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(pool_stack, pnat, cbi)` as the call leaves them, on copies; mutates nothing.
        `block_n` is ignored."""
        pool, pn, cb = pool_stack.clone(), pnat.clone(), cbi.clone()
        n = slots.shape[0]
        ridx = (slots if replay_slots is None else replay_slots)[:n].tolist()
        for g, (slot, row, k) in enumerate(
            zip(slots.tolist(), src_rows[:n].tolist(), accepted[:n].tolist())
        ):
            if slot == pad_slot_id:
                continue
            full = torch.cat([pool_stack[:, slot], xsave[:, row].transpose(1, 2)], dim=2)
            pool[:, slot] = full[:, :, k : k + WIDTH - 1]
            if not bool(dummy[g]):
                r = ridx[g]
                wrote = int(pnat[r]) + step_width > history
                pn[r] = k if wrote else int(pnat[r]) + k
                cb[r] = 1 - int(cbi[r]) if wrote else int(cbi[r])
        return pool, pn, cb

    def is_valid(
        self,
        pool_stack: torch.Tensor,
        xsave: torch.Tensor,
        src_rows: torch.Tensor,
        slots: torch.Tensor,
        accepted: torch.Tensor,
        dummy: torch.Tensor,
        pnat: torch.Tensor,
        cbi: torch.Tensor,
        replay_slots: torch.Tensor | None,
        step_width: int,
        history: int,
        pad_slot_id: int,
        block_n: int = PROMOTE_BLOCK_N,
    ) -> None:
        n = slots.shape[0]
        assert xsave.is_contiguous() and xsave.shape[0] == pool_stack.shape[0], (
            "xsave must be a contiguous [layers, max_batch, T, C] over the pool's layers"
        )
        assert xsave.shape[3] == pool_stack.shape[2] and pool_stack.shape[3] == WIDTH - 1, (
            f"the pool must be [layers, slots, {xsave.shape[3]}, 3]"
        )
        assert pool_stack.stride(2) == WIDTH - 1 and pool_stack.stride(3) == 1, (
            "the pool's channel and column strides must be (3, 1)"
        )
        assert pool_stack.dtype == xsave.dtype == torch.bfloat16
        assert pnat.dtype == cbi.dtype == torch.int32 and dummy.dtype == torch.bool
        assert step_width == xsave.shape[2] and history >= step_width, (
            f"the replay step width {step_width} must be the saved rows a request "
            f"{xsave.shape[2]}, at most the history {history}"
        )
        idx = (src_rows, accepted, dummy) + (() if replay_slots is None else (replay_slots,))
        assert all(t.shape[0] >= n for t in idx), "an index tensor shorter than the batch"


gdn_promote_x = GdnPromoteX()


def promote_x(
    pool_stack: torch.Tensor,
    xsave: torch.Tensor,
    src_rows: torch.Tensor,
    slots: torch.Tensor,
    accepted: torch.Tensor,
    dummy: torch.Tensor,
    pnat: torch.Tensor,
    cbi: torch.Tensor,
    step_width: int,
    history: int,
    *,
    replay_slots: torch.Tensor | None = None,
    pad_slot_id: int = -1,
) -> None:
    """`gdn_promote_x` with keyword defaults (the target's call)."""
    gdn_promote_x(
        pool_stack, xsave, src_rows, slots, accepted, dummy, pnat, cbi, replay_slots,
        step_width, history, pad_slot_id,
    )  # fmt: skip


def precompile(
    c: int,
    seqlen: int,
    history: int,
    x_row_stride: int,
    device,
    *,
    pad_slot_id: int = -1,
    index_dtypes: tuple = (torch.int32,),
    accepted_dtypes: tuple = (torch.int32, torch.int64),
) -> int:
    """Compile both kernels at a model's constants (both index dtypes the runtime hands over,
    separate replay slots and not, strided bookkeeping) by launching them once on small private
    buffers, so that no serving step or graph capture meets the compile. Returns the number of
    launches."""
    bf16 = torch.bfloat16
    x = torch.zeros((2 * seqlen, x_row_stride), dtype=bf16, device=device)
    w = torch.zeros((c, WIDTH), dtype=bf16, device=device)
    pool = torch.zeros((2, 2, c, WIDTH - 1), dtype=bf16, device=device)  # [slots, layers, C, 3]
    stack = pool.permute(1, 0, 2, 3)
    xsave = torch.zeros((2, 2, seqlen, c), dtype=bf16, device=device)
    meta = torch.zeros((2, 2), dtype=torch.int32, device=device)  # strided bookkeeping
    dummy = torch.zeros(2, dtype=torch.bool, device=device)
    launches = 0
    for it in index_dtypes:
        idx = torch.arange(2, dtype=it, device=device)
        conv_lean(x[:, :c], w, pool[:, 1], idx, idx, xsave[1], seqlen, pad_slot_id=pad_slot_id)
        launches += 1
        for at in accepted_dtypes:
            acc = torch.ones(2, dtype=at, device=device)
            for replay in (None, idx):
                promote_x(
                    stack, xsave, idx, idx, acc, dummy, meta[:, 0], meta[:, 1], seqlen, history,
                    replay_slots=replay, pad_slot_id=pad_slot_id,
                )  # fmt: skip
                launches += 1
    torch.cuda.synchronize(device)
    return launches
