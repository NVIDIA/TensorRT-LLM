# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row hyper-connection site, first half: the previous block's gated
injection and the mixer's first projection (`[down; block_inject]`) as one
Triton GEMV launch that leaves per-stream partial logits and per-stream sums
of squares for `qwen4_exp_hc_decode_mix`.

Tier 5 (Triton written for this target). At decode row counts the first
projection is a GEMV over a 6.6 MB weight; cuBLAS reaches ~1.5 TB/s on it, and
the injection, the norm and the activations were three more launches. The RMS
is a per-(row, stream) scalar, so the projection runs on the *un-normed*
stream scaled by `(1 + norm_weight)`, one stream per program column: each
program's dot is exactly one stream's partial, and the consumer applies
`rstd` when it combines the four. No split-K workspace, atomics or counters.
"""

from __future__ import annotations

import functools
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest row count this entry is certified for (the forward's dispatch threshold
#: must not exceed it).
MAX_ROWS = 16


@functools.lru_cache(maxsize=None)
def pdl_enabled(device_index: int) -> bool:
    """Programmatic dependent launch: sm_90+ unless TRTLLM_ENABLE_PDL=0."""
    props = torch.cuda.get_device_properties(device_index)
    return props.major >= 9 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@triton.jit
def _hc_decode_first_kernel(
    hc_ptr,
    block_ptr,
    inj_ptr,
    shared_ptr,
    gate_ptr,
    topk_w_ptr,
    topk_row_ptr,
    norm_w_ptr,
    w1_ptr,
    hc_out_ptr,
    part_ptr,
    ss_ptr,
    stride_hc,
    stride_block,
    stride_inj,
    stride_shared,
    stride_gate,
    stride_topk_w,
    stride_topk_row,
    stride_hc_out,
    stride_w1,
    M: tl.constexpr,
    HIDDEN: tl.constexpr,
    N_REAL: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KC: tl.constexpr,
    NC: tl.constexpr,
    NC_P2: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    TOPK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    # grid = (BLOCK_N-row tiles of the weight, streams). One program: BLOCK_N weight
    # rows across one whole stream, as a single [BLOCK_N, NC_P2, KC] load issued
    # before any arithmetic. Reductions run along the tile's last two axes, which a
    # warp holds, so no CTA barrier orders a memory round trip behind another.
    pid_n = tl.program_id(0)
    s = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    jc = tl.arange(0, NC_P2)
    kc = tl.arange(0, KC)
    koff = jc[:, None] * KC + kc[None, :]
    kmask = (jc < NC)[:, None] & (kc >= 0)[None, :]
    kcol = s * HIDDEN + koff
    w = tl.load(
        w1_ptr + rn[:, None, None] * stride_w1 + kcol[None, :, :],
        mask=(rn < N_REAL)[:, None, None] & kmask[None, :, :],
        other=0.0,
    )
    gam = 1.0 + tl.load(norm_w_ptr + kcol, mask=kmask, other=0.0).to(tl.float32)
    if LAUNCH_WITH_PDL:
        # The weights are static, so they stream while the previous kernel finishes;
        # every read of an activation and every write comes after the wait.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    lead = (rn == pid_n * BLOCK_N) & (pid_n == 0)
    for m in tl.static_range(M):
        x = tl.load(hc_ptr + m * stride_hc + kcol, mask=kmask, other=0.0).to(tl.float32)
        if HAS_INJECT:
            if TOPK > 0:
                # The routed experts' finalize: this row's TOPK permuted FC2 rows weighted
                # by their routing weights, fp32 FMAs in slot order from zero (a slot mapped
                # to -1 is skipped), stored once in the rows' dtype.
                b = tl.zeros([NC_P2, KC], dtype=tl.float32)
                for k in tl.static_range(TOPK):
                    row = tl.load(topk_row_ptr + m * stride_topk_row + k)
                    wk = tl.load(topk_w_ptr + m * stride_topk_w + k).to(tl.float32)
                    r = tl.load(
                        block_ptr + tl.maximum(row, 0).to(tl.int64) * stride_block + koff,
                        mask=kmask,
                        other=0.0,
                    ).to(tl.float32)
                    b = tl.where(row >= 0, tl.fma(r, wk, b), b)
                b = b.to(block_ptr.dtype.element_ty).to(tl.float32)
            else:
                b = tl.load(block_ptr + m * stride_block + koff, mask=kmask, other=0.0).to(
                    tl.float32
                )
            if HAS_SHARED:
                # The MoE block output, `routed + sigmoid(gate) * shared` in fp32, rounded
                # once to the routed output's dtype -- the value that block used to store.
                sg = tl.sigmoid(tl.load(gate_ptr + m * stride_gate).to(tl.float32))
                sh = tl.load(shared_ptr + m * stride_shared + koff, mask=kmask, other=0.0)
                b = (b + sg * sh.to(tl.float32)).to(block_ptr.dtype.element_ty).to(tl.float32)
            g = tl.load(inj_ptr + m * stride_inj + s).to(tl.float32)
            # Rounded once, like `addcmul`: the norm and every later site read the rounded value.
            xr = (x + b * g).to(hc_out_ptr.dtype.element_ty)
            tl.store(hc_out_ptr + m * stride_hc_out + kcol, xr, mask=kmask & (pid_n == 0))
            x = xr.to(tl.float32)
        p = tl.sum(tl.sum(w.to(tl.float32) * (x * gam)[None, :, :], axis=2), axis=1)
        tl.store(part_ptr + (s * M + m) * N_PAD + rn, p)
        sq = tl.sum(
            tl.sum(tl.broadcast_to((x * x)[None, :, :], [BLOCK_N, NC_P2, KC]), axis=2), axis=1
        )
        tl.store(ss_ptr + s * M + m + 0 * rn, sq, mask=lead)


#: Launch configuration: (BLOCK_N, KC, num_warps). Swept on GB300 (see the note).
DEFAULT_CONFIG = (2, 512, 2)
#: With the MoE combine every program also reads the routed and shared rows, so
#: wider weight tiles (half the programs) win: 0.2-0.3 us per site on GB300.
COMBINE_CONFIG = (4, 512, 2)
#: With the routed experts' finalize folded in (`topk_rows`), every program
#: gathers the row's top-k FC2 rows as well.
FOLD_CONFIG = (4, 512, 2)


def launch(
    hyper_input,
    block_out,
    inject,
    norm_weight,
    first,
    hc_count,
    n_cols,
    shared=None,
    gate_logits=None,
    topk_weights=None,
    topk_rows=None,
    config=DEFAULT_CONFIG,
):
    """The launch behind `Qwen4ExpHcDecodeFirst.__call__`, tile config exposed.
    Returns `(hc_out, part, ss)`."""
    rows, width = hyper_input.shape
    hidden = width // hc_count
    block_n, kc, warps = config
    while hidden % kc:
        kc //= 2
    n_tiles = triton.cdiv(n_cols, block_n)
    n_pad = n_tiles * block_n
    dev, dt = hyper_input.device, hyper_input.dtype
    has_inject = block_out is not None
    has_shared = shared is not None
    top_k = topk_rows.shape[1] if topk_rows is not None else 0
    hc_out = torch.empty((rows, width), dtype=dt, device=dev) if has_inject else hyper_input
    part = torch.empty((hc_count, rows, n_pad), dtype=torch.float32, device=dev)
    ss = torch.empty((hc_count, rows), dtype=torch.float32, device=dev)
    pdl = pdl_enabled(dev.index if dev.index is not None else torch.cuda.current_device())
    if rows > 0:
        _hc_decode_first_kernel[(n_tiles, hc_count)](
            hyper_input,
            block_out if has_inject else hyper_input,
            inject if has_inject else hyper_input,
            shared if has_shared else hyper_input,
            gate_logits if has_shared else hyper_input,
            topk_weights if top_k else hyper_input,
            topk_rows if top_k else hyper_input,
            norm_weight,
            first,
            hc_out,
            part,
            ss,
            hyper_input.stride(0),
            block_out.stride(0) if has_inject else 0,
            inject.stride(0) if has_inject else 0,
            shared.stride(0) if has_shared else 0,
            gate_logits.stride(0) if has_shared else 0,
            topk_weights.stride(0) if top_k else 0,
            topk_rows.stride(0) if top_k else 0,
            hc_out.stride(0),
            first.stride(1),
            M=rows,
            HIDDEN=hidden,
            N_REAL=n_cols,
            N_PAD=n_pad,
            BLOCK_N=block_n,
            KC=kc,
            NC=hidden // kc,
            NC_P2=triton.next_power_of_2(hidden // kc),
            HAS_INJECT=has_inject,
            HAS_SHARED=has_shared,
            TOPK=top_k,
            LAUNCH_WITH_PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
    return hc_out, part, ss


class Qwen4ExpHcDecodeFirst(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `N = n_cols`, `first` the
    `[width, >= N]` view of the load-time `[down; block_inject; 0]` weight::

        hc_out[t, s*H+i] = hyper_input[t, s*H+i] + block_out[t, i] * inject[t, s]   (rounded once)
        part[s, t, n]    = sum_i hc_out[t, s*H+i] * (1 + norm_weight[s*H+i]) * first[s*H+i, n]
        ss[s, t]         = sum_i hc_out[t, s*H+i]^2

    Returns `(hc_out, part, ss)`: `hc_out` fresh in the activation dtype (it
    *is* `hyper_input` when `block_out` / `inject` are None -- the norm-only
    site); `part` fp32 `[C, T, N]` (a column slice of a padded buffer) and `ss`
    fp32 `[C, T]`. The grouped RMS `rstd[t, s] = (ss[s, t] / H + eps)^-1/2`
    scales stream `s`'s partial: `normed @ first[:, :N] = sum_s rstd[:, s] *
    part[s]`. `qwen4_exp_hc_decode_mix` makes that combination.

    With `shared` and `gate_logits` (the site after a MoE block), `block_out`
    is the routed experts' output and the block output injected above is
    `block_out[t, i] + sigmoid(gate_logits[t]) * shared[t, i]` in fp32, rounded
    once to `block_out`'s dtype -- bitwise what `fused_sigmoid_gate_mul_add`
    stores. `gate_logits` is one pre-sigmoid scalar per row, `[T, 1]` or `[T]`.

    With `topk_weights` / `topk_rows` as well (the MoE runner left unfinalized),
    `block_out` is its permuted FC2 output `[P, H]` and the routed row is the
    runner's finalize, formed here: `routed[t, i] = sum_k topk_weights[t, k] *
    block_out[topk_rows[t, k], i]` as fp32 FMAs in slot order from zero (a slot
    whose row is -1 is skipped), rounded once to `block_out`'s dtype -- bitwise
    what the runner's finalize stores. `topk_rows` is the runner's
    expanded->permuted index map `[T, k]` int32, `topk_weights` its `[T, k]`
    routing weights. The fold needs the combine (`shared`, `gate_logits`).

    Fusion boundary: the routed experts' finalize (optional), the
    shared-expert combine (optional), the previous block's
    injection, the `(1 + w)` scale and the first projection per stream, plus
    the per-stream sum of squares. The routed and shared experts and the gate
    projection, the RMS itself, the activations and everything after are not
    inside.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True), moe (False: shared + gate_logits too), gate
    # "column" [T, 1] | "flat" [T], fold (0: top-k of the finalize folded in,
    # block_out then the permuted rows), drop (False: some slots -1), n_cols
    # (324), hc_count (4), hidden (2560), n_rows (384: the padded weight),
    # dtype (bf16).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560, 324 cols of 384, bf16, inject, {why}", spec=dict(rows=t))
        for t, why in (
            (1, "decode 1 row (con 1)"),
            (2, "decode 2 rows"),
            (4, "decode 4 rows"),
            (8, "decode 8 rows"),
            (16, "decode 16 rows (MAX_ROWS)"),
        )
    ) + (
        Cell(why="MoE combine, [T, 1] gate, 1 row (con 1)", spec=dict(rows=1, moe=True)),
        Cell(why="MoE combine, flat [T] gate, 4 rows", spec=dict(rows=4, moe=True, gate="flat")),
        Cell(why="MoE combine, 16 rows (MAX_ROWS)", spec=dict(rows=16, moe=True)),
        Cell(
            why="MoE combine into the stack-final mixer: 320 cols, 1 row",
            spec=dict(rows=1, moe=True, n_cols=320),
        ),
        Cell(why="finalize fold, top-10, 1 row (con 1)", spec=dict(rows=1, moe=True, fold=10)),
        Cell(
            why="finalize fold into the stack-final mixer: 320 cols, 1 row",
            spec=dict(rows=1, moe=True, fold=10, n_cols=320),
        ),
        Cell(
            why="finalize fold, 4 rows, slots mapped to -1, flat gate",
            spec=dict(rows=4, moe=True, fold=10, drop=True, gate="flat"),
        ),
        Cell(why="finalize fold, 16 rows (MAX_ROWS)", spec=dict(rows=16, moe=True, fold=10)),
        Cell(
            why="norm-only site (stack start / after PLE), 1 row", spec=dict(rows=1, inject=False)
        ),
        Cell(why="norm-only site, 8 rows", spec=dict(rows=8, inject=False)),
        Cell(why="stack-final mixer: 320 cols, 1 row", spec=dict(rows=1, n_cols=320)),
        Cell(why="stack-final mixer: 320 cols, 16 rows", spec=dict(rows=16, n_cols=320)),
        Cell(why="hyper_input a row slice buf[::2], 4 rows", spec=dict(rows=4, layout="row_slice")),
        Cell(why="unpadded weight [width, 324], 3 rows", spec=dict(rows=3, n_rows=324)),
        Cell(why="hc_count 2 at hidden 2560, 2 rows", spec=dict(rows=2, hc_count=2)),
        Cell(
            why="hidden 128, 36 cols (not the checkpoint), 5 rows",
            spec=dict(rows=5, hidden=128, n_cols=36),
        ),
        Cell(why="float16 operands, 4 rows", spec=dict(rows=4, dtype=torch.float16)),
        Cell(why="zero rows", spec=dict(rows=0)),
    )

    note = """
    Metadata consumed: none. Call sites: `DecodeTarget`'s trunk, every
    hyper-connection site (97 per forward) when the decode batch has at most
    `_HC_DECODE_GEMV_MAX_ROWS` rows (<= MAX_ROWS = 16), followed by
    `qwen4_exp_hc_decode_mix`. Larger batches and `PrefillTarget` keep
    `qwen4_exp_hc_inject_norm` + `cublas_mm` + `qwen4_exp_hc_lowrank_act`.
    The 47 sites after a MoE block other than the one before the PLE add pass
    `shared` and `gate_logits`, and with them the unfinalized runner's
    `topk_weights` / `topk_rows` (`block_out` its permuted FC2 rows).

    Preconditions: 2-D operands with unit column stride (row strides free);
    `first` `[width, >= n_cols]` with unit stride along `width` (the `.t()`
    view of a row-major `[N, width]` weight); `hidden` a multiple of 16. Row
    count is a compile-time constant (one Triton specialization per decode
    batch size); at most MAX_ROWS rows certified. Columns past `n_cols` are
    never read, so the zero padding of the load-time weight costs no traffic.

    Numerics: `hc_out` as `qwen4_exp_hc_inject_norm` (fp32 accumulate, one
    rounding); the projection multiplies the fp32 `hc_out * (1 + w)` by the
    16-bit weight in fp32 (the composition it replaces rounded the normed
    stream to bf16 first); deterministic (no atomics, fixed reduction order).
    The optional combine is fp32 rounded once to `block_out`'s dtype, which
    against the float64 reference can land one ulp of the block output away
    (its rounding midpoint); `hc_out` carries that scaled by `inject`, and the
    test checks `part` / `ss` against the projection of the `hc_out` the op
    stored. Every program recomputes the combine for its stream's columns: the
    extra `[hidden]` read per program is L2-resident. The optional finalize is
    the runner's own arithmetic (fp32 FMAs in slot order from zero, one
    rounding), so `hc_out` and `ss` are bitwise the finalize + combine
    composition at the same tile, and so is `part` from 2 rows; at 1 row the
    fold's Triton specialization reduces the dot in another fp32 order (~12% of
    `part` one fp32 ulp apart, < 1e-7 relative; GB300). The
    float64 reference rounds the finalize from an exact sum, which can land one
    ulp of the routed row away, and the test's band carries that term too.
    With the fold every program also gathers the row's top-k FC2 rows (`k x
    [hidden]`, L2-resident: the runner's GEMM wrote them just before).
    Graph-capturable: no host sync, outputs allocated per call. Launched with
    programmatic dependent launch on sm_90+ (`TRTLLM_ENABLE_PDL=0` turns it
    off): the static weight tile streams before `gdc_wait`, every activation
    read and every write after it, and dependents are released at once so
    `qwen4_exp_hc_decode_mix` can stream its weights meanwhile.

    Grid: (cdiv(n_cols, 2), hc_count) programs of 2 warps, each one load of 2
    weight rows x one stream (10 KB); with the combine, 4 weight rows
    (`COMBINE_CONFIG`; it sums `part` in a different fp32 order than the
    default tile, inside the same band). Measured on GB300 (graph replay, cold
    weights, with `qwen4_exp_hc_decode_mix`): 1 row is fast; each further row
    costs several us (the row loop serializes its memory round trips), so the
    forward dispatches only 1-row batches here. Hard limits: int32 offsets, far below 2^31 at <= 16 rows.
    """

    def __call__(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        n_cols: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if topk_rows is not None:
            config = FOLD_CONFIG
        elif shared is not None:
            config = COMBINE_CONFIG
        else:
            config = DEFAULT_CONFIG
        hc_out, part, ss = launch(
            hyper_input,
            block_out,
            inject,
            norm_weight,
            first,
            hc_count,
            n_cols,
            shared,
            gate_logits,
            topk_weights,
            topk_rows,
            config=config,
        )
        return hc_out, part[:, :, :n_cols], ss

    def reference(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        n_cols: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """float64; the finalize (with `topk_rows`) rounded once to
        `block_out`'s dtype, the combine (with `shared`) rounded once to `block_out`'s
        dtype, `hc_out` rounded once to the activation dtype, `part` and `ss`
        rounded once to fp32."""
        dtype = hyper_input.dtype
        rows, width = hyper_input.shape
        hidden = width // hc_count
        if block_out is None:
            hc_out = hyper_input
        else:
            if topk_rows is not None:
                valid = (topk_rows >= 0).unsqueeze(-1)
                rows_ = block_out.double()[topk_rows.clamp(min=0).long()]
                w = topk_weights.double().unsqueeze(-1)
                block_out = torch.where(valid, rows_ * w, 0.0).sum(1).to(block_out.dtype)
            if shared is not None:
                sg = torch.sigmoid(gate_logits.double().reshape(rows, 1))
                block_out = (block_out.double() + sg * shared.double()).to(block_out.dtype)
            streams = hyper_input.double().unflatten(-1, (hc_count, hidden))
            gated = block_out.double().unsqueeze(-2) * inject.double().unsqueeze(-1)
            hc_out = (streams + gated).flatten(-2).to(dtype)
        x = hc_out.double().unflatten(-1, (hc_count, hidden))
        y = x * (1.0 + norm_weight.double()).unflatten(-1, (hc_count, hidden))
        w = first[:, :n_cols].double().unflatten(0, (hc_count, hidden))
        part = torch.einsum("tsi,sin->stn", y, w).float()
        ss = x.pow(2).sum(dim=-1).t().float()
        return hc_out, part, ss

    def is_valid(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        n_cols: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        topk_rows: Optional[torch.Tensor] = None,
    ) -> None:
        # Raw row strides with unit column stride everywhere; a contiguous
        # [width, N] weight would be read with its rows and columns swapped.
        assert hyper_input.dim() == 2 and hyper_input.stride(1) == 1, "hyper_input [T, width]"
        rows, width = hyper_input.shape
        assert width % hc_count == 0 and (width // hc_count) % 16 == 0, "hidden multiple of 16"
        assert (block_out is None) == (inject is None), "block_out and inject come together"
        assert (shared is None) == (gate_logits is None), "shared and gate_logits come together"
        assert shared is None or block_out is not None, "the combine needs block_out"
        assert (topk_weights is None) == (topk_rows is None), (
            "topk_weights and topk_rows come together"
        )
        assert topk_rows is None or shared is not None, "the finalize fold needs the combine"
        if block_out is not None:
            if topk_rows is None:
                assert block_out.shape == (rows, width // hc_count) and block_out.stride(1) == 1
            else:
                # The permuted FC2 rows: any row count, read through topk_rows.
                assert block_out.dim() == 2 and block_out.shape[1] == width // hc_count
                assert block_out.stride(1) == 1, "permuted rows [P, hidden]"
                assert topk_rows.dtype == torch.int32, "topk_rows int32 (the runner's index map)"
                assert topk_rows.dim() == 2 and topk_rows.shape[0] == rows, "topk_rows [T, k]"
                assert topk_rows.stride(1) == 1 and topk_weights.stride(1) == 1
                assert topk_weights.shape == topk_rows.shape, "topk_weights [T, k]"
            assert inject.shape == (rows, hc_count) and inject.stride(1) == 1
        if shared is not None:
            assert shared.shape == (rows, width // hc_count), "shared [T, hidden]"
            assert shared.stride(1) == 1, "shared [T, hidden]"
            assert shared.dtype == block_out.dtype, "shared in block_out's dtype"
            assert gate_logits.shape in ((rows,), (rows, 1)), "gate_logits [T, 1] or [T]"
            assert rows <= 1 or gate_logits.stride(0) != 0, "one gate logit per row"
        assert norm_weight.shape == (width,) and norm_weight.is_contiguous(), "norm_weight [width]"
        assert first.shape[0] == width and first.shape[1] >= n_cols, "first [width, >= n_cols]"
        assert first.stride(0) == 1, "first must be the .t() view of a row-major [N, width] weight"


qwen4_exp_hc_decode_first = Qwen4ExpHcDecodeFirst()
