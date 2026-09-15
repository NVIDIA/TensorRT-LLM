# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone driver + pytest sweep for the cuTe DSL GVR Top-K kernel.

Compares the kernel output against ``torch.topk`` using tie-aware set
equality. This file exposes every knob
(T, V, ``min_blocks_per_mp``, warp-parallel reduce, both unroll switches,
``use_constant_hint``, ``compress_ratio``, ``max_seq_len`` hint) so bench
scripts can override the heuristic.

**Not in CI** - this file imports the DSL kernel module directly
(no ``trtllm`` runtime dep), to enable knob-A/B development outside the
production op.

Two usage modes:

* `python -m pytest run_gvr_topk.py`` - exhaustive parameterized correctness sweep
  (dtype x K x N x seed x next_n x T x V x warp-parallel-reduce).
* ``python run_gvr_topk.py --dtype bf16 --top_k 1024 --N 8192`` -
  single-case correctness verification on user-specified shape; knob
  overrides via ``--num_threads`` / ``--use_256bit_load`` / etc.
"""

import argparse
import functools
import os
import sys
from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
import pytest
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import GvrTopKKernel
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[4] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # type: ignore[no-redef]


_DTYPE_TORCH_TO_CUTE = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


@functools.cache
def _compile(
    cute_dtype,
    top_k: int,
    next_n: int,
    enable_unroll_4: bool,
    enable_phase3_unroll: bool,
    use_constant_hint: bool,
    min_blocks_per_mp: int,
    use_256bit_load: bool,
    num_threads_per_block: int,
    enable_warp_parallel_reduce: bool,
    compress_ratio: int,
    return_output_values: bool,
    cluster_size: int = 1,
    seqlen_sorted: bool = False,
    p4_warp_redundant: bool = True,
    p2_warp_redundant: bool = True,
    enable_block_skip: bool = False,
    pdl_wait_late: bool = True,
    p4_tail_v3: "bool | None" = None,
    p4_no_fine: "bool | None" = None,
    p4_exact_tail: "bool | None" = None,
    p4_tail_fast: "bool | None" = None,
    p1r_rescue: bool = True,
    num_bins: "int | None" = None,
    p4_fine_rangetest: "bool | None" = None,
    p4_scat_rangetest: bool = False,
    use_ext_counts: bool = False,
    emit_xstate: bool = False,
    use_ext_cand: bool = False,
    ext_rungs: bool = False,
    cand_cap: int = 5120,
    accept_cap: "int | None" = None,
    kc_override: "int | None" = None,
    self_scan: bool = False,
    cap_c: "int | None" = None,
):
    """JIT-compile the GVR kernel for a specific knob combination.

    ``functools.cache`` keys on all args so repeated calls in the same
    process reuse the compiled kernel without an explicit module-level dict.
    """
    n_rows = cute.sym_int()
    n_cols = cute.sym_int()
    n_batch = cute.sym_int()
    in_align = 32 if use_256bit_load else 16
    input_fake = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        (n_rows, n_cols),
        stride_order=(1, 0),
        assumed_align=in_align,
    )
    pre_idx_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_batch,),
        stride_order=(0,),
    )
    # When return_output_values=False the kernel skips all STG.value
    # writes; pass None so cute.compile doesn't materialize the value
    # output placeholder.
    out_values_fake = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype,
            (n_rows, top_k),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if return_output_values
        else None
    )
    out_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n_rows, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    # When seqlen_sorted=False the kernel never reads order_row (the
    # const_expr branch elides the indirection); pass None so cute.compile
    # doesn't materialize a placeholder buffer.
    order_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        if seqlen_sorted
        else None
    )
    block_max_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (n_rows, cute.sym_int()),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if enable_block_skip
        else None
    )
    # ext counts ride PACKED with the lines ([rows, 8] fp32: lines at
    # [0..2], counts as floats at [3..5]) - one 32B sector per row
    seed_thr_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (n_rows, 8 if use_ext_counts else 3),
            stride_order=(1, 0),
            assumed_align=4,
        )
        if (use_ext_counts or ext_rungs)
        else None
    )
    seed_counts_fake = None
    cand_vals_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (n_rows, cand_cap), stride_order=(1, 0), assumed_align=4
        )
        if use_ext_cand
        else None
    )
    cand_idx_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, cand_cap), stride_order=(1, 0), assumed_align=4
        )
        if (use_ext_cand or self_scan)
        else None
    )
    cand_ctl_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, 4), stride_order=(1, 0), assumed_align=8
        )
        if use_ext_cand
        else None
    )
    xstate_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (n_rows, 8), stride_order=(1, 0), assumed_align=4
        )
        if emit_xstate
        else None
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = GvrTopKKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads_per_block,
        enable_unroll_4=enable_unroll_4,
        enable_phase3_unroll=enable_phase3_unroll,
        use_constant_hint=use_constant_hint,
        min_blocks_per_mp=min_blocks_per_mp,
        use_256bit_load=use_256bit_load,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        seqlen_sorted=seqlen_sorted,
        p4_warp_redundant=p4_warp_redundant,
        p2_warp_redundant=p2_warp_redundant,
        enable_block_skip=enable_block_skip,
        pdl_wait_late=pdl_wait_late,
        p4_tail_v3=p4_tail_v3,
        p4_no_fine=p4_no_fine,
        p4_exact_tail=p4_exact_tail,
        p4_tail_fast=p4_tail_fast,
        p1r_rescue=p1r_rescue,
        num_bins=num_bins,
        p4_fine_rangetest=p4_fine_rangetest,
        p4_scat_rangetest=p4_scat_rangetest,
        use_ext_counts=use_ext_counts,
        emit_xstate=emit_xstate,
        use_ext_cand=use_ext_cand,
        ext_rungs=ext_rungs,
        cand_cap=cand_cap,
        accept_cap=accept_cap,
        kc_override=kc_override,
        self_scan=self_scan,
        cap_c=cap_c,
        # ext counts need 3 rung slots (M_thr == 3): 2 qfracs + vseed;
        # the qfrac values are unused here - only the slot count matters.
        r0_qfracs=(0.85, 0.35) if (use_ext_counts or ext_rungs) else None,
    )
    return cute.compile(
        kernel,
        input_fake,
        pre_idx_fake,
        seq_lens_fake,
        out_values_fake,
        out_indices_fake,
        order_row_fake,
        stream=fake_stream,
        block_max=block_max_fake,
        seed_thr=seed_thr_fake,
        seed_counts=seed_counts_fake,
        xstate=xstate_fake,
        cand_vals=cand_vals_fake,
        cand_idx=cand_idx_fake,
        cand_ctl=cand_ctl_fake,
        options="--enable-tvm-ffi",
    )


_FLT_MAX = torch.finfo(torch.float32).max
_META_BLOCK = 128
_META_RECS_PER_BLOCK = 4  # warp-partial records per block (indexer layout)


def _row_n_eff(
    seq_lens: torch.Tensor,
    num_rows: int,
    next_n: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Per-row effective scan length, mirroring the kernel formula."""
    dev = seq_lens.device
    rows = torch.arange(num_rows, device=dev)
    sl = seq_lens[rows // next_n].to(torch.int64)
    actual = sl - next_n + (rows % next_n) + 1
    return actual // compress_ratio


def emu_block_max(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
    tail_mode: str = "pad_inf",
    records: str = "positional",
) -> torch.Tensor:
    """``[num_rows, nb_pad*4] fp32`` warp-partial upper-bound records.

    tail_mode:
      "exact":   tight bound - max over valid positions only.
      "pad_inf": a partially-valid tail unit is forced to +FLT_MAX, the
                 worst legal inflation (the indexer masks by request-level
                 ctx >= N_eff, so tail positions can inflate the bound).
    records:
      "rotate":     fold-correctness fixture - the 128-block max lands in
                    ONE slot rotated by blk % 4, the other 3 hold
                    -FLT_MAX. Valid ONLY for grain-128 consumers (slots
                    are NOT positional).
      "positional": production semantics - record r is the exact max of
                    positions [r*32, r*32+32) (the indexer's TMEM T2R
                    partition gives warp w of a tile the contiguous
                    positions [tile*128 + w*32, +32)). Required for
                    skip_grain=32; also a legal grain-128 input (its
                    fold is the block max).
    """
    assert tail_mode in ("exact", "pad_inf")
    assert records in ("rotate", "positional")
    R, C = logits.shape
    nb = (C + _META_BLOCK - 1) // _META_BLOCK
    dev = logits.device
    n_eff = _row_n_eff(seq_lens, R, next_n, compress_ratio).unsqueeze(1)
    lf = logits.to(torch.float32)
    pad = nb * _META_BLOCK - C
    if pad:
        lf = torch.nn.functional.pad(lf, (0, pad), value=float("-inf"))
    pos = torch.arange(nb * _META_BLOCK, device=dev).unsqueeze(0)
    masked = torch.where(pos < n_eff, lf, torch.full_like(lf, float("-inf")))
    if records == "positional":
        sub = _META_BLOCK // _META_RECS_PER_BLOCK  # 32 positions/record
        nrec = nb * _META_RECS_PER_BLOCK
        rmax = masked.view(R, nrec, sub).amax(-1)
        if tail_mode == "pad_inf":
            rec_start = torch.arange(nrec, device=dev).unsqueeze(0) * sub
            partial = (rec_start < n_eff) & (rec_start + sub > n_eff)
            rmax = torch.where(partial, torch.full_like(rmax, _FLT_MAX), rmax)
        return rmax.contiguous()
    bmax = masked.view(R, nb, _META_BLOCK).amax(-1)
    if tail_mode == "pad_inf":
        blk_start = torch.arange(nb, device=dev).unsqueeze(0) * _META_BLOCK
        partial = (blk_start < n_eff) & (blk_start + _META_BLOCK > n_eff)
        bmax = torch.where(partial, torch.full_like(bmax, _FLT_MAX), bmax)
    out = torch.full((R, nb, _META_RECS_PER_BLOCK), -_FLT_MAX, dtype=torch.float32, device=dev)
    slot = torch.arange(nb, device=dev) % _META_RECS_PER_BLOCK
    out[:, torch.arange(nb, device=dev), slot] = bmax
    return out.reshape(R, nb * _META_RECS_PER_BLOCK).contiguous()


# Rung offsets below each 32-position record's max for the meta-seed
# metadata.
_META_DELTAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def emu_block_meta(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int = 1,
    next_n: int = 1,
) -> torch.Tensor:
    """Emulate the indexer-side per-32-block rung-count metadata.

    Record r packs, for each rung offset ``delta_j`` in
    ``_META_DELTAS``, ``count(v >= record_max - delta_j)`` over
    positions ``[32r, 32r+32)`` as a 5-bit saturating field at bits
    ``[5j, 5j+5)`` (31 = "31 or 32"; the kernel decodes 31 as the safe
    upper bound 32). Positions beyond the row's effective length are
    excluded. Layout matches ``block_max``: ``[num_rows, nrec]`` int32
    with ``nrec = ceil(N/128)*4`` (one record per 32 positions).
    """
    assert next_n == 1, "emu_block_meta: next_n == 1 only"
    x = logits.float()
    R, N = x.shape
    nrec = ((N + _META_BLOCK - 1) // _META_BLOCK) * _META_RECS_PER_BLOCK
    npad = nrec * 32
    if npad > N:
        x = torch.nn.functional.pad(x, (0, npad - N), value=float("-inf"))
    n_eff = seq_lens.long() // compress_ratio
    ar = torch.arange(npad, device=x.device)[None, :]
    x = x.masked_fill(ar >= n_eff[:, None], float("-inf"))
    xb = x.view(R, nrec, 32)
    m = xb.amax(2, keepdim=True)
    meta = torch.zeros(R, nrec, dtype=torch.int32, device=x.device)
    for j, d in enumerate(_META_DELTAS):
        cj = (xb >= (m - d)).sum(2).clamp(max=31).to(torch.int32)
        meta |= cj << (5 * j)
    return meta.contiguous()


def derive_seed_rungs(
    prev_thr: torch.Tensor,
    prev_sthr: "torch.Tensor | None" = None,
    prev_counts: "torch.Tensor | None" = None,
    count_octaves: float = 2.0,
    fallback_spread: float = 0.5,
    top_k: "int | None" = None,
) -> torch.Tensor:
    """Host-side slope-adaptive seed rung derivation (waterfall closed loop).

    Estimates the per-row local slope of log2(count) vs threshold from the
    PREVIOUS step's 3 rung measurements and places the next step's guard
    rungs ``count_octaves`` octaves away from the mid rung (= the previous
    accepted threshold).

    Args:
        prev_thr: [rows] previous accepted threshold (xstate[:, 2]).
        prev_sthr: [rows, 3] previous step's rung thresholds (or None).
        prev_counts: [rows, 3] previous step's rung counts (or None).

    Returns:
        [rows, 3] fp32 seed thresholds (ascending).
    """
    if prev_sthr is None or prev_counts is None:
        d = torch.full_like(prev_thr, fallback_spread)
        d_lo = d
    else:
        c_lo = prev_counts[:, 0].float().clamp(min=1.0)
        c_hi = prev_counts[:, 2].float().clamp(min=1.0)
        dthr = (prev_sthr[:, 2] - prev_sthr[:, 0]).clamp(min=1e-3)
        slope = (torch.log2(c_lo) - torch.log2(c_hi)) / dthr
        d = torch.where(
            slope > 0.05,
            count_octaves / slope.clamp(min=0.05),
            torch.full_like(slope, fallback_spread),
        ).clamp(0.1, 4.0)
        # undershoot hysteresis: a row whose previous loose rung caught
        # fewer than K widens only its next down-guard by +2 octaves.
        oct_lo = torch.full_like(slope, count_octaves)
        if top_k is not None:
            oct_lo = torch.where(prev_counts[:, 0].float() < float(top_k), oct_lo + 2.0, oct_lo)
        d_lo = torch.where(
            slope > 0.05,
            oct_lo / slope.clamp(min=0.05),
            torch.full_like(slope, fallback_spread),
        ).clamp(0.1, 6.0)
    return torch.stack([prev_thr - d_lo, prev_thr, prev_thr + d], dim=1).contiguous()


def emu_cand_bucketed(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    seed_thr: torch.Tensor,
    cap: int,
    seg_cap: int = 8192,
    next_n: int = 1,
    compress_ratio: int = 1,
    sentinel_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bucketed SoA candidate emission (v5 contract).

    Three fixed segments in one buffer: A = [0, seg_cap) holds >= t2,
    B = [seg_cap, 2*seg_cap) holds [t1, t2), C = [2*seg_cap, 2*seg_cap
    + cap) holds [t0, t1). A full segment spills to the next looser one
    (never drops an entry), so the union always equals the full >= t0
    set, and a segment group is complete exactly when its line's count
    fits the acceptance band (seg_cap = B*). Sentinel pads land in C.
    ctl = {n0 (claimed incl pads), void, n1, n2}.
    """
    R, _C = logits.shape
    dev = logits.device
    n_eff = _row_n_eff(seq_lens, R, next_n, compress_ratio)
    lf = logits.to(torch.float32)
    width = 2 * seg_cap + cap
    cand_vals = torch.full((R, width), float("-inf"), dtype=torch.float32, device=dev)
    cand_idx = torch.full((R, width), -1, dtype=torch.int32, device=dev)
    ctl = torch.zeros((R, 4), dtype=torch.int32, device=dev)
    for r in range(R):
        ne = int(n_eff[r])
        row = lf[r, :ne]
        hits = torch.nonzero(row >= seed_thr[r, 0], as_tuple=False).flatten()
        cnt = hits.numel()
        ctl[r, 2] = int((row >= seed_thr[r, 1]).sum())
        ctl[r, 3] = int((row >= seed_thr[r, 2]).sum())
        # emission order is value-blind: shuffle, then classify
        perm = hits[torch.randperm(cnt, device=dev)]
        v = row[perm]
        seg = torch.where(v >= seed_thr[r, 2], 0, torch.where(v >= seed_thr[r, 1], 1, 2))
        # vectorized spill-to-looser: stream s = native entries + spill
        # from s-1 (in emission order); ordinal beyond the cap spills on.
        in_a = seg == 0
        ord_a = torch.cumsum(in_a.int(), 0)
        stay_a = in_a & (ord_a <= seg_cap)
        in_b = (seg == 1) | (in_a & ~stay_a)
        ord_b = torch.cumsum(in_b.int(), 0)
        stay_b = in_b & (ord_b <= seg_cap)
        in_c = (seg == 2) | (in_b & ~stay_b)
        ord_c = torch.cumsum(in_c.int(), 0)
        stay_c = in_c & (ord_c <= cap)
        voided = int(in_c.sum()) > cap
        slot = torch.full_like(seg, -1)
        slot[stay_a] = ord_a[stay_a] - 1
        slot[stay_b] = seg_cap + (ord_b[stay_b] - 1)
        slot[stay_c] = 2 * seg_cap + (ord_c[stay_c] - 1)
        live = slot >= 0
        cand_vals[r, slot[live].long()] = v[live]
        cand_idx[r, slot[live].long()] = perm[live].int()
        claimed = cnt + sentinel_pad
        ctl[r, 0] = claimed
        ctl[r, 1] = 1 if (voided or claimed > cap) else 0
    return cand_vals, cand_idx, ctl


def derive_seed_lines_v4(
    prev_anchor: torch.Tensor,
    prev_sthr: "torch.Tensor | None" = None,
    prev_ctl: "torch.Tensor | None" = None,
    targets: "tuple[float, float, float]" = (8192.0, 5120.0, 2048.0),
    fallback_spread: float = 0.5,
) -> torch.Tensor:
    """v4 host-side line placement: put [t0, t1, t2] at target COUNTS.

    Slope of log2(count) vs threshold is fit from the previous step's
    (t0, n0) / (t2, n2) pairs (counts ride in the widened control words);
    each new line lands where the fit predicts its target count. Targets
    descend (t0 loosest / largest count, t2 tightest).

    Args:
        prev_anchor: [rows] previous accepted cut value (xstate[:, 2]).
        prev_sthr: [rows, 3] previous lines (or None -> fixed spread).
        prev_ctl: [rows, 4] previous control words {n0, void, n1, n2}.
        targets: (T0, T1, T2) target counts, T0 > T1 > T2.

    Returns:
        [rows, 3] fp32 lines ascending [t0, t1, t2].
    """
    t0_t, t1_t, t2_t = targets
    if prev_sthr is None or prev_ctl is None:
        d = torch.full_like(prev_anchor, fallback_spread)
        return torch.stack([prev_anchor - d, prev_anchor, prev_anchor + d], dim=1).contiguous()
    c0 = prev_ctl[:, 0].float().clamp(min=1.0)
    c2 = prev_ctl[:, 3].float().clamp(min=1.0)
    dthr = (prev_sthr[:, 2] - prev_sthr[:, 0]).clamp(min=1e-3)
    slope = ((torch.log2(c0) - torch.log2(c2)) / dthr).clamp(min=0.05, max=64.0)
    # anchor count estimate: slide the anchor onto the prev line fit
    anch_c = (c2 * torch.exp2(-(prev_anchor - prev_sthr[:, 2]) * slope)).clamp(min=1.0, max=1e6)
    lines = [prev_anchor + torch.log2(anch_c / tgt) / slope for tgt in (t0_t, t1_t, t2_t)]
    out = torch.stack(lines, dim=1)
    # enforce strictly ascending (degenerate slope guards)
    out[:, 1] = torch.maximum(out[:, 1], out[:, 0] + 1e-4)
    out[:, 2] = torch.maximum(out[:, 2], out[:, 1] + 1e-4)
    return out.contiguous()


def pack_seed(
    seed_thr: torch.Tensor,
    seed_counts: torch.Tensor,
    block_max: torch.Tensor = None,
) -> torch.Tensor:
    """Pack lines + exact counts into one [rows, 8] fp32 seed row.

    Lines land at [0..2], counts as floats at [3..5] (exact to 2^24);
    col 6 optionally carries the adaptive-skip pass count (32-grain
    block records clearing t_0; 0 = not provided). One 32B sector per
    row. Build ONCE per step, outside any timed region.
    """
    pack = torch.zeros((seed_thr.shape[0], 8), dtype=torch.float32, device=seed_thr.device)
    pack[:, 0:3] = seed_thr
    pack[:, 3:6] = seed_counts.float()
    if block_max is not None:
        pack[:, 6] = (block_max >= seed_thr[:, 0:1]).sum(dim=1).float()
    return pack.contiguous()


def emu_seed_counts(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    seed_thr: torch.Tensor,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """L1 emu: exact per-row threshold counts.

    counts[r][j] = |{i < N_eff(r) : logits[r, i] >= t_j}| on the
    post-conversion values (contract: epilogue_topk_interface.md).
    """
    R, C = logits.shape
    n_eff = _row_n_eff(seq_lens, R, next_n, compress_ratio).unsqueeze(1)
    pos = torch.arange(C, device=logits.device).unsqueeze(0)
    lf = logits.to(torch.float32)
    valid = pos < n_eff
    counts = torch.empty((R, seed_thr.shape[1]), dtype=torch.int32, device=logits.device)
    for j in range(seed_thr.shape[1]):
        counts[:, j] = ((lf >= seed_thr[:, j : j + 1]) & valid).sum(-1, dtype=torch.int32)
    return counts


def emu_cand(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    seed_thr: torch.Tensor,
    cap: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    sentinel_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L2 emu: unordered candidate pre-collect (SoA).

    Unordered (value fp32-bits, index) pairs of all valid positions
    >= t_0 = seed_thr[:, 0]; ctl = {claimed, void}. claimed may
    over-approximate the true count (window sentinels, idx word = -1) -
    ``sentinel_pad`` injects that legally. void=1 when claimed > cap; on
    overflow only the first ``cap`` entries are materialized (contract v2:
    consumers scan [0, min(claimed, cap)) skipping sentinels).
    """
    R, C = logits.shape
    dev = logits.device
    n_eff = _row_n_eff(seq_lens, R, next_n, compress_ratio)
    lf = logits.to(torch.float32)
    cand_vals = torch.full((R, cap), float("-inf"), dtype=torch.float32, device=dev)
    cand_idx = torch.full((R, cap), -1, dtype=torch.int32, device=dev)
    ctl = torch.zeros((R, 4), dtype=torch.int32, device=dev)
    for r in range(R):
        ne = int(n_eff[r])
        hits = torch.nonzero(lf[r, :ne] >= seed_thr[r, 0], as_tuple=False).flatten()
        cnt = hits.numel()
        # emitter-side counts: two extra compares per EMITTED element
        # (t1, t2 > t0 so counting over the list == counting over the row)
        ctl[r, 2] = int((lf[r, :ne] >= seed_thr[r, 1]).sum())
        ctl[r, 3] = int((lf[r, :ne] >= seed_thr[r, 2]).sum())
        # unordered contract: shuffle, then interleave sentinels
        perm = hits[torch.randperm(cnt, device=dev)]
        ent = torch.full((cnt + sentinel_pad,), -1, dtype=torch.int64, device=dev)
        if sentinel_pad:
            slots = torch.randperm(cnt + sentinel_pad, device=dev)[:cnt]
            slots = slots.sort().values
        else:
            slots = torch.arange(cnt, device=dev)
        ent[slots] = perm
        claimed = int(ent.numel())
        nwr = min(claimed, cap)
        live = ent[:nwr] >= 0
        cand_idx[r, :nwr] = ent[:nwr].to(torch.int32)
        cand_vals[r, :nwr][live] = lf[r, ent[:nwr][live]]
        ctl[r, 0] = claimed
        ctl[r, 1] = 1 if claimed > cap else 0
    return cand_vals, cand_idx, ctl


def enc_ordered_f32(t: torch.Tensor) -> torch.Tensor:
    """Order-preserving int encoding of fp32 (an involution).

    Stored back in fp32 slots - matches the indexer's encoded-int atomic
    min/max.
    """
    bits = t.float().contiguous().view(torch.int32)
    enc = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return enc.view(torch.float32)


def hit_agg_identities(num_rows: int, device) -> torch.Tensor:
    """Identity-initialized per-row hit aggregate.

    {enc(+FLT_MAX), enc(-FLT_MAX), 0, 0} - the required initial state of
    the buffer the indexer atomically merges into.
    """
    ident = torch.tensor([_FLT_MAX, -_FLT_MAX], dtype=torch.float32, device=device)
    enc = enc_ordered_f32(ident)
    out = torch.zeros((num_rows, 4), dtype=torch.float32, device=device)
    out[:, 0] = enc[0]
    out[:, 1] = enc[1]
    return out.contiguous()


def gvr_topk_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    num_sms: int = 148,  # default number of sms in a B200
    enable_unroll_4: Optional[bool] = None,
    enable_phase3_unroll: Optional[bool] = None,
    use_constant_hint: bool = False,
    min_blocks_per_mp: Optional[int] = None,
    use_256bit_load: Optional[bool] = None,
    num_threads_per_block: Optional[int] = None,
    enable_warp_parallel_reduce: Optional[bool] = None,
    compress_ratio: int = 1,
    max_seq_len: Optional[int] = None,
    return_output_values: bool = False,
    cluster_size: int = 1,
    seqlen_sorted: bool = False,
    order_row: Optional[torch.Tensor] = None,
    p4_warp_redundant: bool = True,
    p2_warp_redundant: bool = True,
    pdl_wait_late: bool = True,
    p4_tail_v3: "bool | None" = None,
    p4_no_fine: "bool | None" = None,
    p4_exact_tail: "bool | None" = None,
    p4_tail_fast: "bool | None" = None,
    p1r_rescue: bool = True,
    num_bins: "int | None" = None,
    p4_fine_rangetest: Optional[bool] = None,
    p4_scat_rangetest: bool = False,
    block_max: Optional[torch.Tensor] = None,
    skip_min_n: Optional[int] = 200_000,
    seed_thr: Optional[torch.Tensor] = None,
    seed_counts: Optional[torch.Tensor] = None,
    xstate: Optional[torch.Tensor] = None,
    cand_vals: Optional[torch.Tensor] = None,
    cand_idx: Optional[torch.Tensor] = None,
    cand_ctl: Optional[torch.Tensor] = None,
    self_scan: bool = False,
    cap_c: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL GVR Top-K wrapper with every tuning knob exposed.

    ``None``-valued knobs are resolved via the production auto-heuristic
    (same rules as ``CuteDSLGvrTopKDecodeRunner.forward``); concrete values
    override the heuristic for A/B testing.

    Args:
        logits:    ``[num_rows, max_S]`` float32 / bfloat16 / float16.
        pre_idx:   ``[num_rows // next_n, pre_idx_count]`` int32.
                   ``pre_idx[..., 0]`` must be the argmax index - indexer invariant.
        seq_lens:  ``[num_rows // next_n]`` int32 (uncompressed-token space).
        top_k:     K in {512, 1024, 2048} - compile-time specialized.
        next_n:    Temporal stride for V3.2 ``preIdxOffset = (row % next_n) + 1``.
        compress_ratio: KV-indexer compression factor (1 = DSv3.2, 4 = DSv4).
                   When != 1, logits/preIdx live in compressed-token-index space:
                   ``N`` is divided by ``compress_ratio`` and ``preIdxOffset``
                   is forced to 0. Mirrors heuristicTopKDecode.cu PR #14219.
        max_seq_len: Graph-safe hint for peak ``logits.shape[1]`` at replay
                   (same compressed-token-index space as ``logits``).
        seqlen_sorted: When True, the kernel uses ``order_row`` (an LJF
                   request-level dispatch order) to resolve which row a
                   given CTA processes, so longer rows land in earlier
                   waves. Use together with :func:`gvr_topk_sort_prepare`.
                   Compatible with ``cluster_size > 1``.
        p4_warp_redundant: Default True. Phase 4 redundant-warp cadence:
                   every warp replays the k-th bin search reduce and the
                   snap-loop decision from the staged SMEM partials
                   (bit-identical across warps), removing the publish
                   barriers and keeping threshold/convergence state in
                   registers. False restores the leader-thread cadence.
        p2_warp_redundant: Default True. Phase 2 redundant-warp secant
                   cadence (cluster_size == 1 only): one barrier per
                   round; every warp reduces the staged warp counts and
                   replays the classify + secant update in registers.
                   False restores the leader cadence.
        order_row: Required iff ``seqlen_sorted=True``. Request-level -
                   ``int32[batch_size = num_rows // next_n]`` on the same
                   device as ``logits``; ``order_row[i]`` is the original
                   request_id of the i-th-priority request. The kernel
                   expands to row level via
                   ``order_row[req] * next_n + nn``.

    Returns:
        ``(out_values, out_indices)`` both shaped ``[num_rows, top_k]``.
    """
    assert logits.is_cuda, "logits must be on CUDA"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.shape}"
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    if seqlen_sorted:
        # order_row is request-level (length = seq_lens.shape[0] =
        # num_rows // next_n), NOT row-level.
        assert (
            order_row is not None
            and order_row.dtype == torch.int32
            and order_row.is_cuda
            and order_row.shape == seq_lens.shape
        ), (
            "seqlen_sorted=True requires order_row: int32[batch_size] on CUDA"
            f" (expected shape {tuple(seq_lens.shape)}, got "
            f"{tuple(order_row.shape) if order_row is not None else None})"
        )

    if logits.dtype not in _DTYPE_TORCH_TO_CUTE:
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]

    num_rows = logits.shape[0]
    # Host dispatch gate: below skip_min_n (compressed-index space,
    # shape-based so no device sync) drop block_max; None disables the gate.
    if block_max is not None and skip_min_n is not None and logits.shape[1] < skip_min_n:
        block_max = None
    # K > 512 at tiny batch: drop block_max (stock path) unless the admitted
    # line is known up front (ext counts / packed seed / self_scan).
    if (
        block_max is not None
        and num_rows < 8
        and top_k > 512
        and not self_scan
        and not (seed_thr is not None and seed_counts is not None)
        and not (seed_thr is not None and seed_thr.shape[1] >= 6)
    ):
        block_max = None
    if self_scan:
        # fused self-contained mode: kernel scans/buckets the row itself.
        # Inputs: seed_thr (three closed-loop lines) + a write-only gmem
        # POSITION column passed through the cand_idx slot; seed_counts is
        # a dummy (zeros never pass the [K, kC] admission).
        assert seed_thr is not None, "self_scan requires seed_thr"
        assert cand_vals is None and cand_ctl is None, (
            "self_scan excludes external candidate values/control"
        )
        assert "GVR_BSTAR" in os.environ, (
            "self_scan requires GVR_BSTAR (accept_cap) to size the position column"
        )
        if seed_counts is None:
            seed_counts = torch.zeros((num_rows, 3), dtype=torch.int32, device=logits.device)
        _bstar = int(os.environ["GVR_BSTAR"])
        _capc = cap_c if cap_c is not None else int(os.environ.get("GVR_CAPC", "16384"))
        _segtot = 2 * _bstar + _capc
        if cand_idx is None:
            cand_idx = torch.empty((num_rows, _segtot), dtype=torch.int32, device=logits.device)
        assert (
            cand_idx.dtype == torch.int32
            and cand_idx.is_cuda
            and cand_idx.is_contiguous()
            and cand_idx.shape == (num_rows, _segtot)
        ), f"self_scan position column must be int32 [num_rows, {_segtot}]"
    # packed seed row ([rows, >=6] fp32: lines + counts-as-floats) is the
    # native ext-counts input; separate seed_counts is the compat path and
    # pays a per-call pack build - pre-pack with pack_seed() instead.
    pre_packed = seed_thr is not None and seed_thr.shape[1] >= 6
    use_ext_counts = seed_thr is not None and (seed_counts is not None or pre_packed)
    # variant B (two-pass): thresholds without counts -> the kernel counts
    # the rungs itself (stock R0 multi-count) and admits in-kernel
    ext_rungs = seed_thr is not None and seed_counts is None and not pre_packed
    if use_ext_counts:
        assert (
            seed_thr.dtype == torch.float32
            and seed_thr.is_cuda
            and seed_thr.is_contiguous()
            and seed_thr.shape[0] == num_rows
            and (pre_packed or seed_thr.shape[1] == 3)
        ), "seed_thr must be contiguous CUDA fp32 [num_rows, 3|8]"
        assert pre_packed or (
            seed_counts.dtype == torch.int32
            and seed_counts.is_cuda
            and seed_counts.is_contiguous()
            and seed_counts.shape == (num_rows, 3)
        ), "seed_counts must be contiguous CUDA int32 [num_rows, 3]"
    use_ext_cand = cand_vals is not None and cand_idx is not None and cand_ctl is not None
    cand_cap = 5120
    if self_scan:
        cand_cap = cand_idx.shape[1]
    if use_ext_cand:
        assert (
            cand_vals.dtype == torch.float32
            and cand_vals.is_cuda
            and cand_vals.is_contiguous()
            and cand_vals.dim() == 2
            and cand_vals.shape[0] == num_rows
        ), "cand_vals must be contiguous CUDA fp32 [num_rows, CAP]"
        assert (
            cand_idx.dtype == torch.int32
            and cand_idx.is_cuda
            and cand_idx.is_contiguous()
            and cand_idx.shape == cand_vals.shape
        ), "cand_idx must be contiguous CUDA int32 [num_rows, CAP]"
        assert (
            cand_ctl.dtype == torch.int32
            and cand_ctl.is_cuda
            and cand_ctl.is_contiguous()
            and cand_ctl.shape == (num_rows, 4)
        ), "cand_ctl must be contiguous CUDA int32 [num_rows, 4]"
        cand_cap = cand_vals.shape[1]
    emit_xstate = xstate is not None
    if emit_xstate:
        assert (
            xstate.dtype == torch.float32
            and xstate.is_cuda
            and xstate.is_contiguous()
            and xstate.shape == (num_rows, 8)
        ), "xstate must be contiguous CUDA fp32 [num_rows, 8]"
    enable_block_skip = block_max is not None
    if enable_block_skip:
        assert (
            block_max.dtype == torch.float32
            and block_max.is_cuda
            and block_max.is_contiguous()
            and block_max.dim() == 2
            and block_max.shape[0] == num_rows
            and block_max.shape[1] % 4 == 0
            and block_max.shape[1] >= (logits.shape[1] + 31) // 32
        ), "block_max must be contiguous CUDA fp32 [num_rows, nb_pad*4] covering the row"

    if return_output_values:
        if out_values is None:
            out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)

    # Resolve None defaults via the same heuristic as the production Runner.
    if enable_unroll_4 is None:
        enable_unroll_4 = True
    if enable_phase3_unroll is None:
        enable_phase3_unroll = True

    N_cols = logits.shape[1]
    N_dec = max_seq_len if max_seq_len is not None else N_cols
    if num_threads_per_block is None:
        if use_ext_cand and top_k <= 512:
            # list-hit rows do O(list) work, not O(N): use 512 threads
            # (K=1024 lists are big enough to keep the N-keyed pick).
            num_threads_per_block = 512
        elif self_scan:
            # self_scan scans the whole row in one CTA; the phase-0
            # cp.async pipeline scales with warp count at every N.
            num_threads_per_block = 1024
        else:
            if max_seq_len is not None and logits.dtype != torch.float32:
                n_thresh_t = 131072
            else:
                n_thresh_t = 65536
            num_threads_per_block = 1024 if (num_rows <= num_sms and N_dec >= n_thresh_t) else 512
    if use_256bit_load is None:
        use_256bit_load = logits.dtype == torch.float32 and N_dec >= 16384
    if enable_warp_parallel_reduce is None:
        enable_warp_parallel_reduce = num_threads_per_block == 1024

    if min_blocks_per_mp is None:
        vec_bits_host = 256 if use_256bit_load else 128
        vec_w_host = vec_bits_host // (32 if logits.dtype == torch.float32 else 16)
        n_vec_iters = max(1, N_dec // (num_threads_per_block * vec_w_host))
        is_fp32 = logits.dtype == torch.float32
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and N_dec <= 32768:
                # Wave-fit + latency-bound; at N>=65K mb=2 wins (bandwidth-bound).
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

    seed_pack = None
    if use_ext_counts:
        if pre_packed:
            seed_pack = seed_thr
        else:
            seed_pack = pack_seed(seed_thr, seed_counts)

    compiled = _compile(
        cute_dtype,
        top_k,
        next_n,
        enable_unroll_4,
        enable_phase3_unroll,
        use_constant_hint,
        min_blocks_per_mp,
        use_256bit_load,
        num_threads_per_block,
        enable_warp_parallel_reduce,
        compress_ratio,
        return_output_values,
        cluster_size,
        seqlen_sorted,
        p4_warp_redundant,
        p2_warp_redundant,
        enable_block_skip,
        pdl_wait_late,
        p4_tail_v3,
        p4_no_fine,
        p4_exact_tail,
        p4_tail_fast,
        p1r_rescue,
        num_bins,
        p4_fine_rangetest,
        p4_scat_rangetest,
        use_ext_counts,
        emit_xstate,
        use_ext_cand,
        ext_rungs,
        cand_cap,
        int(os.environ["GVR_BSTAR"]) if "GVR_BSTAR" in os.environ else None,
        int(os.environ["GVR_KC"]) if "GVR_KC" in os.environ else None,
        self_scan,
        cap_c
        if cap_c is not None
        else (int(os.environ.get("GVR_CAPC", "16384")) if self_scan else None),
    )
    # When return_output_values=False the kernel was compiled to skip
    # STG.value and accepts None for the value-output slot.
    # When seqlen_sorted=False the const_expr branch elides the order_row
    # read so the kernel accepts None for that slot as well.
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values if return_output_values else None,
        out_indices,
        order_row if seqlen_sorted else None,
        block_max if enable_block_skip else None,
        seed_pack if use_ext_counts else (seed_thr if ext_rungs else None),
        None,
        xstate if emit_xstate else None,
        cand_vals if use_ext_cand else None,
        cand_idx if (use_ext_cand or self_scan) else None,
        cand_ctl if use_ext_cand else None,
    )
    if return_output_values:
        return out_values, out_indices
    else:
        return None, out_indices


def gvr_topk_sort_prepare(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build the LJF dispatch order for :func:`gvr_topk_decode`.

    Returns ``int32[batch_size]`` (= ``seq_lens.shape[0]`` =
    ``num_rows // next_n``) whose i-th entry is the original-batch index
    of the i-th longest request - request-level, NOT row-level. The
    kernel expands to row level via ``order_row[req] * next_n + nn``
    inside the const_expr ``seqlen_sorted`` branch. Run once per decode
    step; the same
    ``order_row`` is reused across all per-layer ``gvr_topk_decode``
    calls with ``seqlen_sorted=True`` (seq_lens is layer-invariant
    within a decode step). For an LB-style two-bucket partition, use
    :func:`gvr_topk_lb_prepare` instead.
    """
    assert seq_lens.is_cuda and seq_lens.dim() == 1 and seq_lens.dtype == torch.int32
    return torch.argsort(seq_lens, descending=True, stable=False).to(torch.int32)


# ---- Load-Balance (hybrid multi-CTA + single-CTA) wrappers ------------------
try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_load_balance import (
        GvrTopKLBKernel,
        GvrTopKLBPrepareKernel,
    )
except (ModuleNotFoundError, ImportError):
    from blackwell.top_k.gvr_topk_decode_load_balance import (  # type: ignore[no-redef]
        GvrTopKLBKernel,
        GvrTopKLBPrepareKernel,
    )


@functools.cache
def _compile_lb_prepare(
    num_threads: int, batch_size: int, long_threshold: int, compress_ratio: int
):
    """JIT-compile the LB prepare kernel for a specific tuple.

    Specialized over ``(num_threads, batch_size, threshold,
    compress_ratio)``.

    ``num_threads`` = kernel block size + ``order_row`` length.
    ``batch_size``  = compile-time seq_lens shape (must equal runtime
                      ``seq_lens.shape[0]`` for TVM-FFI marshalling).
    ``compress_ratio`` = KV-indexer compression factor; the classifier
                      divides ``seq_lens`` by this before comparing
                      against ``long_threshold`` so the threshold stays
                      in scan-length (post-compress) space.
    """
    prep = GvrTopKLBPrepareKernel(
        long_threshold=long_threshold,
        compress_ratio=compress_ratio,
        num_threads=num_threads,
    )
    fake_seq = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch_size,), stride_order=(0,)
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_threads,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        prep,
        fake_seq,
        fake_order,
        fake_ctr,
        cutlass.Int32(0),
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def gvr_topk_lb_prepare(
    seq_lens: torch.Tensor,
    max_batch_size: int = 1024,
    long_threshold: int = 64 * 1024,
    compress_ratio: int = 1,
    order_row: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the LB prepare kernel.

    ``seq_lens`` keeps its actual shape ``(batch_size,)`` - the kernel
    is compiled to match that exact shape; ``max_batch_size`` only
    determines the prepare kernel's block size and the ``order_row``
    buffer length. ``long_threshold`` is in SCAN-LENGTH space
    (= seq_lens / compress_ratio), matching the GVR kernel's actual
    work; the classifier inside the prepare kernel divides by
    ``compress_ratio`` automatically so callers can pass raw
    ``seq_lens`` regardless of cr.

    Returns ``(order_row, counters)`` which the caller feeds into
    :func:`gvr_topk_lb_decode` for every per-layer Top-K call within
    the same decode step.
    """
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    # block_prefix_sum_kernel (used inside LB prepare) constraints:
    # num_warps = max_batch_size / 32 must be > 1 and a power of 2 ->
    # max_batch_size in {64, 128, 256, 512, 1024}.
    if not (64 <= max_batch_size <= 1024) or (max_batch_size & (max_batch_size - 1)) != 0:
        raise ValueError(
            f"max_batch_size must be a power of 2 in [64, 1024] "
            f"(block_prefix_sum_kernel constraint); got {max_batch_size}"
        )
    batch_size = seq_lens.shape[0]
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= max_batch_size "
            f"({max_batch_size}): the LB prepare kernel hard-wires a block "
            f"of max_batch_size threads and order_row[max_batch_size]; "
            f"tail requests beyond that would never be classified into the "
            f"long/short partition and order_row[batch_size:] would be -1, "
            f"so the decode path's order_row lookup would return invalid "
            f"row indices."
        )
    if order_row is None:
        order_row = torch.full((max_batch_size,), -1, dtype=torch.int32, device=seq_lens.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=seq_lens.device)

    compiled = _compile_lb_prepare(max_batch_size, batch_size, long_threshold, compress_ratio)
    compiled(seq_lens, order_row, counters, cutlass.Int32(batch_size))
    return order_row, counters


@functools.cache
def _compile_lb(
    cute_dtype,
    top_k: int,
    next_n: int,
    num_rows: int,
    N: int,
    compress_ratio: int,
    max_batch_size: int,
    num_threads: int,
    cluster_size: int,
    return_output_values: bool,
):
    """JIT-compile the LB main kernel.

    ``num_rows`` baked in via fake tensors; ``seq_lens`` fake shape
    uses ``n_groups = num_rows // next_n`` so the caller can feed the
    actual seq_lens without padding. ``max_batch_size`` drives the
    grid (``* next_n * cluster_size`` CTAs) for CUDA Graph
    compatibility.
    """
    kernel = GvrTopKLBKernel(
        dtype=cute_dtype,
        top_k=top_k,
        next_n=next_n,
        num_threads=num_threads,
        compress_ratio=compress_ratio,
        return_output_values=return_output_values,
        cluster_size=cluster_size,
        max_batch_size=max_batch_size,
    )
    n_groups = num_rows // next_n
    fake_logits = cute.runtime.make_fake_compact_tensor(
        cute_dtype, (num_rows, N), stride_order=(1, 0), assumed_align=16
    )
    fake_pre_idx = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (n_groups, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_seq = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n_groups,), stride_order=(0,))
    fake_out_v = (
        cute.runtime.make_fake_compact_tensor(
            cute_dtype, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
        )
        if return_output_values
        else None
    )
    fake_out_i = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, top_k), stride_order=(1, 0), assumed_align=16
    )
    fake_order = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (max_batch_size,), stride_order=(0,)
    )
    fake_ctr = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (2,), stride_order=(0,))
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        fake_logits,
        fake_pre_idx,
        fake_seq,
        fake_out_v,
        fake_out_i,
        fake_order,
        fake_ctr,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def gvr_topk_lb_decode(
    logits: torch.Tensor,
    pre_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    order_row: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
    cluster_size: int = 4,
    max_batch_size: int = 1024,
    num_threads: int = 512,
    return_output_values: bool = False,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run the LB (hybrid multi-CTA + single-CTA) main kernel.

    ``order_row`` and ``counters`` MUST already be populated by a prior
    call to :func:`gvr_topk_lb_prepare` for the current ``seq_lens``
    (the metadata is invariant across per-layer Top-K calls within one
    decode step, so callers run prepare once and reuse).
    """
    assert logits.is_cuda and logits.dim() == 2
    assert pre_idx.dim() == 2 and pre_idx.dtype == torch.int32
    assert seq_lens.dim() == 1 and seq_lens.dtype == torch.int32

    if logits.dtype not in _DTYPE_TORCH_TO_CUTE:
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    cute_dtype = _DTYPE_TORCH_TO_CUTE[logits.dtype]

    num_rows = logits.shape[0]
    N = logits.shape[1]
    if out_indices is None:
        out_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=logits.device)
    if return_output_values and out_values is None:
        out_values = torch.empty((num_rows, top_k), dtype=logits.dtype, device=logits.device)
    if not return_output_values:
        out_values = None  # passed to kernel as ``None``

    compiled = _compile_lb(
        cute_dtype,
        top_k,
        next_n,
        num_rows,
        N,
        compress_ratio,
        max_batch_size,
        num_threads,
        cluster_size,
        return_output_values,
    )
    compiled(
        logits,
        pre_idx,
        seq_lens,
        out_values,
        out_indices,
        order_row,
        counters,
    )
    return out_values, out_indices


# ---- Correctness helpers ----------------------------------------------------
def _make_inputs(
    num_rows: int,
    N: int,
    top_k: int,
    dtype: torch.dtype,
    seed: int,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (logits, pre_idx, seq_lens) for a multi-row test.

    Shapes:
      logits  : [num_rows, N]                 - compressed-token-index space
      pre_idx : [num_rows // next_n, top_k]   - argmax in slot 0 (indexer invariant)
      seq_lens: [num_rows // next_n]          - UNCOMPRESSED-token space

    Kernel divides ``seq_lens`` by ``compress_ratio`` internally. Setting
    ``seq_lens = N * cr`` makes the kernel's
    ``N_kernel = (seq_lens - next_n + ofs + 1) // cr`` match the reference
    ``N_eff = N - next_n + ofs + 1`` for next_n in {1, 2} (covers the
    current sweep). For cr=1 this reduces to ``seq_lens = N``.

    ``pre_idx.shape[1] == top_k`` per CUDA invariant (heuristic_topk.cuh:810:
    ``preIdxCount == topK`` is a dispatch precondition).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    logits = logits_f32.to(dtype)
    num_groups = num_rows // next_n
    # argmax must come from the effective scan range, not full N - for
    # next_n>1 the kernel's row-0 N_eff is only (N - next_n + 1) cols.
    effective_len = N - next_n + 1
    argmax_idx = logits[::next_n, :effective_len].argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j
    # seq_lens is uncompressed; ``N * cr`` makes the kernel's
    # ``N_kernel = (seq_lens - next_n + ofs + 1) // cr`` match ref's
    # ``N_eff = N - next_n + ofs + 1`` for next_n in {1, 2}. (For cr=1
    # this reduces to seq_lens = N.)
    seq_lens_val = N * compress_ratio
    seq_lens = torch.full((num_groups,), seq_lens_val, dtype=torch.int32, device="cuda")
    return logits, pre_idx, seq_lens


def _tie_aware_correct(
    kernel_idxs: torch.Tensor,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int = 1,
) -> tuple[bool, str]:
    """Multi-row tie-aware correctness check with strict sort+allclose.

    Per row r: scan range mirrors the kernel formula (see
    ``GvrTopKKernel.gvr_topk_kernel``):

        actual_kv_len = seq_lens[r // next_n] - next_n + (r % next_n) + 1
        N_eff = actual_kv_len // compress_ratio   # cr=1 is identity

    Reference ``torch.topk`` is masked to this range so reference and
    kernel scan exactly the same columns under any (next_n, cr) combo.

    Returns ``(False, message)`` on the first failing row; ``(True, "ok")``
    when all rows pass. Sort+allclose catches the "drop-strictly-above +
    add-tied-at-kth" bug that count-below-kth alone misses on ties.
    """
    num_rows = kernel_idxs.shape[0]
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    for row in range(num_rows):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            # Degenerate path - skip; caller's main() guards against this.
            continue
        row_logits = logits_f32[row, :N_eff]
        topk_vals, _ = torch.topk(row_logits, k=top_k, largest=True, sorted=True)
        kth_value = topk_vals[-1].item()
        sel = [int(i) for i in kernel_idxs[row].cpu().tolist() if i >= 0]
        if any(i >= N_eff for i in sel):
            return False, f"row={row}: out-of-range index"
        if len(set(sel)) != len(sel):
            return False, f"row={row}: duplicate indices"
        if len(sel) != top_k:
            return False, f"row={row}: returned {len(sel)} indices, expected {top_k}"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        n_below = int((sel_vals < kth_value).sum().item())
        if n_below > 0:
            return False, (f"row={row}: {n_below} selected values < Kth-rank ({kth_value:.6f})")
        # Strict: sorted-value multiset must match torch.topk.
        sel_sorted, _ = sel_vals.sort(descending=True)
        if not torch.allclose(sel_sorted, topk_vals, rtol=1e-5, atol=1e-5):
            max_diff = (sel_sorted - topk_vals).abs().max().item()
            return False, f"row={row}: sorted-value mismatch (max diff {max_diff:.4e})"
    return True, "ok"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        # Production cells: (bf16, K=512/1024) and (fp32, K=2048) match
        # the deployed K -> dtype mapping. (fp16, K=1024) is added to keep
        # the fp16 convert-to-fp32 tail path under test even though it is
        # not a current production cell.
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 65536])
@pytest.mark.parametrize("next_n", [1])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("use_256bit_load", [False, True])
@pytest.mark.parametrize("num_threads_per_block", [512, 1024])
@pytest.mark.parametrize("enable_warp_parallel_reduce", [False, True])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_gvr_topk_decode(
    dtype: torch.dtype,
    top_k: int,
    N: int,
    next_n: int,
    batch_size: int,
    use_256bit_load: bool,
    num_threads_per_block: int,
    enable_warp_parallel_reduce: bool,
    cluster_size: int,
) -> None:
    # Kernel scans `N_eff = seq_lens[0] - next_n + (row_idx % next_n) + 1`
    # columns. Smallest row's N_eff = N - next_n + 1. Degenerate path
    # (N_eff <= top_k) is a separate code branch - skip here.
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k is degenerate; the kernel requires N_eff >= top_k")
    seed = 42
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        N,
        top_k,
        dtype,
        seed,
        next_n=next_n,
        compress_ratio=1,
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _, out_idxs = gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        next_n=next_n,
        num_sms=num_sms,
        use_256bit_load=use_256bit_load,
        num_threads_per_block=num_threads_per_block,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        return_output_values=False,
        cluster_size=cluster_size,
    )
    torch.cuda.synchronize()
    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, (
        f"dtype={dtype} K={top_k} N={N} seed={seed} next_n={next_n} "
        f"batch_size={batch_size} use_256bit_load={use_256bit_load} "
        f"num_threads_per_block={num_threads_per_block} "
        f"enable_warp_parallel_reduce={enable_warp_parallel_reduce} "
        f"cluster_size={cluster_size}: {msg}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("cluster_size", [1, 4])
@pytest.mark.parametrize(
    "p4_warp_redundant,p2_warp_redundant",
    [
        # Leader-path coverage: the redundant-warp knobs default ON, so
        # the main sweep above exercises the redundant cadences; these
        # combinations keep the knob-off (pre-redundant leader) paths and
        # the two mixed configurations compiling and exact.
        (False, False),
        (False, True),
        (True, False),
    ],
)
def test_gvr_topk_decode_leader_paths(
    dtype: torch.dtype,
    top_k: int,
    cluster_size: int,
    p4_warp_redundant: bool,
    p2_warp_redundant: bool,
) -> None:
    N = 65536
    batch_size = 32
    seed = 42
    logits, pre_idx, seq_lens = _make_inputs(
        batch_size,
        N,
        top_k,
        dtype,
        seed,
        next_n=1,
        compress_ratio=1,
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _, out_idxs = gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        next_n=1,
        num_sms=num_sms,
        return_output_values=False,
        cluster_size=cluster_size,
        p4_warp_redundant=p4_warp_redundant,
        p2_warp_redundant=p2_warp_redundant,
    )
    torch.cuda.synchronize()
    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, 1)
    assert ok, (
        f"dtype={dtype} K={top_k} cluster_size={cluster_size} "
        f"p4_warp_redundant={p4_warp_redundant} "
        f"p2_warp_redundant={p2_warp_redundant}: {msg}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--top_k", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--next_n", type=int, default=1)
    p.add_argument("--num_sms", type=int, default=148)
    p.add_argument("--compress_ratio", type=int, default=1, choices=[1, 4])
    p.add_argument("--num_threads", type=int, default=512)
    p.add_argument("--use_256bit_load", action="store_true")
    p.add_argument("--min_blocks_per_mp", type=int, default=None)
    p.add_argument("--enable_warp_parallel_reduce", action="store_true")
    p.add_argument("--disable_unroll_4", action="store_true")
    p.add_argument("--disable_phase3_unroll", action="store_true")
    p.add_argument("--use_constant_hint", action="store_true")
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument(
        "--cluster_size",
        type=int,
        default=1,
        help="CTAs per row (1=V5 single-CTA, 2/4=DSMEM cluster).",
    )
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    effective_len = args.N - args.next_n + 1
    if effective_len < args.top_k:
        print(f"FAIL: N_eff={effective_len} < top_k={args.top_k} (degenerate path)")
        sys.exit(1)

    seed = 42
    num_rows = args.batch_size * args.next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows,
        args.N,
        args.top_k,
        dtype,
        seed,
        next_n=args.next_n,
        compress_ratio=args.compress_ratio,
    )
    knobs = dict(
        num_threads_per_block=args.num_threads,
        use_256bit_load=args.use_256bit_load,
        min_blocks_per_mp=args.min_blocks_per_mp,
        enable_warp_parallel_reduce=args.enable_warp_parallel_reduce,
        enable_unroll_4=not args.disable_unroll_4,
        enable_phase3_unroll=not args.disable_phase3_unroll,
        use_constant_hint=args.use_constant_hint,
        compress_ratio=args.compress_ratio,
        max_seq_len=args.max_seq_len,
        return_output_values=False,
        cluster_size=args.cluster_size,
    )
    print(
        f"config: dtype={args.dtype} top_k={args.top_k} N={args.N} "
        f"batch_size={args.batch_size} next_n={args.next_n}, num_sms={args.num_sms}"
    )
    print(f"knobs: {knobs}")

    _, out_idxs = gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        args.top_k,
        next_n=args.next_n,
        num_sms=args.num_sms,
        **knobs,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(
        out_idxs,
        logits,
        seq_lens,
        args.top_k,
        args.next_n,
        compress_ratio=args.compress_ratio,
    )
    print(f"correctness: {'PASS' if ok else f'FAIL ({msg})'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
