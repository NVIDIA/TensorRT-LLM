# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Align the pure-PyTorch module references against the checkpoint's own forward.

This closes rung 2 of the reference ladder. ``refmods.py`` implements every
module of DeepSeek-V4.1-Flash from the checkpoint's semantics; this file drives
the checkpoint's *own* ``inference/`` forward, observes what each module
actually computed, and checks the pure-PyTorch implementation against it on the
same inputs and the same weights. A module is only a trusted reference after it
aligns here, and every later module Goal's parity condition cites that.

Two modes, cheapest first:

``--mode small``
    A randomly-initialized model with the released *structural* constants
    (hc_mult 4, 20 Sinkhorn iterations, sqrtsoftplus routing, clamped SwiGLU,
    ratios 0/1/2, a candidate hierarchy, two Engram layers, sinks) at toy
    widths, on one GPU, no distribution. Seconds to minutes. It exercises
    regimes the released prompts cannot reach -- a sliding window that actually
    wraps during decode, a top-k that actually binds, candidate blocks that
    actually filter -- so a semantic error surfaces here rather than an hour
    later.

``--mode full``
    The real checkpoint at model-parallel 4 on the five frozen greedy fixtures.
    Before anything is compared, the instrumented run must still reproduce each
    fixture's continuation in full -- all 129 tokens of 7 / 24 / 48 / 2 / 48,
    with the lengths recorded next to the verdict. That is what proves the
    recording wrappers changed nothing, and without it an aligned module would
    only mean the harness agrees with itself. A prefix will not do: the early
    tokens are the ones least likely to diverge.

Every comparison also drives *controls* -- deliberately wrong variants of the
same computation. A tolerance is only meaningful next to the distance at which
a wrong answer lands, so the report carries both. A control that fails to
discriminate has to come with a measured statement of why the input cannot
reach the behaviour; without one it is a blind harness and the run fails.

Three things are checked before any boundary result is believed:

  * the comparator itself, against a defect it must see at this checkpoint's
    integer magnitudes (:func:`comparator_selftest`);
  * the set of boundaries and controls the run owes
    (:data:`_REQUIRED_BOUNDARIES`, :data:`_REQUIRED_CONTROLS`), so a reference
    that is never driven fails the command instead of being absent from a
    green report;
  * in ``--mode full``, that the instrumented model still reproduces every
    frozen fixture in full, with the checked and frozen lengths recorded.

The reference leg never imports ``tensorrt_llm``; it runs under the checkpoint's
own pinned environment.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refmods as R  # noqa: E402  # ty: ignore[unresolved-import]  (after the sys.path fix-up)

# ---------------------------------------------------------------------------
# comparison bookkeeping
# ---------------------------------------------------------------------------

#: ``torch.testing.assert_close``'s dtype defaults, which is where every
#: floating gate starts. Anything looser has to be earned with a measured
#: correct-result error and controls that clearly fail.
_DTYPE_TOL = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float16: (1e-3, 1e-5),
}

#: THE ONE LOOSENING, and the measurement that earned it.
#:
#: Three boundaries -- ``attn.sparse_attn``, ``engram.module``, ``moe.module``
#: -- are sums whose terms cancel: a routed-expert sum of six weighted
#: contributions plus a shared one, an attention output over gathered rows, a
#: gated residual. Where they nearly cancel the *result* is near zero while the
#: *terms* are order one, so an elementwise ``atol + rtol * |w|`` test asks the
#: implementation to reproduce a cancellation to a precision neither side
#: carries. They miss it while their absolute distance is a fraction of one
#: bf16 step at the tensor's own magnitude.
#:
#: So the gate gains a second, scale-relative term: a difference no larger than
#: half a step at the tensor's own magnitude also passes.
#:
#: PROVENANCE, and every number below is a multiple of the row's own gate as the
#: report prints it. Measured on the real checkpoint at mp4, job 746617, and
#: independently reproduced to the digit by job 746594 on all four ranks, which
#: is why these are quoted rather than re-derived. Across the boundaries held to
#: this floor the worst correct result is 0.977 (``block.ffn_input``), then
#: 0.653 (``linear.fp4``, ``moe.expert``); the nearest wrong one is 2.06
#: (``norm.rmsnorm/bf16-statistic``, accumulating the norm statistic in bf16
#: instead of fp32), then 4.90 (``mean-centered``) and 28.28
#: (``hc.post/transposed-comb``). At toy widths the ordering is tighter and the
#: pair is different -- worst correct 0.518 (``attn.sparse_attn``), nearest
#: wrong 1.90 (``moe.module/no-shared``, job 746631) -- which is the reason both
#: regimes run: neither one alone bounds this floor.
#:
#: 2.06x is the tightest separation in the whole ladder and it is deliberate:
#: `bf16-statistic` is a *precision* control, the smallest real defect this
#: gate is asked to see, so it sits closer than any semantic one. A floor that
#: cleared it by more would not be measuring precision any more.
#:
#: fp32 boundaries get a far tighter floor: they measure 0.02 of it while their
#: nearest control is 9460x (``hc.split_sinkhorn.comb/transposed-comb``), so
#: 2**-16 keeps orders of headroom on both sides.
_CANCELLATION_FLOOR = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-10,
    torch.float32: 2.0**-16,
    torch.float64: 2.0**-40,
}


#: Boundaries held to ONE representable step at the tensor's scale instead of a
#: half step, because they chain several rounded GEMM-class results rather than
#: checking one.
#:
#: PROVENANCE. Measured on the real checkpoint at mp4, job 746617, and
#: independently reproduced to the digit by job 746594 on all four ranks (both
#: sides are post-all-reduce, so every rank compares the same values). Numbers
#: below are multiples of each row's own gate, as the report prints them.
#:
#: What motivated the widening: against the half-step floor these four measured
#: 0.859 at the grouped output LoRA and 0.547 / 0.762 / 1.169 at the three
#: attention-ratio modules, so ratio 1 sat just outside. Every single-op link
#: feeding them was far inside on the same run -- `linear.fp8` 0.131,
#: `attn.sparse_attn` 0.181, both rope boundaries exactly 0.000 -- and the
#: ordering of the three ratios is the explanation: ratio 1 publishes one
#: compressed row per token, so its gathered set is 128 window + 512 compressed
#: rows against ratio 2's 128 + 250, which is ten tiles of online softmax
#: rescaling instead of six, followed by two more bf16 GEMMs. Four chained
#: roundings landing inside one representable step is the dtype agreeing with
#: itself, not a semantic gap.
#:
#: The separation that makes this a gate rather than a concession, against the
#: widened floor these rows now carry: worst correct result 0.584
#: (`attn.module.r1`), then 0.430 and 0.381 and 0.274. Nearest wrong result
#: 9.61 (`attn.module.r2/flat-q-norm-weight`), then 11.66, 16.99, and
#: 42.80-201.09 for the missing inverse rotation and the rank-major LoRA
#: groups. So the bound sits 1.71x above the worst correct result and 9.61x
#: below the nearest wrong one.
#:
#: WHY THIS LIST EXISTS AT ALL, stated as a rule so the next addition is not a
#: guess. `rel_scale` is `max_abs / absmax`: the worst single disagreement,
#: normalized by the tensor's peak. Two implementations that accumulate the
#: same reduction in different fp32 orders differ by about 1e-6 relative, which
#: crosses a bf16 rounding boundary roughly once in 4,000 elements. When one of
#: those elements happens to sit at the tensor's peak, `rel_scale` reads one
#: full representable step -- exactly 2.0 half-steps. So for any bf16 boundary
#: whose two sides reduce in different orders, reaching 2.0 is a matter of how
#: many elements were compared, not of correctness, and a half-step gate is
#: satisfiable only by luck of sample size.
#:
#: That is not hypothetical either. Lengthening the frozen-fixture check from a
#: 45-step prefix to the full 124 decode steps took `linear.fp4` from 108 calls
#: to 387 and from 0.653 to 1.133, on rank 0 alone, with `over_elementwise`
#: still 0 -- every element inside the dtype's own default tolerance, one of
#: them one step away at the peak.
#:
#: A boundary joins this list when BOTH hold, and each entry below records its
#: own numbers rather than inheriting the rule's:
#:   * its reference side reduces in a different order from the kernel, so
#:     one-step disagreements are expected rather than diagnostic; and
#:   * its measured separation stays wide at the doubled gate.
#:
#: Measured in two steps, and the normalization is stated for each because
#: mixing them is how a stale margin gets quoted. Every number is a multiple of
#: the floor in force for that row AT THAT TIME.
#:
#: What put the last three on this list -- against the HALF-STEP floor they
#: carried then, full mode job 746735, worst of four ranks:
#:   linear.fp4        1.133   over_elementwise 0 (rank 0 only, 387 calls)
#:   block.ffn_input   0.977
#:   attn.sparse_attn  0.941
#:
#: Where all seven sit now -- against the DOUBLED gate they carry today, full
#: mode job 746795, worst of four ranks:
#:   attn.module.r0/r1/r2  0.381 / 0.584 / 0.274  nearest wrong  9.61
#:   attn.output_lora.a    0.481                  nearest wrong 211.04
#:   linear.fp4            0.566                  nearest wrong 237.47
#:   attn.sparse_attn      0.471                  nearest wrong 128.00
#:   block.ffn_input       0.489                  nearest wrong 250.03
#: So each gate sits 1.7x-3.6x above its own worst correct result and 9.61x-250x
#: below its own nearest wrong one. The 9.61x belongs to the attention modules
#: and is the tightest of the seven; it is a precision control (`flat-q-norm-
#: weight`), not a semantic one, which is why it sits closest.
#:
#: WHAT STAYS AT THE HALF STEP, and why it matters. `moe.module` is a
#: composition too and would qualify on the first clause, but its `no-shared`
#: control -- dropping an entire expert -- measures 1.896 at toy widths against
#: a correct result of 0.219. Doubling its gate to 2.0 would make the ladder's
#: single tightest control blind, so it stays where it is. That control is the
#: one place in this ladder where the statistic cannot cleanly separate a real
#: defect from a one-step rounding, and it is only 1.05x under the ceiling; its
#: real discrimination is the 339x it measures at full scale.
#:
#: `block.module` (0.692) and `moe.expert` (0.653) are the next closest to the
#: half-step line, and are the whole ladder's worst correct results as of job
#: 746795. They are named here so that a future trip reads as this same
#: mechanism and is checked against these numbers -- and against the `coherent`
#: column, which separates the two cases directly: one element rounding reads
#: near zero there (`linear.fp4` measured 0.0002 while tripping at 1.133),
#: where a real defect does not.
#: THE EIGHTH AND NINTH ENTRIES, added by Goal 1.2's rung-3 parity driver. Both
#: are TIGHTER than anything above them -- named as such so they are checked
#: rather than assumed. Neither widens a boundary this ladder itself measures:
#: `linear.fp8` keeps its half step, and its `reference-vs-native` rows still
#: sit at 0.131 there. The two names below exist precisely so the widening
#: applies only to comparisons that span the TARGET's own kernel, which Goal 1.1
#: never ran.
#:
#: `linear.fp8.target` covers `target-vs-reference` and `target-vs-native` for
#: the replicated and column-parallel FP8 projections, plus their rolled-scale
#: control. Its two sides are two different fp32 reductions -- the CUTLASS
#: MXFP8 GEMM against `refmods`' blocked scaled matmul, and against the
#: reference implementation's own `fp8_gemm` -- so this is the ordinary
#: different-order case the rule below was written for.
#:
#: PROVENANCE: job 749240's captures replayed across five pinned prompts and 44
#: FP8 projections. Exactly one row reaches the half-step line --
#: `layers.39.attn.wq_a` on three of five prompts -- and everything about it is
#: the signature this file already names:
#:   ULP@scale 1.83, coherent 0.0003, over_elementwise 0
#:   max_abs 1.953e-03, which is EXACTLY one bf16 ulp at that tensor's
#:   2.734e-01 peak -- one element at the peak rounding one step
#: The precedent is recorded above: `linear.fp4` read coherent 0.0002 while
#: tripping at 1.133. A semantic gap does not read 0.0003 there.
#:
#: Two further facts make this a rounding artefact rather than a tolerance
#: problem. `over_elementwise 0` means every element is inside the dtype's own
#: default tolerance, so the `mxfp8_mxfp8_gemm` entry's certified elementwise
#: bound is satisfied -- only this ladder's EXTRA tensor-level `rel_scale` term
#: trips, and that term was added to catch a dropped expert, not one ulp. And
#: layer 39 is the one FP8 projection whose output peak is unusually small
#: (2.734e-01 against 0.4-55 elsewhere), which is exactly when a peak-normalized
#: statistic is most sensitive to a single element.
#:
#: Separation at the DOUBLED gate, over all five prompts: worst correct 0.914;
#: nearest wrong 27.2 (`control:rolled-weight-scale` at its tightest width,
#: w1280), ranging to 174.6 at w4096. So 1.09x above the worst correct result
#: and 27.2x below the nearest wrong one. `linear.fp8` itself, which keeps the
#: half step, measures 0.631 over the same run -- it did not need widening and
#: did not get it.
#:
#: THE EIGHTH ENTRY.
#:
#: `linear.fp8.tp_reduced` is the only boundary where the two sides do not
#: merely reduce in a different ORDER but in a different STRUCTURE.
#: `RowParallelLinear.forward` computes `y = linear(x_r, w_r)` per rank, which
#: returns BF16, and only then does `.float()` + `all_reduce` -- so the native
#: value is `bf16(sum of four bf16-rounded partials)`. The staircase target
#: replicates `wo_b` and performs one fp32 reduction over the full K=8192 with a
#: single rounding at the end. Four intermediate roundings of a half ulp each
#: bound the difference at two representable steps; the doubled gate is one, and
#: the measurement below is what says the smaller bound is the right one here.
#:
#: PROVENANCE: full-mode captures from job 749240, replayed by `parity_dense.py`
#: across five pinned prompts and eleven `wo_b` layers. Against the DOUBLED gate
#: this row carries:
#:   worst correct result   0.955  (layer 39; 7.463e-03 against a 7.812e-03 gate)
#:   nearest wrong result 235.4    (`control:reversed-rank-order`, 1.839)
#:
#: THE TARGET CONTRIBUTES NONE OF THAT 0.914, and the driver measures it rather
#: than asserting it: `reference-vs-native:fullwidth` runs the identical
#: comparison with the target absent and reads the SAME value to the digit on
#: every one of the eleven layers, while `target-vs-reference` at the same full
#: width reads 7.1e-04 -- 0.18 of the HALF-step floor. So the gap is the
#: topology's rounding structure and not the port's arithmetic.
#:
#: NOTED RISK, and it is the sharpest in this file: 1.05x of headroom, against
#: 1.7x-3.6x for the seven entries above and 1.09x for `linear.fp8.target`. Only
#: `moe.module`'s toy-width control sits as close, and this file already names
#: that as its tightest case. The rule above says plainly that `rel_scale` grows
#: with how many elements were compared, so a sixth prompt or a twelfth layer
#: could cross it. The derivation says where the real ceiling is: four
#: intermediate roundings of a half ulp bound the difference at TWO
#: representable steps, which is twice this gate. If a later run trips this row,
#: the first thing to check is `reference-vs-native:fullwidth` on the same run:
#: if it moved by the same amount, the topology is still the whole explanation
#: and the bound -- not the target -- is what needs re-deriving against the
#: two-representable-step ceiling above.
_MULTI_STEP_BOUNDARIES = frozenset(
    {
        "attn.module.r0",
        "attn.module.r1",
        "attn.module.r2",
        "attn.output_lora.a",
        "linear.fp4",
        "attn.sparse_attn",
        "block.ffn_input",
        "linear.fp8.target",
        "linear.fp8.tp_reduced",
    }
)


def _category(t: torch.Tensor) -> str:
    """Which comparison a tensor belongs to: ``bool``/``integer``/``floating``/``complex``.

    The category decides the *kind* of check, and torch will silently convert
    across all four if asked -- which is how two defects reached review here.
    Comparing a float32 ``1.9`` against an int64 ``1`` truncated the float and
    reported a match, and comparing a complex64 ``1+5j`` against a float32 ``1``
    dropped the imaginary part (with a warning nobody reads) and reported a
    match. Neither conversion is the harness's to make: a boundary whose two
    sides are in different categories is a harness defect, not a measurement,
    so it is refused rather than coerced.
    """
    if t.is_complex():
        return "complex"
    if t.is_floating_point():
        return "floating"
    if t.dtype == torch.bool:
        return "bool"
    return "integer"


def _tol_for(dtype: torch.dtype) -> tuple[float, float]:
    return _DTYPE_TOL.get(dtype, (1.6e-2, 1e-5))


def _floor_for(dtype: torch.dtype, tag: str = "") -> float:
    floor = _CANCELLATION_FLOOR.get(dtype, 2.0**-8)
    return floor * 2.0 if tag in _MULTI_STEP_BOUNDARIES else floor


class Report:
    """Per-boundary error statistics, plus the controls that discriminate them.

    One row per ``(tag, kind)``: ``kind`` is ``"ref"`` for the pure-PyTorch
    reference and a control name otherwise. Rows accumulate the worst case over
    every call, because a boundary that aligns on nine calls and diverges on the
    tenth is a boundary that does not align.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict] = {}
        self.notes: list[str] = []
        #: Boundaries whose controls provably cannot bind on this input, each
        #: recorded once with the measurement that establishes it. ``_finish``
        #: requires membership here before it will accept a control that failed
        #: to discriminate -- otherwise a blind harness and an unreachable
        #: regime are the same green column.
        self.regimes: set[str] = set()
        self._noted: set[str] = set()

    def add(self, tag: str, kind: str, got: torch.Tensor, want: torch.Tensor) -> dict:
        if got.shape != want.shape:
            row = self.rows.setdefault((tag, kind), self._empty(str(want.dtype)))
            row["calls"] += 1
            row["shape_mismatch"] = f"{tuple(got.shape)} vs {tuple(want.shape)}"
            row["passes"] = False
            return row
        g, w = got.detach(), want.detach()
        if g.device != w.device:
            g = g.to(w.device)
        gc, wc = _category(g), _category(w)
        if gc != wc:
            # Refused, not coerced. See :func:`_category`: every cross-category
            # conversion torch offers here is lossy in the direction that hides
            # a difference, so the only honest outcome is to record what was
            # handed over and fail. ``_finish`` surfaces these separately from
            # numerical failures, because a category mismatch is a defect in the
            # harness rather than a result about the checkpoint.
            row = self.rows.setdefault((tag, kind), self._empty(str(want.dtype)))
            row["calls"] += 1
            row["dtype_mismatch"] = f"got {got.dtype} ({gc}) vs want {want.dtype} ({wc})"
            row["passes"] = False
            return row
        if wc in ("integer", "bool"):
            # EXACT, AND IN THE ORIGINAL DTYPE. Widening to float32 first is what
            # this comparison used to do, and it is blind above 2**24: the
            # adjacent Engram hash ids 384006167 and 384006168 both round to
            # 384006176.0 and compare equal. 643 of the 672 hash ids in the
            # saved full capture are above that range, as are every hash
            # multiplier and every table offset, so the blindness covered
            # exactly the values the Engram module is indexed by. Integers and
            # indices are compared as int64 -- lossless from every signed width
            # and from uint8 -- and never through a float. uint64 is the one
            # width int64 cannot hold, so a mixed-width comparison involving it
            # is refused rather than wrapped.
            if g.dtype != w.dtype:
                if torch.uint64 in (g.dtype, w.dtype):
                    row = self.rows.setdefault((tag, kind), self._empty(str(want.dtype)))
                    row["calls"] += 1
                    row["dtype_mismatch"] = (
                        f"got {got.dtype} vs want {want.dtype}: no lossless common integer width"
                    )
                    row["passes"] = False
                    return row
                g, w = g.to(torch.int64), w.to(torch.int64)
            neq = g != w
            bad = int(neq.sum().item())
            row = self.rows.setdefault((tag, kind), self._empty(str(want.dtype)))
            row["calls"] += 1
            row["mismatched"] += bad
            row["numel"] += w.numel()
            row["compared_as"] = str(w.dtype)
            if bad and not row["first_mismatch"]:
                at = int(neq.flatten().nonzero()[0].item())
                row["first_mismatch"] = (
                    f"flat[{at}] got {g.flatten()[at].item()} want {w.flatten()[at].item()}"
                )
            row["passes"] = row["passes"] and bad == 0
            return row
        # Both sides are floating or both complex from here: the category check
        # above has already refused every other pairing.
        gate_dtype = want.dtype
        if wc == "complex":
            # A complex table is two real planes: comparing them as one real
            # tensor keeps the dtype-aware gate meaningful (a phase error shows
            # up in both) without inventing a complex tolerance.
            gate_dtype = torch.float64 if want.dtype == torch.complex128 else torch.float32
            g, w = torch.view_as_real(g), torch.view_as_real(w)
        # Accumulate at least as wide as the widest side, so the arithmetic of
        # the comparison never costs more precision than the values carry. fp32
        # is the floor rather than the rule: a float64 boundary compared through
        # fp32 would be measuring the cast, not the boundary.
        work = (
            torch.float64
            if torch.float64 in (g.dtype, w.dtype) or gate_dtype == torch.float64
            else torch.float32
        )
        g, w = g.to(work), w.to(work)
        rtol, atol = _tol_for(gate_dtype)
        floor = _floor_for(gate_dtype, tag)
        diff = (g - w).abs()
        denom = w.abs().clamp_min(1e-12)
        absmax = float(w.abs().max().item()) if w.numel() else 0.0
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        # The scale-relative error: how far apart the two are measured against
        # the tensor's own magnitude rather than each element's. This is the
        # number that separates a cancellation from a semantic gap, so it is
        # recorded whether or not the row passes.
        rel_scale = max_abs / absmax if absmax > 0 else 0.0
        # The COHERENT-SHIFT statistic, recorded beside the worst-element one
        # because they answer different questions and only one of them is a
        # measure of correctness.
        #
        # `rel_scale` is driven by a single element. Two implementations that
        # accumulate the same reduction in different fp32 orders disagree by
        # about 1e-6 relative, which crosses a bf16 rounding boundary roughly
        # once in every 4,000 elements; when one of those elements happens to
        # sit near the tensor's absmax, `rel_scale` reads exactly one
        # representable step. That is the smallest disagreement bf16 can
        # express, and with enough calls it is not a risk but a certainty.
        #
        # `mean_rel` is driven by all of them. It is RECORDED BUT NOT GATED, and
        # the measurement that decided that is worth keeping: the obvious idea
        # is to replace the worst-element term with this one, on the theory
        # that a semantic error moves every element coherently while a few
        # roundings do not. Measured at toy widths (job 745602), that theory is
        # wrong here. `moe.module/no-shared` -- dropping an entire expert --
        # reads 1.896 on the worst-element statistic but only 0.0801 on this
        # one, because the shared expert is large on a few elements rather than
        # coherent across the tensor. A mean-based gate would have made the
        # ladder's tightest control blind. So this stays a diagnostic: when a
        # boundary trips, it says whether the cause was one element or all of
        # them.
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        mean_rel = mean_abs / absmax if absmax > 0 else 0.0
        if diff.numel():
            base = diff <= atol + rtol * w.abs()
            n_over = int((~base).sum().item())
            # TWO conditions, and the second is what makes the first safe.
            #
            #   elementwise  no element may miss BOTH the dtype-default
            #                tolerance and the cancellation floor
            #   tensor-level the whole tensor's scale-relative error must stay
            #                inside the floor
            #
            # The elementwise half alone is not a gate: measured on the toy
            # model, dropping the shared expert entirely left every
            # base-violating element below the floor (its diffs there were
            # small) while the elements with the large diffs cleared the base
            # rtol against a large |w|. The control passed. Adding the
            # tensor-level term put it back outside at 1.90 ULP-at-scale
            # against the correct result's 0.22.
            ok = bool(torch.all(base | (diff <= floor * absmax)).item()) and rel_scale <= floor
        else:
            n_over, ok = 0, True
        row = self.rows.setdefault((tag, kind), self._empty(str(want.dtype)))
        row["calls"] += 1
        row["numel"] += w.numel()
        row["max_abs"] = max(row["max_abs"], max_abs)
        row["max_rel"] = max(row["max_rel"], max_rel)
        row["rel_scale"] = max(row["rel_scale"], rel_scale)
        row["mean_rel"] = max(row["mean_rel"], mean_rel)
        row["over_elementwise"] += n_over
        row["ref_absmax"] = max(row["ref_absmax"], absmax)
        row["passes"] = row["passes"] and ok
        row["rtol"], row["atol"], row["floor"] = rtol, atol, floor
        return row

    @staticmethod
    def _empty(dtype: str) -> dict:
        return {
            "dtype": dtype,
            "calls": 0,
            "numel": 0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "rel_scale": 0.0,
            "mean_rel": 0.0,
            "over_elementwise": 0,
            "ref_absmax": 0.0,
            "mismatched": 0,
            "passes": True,
            "rtol": 0.0,
            "atol": 0.0,
            "floor": 0.0,
            "compared_as": "",
            "first_mismatch": "",
            "dtype_mismatch": "",
        }

    def note(self, message: str) -> None:
        self.notes.append(message)

    def note_once(self, key: str, message: str) -> None:
        """Record a note the first time only.

        Some notes describe a behaviour that recurs once per decode step. With
        the full frozen continuations that is 124 steps, and 124 copies of the
        same sentence buries the four notes that carry a measurement.
        """
        if key not in self._noted:
            self._noted.add(key)
            self.notes.append(message)

    def dtype_mismatches(self) -> list[str]:
        """Boundaries handed two tensors that are not comparable at all.

        Separate from :meth:`failures` on purpose: this is a defect in the
        harness, not a result about the checkpoint, and reading it as either a
        numerical failure or (on a control row) as successful discrimination
        would be wrong in both directions.
        """
        return [
            f"{tag}/{kind}: {row['dtype_mismatch']}"
            for (tag, kind), row in sorted(self.rows.items())
            if row["dtype_mismatch"]
        ]

    def as_dict(self) -> dict:
        out: dict[str, dict] = {}
        for (tag, kind), row in sorted(self.rows.items()):
            out.setdefault(tag, {})[kind] = row
        return out

    def failures(self) -> list[str]:
        return [
            f"{tag}/{kind}"
            for (tag, kind), row in sorted(self.rows.items())
            if kind == "ref" and not row["passes"]
        ]

    def indistinct_controls(self) -> list[str]:
        """Controls a wrong answer got away with -- the harness could not see them.

        Not a failure on its own: some controls are genuinely degenerate on a
        given input (an epsilon that never bites, a permutation that is the
        identity for one head count). But a boundary whose only evidence is a
        tolerance and no discriminating control is evidence of nothing, so these
        are surfaced rather than dropped.
        """
        return [
            f"{tag}/{kind}"
            for (tag, kind), row in sorted(self.rows.items())
            if kind != "ref" and not kind.startswith("probe-") and row["passes"]
        ]

    def render(self) -> str:
        lines = [
            f"{'boundary':<36} {'kind':<20} {'calls':>5} {'max_abs':>11} "
            f"{'ULP@scale':>10} {'coherent':>10} {'verdict':>8}",
            "-" * 107,
        ]
        for tag, kinds in sorted(self.as_dict().items()):
            for kind, row in sorted(kinds.items(), key=lambda kv: (kv[0] != "ref", kv[0])):
                if row["dtype_mismatch"]:
                    metric = f"{'-':>11} {'-':>10} {'-':>10}"
                elif row["numel"] and not row["rtol"]:
                    metric = f"{row['mismatched']:>11d} {'exact':>10} {'-':>10}"
                else:
                    ulp = row["rel_scale"] / row["floor"] if row["floor"] else 0.0
                    coh = row["mean_rel"] / row["floor"] if row["floor"] else 0.0
                    metric = f"{row['max_abs']:>11.3e} {ulp:>10.2f} {coh:>10.4f}"
                if row["dtype_mismatch"]:
                    # Neither a numerical result nor discrimination: the two
                    # sides were never comparable, whatever the row's kind.
                    verdict = "DTYPE"
                elif kind == "ref":
                    verdict = "pass" if row["passes"] else "FAIL"
                elif kind.startswith("probe-"):
                    # a sensitivity measurement, not a discriminating control
                    verdict = "probe" if row["passes"] else "probe!"
                else:
                    # a control is doing its job when it lands OUTSIDE the gate
                    verdict = "INDIST" if row["passes"] else "discrim"
                lines.append(f"{tag:<36} {kind:<20} {row['calls']:>5} {metric} {verdict:>8}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# comparator self-test
#
# Every exact claim in this file rests on one method, and a blind comparator
# would report every one of them green. So before any boundary is compared, the
# comparator is shown a defect it must see and a match it must accept, at the
# magnitudes this checkpoint actually uses.
#
# This is not a hypothetical. The version this replaces widened integers to
# float32 first, and float32 has 24 mantissa bits: at the Engram table's scale
# the adjacent row ids 384006167 and 384006168 both become 384006176.0 and
# compare equal. 643 of the 672 hash ids in the saved full capture sit above
# that range, and so does every hash multiplier (about 4.6e13). The whole
# Engram indexing surface was inside the blind spot, and the column read
# "exact".
# ---------------------------------------------------------------------------

#: Straddles this checkpoint's two Engram table sizes (384,006,168 and
#: 384,016,682 rows) and is far above float32's 2**24 exact range.
_ADJACENT_HIGH_INT = 384006167


def comparator_selftest() -> tuple[bool, list[str]]:
    """Prove :meth:`Report.add` can see the defects it claims to gate on.

    Returns ``(ok, lines)``. Each line records a case, what it expected and what
    it got, so the evidence is in the report rather than in this function's
    control flow.
    """
    lines: list[str] = []
    ok = True

    def case(
        name: str,
        got: torch.Tensor,
        want: torch.Tensor,
        should_pass: bool,
        want_dtype_mismatch: bool = False,
    ) -> None:
        nonlocal ok
        row = Report().add("selftest", "ref", got, want)
        good = bool(row["passes"]) == should_pass
        # For the cross-category cases it is not enough that they fail: they
        # have to fail *for the stated reason*. A coerced comparison that
        # happened to come out unequal would satisfy `passes=False` while
        # leaving the lossy conversion in place.
        good = good and bool(row["dtype_mismatch"]) == want_dtype_mismatch
        ok = ok and good
        detail = row["dtype_mismatch"] or f"compared_as={row['compared_as'] or 'float'}"
        lines.append(
            f"{'ok  ' if good else 'FAIL'} {name}: passes={row['passes']} "
            f"expected={should_pass} mismatched={row['mismatched']} {detail}"
        )

    for dtype in (torch.int64, torch.int32):
        hi = torch.tensor([_ADJACENT_HIGH_INT], dtype=dtype)
        case(f"adjacent {dtype} above 2**24 differs", hi + 1, hi, False)
        case(f"identical {dtype} above 2**24 matches", hi.clone(), hi, True)
    # int32 cannot hold a multiplier, so the multiplier magnitude is checked in
    # int64 only -- it is the other value class that lived in the blind spot.
    mult = torch.tensor([4652297985743381], dtype=torch.int64)
    case("adjacent int64 multiplier differs", mult + 1, mult, False)
    flags = torch.tensor([True, False, True])
    case("bool mask differs", flags.logical_not(), flags, False)
    case("bool mask matches", flags.clone(), flags, True)
    # The float path is not exact and must not be: it has to accept its own
    # dtype's rounding and reject a difference the cancellation floor does not
    # cover. Both directions are checked so a change to the floor cannot make
    # the gate vacuous in either one.
    base = torch.full((64,), 2.0, dtype=torch.bfloat16)
    case("bf16 identical matches", base.clone(), base, True)
    case("bf16 one-in-eight scale error differs", base * 1.125, base, False)
    # CROSS-CATEGORY, the two cases the review probe drove. Both used to report
    # a match: the float was truncated to the integer grid, and the complex
    # tensor lost its imaginary part to a warning. Neither may be coerced now,
    # and each must say so rather than merely come out unequal.
    case(
        "float32 1.9 against int64 1 is refused",
        torch.tensor([1.9], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int64),
        False,
        want_dtype_mismatch=True,
    )
    case(
        "complex64 1+5j against float32 1 is refused",
        torch.tensor([1 + 5j], dtype=torch.complex64),
        torch.tensor([1.0], dtype=torch.float32),
        False,
        want_dtype_mismatch=True,
    )
    # The mirror images, so the check is on the category pair and not on which
    # side happens to be the reference.
    case(
        "int64 1 against float32 1.9 is refused",
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1.9], dtype=torch.float32),
        False,
        want_dtype_mismatch=True,
    )
    case(
        "bool against int64 is refused",
        torch.tensor([True]),
        torch.tensor([1], dtype=torch.int64),
        False,
        want_dtype_mismatch=True,
    )
    # ...and the comparisons that ARE legitimate across widths stay legitimate,
    # so the category rule cannot have been implemented as "dtypes must match".
    case(
        "int32 against int64 of equal value matches",
        torch.tensor([_ADJACENT_HIGH_INT], dtype=torch.int32),
        torch.tensor([_ADJACENT_HIGH_INT], dtype=torch.int64),
        True,
    )
    case(
        "bf16 against fp32 of equal value matches",
        torch.full((8,), 2.0, dtype=torch.bfloat16),
        torch.full((8,), 2.0, dtype=torch.float32),
        True,
    )
    return ok, lines


# ---------------------------------------------------------------------------
# transparent recording wrappers
#
# The reference's module-level functions and a few Block/Attention methods are
# not nn.Modules, so a forward hook cannot see them. They are wrapped instead:
# each wrapper calls the original, compares, and returns the original's result
# unchanged. Transparency is not asserted by inspection -- ``--mode full``
# regenerates every frozen fixture's whole continuation with the wrappers
# installed and compares all of it.
# ---------------------------------------------------------------------------


class Harness:
    """Installs the recorders, holds the active layer, and owns the report."""

    def __init__(
        self, cfg: R.RefConfig, report: Report, layers: set[int] | None, max_calls: int
    ) -> None:
        self.cfg = cfg
        self.report = report
        self.layers = layers
        self.max_calls = max_calls
        self.layer: int | None = None
        self.sublayer: str = "?"
        self.counts: dict[str, int] = {}
        #: Budgets that survive ``counts.clear()`` between decode steps. The
        #: per-forward budget is what gives decode its own coverage, but an
        #: expensive composition driven once per step for five fixtures is
        #: hundreds of full-width reference GEMMs. The module compositions
        #: therefore carry a run-long cap on a tag that names the regime and the
        #: phase, so every regime is measured exactly once in each phase.
        self.global_counts: dict[str, int] = {}
        self._undo: list = []
        self.saved: dict[str, list] = {}
        #: Per-layer stashes of what the native forward produced, so a
        #: composition running in a later hook sees the same values the source
        #: used. Cloned: the sliding-window KV and the residual stream are both
        #: overwritten by the next layer.
        self.native: dict[tuple[str, int], Any] = {}
        #: Which pinned prompt is running, and whether this is its prefill.
        #: The parity captures budget one prefill per fixture rather than one
        #: per run, so that rung 3 replays every pinned prompt instead of
        #: freezing the first. Every rank sets these from the same loop over the
        #: same fixture list, which is what makes the per-rank captures
        #: positionally aligned -- and `parity_dense.py` asserts that alignment
        #: from the recorded index rather than assuming it.
        self.fixture: int = -1
        self.is_prefill: bool = False

    # -- plumbing ---------------------------------------------------------
    def active(self, tag: str) -> bool:
        if self.layers is not None and self.layer is not None and self.layer not in self.layers:
            return False
        n = self.counts.get(tag, 0)
        if n >= self.max_calls:
            return False
        self.counts[tag] = n + 1
        return True

    def active_once(self, tag: str, cap: int = 1) -> bool:
        """Budget a boundary over the whole run rather than per forward."""
        if self.layers is not None and self.layer is not None and self.layer not in self.layers:
            return False
        n = self.global_counts.get(tag, 0)
        if n >= cap:
            return False
        self.global_counts[tag] = n + 1
        return True

    def compare(self, tag: str, got: torch.Tensor, want: torch.Tensor, kind: str = "ref") -> None:
        self.report.add(tag, kind, got, want)

    def in_scope(self, layer: int | None) -> bool:
        return layer is not None and (self.layers is None or layer in self.layers)

    def stash(self, name: str, layer: int, value) -> None:
        """Keep a native value for a composition running in a later hook.

        Guarded by the layer filter: at full scale every stashed residual
        stream is several MiB and there are forty layers, so stashing outside
        the recorded set buys nothing and costs about a gigabyte.
        """
        if self.in_scope(layer):
            self.native[(name, layer)] = value

    def regime(self, tag: str, message: str) -> None:
        """Record, once per boundary, why a control could not bind on this input.

        A control that passes is either a blind harness or an input regime that
        does not reach the behaviour. Those two look identical in a pass/fail
        column and are completely different findings, so the second one is
        stated with the measurement that establishes it -- and ``_finish``
        rejects the first by requiring this record.
        """
        if tag not in self.report.regimes:
            self.report.regimes.add(tag)
            self.report.note(f"{tag}: {message}")

    def save(self, tag: str, payload: dict, cap: int | None = None) -> None:
        """Keep a payload under ``tag``, capped at ``max_calls`` unless told otherwise.

        The parity captures pass an explicit ``cap``: they budget one prefill per
        FIXTURE rather than per call, so their limit is the fixture count and not
        the per-forward call budget the boundary comparisons use.
        """
        bucket = self.saved.setdefault(tag, [])
        if len(bucket) < (self.max_calls if cap is None else cap):
            bucket.append(
                {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in payload.items()}
            )

    def patch(self, obj, name: str, factory) -> None:
        original = getattr(obj, name)
        setattr(obj, name, factory(original))
        self._undo.append((obj, name, original))

    def hook(self, module, fn) -> None:
        handle = module.register_forward_hook(fn)
        self._undo.append((handle, None, None))

    def close(self) -> None:
        for obj, name, original in reversed(self._undo):
            if name is None:
                obj.remove()
            else:
                setattr(obj, name, original)
        self._undo.clear()


def _roll_last(x: torch.Tensor) -> torch.Tensor:
    """Shift a tensor one place along its last axis, wrapping around.

    The float8 scale dtypes carry almost no CUDA kernels: measured here, both
    ``torch.roll`` and ``torch.cat`` raise ``NotImplementedError ... for
    'Float8_e8m0fnu'``. Since a scale is one byte, the shift is done on a uint8
    view -- a pure reinterpretation of the same storage -- and viewed back.
    """
    if x.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raw = x.view(torch.uint8)
        rolled = torch.cat([raw[..., -1:], raw[..., :-1]], dim=-1).contiguous()
        return rolled.view(x.dtype)
    return torch.cat([x[..., -1:], x[..., :-1]], dim=-1)


def _split_engram_kv(kv: torch.Tensor, hc_mult: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    return key.unflatten(-1, (hc_mult, dim)), value


def install(harness: Harness, model, ref_module, tokenizer=None) -> None:
    """Wrap the reference's functions and hook its modules.

    ``ref_module`` is the checkpoint's imported ``model`` module; the wrappers
    are installed into its namespace so that the call sites inside it resolve to
    them without any call site being edited. ``tokenizer`` is what the
    compressed-token-map comparison rebuilds from; without it that boundary
    cannot run, and the required-boundary manifest then fails the command.
    """
    cfg = harness.cfg
    H = harness

    # -- quantization grids ----------------------------------------------
    def wrap_act_quant(orig):
        def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
            before = x.detach().clone()
            out = orig(x, block_size, scale_fmt, scale_dtype, inplace)
            if H.active("quant.act_quant.inplace" if inplace else "quant.act_quant"):
                if inplace:
                    want = R.act_quant_dequant_ref(
                        before, block_size, round_scale=scale_fmt is not None
                    )
                    H.compare("quant.act_quant.inplace", want, out)
                else:
                    q, s = R.act_quant_ref(before, block_size, round_scale=scale_fmt is not None)
                    H.compare("quant.act_quant.values", q, out[0].float())
                    H.compare("quant.act_quant.scales", s, out[1].float())
            return out

        return act_quant

    def wrap_fp4_act_quant(orig):
        def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
            before = x.detach().clone()
            out = orig(x, block_size, inplace, scale_dtype)
            e4m3 = scale_dtype == torch.float8_e4m3fn
            tag = f"quant.fp4_act_quant.{'e4m3' if e4m3 else 'e8m0'}"
            if inplace and H.active(tag):
                want = R.fp4_act_quant_dequant_ref(before, block_size, e4m3_scale=e4m3)
                H.compare(tag, want, out)
                # control: the other scale format, which differs in both the
                # floor and the rounding grid of the scale itself
                other = R.fp4_act_quant_dequant_ref(before, block_size, e4m3_scale=not e4m3)
                H.compare(tag, other, out, kind="wrong-scale-format")
            return out

        return fp4_act_quant

    # -- linear -----------------------------------------------------------
    def wrap_linear(orig):
        def linear(x, weight, bias=None):
            out = orig(x, weight, bias)
            kind = {
                torch.float4_e2m1fn_x2: "linear.fp4",
                torch.float8_e4m3fn: "linear.fp8",
            }.get(weight.dtype, "linear.plain")
            if H.active(kind):
                scale = getattr(weight, "scale", None)
                want = R.linear_ref(x, weight, scale, out_dtype=out.dtype)
                H.compare(kind, want, out)
                # The two quantized branches are the only float boundaries whose
                # correct result is nonzero and whose semantics are not covered
                # by a consumer's control, so each gets one here. Both attack
                # the scale grid rather than the values: mis-associating a
                # per-block scale is the mistake this layout invites, and it is
                # invisible in a shape check because the wrong grid has the
                # right shape.
                if scale is not None and kind == "linear.fp8":
                    # the fp8 grid is [ceil(N/32), K/32]; rolling it along K
                    # pairs every block with its neighbour's scale
                    H.compare(
                        kind,
                        R.linear_fp8_ref(x, weight, _roll_last(scale), out_dtype=out.dtype),
                        out,
                        kind="rolled-weight-scale",
                    )
                elif scale is not None and kind == "linear.fp4":
                    H.compare(
                        kind,
                        R.linear_fp4_ref(x, weight, _roll_last(scale), out_dtype=out.dtype),
                        out,
                        kind="rolled-weight-scale",
                    )
            return out

        return linear

    # -- rope -------------------------------------------------------------
    def wrap_rope(orig):
        def apply_rotary_emb(x, freqs_cis, inverse=False):
            before = x.detach().clone()
            out = orig(x, freqs_cis, inverse)
            tag = "rope.inverse" if inverse else "rope.forward"
            if H.active(tag):
                H.compare(tag, R.apply_rope_ref(before, freqs_cis, inverse), out)
                # control: NeoX half-split pairing instead of adjacent pairs
                half = before.size(-1) // 2
                xc = torch.complex(before.float()[..., :half], before.float()[..., half:])
                f = freqs_cis.conj() if inverse else freqs_cis
                f = (
                    f.view(1, xc.size(1), xc.size(-1))
                    if xc.ndim == 3
                    else f.view(1, xc.size(1), 1, xc.size(-1))
                )
                rot = xc * f
                neox = torch.cat([rot.real, rot.imag], dim=-1).to(before.dtype)
                H.compare(tag, neox, out, kind="neox-pairing")
                if not inverse:
                    H.compare(
                        tag, R.apply_rope_ref(before, freqs_cis, True), out, kind="conjugated"
                    )
            return out

        return apply_rotary_emb

    # -- hyper-connections -------------------------------------------------
    def wrap_sinkhorn(orig):
        def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
            pre, post, comb = orig(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
            if H.active("hc.split_sinkhorn"):
                r_pre, r_post, r_comb = R.hc_split_sinkhorn_ref(
                    mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
                )
                H.compare("hc.split_sinkhorn.pre", r_pre, pre)
                H.compare("hc.split_sinkhorn.post", r_post, post)
                H.compare("hc.split_sinkhorn.comb", r_comb, comb)
                # controls: the transposed comb, and stopping Sinkhorn early
                H.compare(
                    "hc.split_sinkhorn.comb", r_comb.transpose(-1, -2), comb, kind="transposed-comb"
                )
                one = R.hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, 1, eps)[2]
                H.compare("hc.split_sinkhorn.comb", one, comb, kind="one-iteration")
                # control: post without the factor of two
                H.compare("hc.split_sinkhorn.post", r_post / 2, post, kind="post-without-2x")
                # control: pre built like post -- 2*sigmoid instead of
                # sigmoid + eps. The two coefficient sets come out of adjacent
                # slices of one projection and differ only in this transform,
                # which is exactly why swapping them is plausible.
                H.compare("hc.split_sinkhorn.pre", r_post, pre, kind="pre-built-like-post")
            return pre, post, comb

        return hc_split_sinkhorn

    def wrap_hc_mixes(orig):
        def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
            out = orig(self, x, hc_fn, hc_scale, hc_base)
            # A block calls this twice, in order: first on its input, then on
            # the stream between the two sublayers. The second input is the
            # only place that intermediate stream is visible from outside, and
            # the block composition needs it to keep each of its links anchored
            # on the source's own value rather than on its own previous one.
            if H.in_scope(self.layer_id):
                seen = H.native.setdefault(("hc_mix_in", self.layer_id), [])
                if len(seen) >= 2:
                    seen.clear()
                seen.append(x.detach().clone())
            if H.active("hc.mixes"):
                mixes = R.hc_mix_projection_ref(x, hc_fn, self.norm_eps)
                r = R.hc_split_sinkhorn_ref(
                    mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
                )
                for name, a, b in zip(("pre", "post", "comb"), r, out):
                    H.compare(f"hc.mixes.{name}", a, b)
                H.compare("hc.mixes.post", r[1] / 2, out[1], kind="post-without-2x")
                # Per-copy normalization is NOT a control here: with equal-width
                # copies the mean of per-copy means is the mean over the flat
                # stream, and it measured identical. Omitting the statistic
                # entirely is the real failure mode -- a port that projects the
                # unnormalized stream.
                xf = x.flatten(2).float()
                bad = torch.nn.functional.linear(xf, hc_fn.float())
                rb = R.hc_split_sinkhorn_ref(
                    bad, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
                )
                H.compare("hc.mixes.pre", rb[0], out[0], kind="no-rsqrt")
                H.compare("hc.mixes.comb", rb[2], out[2], kind="no-rsqrt")
            return out

        return hc_mixes

    def wrap_hc_pre(orig):
        def hc_pre(self, x, pre_mix):
            out = orig(self, x, pre_mix)
            if H.active("hc.pre"):
                H.compare("hc.pre", R.hc_pre_ref(x, pre_mix), out)
                H.compare("hc.pre", R.hc_pre_ref(x, pre_mix.flip(-1)), out, kind="permuted-mix")
            return out

        return hc_pre

    def wrap_hc_post(orig):
        def hc_post(self, x, residual, post, comb):
            out = orig(self, x, residual, post, comb)
            if H.active("hc.post"):
                H.compare("hc.post", R.hc_post_ref(x, residual, post, comb), out)
                H.compare(
                    "hc.post",
                    R.hc_post_ref(x, residual, post, comb.transpose(-1, -2)),
                    out,
                    kind="transposed-comb",
                )
                H.compare(
                    "hc.post",
                    R.hc_post_ref(x, residual, post, comb * 0 + 0.25),
                    out,
                    kind="uniform-comb",
                )
            return out

        return hc_post

    # -- attention --------------------------------------------------------
    def wrap_sparse_attn(orig):
        def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
            out = orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            # Cloned and stashed before any gate, because the attention
            # composition runs in a later hook and both of these are cache
            # state the next layer overwrites: `kv` is a view of the sliding
            # window ring during decode, and the shared compressed pool is
            # rewritten at every source layer.
            if H.layer is not None:
                H.stash("sparse_kv", H.layer, (kv.detach().clone(), topk_idxs.detach().clone()))
            if H.active("attn.sparse_attn"):
                H.compare(
                    "attn.sparse_attn",
                    R.sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale),
                    out,
                )
                no_sink = R.sparse_attn_ref(
                    q, kv, torch.full_like(attn_sink, -1e30), topk_idxs, softmax_scale
                )
                H.compare("attn.sparse_attn", no_sink, out, kind="no-sink")
                # Permuting a row's indices is NOT a control: the kernel
                # handles every gathered slot independently, so index order is
                # semantically free and a rolled row measured identical. What
                # does change the answer is changing the SET -- here by
                # dropping each row's first reachable position.
                dropped = topk_idxs.clone()
                first = (dropped >= 0).float().argmax(dim=-1, keepdim=True)
                dropped.scatter_(-1, first, torch.full_like(first, -1, dtype=dropped.dtype))
                H.compare(
                    "attn.sparse_attn",
                    R.sparse_attn_ref(q, kv, attn_sink, dropped, softmax_scale),
                    out,
                    kind="dropped-row",
                )
                H.save(
                    "attn.sparse_attn",
                    {
                        "q": q,
                        "kv": kv,
                        "attn_sink": attn_sink,
                        "topk_idxs": topk_idxs,
                        "softmax_scale": softmax_scale,
                        "out": out,
                        "layer": H.layer,
                    },
                )
            return out

        return sparse_attn

    def wrap_candidates(orig):
        def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
            out = orig(logits, compress_lens, topk_blocks, block_size)
            if H.active("attn.candidates"):
                H.compare(
                    "attn.candidates",
                    R.select_candidate_blocks_ref(logits, compress_lens, topk_blocks, block_size),
                    out,
                )
                width = logits.size(-1)
                pad = torch.nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
                scores = pad.unflatten(-1, (-1, block_size)).amax(dim=-1)
                top = scores.topk(min(topk_blocks, scores.size(-1)), dim=-1)
                keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                    -1, top.indices, top.values > -torch.inf
                )
                H.compare(
                    "attn.candidates",
                    keep.repeat_interleave(block_size, dim=-1)[..., :width],
                    out,
                    kind="no-force-newest",
                )
                # A control that cannot bind is not a blind harness, it is a
                # regime fact -- and the number is what makes the difference
                # legible. With fewer blocks than topk_blocks every block is
                # kept, so forcing the newest one in changes nothing.
                n_blocks = (width + block_size - 1) // block_size
                if n_blocks <= topk_blocks:
                    H.regime(
                        "attn.candidates",
                        f"{n_blocks} blocks of {block_size} over {width} compressed positions, "
                        f"topk_blocks={topk_blocks}: every block is reachable, so the candidate "
                        f"filter is inert at this length and no control here can discriminate",
                    )
            return out

        return select_candidate_blocks

    def wrap_window_idxs(orig):
        def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
            out = orig(window_size, bsz, seqlen, start_pos)
            if H.active("attn.window_idxs"):
                H.compare(
                    "attn.window_idxs",
                    R.window_topk_idxs_ref(window_size, bsz, seqlen, start_pos).to(out.device),
                    out,
                )
            return out

        return get_window_topk_idxs

    # -- module hooks ------------------------------------------------------
    def rmsnorm_hook(module, args, output):
        if H.active("norm.rmsnorm"):
            x = args[0]
            H.compare("norm.rmsnorm", R.rms_norm_ref(x, module.weight, module.eps), output)
            # A sensitivity probe, not a control: 1e-20 against 1e-6 is below
            # bf16 rounding at these magnitudes, so it is recorded as a measured
            # fact rather than counted as discrimination.
            H.compare(
                "norm.rmsnorm",
                R.rms_norm_ref(x, module.weight, 1e-6),
                output,
                kind="probe-eps-1e-6",
            )
            # control: LayerNorm-style mean centering, which is a different norm
            centered = x.float() - x.float().mean(-1, keepdim=True)
            H.compare(
                "norm.rmsnorm",
                R.rms_norm_ref(centered.to(x.dtype), module.weight, module.eps),
                output,
                kind="mean-centered",
            )
            # control: accumulating the statistic in bf16 rather than fp32,
            # which is the precision mistake a port actually makes
            xb = x.to(torch.bfloat16)
            var = xb.square().mean(-1, keepdim=True)
            H.compare(
                "norm.rmsnorm",
                (module.weight.to(torch.bfloat16) * (xb * torch.rsqrt(var + module.eps))).to(
                    x.dtype
                ),
                output,
                kind="bf16-statistic",
            )

    def embed_hook(module, args, output):
        if H.active("embed.shard"):
            want = R.embedding_shard_ref(
                args[0], module.weight, module.vocab_start_idx, module.vocab_end_idx
            )
            if _world() > 1:
                torch.distributed.all_reduce(want)
            H.compare("embed.shard", want, output)

    def head_hook(module, args, output):
        if H.active("head.logits"):
            x = args[0]
            x = x if len(args) > 1 and args[1] else x[:, -1]
            want = R.head_shard_ref(x, module.weight)
            if _world() > 1:
                parts = [torch.empty_like(want) for _ in range(_world())]
                torch.distributed.all_gather(parts, want)
                want = torch.cat(parts, dim=-1)
            H.compare("head.logits", want, output)

    def gate_hook(module, args, output):
        if H.active("moe.gate"):
            weights, indices = output
            r_w, r_i = R.gate_ref(
                args[0],
                module.weight,
                module.bias,
                module.topk,
                module.score_func,
                module.gate_temp,
                module.norm_topk_prob,
                module.route_scale,
            )
            H.compare("moe.gate.indices", r_i, indices)
            H.compare("moe.gate.weights", r_w, weights)
            s_w, s_i = R.gate_ref(
                args[0],
                module.weight,
                module.bias,
                module.topk,
                "sigmoid",
                module.gate_temp,
                module.norm_topk_prob,
                module.route_scale,
            )
            H.compare("moe.gate.indices", s_i, indices, kind="sigmoid-scoring")
            H.compare("moe.gate.weights", s_w, weights, kind="sigmoid-scoring")
            # control: letting the correction bias scale the weights as well as
            # steer the selection, which is the single most plausible misreading
            # of this router
            biased = (
                torch.nn.functional.softplus(
                    torch.nn.functional.linear(args[0].float(), module.weight.float())
                    / module.gate_temp
                ).sqrt()
                + module.bias.float()
            ).gather(1, indices)
            if module.norm_topk_prob and module.topk > 1:
                biased = biased / (biased.sum(-1, keepdim=True) + 1e-20)
            H.compare(
                "moe.gate.weights", biased * module.route_scale, weights, kind="bias-scaled-weights"
            )
            H.save(
                "moe.gate", {"x": args[0], "weights": weights, "indices": indices, "layer": H.layer}
            )

    def expert_hook(module, args, output):
        if H.active("moe.expert"):
            x = args[0]
            w = args[1] if len(args) > 1 else None
            got = R.expert_ref(
                x,
                module.w1.weight,
                getattr(module.w1.weight, "scale", None),
                module.w2.weight,
                getattr(module.w2.weight, "scale", None),
                module.w3.weight,
                getattr(module.w3.weight, "scale", None),
                module.swiglu_limit,
                w,
            )
            H.compare("moe.expert", got, output)
            swapped = R.expert_ref(
                x,
                module.w3.weight,
                getattr(module.w3.weight, "scale", None),
                module.w2.weight,
                getattr(module.w2.weight, "scale", None),
                module.w1.weight,
                getattr(module.w1.weight, "scale", None),
                module.swiglu_limit,
                w,
            )
            H.compare("moe.expert", swapped, output, kind="swapped-w1-w3")
            unclamped = R.expert_ref(
                x,
                module.w1.weight,
                getattr(module.w1.weight, "scale", None),
                module.w2.weight,
                getattr(module.w2.weight, "scale", None),
                module.w3.weight,
                getattr(module.w3.weight, "scale", None),
                0.0,
                w,
            )
            H.compare("moe.expert", unclamped, output, kind="no-clamp")
            if module.swiglu_limit > 0:
                gate = R.linear_ref(
                    x, module.w1.weight, getattr(module.w1.weight, "scale", None)
                ).float()
                up = R.linear_ref(
                    x, module.w3.weight, getattr(module.w3.weight, "scale", None)
                ).float()
                reach = max(float(gate.max().item()), float(up.abs().max().item()))
                if reach <= module.swiglu_limit:
                    H.regime(
                        "moe.expert",
                        f"the widest branch value here is {reach:.3f} against "
                        f"swiglu_limit={module.swiglu_limit}: the clamp never engages on this "
                        f"input, so removing it is not observable and the pinned prompts do not "
                        f"exercise it",
                    )

    def engram_embed_hook(module, args, output):
        if H.active("engram.lookup"):
            want = R.engram_lookup_shard_ref(
                args[0],
                module.weight,
                module.scale,
                module.vocab_start_idx,
                module.vocab_end_idx,
                module.block_size,
            )
            if _world() > 1:
                torch.distributed.all_reduce(want)
            H.compare("engram.lookup", want, output)
            H.save("engram.lookup", {"hash_ids": args[0], "out": output, "layer": H.layer})

    def engram_hook(module, args, output):
        if H.active("engram.module"):
            x, hash_ids = args[0], args[1]
            token_mask = args[2] if len(args) > 2 else None
            rows = module.embed(hash_ids)
            got = R.engram_module_ref(
                x,
                rows,
                module.wkv.weight,
                getattr(module.wkv.weight, "scale", None),
                module.q_weight,
                module.k_weight,
                module.hc_mult,
                module.dim,
                module.eps,
                module.clamp_value,
                token_mask,
            )
            H.compare("engram.module", got, output)
            kv = R.linear_ref(
                rows.flatten(-2), module.wkv.weight, getattr(module.wkv.weight, "scale", None)
            )
            key, value = _split_engram_kv(kv, module.hc_mult, module.dim)
            # control: unsigned sqrt, i.e. dropping the copysign the training
            # kernel applies before the sigmoid
            weight = module.q_weight.float() * module.k_weight.float()
            h, k = x.float(), key.float()
            rstd = torch.rsqrt(h.square().mean(-1) + module.eps) * torch.rsqrt(
                k.square().mean(-1) + module.eps
            )
            dot = (h * weight * k).sum(-1) * rstd * module.dim**-0.5
            gate = torch.sigmoid(dot.abs().clamp_min(module.clamp_value).sqrt())
            if token_mask is not None:
                gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
            unsigned = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
            nosqrt = (h + torch.sigmoid(dot).unsqueeze(-1) * value.float().unsqueeze(-2)).to(
                x.dtype
            )
            H.compare("engram.module", unsigned, output, kind="unsigned-sqrt")
            H.compare("engram.module", nosqrt, output, kind="no-sqrt")
            # Both gate controls are inert when the n-gram term is negligible
            # against the residual stream it is added to: the gate is bounded in
            # (0, 1), so the *most* any wrong gate can move the output is the
            # magnitude of `value` itself. At toy widths that is far under the
            # scale floor, and the same two controls discriminate clearly on the
            # real checkpoint -- which is why both regimes exist.
            span = max(
                float((unsigned.float() - output.float()).abs().max().item()),
                float((nosqrt.float() - output.float()).abs().max().item()),
            )
            scale = float(output.float().abs().max().item())
            floor = _floor_for(output.dtype)
            if scale > 0 and span <= floor * scale:
                H.regime(
                    "engram.module",
                    f"the widest wrong gate moves the output by {span:.3e} against a stream "
                    f"magnitude of {scale:.3e} ({span / scale:.2e} of scale, floor "
                    f"{floor:.2e}): the n-gram term is negligible against the residual at "
                    f"these widths, so no gate control can discriminate here",
                )

    def ngram_hook(module, args, output):
        if H.active("engram.hash"):
            input_ids, start_pos = args[0], args[1]
            batch, seqlen = input_ids.shape
            # The compressed token map, in full. Its *size* was already checked
            # and that is the weaker half: the size only fixes the hash
            # multipliers, while the contents decide which tokens share an
            # n-gram at all. A normalizer that collapsed one pair differently
            # would keep the size and rehash those positions, so the whole
            # 129,280-entry lookup is compared here, exactly, once.
            if tokenizer is not None and H.active_once("engram.token_map"):
                lookup, size = R.compressed_token_map_ref(tokenizer)
                H.compare(
                    "engram.token_map",
                    torch.tensor(lookup, device=module.token_map.device),
                    module.token_map,
                )
                H.compare(
                    "engram.compressed_vocab",
                    torch.tensor([size]),
                    torch.tensor([cfg.engram_compressed_vocab_size]),
                )
                # The pad id the hash substitutes at a sequence or image
                # boundary is itself a compressed id, not the raw one.
                pad = cfg.engram_pad_id
                H.compare(
                    "engram.pad_id", torch.tensor([lookup[pad]]), torch.tensor([module.pad_id])
                )
                if lookup[pad] == pad:
                    H.regime(
                        "engram.pad_id",
                        f"raw token {pad} compresses to {lookup[pad]}, i.e. to itself, so the "
                        f"raw-versus-compressed control is the identity on this tokenizer and "
                        f"cannot discriminate",
                    )
                H.compare(
                    "engram.pad_id",
                    torch.tensor([pad]),
                    torch.tensor([module.pad_id]),
                    kind="raw-not-compressed",
                )
            # The hash constants are part of the semantics, not setup: every
            # multiplier is derived from the compressed vocab size, and the
            # prime ranges are what keep one flat table's buckets disjoint. They
            # are integers, so this is an exact comparison and the strongest
            # evidence available for this module.
            primes = R.engram_primes_ref(
                len(cfg.engram_layer_ids),
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                cfg.engram_vocab_size,
            )
            H.compare(
                "engram.primes",
                torch.tensor(primes, device=module.primes.device),
                module.primes,
            )
            H.compare(
                "engram.offsets",
                R.engram_offsets_ref(primes).to(module.offsets.device),
                module.offsets,
            )
            H.compare(
                "engram.multipliers",
                R.engram_multipliers_ref(
                    cfg.engram_layer_ids,
                    cfg.engram_max_ngram_size,
                    cfg.engram_compressed_vocab_size,
                ).to(module.multipliers.device),
                module.multipliers,
            )
            want = R.ngram_hash_ref(
                module.cache[:batch],
                start_pos,
                seqlen,
                module.multipliers,
                module.primes,
                module.offsets,
                module.pad_id,
                cfg.engram_max_ngram_size,
            )
            H.compare("engram.hash", want, output)
            H.save("engram.hash", {"input_ids": input_ids, "start_pos": start_pos, "out": output})

    # The compressor carries an incomplete group across decode steps, so a
    # decode comparison needs the state as it stood *before* the call. A
    # pre-hook snapshots it; nothing else can see that value afterwards because
    # the forward overwrites the slot it is about to pool.
    compressor_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def compressor_pre_hook(module, args):
        if module.compress_ratio > 1:
            compressor_state[id(module)] = (
                module.kv_state[:1].clone(),
                module.score_state[:1].clone(),
            )
        return None

    def compressor_hook(module, args, output):
        # Budgeted per ratio, not per "attn.compressor". With one budget for
        # all four kv-source layers the three ratio-2 sources (2, 8, 14)
        # consumed it and layer 20 -- the only ratio-1 source, and a different
        # code path with no pooling and no fp32 promotion -- was never compared
        # in full mode at all. That hole passed as green because nothing
        # required the boundary to exist.
        if not H.active(f"attn.compressor.r{module.compress_ratio}"):
            return
        x, start_pos = args[0], args[1]
        if module.compress_ratio == 1:
            got, _, _ = R.compressor_prefill_ref(
                x, 1, module.wkv.weight, None, module.norm.weight, module.norm.eps, x.dtype
            )
            assert got is not None
            H.compare("attn.compressor.ratio1", got, output)
            return
        if start_pos == 0:
            got, _, _ = R.compressor_prefill_ref(
                x,
                module.compress_ratio,
                module.wkv.weight,
                module.wgate.weight,
                module.norm.weight,
                module.norm.eps,
                x.dtype,
            )
            if output is None:
                H.report.note(f"compressor prefill produced no latent at seqlen {x.size(1)}")
                return
            assert got is not None
            H.compare("attn.compressor.ratio2.prefill", got, output)
            # control: mean pooling instead of the learned softmax gate
            xf = x.float()
            kv = torch.nn.functional.linear(xf, module.wkv.weight.float())
            seqlen = x.size(1)
            cutoff = seqlen - seqlen % module.compress_ratio
            mean = kv[:, :cutoff].unflatten(1, (-1, module.compress_ratio)).mean(dim=2)
            H.compare(
                "attn.compressor.ratio2.prefill",
                R.rms_norm_ref(mean.to(x.dtype), module.norm.weight, module.norm.eps),
                output,
                kind="mean-pooling",
            )
            return
        prior = compressor_state.get(id(module))
        if prior is None:
            return
        kv_state, score_state = prior[0][0].clone(), prior[1][0].clone()
        got = R.compressor_decode_ref(
            x,
            module.compress_ratio,
            start_pos,
            kv_state,
            score_state,
            module.wkv.weight,
            module.wgate.weight,
            module.norm.weight,
            module.norm.eps,
            x.dtype,
        )
        if output is None:
            # Once, not once per step: with the full frozen continuations this
            # fires on about half of 124 decode steps, and the boundary below is
            # the evidence anyway -- the note is only colour.
            H.report.note_once(
                f"compressor-held-r{module.compress_ratio}",
                f"compressor at ratio {module.compress_ratio} holds its incomplete group on the "
                f"steps that do not complete one (first seen at start_pos {start_pos}); every "
                f"such step is checked by attn.compressor.ratio2.decode.held",
            )
            H.compare(
                "attn.compressor.ratio2.decode.held",
                torch.tensor([got is None]),
                torch.tensor([True]),
            )
            return
        assert got is not None
        H.compare("attn.compressor.ratio2.decode", got, output)
        # control: dropping the carried state, i.e. pooling only the newest slot
        only_new = R.rms_norm_ref(
            kv_state[start_pos % module.compress_ratio].view(1, 1, -1).to(x.dtype),
            module.norm.weight,
            module.norm.eps,
        )
        H.compare("attn.compressor.ratio2.decode", only_new, output, kind="newest-slot-only")

    def indexer_hook(module, args, output):
        # Keyed by ratio for the same reason the compressor is: the ratio-1
        # index sources sit behind four ratio-2 ones in layer order, and a
        # single budget never reaches them.
        if not H.active(f"attn.indexer.r{module.compress_ratio}"):
            return
        x, qr, latent, start_pos, offset = args[0], args[1], args[2], args[3], args[4]
        bsz, seqlen, _ = x.size()
        ratio = module.compress_ratio
        end_pos = start_pos + seqlen
        rd = module.rope_head_dim
        assert module.freqs_cis is not None

        if module.owns_k and latent is not None:
            freqs = (
                module.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else module.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = R.index_k_ref(
                latent,
                module.wk.weight,
                getattr(module.wk.weight, "scale", None),
                module.k_norm.weight,
                module.k_norm.eps,
                freqs,
                rd,
            )
            lo = start_pos // ratio
            H.compare("attn.indexer.k", k, module.k_cache[:bsz, lo : lo + k.size(1)])

        q = R.index_q_ref(
            qr,
            module.wq_b.weight,
            getattr(module.wq_b.weight, "scale", None),
            module.n_local_heads,
            module.index_head_dim,
            module.freqs_cis[start_pos:end_pos],
            rd,
        )
        index_k = ref_module.shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = R.linear_ref(
            x, module.weights_proj.weight, getattr(module.weights_proj.weight, "scale", None)
        )
        weights = weights * (module.softmax_scale * module.n_heads**-0.5)
        score = R.indexer_scores_ref(q, index_k, weights)
        if _world() > 1:
            torch.distributed.all_reduce(score)
        candidates = ref_module.shared_attn.candidates if module.uses_candidates else None
        score, lens = R.indexer_mask_ref(
            score, start_pos, seqlen, ratio, end_pos, candidates, x.device
        )
        got = R.indexer_topk_ref(score, lens, module.index_topk, end_pos, ratio, offset)
        H.compare("attn.indexer.topk", got, output)
        available = end_pos // ratio
        if module.index_topk >= available:
            H.regime(
                "attn.indexer.topk",
                f"index_topk={module.index_topk} >= {available} reachable compressed positions: "
                f"the selection takes all of them, so the scores do not order anything and no "
                f"scoring control can discriminate at this length",
            )
        # control: no rectification before the head weighting, which lets a
        # negative per-head score cancel a positive one
        raw = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        unrect = (raw * weights.float().unsqueeze(-1)).sum(dim=2)
        if _world() > 1:
            torch.distributed.all_reduce(unrect)
        unrect, _ = R.indexer_mask_ref(
            unrect, start_pos, seqlen, ratio, end_pos, candidates, x.device
        )
        H.compare(
            "attn.indexer.topk",
            R.indexer_topk_ref(unrect, lens, module.index_topk, end_pos, ratio, offset),
            output,
            kind="no-relu",
        )
        H.save(
            "attn.indexer",
            {"out": output, "layer": H.layer, "start_pos": start_pos, "offset": offset},
        )

    def wo_b_hook(attn):
        """Stash the grouped output LoRA's A result, which is wo_b's input.

        The A projection has no other observable: the reference computes it
        inline inside ``Attention.forward`` and immediately feeds it to the
        row-parallel B projection, so the only place its value is visible from
        outside is as that call's argument.
        """

        def hook(module, args, output):
            H.stash("lora_a", attn.layer_id, args[0].detach().clone())

        return hook

    def _freqs_compare(module) -> None:
        """Check the layer's own rotary table against an independent construction.

        The table is built once at construction and then indexed by every
        rotation in the layer, so a wrong theta or a wrongly-applied YaRN ramp
        is invisible at the ``apply_rotary_emb`` boundary -- both sides would
        use the same wrong table. It is compared here instead, once per ratio,
        against the reference built from the config's own numbers.
        """
        ratio = module.compress_ratio
        tag = f"rope.freqs_cis.r{ratio}"
        if not H.active_once(tag):
            return
        table = module.freqs_cis
        seqlen, device = table.size(0), table.device
        rd = module.rope_head_dim
        # A compressed layer uses YaRN over the training context at the longer
        # theta; a window-only layer disables YaRN entirely and keeps the base
        # theta. Both halves of that rule are checked, each against the other
        # as its control.
        yarn_len = cfg.original_seq_len if ratio else 0
        theta = cfg.compress_rope_theta if ratio else cfg.rope_theta
        want = R.precompute_freqs_cis_ref(
            rd, seqlen, yarn_len, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow, device
        )
        H.compare(tag, want, table)
        other_len = 0 if ratio else cfg.original_seq_len
        other_theta = cfg.rope_theta if ratio else cfg.compress_rope_theta
        H.compare(
            tag,
            R.precompute_freqs_cis_ref(
                rd,
                seqlen,
                other_len,
                other_theta,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
                device,
            ),
            table,
            kind="other-rope-policy",
        )

    def attention_hook(module, args, output):
        x, start_pos = args[0], args[1]
        H.stash("attn_in", module.layer_id, x.detach().clone())
        H.stash("attn_out", module.layer_id, output.detach().clone())
        _freqs_compare(module)
        ratio = module.compress_ratio
        phase = "prefill" if start_pos == 0 else "decode"
        # Run-long budget: this composition is four full-width reference GEMMs
        # plus a tiled sparse attention, and the per-forward budget would drive
        # it once per decode step per fixture. One measurement per (ratio,
        # phase) covers every regime the forward has.
        if not H.active_once(f"attn.module.r{ratio}.{phase}"):
            return
        stashed = H.native.get(("sparse_kv", module.layer_id))
        if stashed is None:
            H.report.note(
                f"attn.module.r{ratio}: no sparse_attn call was seen at layer {module.layer_id}"
            )
            return
        kv, topk_idxs = stashed
        seqlen = x.size(1)
        freqs = module.freqs_cis[start_pos : start_pos + seqlen]
        wo_b_scale = getattr(module.wo_b.weight, "scale", None)
        partial, lora_a, o_raw = R.attention_module_ref(
            x,
            module.wq_a.weight,
            getattr(module.wq_a.weight, "scale", None),
            module.q_norm.weight,
            module.wq_b.weight,
            getattr(module.wq_b.weight, "scale", None),
            module.attn_sink,
            module.wo_a.weight,
            module.wo_b.weight,
            wo_b_scale,
            kv,
            topk_idxs,
            freqs,
            module.n_local_heads,
            module.head_dim,
            module.rope_head_dim,
            module.n_local_groups,
            module.o_lora_rank,
            module.softmax_scale,
            module.eps,
        )

        def reduced(value: torch.Tensor) -> torch.Tensor:
            value = value.clone()
            if _world() > 1:
                torch.distributed.all_reduce(value)
            return value.to(x.dtype)

        tag = f"attn.module.r{ratio}"
        H.compare(tag, reduced(partial), output)
        # control: the query's rotation left on the attention output. The
        # inverse rotation is the step a port omits, and omitting it is
        # invisible in every other boundary here.
        bad_a = R.output_lora_a_ref(
            o_raw.flatten(2), module.wo_a.weight, module.n_local_groups, module.o_lora_rank
        )
        bad = R.linear_ref(
            bad_a.flatten(2), module.wo_b.weight, wo_b_scale, out_dtype=x.dtype
        ).float()
        H.compare(tag, reduced(bad), output, kind="no-inverse-rope")
        # control: the q-LoRA norm dropped. A second, independent way to be
        # wrong here, and the reason this boundary's bound is defensible: it
        # perturbs the chain at its start rather than at its end, so it tests
        # that the composition is not merely insensitive to its own inputs.
        no_qnorm, _, _ = R.attention_module_ref(
            x,
            module.wq_a.weight,
            getattr(module.wq_a.weight, "scale", None),
            torch.ones_like(module.q_norm.weight),
            module.wq_b.weight,
            getattr(module.wq_b.weight, "scale", None),
            module.attn_sink,
            module.wo_a.weight,
            module.wo_b.weight,
            wo_b_scale,
            kv,
            topk_idxs,
            freqs,
            module.n_local_heads,
            module.head_dim,
            module.rope_head_dim,
            module.n_local_groups,
            module.o_lora_rank,
            module.softmax_scale,
            module.eps,
        )
        H.compare(tag, reduced(no_qnorm), output, kind="flat-q-norm-weight")

        native_a = H.native.get(("lora_a", module.layer_id))
        if native_a is not None and native_a.shape == lora_a.flatten(2).shape:
            H.compare("attn.output_lora.a", lora_a.flatten(2), native_a)
            # Goal 1.2's parity leg replays this one through `gemm/bmm_out`, so
            # it needs the grouped A projection's real input -- the inverse-
            # rotated sparse-attention output -- and the group geometry. The
            # weight is not saved; the parity driver reads
            # `layers.<L>.attn.wo_a.weight` from the raw checkpoint.
            if _parity_wanted(H, "parity.dense.output_lora_a"):
                H.save(
                    "parity.dense.output_lora_a",
                    {
                        "module": f"layers.{module.layer_id}.attn.wo_a",
                        "kind": "output_lora_a",
                        "x": R.apply_rope_tail_ref(
                            o_raw, freqs, module.rope_head_dim, inverse=True
                        ).flatten(2),
                        "out": native_a,
                        "n_local_groups": module.n_local_groups,
                        "o_lora_rank": module.o_lora_rank,
                        "fixture": H.fixture,
                        "rank": _rank(),
                        "world_size": _world(),
                    },
                    cap=_PARITY_MAX_FIXTURES,
                )
            # control: the flat [groups*rank, heads_per_group*head_dim] weight
            # unpacked rank-major instead of group-major. Reading it as one
            # dense projection -- the V3-family output projection -- is the
            # other misreading, but that one does not even multiply here
            # (measured: 21x2048 against 256x512), so the shape catches it and
            # it is not a control. This one reads the same bytes into the same
            # shape and differs only in which head group each rank row belongs
            # to, which is exactly the mistake a shape check cannot see.
            g, r = module.n_local_groups, module.o_lora_rank
            wrong_a = module.wo_a.weight.reshape(r, g, -1).transpose(0, 1).reshape(g * r, -1)
            H.compare(
                "attn.output_lora.a",
                R.output_lora_a_ref(
                    R.apply_rope_tail_ref(o_raw, freqs, module.rope_head_dim, inverse=True).flatten(
                        2
                    ),
                    wrong_a,
                    g,
                    r,
                ).flatten(2),
                native_a,
                kind="rank-major-groups",
            )

    def block_hook(module, args, output):
        layer = module.layer_id
        x, pre_mix = args[0], args[2]
        if layer == 0 and H.active("hc.identity_pre_mix"):
            # The first block's attention consumes a one-hot stream selector
            # rather than a learned mix; every later block consumes what the
            # previous FFN produced. Getting this wrong starts the whole
            # residual schedule on the wrong stream.
            H.compare(
                "hc.identity_pre_mix",
                R.identity_pre_mix_ref(x.size(0), x.size(1), module.hc_mult, x.device),
                pre_mix,
            )
        if not H.active("block.module"):
            return
        attn_in = H.native.get(("attn_in", layer))
        attn_out = H.native.get(("attn_out", layer))
        ffn_in = H.native.get(("ffn_in", layer))
        ffn_out = H.native.get(("ffn_out", layer))
        mixed = H.native.get(("hc_mix_in", layer)) or []
        if (
            attn_in is None
            or attn_out is None
            or ffn_in is None
            or ffn_out is None
            or len(mixed) != 2
        ):
            H.report.note(f"block.module: layer {layer} did not expose both sublayers")
            return
        mid = mixed[1]
        kwargs = dict(
            hc_attn=(module.hc_attn_fn, module.hc_attn_scale, module.hc_attn_base),
            hc_ffn=(module.hc_ffn_fn, module.hc_ffn_scale, module.hc_ffn_base),
            attn_norm_weight=module.attn_norm.weight,
            ffn_norm_weight=module.ffn_norm.weight,
            norm_eps=module.norm_eps,
            hc_mult=module.hc_mult,
            sinkhorn_iters=module.hc_sinkhorn_iters,
            hc_eps=module.hc_eps,
        )
        comp = R.block_module_ref(x, pre_mix, attn_out, ffn_out, mid=mid, **kwargs)
        H.compare("block.mid", comp["mid_composed"], mid)
        H.compare("block.attn_input", comp["attn_input"], attn_in)
        H.compare("block.ffn_input", comp["ffn_input"], ffn_in)
        H.compare("block.module", comp["out"], output[0])
        H.compare("block.next_pre_mix", comp["next_pre_mix"], output[1])
        # control: the schedule not delayed -- each sublayer consuming the mix
        # it produced itself. This is the failure mode the source's own comment
        # warns about, and it changes nothing else in the block, so the two
        # sublayer inputs are the only place it can show up.
        bad = R.block_module_ref(
            x, pre_mix, attn_out, ffn_out, mid=mid, immediate_pre=True, **kwargs
        )
        H.compare("block.attn_input", bad["attn_input"], attn_in, kind="immediate-pre")
        H.compare("block.ffn_input", bad["ffn_input"], ffn_in, kind="immediate-pre")
        # control: the block returning the attention's pre instead of the FFN's,
        # i.e. handing the next block a mix from one sublayer too early.
        H.compare(
            "block.next_pre_mix",
            R.hc_split_sinkhorn_ref(
                R.hc_mix_projection_ref(x, module.hc_attn_fn, module.norm_eps),
                module.hc_attn_scale,
                module.hc_attn_base,
                module.hc_mult,
                module.hc_sinkhorn_iters,
                module.hc_eps,
            )[0],
            output[1],
            kind="attn-pre-not-ffn-pre",
        )
        # control: comb contracted over the destination copy instead of the
        # source copy. Square and doubly stochastic, so neither the shape nor a
        # row-sum check can see it.
        H.compare(
            "block.module",
            R.hc_post_ref(ffn_out, mid, comp["ffn_post"], comp["ffn_comb"].transpose(-1, -2)),
            output[0],
            kind="transposed-comb",
        )

    def moe_hook(module, args, output):
        # Stashed before the gate: the block composition needs both sublayer
        # results whether or not the MoE boundary itself drew a budget slot.
        H.stash("ffn_in", module.layer_id, args[0].detach().clone())
        H.stash("ffn_out", module.layer_id, output.detach().clone())
        if not H.active("moe.module"):
            return
        x = args[0].view(-1, module.dim)

        def params(expert_id):
            e = module.experts[expert_id]
            if e is None:
                return None
            return (
                e.w1.weight,
                getattr(e.w1.weight, "scale", None),
                e.w2.weight,
                getattr(e.w2.weight, "scale", None),
                e.w3.weight,
                getattr(e.w3.weight, "scale", None),
            )

        se = module.shared_experts
        routed, shared, _, _ = R.moe_module_ref(
            x,
            module.gate.weight,
            module.gate.bias,
            params,
            module.experts_start_idx,
            module.experts_end_idx,
            (
                se.w1.weight,
                getattr(se.w1.weight, "scale", None),
                se.w2.weight,
                getattr(se.w2.weight, "scale", None),
                se.w3.weight,
                getattr(se.w3.weight, "scale", None),
            ),
            module.gate.topk,
            module.gate.score_func,
            module.gate.gate_temp,
            module.gate.norm_topk_prob,
            module.gate.route_scale,
            se.swiglu_limit,
        )
        # The reference all-reduces the routed partials and only then adds the
        # replicated shared expert; folding it in before the reduction would
        # count it once per rank.
        if _world() > 1:
            torch.distributed.all_reduce(routed)
        y = routed + shared
        H.compare("moe.module", y.type_as(args[0]).view(args[0].shape), output)
        H.compare(
            "moe.module",
            routed.type_as(args[0]).view(args[0].shape),
            output,
            kind="no-shared",
        )

    def block_pre_hook(module, args):
        H.layer = module.layer_id
        return None

    # -- install -----------------------------------------------------------
    H.patch(ref_module, "act_quant", wrap_act_quant)
    H.patch(ref_module, "fp4_act_quant", wrap_fp4_act_quant)
    H.patch(ref_module, "linear", wrap_linear)
    H.patch(ref_module, "apply_rotary_emb", wrap_rope)
    H.patch(ref_module, "hc_split_sinkhorn", wrap_sinkhorn)
    H.patch(ref_module, "sparse_attn", wrap_sparse_attn)
    H.patch(ref_module, "select_candidate_blocks", wrap_candidates)
    H.patch(ref_module, "get_window_topk_idxs", wrap_window_idxs)
    H.patch(ref_module.Block, "hc_mixes", wrap_hc_mixes)
    H.patch(ref_module.Block, "hc_pre", wrap_hc_pre)
    H.patch(ref_module.Block, "hc_post", wrap_hc_post)

    # Walk the TEXT BACKBONE ONLY, by reaching the roots this task's scope
    # names rather than by scanning every module in the model.
    #
    # This is not tidiness. `inference/vision.py` defines its own `Attention`,
    # `Block`, `RMSNorm` and `MLP`, and a whole-model scan by class name
    # matches the ViT's classes too. Measured: at mp4 the walk reached the ViT
    # and died with `'Attention' object has no attribute 'wo_b'` -- the vision
    # tower's attention is a different module with the same name. It never
    # runs in the text path, so until a composition needed a submodule the
    # collision was invisible.
    #
    # `model.mtp` (DSpark) is excluded for the same reason it is out of scope:
    # it is a draft path this Stage does not build. Its stages alias the
    # backbone embed and head, which the id() dedupe keeps from being hooked
    # twice.
    roots = [model.embed, model.norm, model.head, *model.layers]
    if getattr(model, "engram_hash", None) is not None:
        roots.append(model.engram_hash)
    walked: set[int] = set()
    backbone = []
    for root in roots:
        for module in root.modules():
            if id(module) not in walked:
                walked.add(id(module))
                backbone.append(module)

    _install_dense_parity_capture(H, model)

    for module in backbone:
        name = type(module).__name__
        if name == "RMSNorm":
            H.hook(module, rmsnorm_hook)
        elif name == "ParallelEmbedding":
            H.hook(module, embed_hook)
        elif name == "ParallelHead":
            H.hook(module, head_hook)
        elif name == "Gate":
            H.hook(module, gate_hook)
        elif name == "Expert":
            H.hook(module, expert_hook)
        elif name == "MoE":
            H.hook(module, moe_hook)
        elif name == "ParallelEngramEmbedding":
            H.hook(module, engram_embed_hook)
        elif name == "Engram":
            H.hook(module, engram_hook)
        elif name == "NgramHashState":
            H.hook(module, ngram_hook)
        elif name == "Compressor":
            H._undo.append((module.register_forward_pre_hook(compressor_pre_hook), None, None))
            H.hook(module, compressor_hook)
        elif name == "Indexer":
            H.hook(module, indexer_hook)
        elif name == "Attention":
            H.hook(module, attention_hook)
            H.hook(module.wo_b, wo_b_hook(module))
        elif name == "Block":
            H._undo.append((module.register_forward_pre_hook(block_pre_hook), None, None))
            H.hook(module, block_hook)


#: Module-name suffixes whose real prefill activation Goal 1.2's parity leg
#: replays through the target's catalog wrappers. Each is captured under its own
#: tag so one call per module survives the run-long budget, and each name is
#: exactly the RAW checkpoint key prefix the target's `weights.py` reads
#: (`layers.2.attn.wq_a` -> `layers.2.attn.wq_a.weight`), so the parity driver
#: never has to guess which weight produced a captured activation.
#:
#: Only the dense/norm/embedding/head/output-LoRA set: this is Goal 1.2's
#: module, and capturing more would be another Goal's evidence.
_DENSE_PARITY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("attn_norm", "rmsnorm"),
    ("ffn_norm", "rmsnorm"),
    ("attn.q_norm", "rmsnorm"),
    ("attn.kv_norm", "rmsnorm"),
    ("attn.indexer.k_norm", "rmsnorm"),
    ("attn.compressor.norm", "rmsnorm"),
    ("attn.wq_a", "linear"),
    ("attn.wq_b", "linear"),
    ("attn.wkv", "linear"),
    ("attn.wo_b", "linear"),
    ("attn.indexer.wq_b", "linear"),
    ("ffn.shared_experts.w1", "linear"),
    ("ffn.shared_experts.w2", "linear"),
    ("ffn.shared_experts.w3", "linear"),
)


#: Upper bound on parity captures per module: one prefill per pinned prompt.
#: `full` mode runs five, and the bound is stated rather than inferred so a
#: `--num-prompts` above it truncates visibly instead of silently.
_PARITY_MAX_FIXTURES = 5


def _parity_wanted(harness: "Harness", tag: str) -> bool:
    """One capture per (module, fixture), and only from that fixture's PREFILL.

    An earlier revision budgeted `cap=1` per module over the whole run, which
    froze the first 12-token prompt and silently made rung 3 a one-prompt
    check -- a capture audit found all 138 dense tags with cardinality exactly
    one. Budgeting per fixture is what makes "replayed on the pinned prompts"
    true rather than nearly true.

    Decode steps are excluded deliberately: this module's parity is about wiring
    and geometry, and a one-token decode row exercises neither the multi-row
    path nor anything a prefill row does not. Goal 1.7's state canary is where
    decode lifecycle belongs.
    """
    if not harness.is_prefill or harness.fixture < 0:
        return False
    return harness.active_once(f"{tag}@f{harness.fixture}", cap=1)


def _install_dense_parity_capture(harness: "Harness", model) -> None:
    """Save one real prefill activation per dense/norm/head module, keyed by checkpoint name.

    Goal 1.2's parity half needs the target's composition driven on the SAME
    inputs the native implementation saw on the pinned prompts, and the existing
    boundary comparisons run in place without keeping them. This is purely
    additive -- it registers its own hooks, compares nothing, and changes no
    verdict -- so the 56 boundary records this run produces are the same records
    it produced before.

    Weights are deliberately NOT saved. They are hundreds of megabytes per call
    and the parity driver has a better source for them: the raw checkpoint,
    which is what the target itself loads. The module's qualified name is saved
    instead, and it is the checkpoint key prefix.
    """

    def make_hook(name: str, kind: str):
        def hook(module, args, output):
            if not _parity_wanted(harness, f"parity.dense.{name}"):
                return
            payload = {
                "module": name,
                "kind": kind,
                "x": args[0],
                "out": output,
                "fixture": harness.fixture,
                "rank": _rank(),
                "world_size": _world(),
            }
            if kind == "rmsnorm":
                payload["eps"] = float(module.eps)
            else:
                payload["weight_dtype"] = str(module.weight.dtype)
                payload["weight_shape"] = tuple(module.weight.shape)
            harness.save(f"parity.dense.{name}", payload, cap=_PARITY_MAX_FIXTURES)

        return hook

    for name, module in model.named_modules():
        for suffix, kind in _DENSE_PARITY_SUFFIXES:
            if name.endswith("." + suffix) or name == suffix:
                harness.hook(module, make_hook(name, kind))
                break

    def embed_parity(module, args, output):
        if _parity_wanted(harness, "parity.dense.embed"):
            harness.save(
                "parity.dense.embed",
                {
                    "module": "embed",
                    "kind": "embed",
                    "ids": args[0],
                    "out": output,
                    "vocab_start": module.vocab_start_idx,
                    "vocab_end": module.vocab_end_idx,
                    "fixture": harness.fixture,
                    "rank": _rank(),
                    "world_size": _world(),
                },
                cap=_PARITY_MAX_FIXTURES,
            )

    def head_parity(module, args, output):
        if _parity_wanted(harness, "parity.dense.head"):
            x = args[0]
            harness.save(
                "parity.dense.head",
                {
                    "module": "head",
                    "kind": "head",
                    "x": x if len(args) > 1 and args[1] else x[:, -1],
                    "out": output,
                    "fixture": harness.fixture,
                    "rank": _rank(),
                    "world_size": _world(),
                },
                cap=_PARITY_MAX_FIXTURES,
            )

    harness.hook(model.embed, embed_parity)
    harness.hook(model.head, head_parity)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _world() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


# ---------------------------------------------------------------------------
# small mode: a toy model with the released structural constants
# ---------------------------------------------------------------------------

#: Widths are toy; every structural constant is the released one. The three
#: things this buys that the released prompts cannot: a sliding window that
#: wraps inside eight decode steps, an index top-k smaller than the number of
#: compressed positions, and candidate blocks that actually filter.
SMALL_OVERRIDES: dict[str, Any] = dict(
    max_batch_size=2,
    max_seq_len=128,
    dtype="fp8",
    expert_dtype="fp4",
    vocab_size=1024,
    dim=256,
    moe_inter_dim=256,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=16,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=128,
    head_dim=128,
    rope_head_dim=32,
    norm_eps=1e-20,
    o_groups=8,
    o_lora_rank=64,
    window_size=8,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 3, 4, 5),
    compress_rope_theta=160000.0,
    original_seq_len=65536,
    rope_theta=10000.0,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=16,
    index_head_dim=64,
    index_topk=4,
    candidate_source_layer=4,
    candidate_topk_blocks=2,
    candidate_block_size=2,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=(1, 3),
    engram_max_ngram_size=4,
    engram_n_heads=8,
    engram_head_dim=64,
    engram_pad_id=2,
    engram_vocab_size=97,
    vision_n_layers=0,
    dspark_block_size=0,
)


def _fill_small(model, seed: int) -> None:
    """Give the toy model finite, in-range weights for every stored dtype.

    ``Transformer`` allocates with ``torch.empty``, so an uninitialized fp8
    tensor is as likely to hold NaN as a number. Each dtype is filled on its own
    terms: E8M0 scales are written as biased exponent bytes (they have no
    arithmetic in torch), packed E2M1 as random nibble pairs, and norm-like
    parameters near one.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def fill(param, name):
        if param.dtype == torch.float8_e4m3fn:
            src = torch.randn(param.shape, generator=gen).mul_(0.05)
            param.data.copy_(src.to(torch.float8_e4m3fn))
        elif param.dtype == torch.float8_e8m0fnu:
            exps = torch.randint(121, 128, param.shape, generator=gen, dtype=torch.uint8)
            param.data.view(torch.uint8).copy_(exps)
        elif param.dtype == torch.float4_e2m1fn_x2:
            param.data.view(torch.uint8).copy_(
                torch.randint(0, 256, param.shape, generator=gen, dtype=torch.uint8)
            )
        elif name.endswith(("norm.weight", "q_weight", "k_weight")) or ".norm." in name:
            param.data.copy_((1.0 + 0.02 * torch.randn(param.shape, generator=gen)).to(param.dtype))
        elif name.endswith("_scale"):
            param.data.copy_(torch.ones(param.shape).to(param.dtype))
        elif name.endswith("_base"):
            param.data.copy_((0.1 * torch.randn(param.shape, generator=gen)).to(param.dtype))
        elif name.endswith("attn_sink"):
            param.data.copy_(torch.randn(param.shape, generator=gen).to(param.dtype))
        else:
            param.data.copy_((0.02 * torch.randn(param.shape, generator=gen)).to(param.dtype))

    for name, param in model.named_parameters():
        fill(param, name)
    for name, param in model.named_parameters():
        scale = getattr(param, "scale", None)
        if scale is not None and scale is not param:
            fill(scale, name + "_qscale")


def _small_args(ref_module, tokenizer, report: Report, declared: int):
    """Build the toy ``ModelArgs``, sizing the Engram tables from their own primes.

    The compressed vocab size is computed by the *reference* module's own
    normalizer chain here, and the value it produces is checked against the
    released config's ``engram_compressed_vocab_size``. That number is not a
    bound: every hash multiplier is derived from it, so a normalizer that
    collapsed one more or one fewer token would rehash the entire table.
    """
    overrides = dict(SMALL_OVERRIDES)
    _, compressed = R.compressed_token_map_ref(tokenizer)
    report.note(
        f"compressed token map: refmods {compressed} vs released config {declared}"
        f" -- {'agree' if compressed == declared else 'DISAGREE'}"
    )
    report.add(
        "engram.compressed_vocab",
        "ref",
        torch.tensor([compressed]),
        torch.tensor([declared]),
    )
    overrides["engram_compressed_vocab_size"] = compressed
    primes = R.engram_primes_ref(
        len(overrides["engram_layer_ids"]),
        overrides["engram_max_ngram_size"],
        overrides["engram_n_heads"],
        overrides["engram_vocab_size"],
    )
    overrides["engram_num_embeddings"] = tuple(
        sum(p for per in layer for p in per) for layer in primes
    )
    return ref_module.ModelArgs(**overrides)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def _log(message: str) -> None:
    if _rank() == 0:
        print(f"[refcapture {time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_small(args) -> int:
    sys.path.insert(0, args.ref_dir)
    import model as ref_model  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_ckpt)
    report = Report()
    declared = json.loads(Path(args.ref_dir, "config.json").read_text())[
        "engram_compressed_vocab_size"
    ]
    margs = _small_args(ref_model, tokenizer, report, declared)
    _log(
        f"small model: {margs.n_layers} layers, dim {margs.dim}, engram rows {margs.engram_num_embeddings}"
    )
    with torch.device("cuda"):
        model = ref_model.Transformer(margs, tokenizer)
    _fill_small(model, seed=1234)
    torch.set_default_device("cuda")

    cfg = R.config_from_dict(
        {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in SMALL_OVERRIDES.items()},
            "engram_compressed_vocab_size": margs.engram_compressed_vocab_size,
            "engram_num_embeddings": list(margs.engram_num_embeddings),
        }
    )
    harness = Harness(cfg, report, layers=None, max_calls=args.max_calls)
    install(harness, model, ref_model, tokenizer)

    torch.manual_seed(7)
    prompt = torch.randint(0, margs.vocab_size, (1, args.small_prompt_len), device="cuda")
    _log(f"prefill {prompt.shape}")
    with torch.inference_mode():
        out_ids, _, _ = model(prompt)
        for step in range(args.decode_steps):
            pos = args.small_prompt_len + step
            harness.counts.clear()
            out_ids, _, _ = model(out_ids.view(1, 1), pos)
    harness.close()
    report.note(
        f"small model: {margs.n_layers} layers, window {margs.window_size}, index_topk {margs.index_topk}"
    )
    return _finish(args, report, harness, "small")


def run_full(args) -> int:
    sys.path.insert(0, args.ref_dir)
    import anchor  # ty: ignore[unresolved-import]
    import model as ref_model  # ty: ignore[unresolved-import]

    fixtures = json.loads(Path(args.fixtures).read_text())["frozen"]
    fixtures = fixtures[: args.num_prompts]
    max_len = max(len(f["generated_token_ids"]) for f in fixtures) + 64
    model, tokenizer, world_size, rank = anchor._build_reference(
        args.ref_dir, args.ckpt_path, 1, max(256, max_len), _log
    )
    with open(os.path.join(args.ref_dir, "config.json")) as f:
        cfg = R.config_from_dict(json.load(f))

    report = Report()
    layers = set(int(x) for x in args.layers.split(",")) if args.layers else None
    harness = Harness(cfg, report, layers=layers, max_calls=args.max_calls)
    install(harness, model, ref_model, tokenizer)

    # EVERY frozen token, not a prefix of one.
    #
    # This loop used to decode `min(args.decode_steps, len(want) - 1)` times and
    # compare against `want[:len(got)]`, with `--decode-steps 8` in the job. On
    # fixtures of 7 / 24 / 48 / 2 / 48 tokens it therefore checked 7 / 9 / 9 / 2
    # / 9 and reported "reproduced token for token" -- a claim about the whole
    # continuation backed by a prefix of it, and the prefix is the part least
    # likely to diverge. The frozen length is the contract, so it drives the
    # loop; `--decode-steps` does not apply to this mode at all. Per-fixture
    # checked-versus-frozen lengths are recorded before any verdict is emitted,
    # so a future truncation is visible in the report rather than inferred.
    mismatched = []
    lengths = []
    for fixture_index, fixture in enumerate(fixtures):
        ids = tokenizer(fixture["prompt"], return_tensors="pt").input_ids.cuda()
        want = fixture["generated_token_ids"]
        got = []
        harness.fixture = fixture_index
        with torch.inference_mode():
            harness.counts.clear()
            harness.is_prefill = True
            out_ids, _, _ = model(ids)
            harness.is_prefill = False
            got.append(int(out_ids[0].item()))
            for step in range(len(want) - 1):
                harness.counts.clear()
                out_ids, _, _ = model(out_ids.view(1, 1), ids.size(1) + step)
                got.append(int(out_ids[0].item()))
        assert len(got) == len(want), (
            f"fixture {fixture['prompt']!r}: generated {len(got)} tokens against a frozen "
            f"continuation of {len(want)} -- the transparency check may not compare a prefix"
        )
        lengths.append((fixture["prompt"], len(got), len(want)))
        if got != want:
            mismatched.append({"prompt": fixture["prompt"], "want": want, "got": got})
        _log(
            f"fixture {fixture['prompt']!r}: {len(got)} of {len(want)} frozen tokens, "
            f"{'match' if got == want else 'MISMATCH'}"
        )
    harness.close()

    report.note(
        "fixture lengths checked/frozen: "
        + ", ".join(f"{n}/{m}" for _, n, m in lengths)
        + f" ({sum(n for _, n, _ in lengths)} tokens generated under instrumentation)"
    )
    if mismatched:
        report.note(
            f"INSTRUMENTATION NOT TRANSPARENT: {len(mismatched)} of {len(fixtures)} fixtures diverged"
        )
        for m in mismatched:
            report.note(f"  {m['prompt']!r} want {m['want']} got {m['got']}")
    elif any(n != m for _, n, m in lengths):
        report.note("INSTRUMENTATION CHECK INCOMPLETE: a fixture was compared on a prefix")
    else:
        report.note(
            f"instrumentation transparent: {len(fixtures)} frozen fixtures reproduced in full, "
            f"every one of their {sum(m for _, _, m in lengths)} tokens"
        )
    rc = _finish(args, report, harness, "full")
    if mismatched or any(n != m for _, n, m in lengths):
        return rc or 3
    return rc


# ---------------------------------------------------------------------------
# what a run must have measured
#
# A ladder rung that exits 0 because it compared *something* is not a rung. The
# previous version required only that at least one boundary ran, and three
# module references -- the rotary table, the grouped output LoRA and the
# identity pre-mix -- were never driven at all while the command stayed green.
# So the set is written down, per mode, and a missing entry fails.
# ---------------------------------------------------------------------------

#: Boundaries every mode must compare. Named as the module Goals will cite them.
_REQUIRED_BOUNDARIES: tuple[str, ...] = (
    # dense / norm / head (Goal 1.2)
    "linear.fp8",
    "linear.fp4",
    "linear.plain",
    "norm.rmsnorm",
    "embed.shard",
    "head.logits",
    "quant.act_quant.values",
    "quant.act_quant.scales",
    "quant.act_quant.inplace",
    "quant.fp4_act_quant.e8m0",
    "quant.fp4_act_quant.e4m3",
    # hyper-connections (Goal 1.3)
    "hc.mixes.pre",
    "hc.mixes.post",
    "hc.mixes.comb",
    "hc.split_sinkhorn.pre",
    "hc.split_sinkhorn.post",
    "hc.split_sinkhorn.comb",
    "hc.pre",
    "hc.post",
    "hc.identity_pre_mix",
    "block.attn_input",
    "block.ffn_input",
    "block.mid",
    "block.module",
    "block.next_pre_mix",
    # engram (Goal 1.4)
    "engram.primes",
    "engram.offsets",
    "engram.multipliers",
    "engram.token_map",
    "engram.compressed_vocab",
    "engram.pad_id",
    "engram.hash",
    "engram.lookup",
    "engram.module",
    # sparse attention (Goal 1.5)
    "rope.forward",
    "rope.inverse",
    "rope.freqs_cis.r0",
    "rope.freqs_cis.r1",
    "rope.freqs_cis.r2",
    "attn.window_idxs",
    "attn.compressor.ratio1",
    "attn.compressor.ratio2.prefill",
    "attn.compressor.ratio2.decode",
    "attn.compressor.ratio2.decode.held",
    "attn.indexer.k",
    "attn.indexer.topk",
    "attn.candidates",
    "attn.sparse_attn",
    "attn.output_lora.a",
    "attn.module.r0",
    "attn.module.r1",
    "attn.module.r2",
    # routed / shared MoE (Goal 1.6)
    "moe.gate.indices",
    "moe.gate.weights",
    "moe.expert",
    "moe.module",
)

#: Controls that must be present. A boundary whose only evidence is a tolerance
#: is evidence of nothing, so the ones that carry each module's real failure
#: mode are named rather than left to whichever branch happened to run.
_REQUIRED_CONTROLS: tuple[tuple[str, str], ...] = (
    ("norm.rmsnorm", "mean-centered"),
    ("norm.rmsnorm", "bf16-statistic"),
    ("linear.fp8", "rolled-weight-scale"),
    ("linear.fp4", "rolled-weight-scale"),
    ("quant.fp4_act_quant.e8m0", "wrong-scale-format"),
    ("quant.fp4_act_quant.e4m3", "wrong-scale-format"),
    ("rope.forward", "neox-pairing"),
    ("rope.inverse", "neox-pairing"),
    ("rope.freqs_cis.r0", "other-rope-policy"),
    ("rope.freqs_cis.r1", "other-rope-policy"),
    ("rope.freqs_cis.r2", "other-rope-policy"),
    ("hc.split_sinkhorn.comb", "transposed-comb"),
    ("hc.split_sinkhorn.comb", "one-iteration"),
    ("hc.pre", "permuted-mix"),
    ("hc.post", "transposed-comb"),
    ("block.attn_input", "immediate-pre"),
    ("block.ffn_input", "immediate-pre"),
    ("block.module", "transposed-comb"),
    ("block.next_pre_mix", "attn-pre-not-ffn-pre"),
    ("engram.module", "unsigned-sqrt"),
    ("engram.module", "no-sqrt"),
    ("engram.pad_id", "raw-not-compressed"),
    ("attn.candidates", "no-force-newest"),
    ("attn.indexer.topk", "no-relu"),
    ("attn.sparse_attn", "no-sink"),
    ("attn.sparse_attn", "dropped-row"),
    ("attn.output_lora.a", "rank-major-groups"),
    ("attn.module.r0", "no-inverse-rope"),
    ("attn.module.r1", "no-inverse-rope"),
    ("attn.module.r2", "no-inverse-rope"),
    ("attn.module.r0", "flat-q-norm-weight"),
    ("attn.module.r1", "flat-q-norm-weight"),
    ("attn.module.r2", "flat-q-norm-weight"),
    ("attn.compressor.ratio2.prefill", "mean-pooling"),
    ("attn.compressor.ratio2.decode", "newest-slot-only"),
    ("moe.gate.indices", "sigmoid-scoring"),
    ("moe.gate.weights", "bias-scaled-weights"),
    ("moe.expert", "swapped-w1-w3"),
    ("moe.expert", "no-clamp"),
    ("moe.module", "no-shared"),
)


def _manifest_failures(report: Report) -> list[str]:
    """Everything a complete run owes that this run did not deliver.

    Three kinds, and the third is the one that used to be a printed warning:

      missing boundary   a required module reference was never driven
      missing control    a required wrong variant was never driven
      blind control      a control ran, did not discriminate, and no regime
                         note explains why. That is indistinguishable from a
                         harness that cannot see the defect, so it fails.
    """
    out: list[str] = []
    measured = {tag for tag, kind in report.rows if kind == "ref"}
    for tag in _REQUIRED_BOUNDARIES:
        if tag not in measured:
            out.append(f"missing boundary {tag}")
    for tag, kind in _REQUIRED_CONTROLS:
        if (tag, kind) not in report.rows:
            out.append(f"missing control {tag}/{kind}")
    for (tag, kind), row in sorted(report.rows.items()):
        if kind == "ref" or kind.startswith("probe-"):
            continue
        if row["passes"] and tag not in report.regimes:
            out.append(f"blind control {tag}/{kind} (passed the gate, no regime note)")
    return out


def _finish(args, report: Report, harness: Harness, mode: str) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selftest_ok, selftest_lines = comparator_selftest()
    manifest = _manifest_failures(report)
    dtype_bad = report.dtype_mismatches()
    payload = {
        "kind": "deepseek-v41-flash-reference-ladder/2",
        "mode": mode,
        "rank": _rank(),
        "world_size": _world(),
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "boundaries": report.as_dict(),
        "notes": report.notes,
        "regimes": sorted(report.regimes),
        "indistinct_controls": report.indistinct_controls(),
        "comparator_selftest": {"ok": selftest_ok, "cases": selftest_lines},
        "manifest_failures": manifest,
        "dtype_mismatches": dtype_bad,
        "required_boundaries": list(_REQUIRED_BOUNDARIES),
        "required_controls": [f"{t}/{k}" for t, k in _REQUIRED_CONTROLS],
    }
    stem = f"refladder-{mode}-{args.tag}-rank{_rank()}"
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if harness.saved and args.save_activations:
        torch.save(harness.saved, out_dir / f"{stem}-activations.pt")
    if _rank() == 0:
        print(report.render(), flush=True)
        for note in report.notes:
            print(f"note: {note}", flush=True)
        print("comparator self-test:", flush=True)
        for line in selftest_lines:
            print(f"  {line}", flush=True)
    failures = report.failures()
    covered = sorted({tag for tag, kind in report.rows if kind == "ref"})
    print(
        f"rank {_rank()}: {len(covered)} boundaries compared, {len(failures)} outside gate, "
        f"{len(manifest)} manifest failures, {len(dtype_bad)} dtype mismatches, "
        f"selftest {'ok' if selftest_ok else 'BROKEN'}",
        flush=True,
    )
    rc = 0
    if not selftest_ok:
        # Nothing else in this file means anything if the comparator is blind,
        # so this is reported before any boundary result.
        print(f"rank {_rank()}: FAIL comparator self-test", flush=True)
        rc = 1
    if failures:
        print(f"rank {_rank()}: FAIL {' '.join(failures)}", flush=True)
        rc = 1
    if dtype_bad:
        # Before the numerical verdict: a boundary handed two incomparable
        # tensors produced no measurement at all, so nothing downstream of it
        # should be read as one.
        for line in dtype_bad:
            print(f"rank {_rank()}: FAIL dtype mismatch {line}", flush=True)
        rc = 1
    if manifest:
        for line in manifest:
            print(f"rank {_rank()}: FAIL {line}", flush=True)
        rc = 1
    if not covered:
        print(f"rank {_rank()}: FAIL no boundary was compared at all", flush=True)
        rc = 1
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("small", "full"), default="small")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--ckpt-path", default="")
    p.add_argument("--hf-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fixtures", default="")
    p.add_argument("--tag", default="t0")
    p.add_argument("--layers", default="", help="comma-separated layer ids to record; empty = all")
    p.add_argument("--max-calls", type=int, default=4, help="records per boundary per forward")
    p.add_argument(
        "--decode-steps",
        type=int,
        default=8,
        help="decode steps in --mode small. --mode full ignores this: its step count is "
        "the frozen fixtures' own length, so the transparency check cannot be truncated "
        "by a job flag",
    )
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--small-prompt-len", type=int, default=21)
    p.add_argument("--save-activations", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "the reference ladder is a GPU measurement"
    if args.mode == "full":
        assert args.fixtures and args.ckpt_path, "--mode full needs --fixtures and --ckpt-path"
        rc = run_full(args)
    else:
        rc = run_small(args)
    with contextlib.suppress(Exception):
        if _world() > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())


# Referenced by the reference ladder's documentation; kept here so the file is
# self-describing about what a "module" means in this codebase's Goal split.
MODULE_GOALS = {
    "1.2 dense/norm/head": (
        "linear.fp8",
        "linear.fp4",
        "linear.plain",
        "norm.rmsnorm",
        "embed.shard",
        "head.logits",
        "quant.act_quant.values",
        "quant.act_quant.scales",
        "quant.act_quant.inplace",
        "quant.fp4_act_quant.e8m0",
        "quant.fp4_act_quant.e4m3",
    ),
    "1.3 hyper-connections": (
        "hc.mixes.pre",
        "hc.mixes.post",
        "hc.mixes.comb",
        "hc.split_sinkhorn.pre",
        "hc.split_sinkhorn.post",
        "hc.split_sinkhorn.comb",
        "hc.pre",
        "hc.post",
        "hc.identity_pre_mix",
        "block.attn_input",
        "block.ffn_input",
        "block.mid",
        "block.module",
        "block.next_pre_mix",
    ),
    "1.4 engram": (
        "engram.primes",
        "engram.offsets",
        "engram.multipliers",
        "engram.token_map",
        "engram.compressed_vocab",
        "engram.pad_id",
        "engram.hash",
        "engram.lookup",
        "engram.module",
    ),
    "1.5 sparse attention": (
        "rope.forward",
        "rope.inverse",
        "rope.freqs_cis.r0",
        "rope.freqs_cis.r1",
        "rope.freqs_cis.r2",
        "attn.window_idxs",
        "attn.compressor.ratio1",
        "attn.compressor.ratio2.prefill",
        "attn.compressor.ratio2.decode",
        "attn.compressor.ratio2.decode.held",
        "attn.indexer.k",
        "attn.indexer.topk",
        "attn.candidates",
        "attn.sparse_attn",
        "attn.output_lora.a",
        "attn.module.r0",
        "attn.module.r1",
        "attn.module.r2",
    ),
    "1.6 routed/shared MoE": ("moe.gate.indices", "moe.gate.weights", "moe.expert", "moe.module"),
}

assert set(_REQUIRED_BOUNDARIES) == {tag for tags in MODULE_GOALS.values() for tag in tags}, (
    "the required-boundary manifest and the module-Goal map must name the same boundaries"
)
