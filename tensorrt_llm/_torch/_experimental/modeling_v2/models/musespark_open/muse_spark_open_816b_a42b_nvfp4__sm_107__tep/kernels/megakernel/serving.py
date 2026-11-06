# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The served decode step's routed layers as whole-layer megakernel launches (the ``megakernel`` fast path).

On the configuration that serves the ``megakernel`` fast path, ``DecodeTarget`` runs target layers ``1 .. n - 2`` (every
NVFP4 routed layer; 1..60 of the 62-layer model) of a decode forward as the whole-layer launches of
``layer_mk.LayerKernels``: a sliding-window layer is one launch, a full-attention layer one (``full_one``). Only the
forward the kernel is built for takes them: sm_107, TP8 with EP8 over two nodes of four, no context rows, one sequence,
8 rows (one token and seven drafts). Every other forward (prefill and mixed batches, other row counts, a step whose
metadata was first seen inside a capture) runs the stock layers. A forward runs the kernel on all replaced layers or on
none: each launch empties the buffers of a later launch in the copy rotation. The ``megakernel`` lever installs it;
the served build carries the numerics-budget levers (``levers.MEGAKERNEL_NUMERICS``), which the lever requires, so
turning one of them off runs the stock layers.

Lifecycle:

* ``install`` (the shell's post-load hook, after every stock fast path and the drafter's, on every rank): compiles
  the library (rank 0 first), builds the kernels' workspaces -- the MoE hops' multicast buffer is created here, after
  every stock multicast workspace of the served path -- and packs each replaced layer's GEMV weights once (repacked
  copies; the stock weights stay for prefill). This happens before the KV cache takes its share of the free memory;
  the bytes are logged.
* ``begin`` (per decode forward): the decision, then the parameter blocks of the forward's binding -- the attention
  metadata's block tables and KV lengths, the positions buffer and the DFlash capture buffer, all bound by address --
  made eagerly the first time the binding is seen (the CUDA-graph runner's eager warm-up forward precedes each capture
  with the same metadata and static inputs); a binding first seen inside a capture runs stock.
* ``scope`` (the forward's layer loop): the layer before the first replaced one does not fork its next-QKV L2 prefetch
  (it would run beside the persistent kernel, which needs every SM).
* ``run``: joins any outstanding dense-GEMM prefetch fork, then launches every replaced layer and returns the
  residual and the next layer's normed attention input, as the stock layer ``n - 2`` would.

``DecodeTarget`` also leaves the unpool and pool direct transports unarmed on a megakernel forward: their buffer
sequence assumes every fused layer calls them, and the only stock caller left (layer 0's post-FFN boundary) then runs
the tagged-inbox path, which is call-count safe.

Edge layers (``ServedMegakernel(edges=True)``; the served configuration keeps them stock): layers 0 and ``n - 1`` (the
BF16-expert layers) run on the kernel too, with the BF16 experts stage (``stages/bf16Experts.cuh``, a ``*_z1`` build),
so every layer of the forward is a kernel launch. Layer 0 starts from the stock pre-QKV norm of the embedding residual
(one op before its launch); layer ``n - 1`` is a one-launch full layer whose last boundary applies the model's final
norm (``kIFlagZ1FinalNorm``), so its x output is the final hidden state and ``core.norm`` is not run.

Copy plan: the launches of a forward cycle the main workspace's 4 copies by ``copy_plan`` (the served 60 launches are
the plain ``p % 4`` rotation; with the edge layers, 62 launches, which 4 does not divide: 14 blocks ``0 1 2 3`` then 2
blocks ``0 1 2``, so any three consecutive launches use distinct copies), and each layer builds only the parameter
block of the copy it always runs on.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch

#: The library build the launches use (``library.MODES``): the lean build (no watchdog, no stamps) with the served flag
#: words fixed at compile time (``icfpoxaft``: a launch with other experts or full-attention bits traps) and the
#: per-boundary levers nb 3, bh 11, pb 15, rp 1, rl 1, ah 903, xp 40, km 8 and lm 2 (the lever comments below). nb, bh
#: and rl are the numerics-budget levers of the ``megakernel`` lever (``levers.MEGAKERNEL_NUMERICS``); ``served_mode``
#: composes another build from mask settings, for the tests and for adding a build.
DEFAULT_MODE = "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_lm2_s4"
# ---- the whole-layer kernel's layer options ----
#: kPIFlags bits ORed into every sliding-window layer: the leading units before the wait, light entry hooks and warp-7
#: issue (``layer_mk.IFLAG_I2B_PRE_WAIT`` / ``IFLAG_I2B_ENTRY_LITE`` / ``IFLAG_I2B_ISSUE_WARP7`` = ``0x700000``).
DEFAULT_I2B_FLAGS = 0x700000
# The sliding-window layers convert their pages to the BF16 tiles before the Q poll
# (``layer_mk.IFLAG_W1_SWA_PRE_CONVERT``, bit 60).
DEFAULT_I2B_FLAGS |= 1 << 60
# ---- end layer options ----
# Layers 0 and n - 1 on the kernel (the BF16 experts stage; ``ServedMegakernel(edges=True)``; the served configuration
# keeps them stock).
#: The build with the edge layers (the BF16 stage needs ``MSO_MK_LANES`` bit 16; four k-blocks per request).
DEFAULT_EDGES_MODE = "lean_z1u4_s4"
#: The final norm epsilon of the kernel's boundary norm (``mhRsqrt(.., 1e-5f)``).
FINAL_NORM_EPS = 1e-5
# Experts flag bits ORed into every layer's experts descriptor (``LayerKernels.moe_flags_extra``).
#: ``experts.moe_w2_cap(6) | experts.MOE_FLAG_W2_FC2_ONE_ELECT``: at most 6 weight units in flight per CTA (the ring
#: cap) and the FC2 h-scale copies inside the MMA block (both bitwise vs flags 0). Replaced by the value below.
DEFAULT_MOE_FLAGS = 0x46000000000
#: ``experts.MOE_FLAG_W2_FC2_ONE_ELECT``: the FC2 h-scale copies inside the MMA block (bitwise vs flags 0). The
#: producer's in-flight bound is ``DEFAULT_CAP_MOE``'s, so the ring cap (``experts.moe_w2_cap``) is not set. Part of the
#: fixed experts word of the served build (``library.CB_MOE_FLAGS``).
DEFAULT_MOE_FLAGS = 1 << 42
#: The table-driven experts issue loops (``experts.MOE_FLAGS_LX``: the producer on the whole of warp 0, the
#: table-driven MMA warp; bitwise vs the default loops), ORed into every layer's experts flags; part of the fixed
#: experts word of the served build.
LX_MOE_FLAGS = 0x600000000  # = experts.MOE_FLAGS_LX (bits 33-34)
#: The stock persistent_experts schedule in the experts stage, ORed into every layer's experts flags. Levels
#: (``experts.MOE_FLAGS_PX_LEVELS``): 1 = the tile queue (bit 35), 2 = + the stock-shape MMA loop (bit 17), 3 = + the
#: 8-stage ring (bit 18).
PX_MOE_FLAGS = {
    0: 0,
    1: 0x800000000,
    2: 0x800020000,
    3: 0x800060000,
}  # = experts.MOE_FLAGS_PX_LEVELS
#: The served layers' level: persistent_experts' static queue of whole 128-row tiles instead of the byte cut (its rows
#: bit for bit); part of the fixed experts word of the served build.
PX_LEVEL = 3
# ---- attention-half plan options of the sliding-window layers ----
#: Comma or + list: ``offmerge`` (no o_proj on the merge CTAs, ``schedule_attn.plan_attention(oproj_off_merge=True)``),
#: a host plan option. The kernel bits of these options (``layer_mk.IFLAG_W1_*``) go through ``DEFAULT_I2B_FLAGS``.
DEFAULT_W1_PLAN = "offmerge"
W1_PLAN_OPTIONS = ("offmerge",)
# ---- end attention-half plan options ----
#: ``kPIFlags`` bits of the FaAttn stage on every one-launch full layer (``LayerKernels(f3_flags=)``): the two
#: warpgroups' softmaxes in tile order and the ring refills off the issuer (``layer_mk.F3_DEFAULT_FLAGS``,
#: ordering-only, value-identical), the FaAttn meta at kernel entry, the Q prep in QkvSum (the same statements on the
#: same BF16 words) and the deferred QKV runs' weights prefetched into L2 (``layer_mk.IFLAG_FA_EARLY_META`` /
#: ``IFLAG_FA_Q_PREP`` / ``IFLAG_FA_RUN_L2``, bits 37-39). The served build fixes bits 28-29 and 37-38 to 1.
FA_FLAGS = 0xE030000000
#: The full layers' plan: no QKV run on the 64 late CTAs before their FaAttn unit, their gate runs as second runs of
#: early CTAs and the gate sums on the late CTAs after their FaAttn unit (``layer_mk.plan_full_one(late_first=True,
#: late_sums_after=True)``; ``ServedMegakernel(fa_late_first=3)``; 1: the gate sums early, 2: the gate runs and sums
#: after the FaAttn unit). Host tables only.
FA_LATE_FIRST = 3
#: Plan options of the one-launch full layer with ``FA_LATE_FIRST`` 3, host tables only: bit 0 the FaAttn slot order
#: (``layer_mk.FH_PLAN_SLOTS``), bit 1 the FaMerge placement (``layer_mk.FH_PLAN_MERGES``), bit 2 both only on the full
#: layers with early x in (``layer_mk.FH_PLAN_STEADY``: not layer 1); bit 3 with bit 0 the slot order with every
#: FaAttn-first CTA on the low slots (``layer_mk.FT_PLAN_LATE64``). The served full layers take bits 0-2.
FH_PLAN = 7
# The per-boundary levers of the build mode (``library.LEVERS``). ``DEFAULT_MODE`` names the served levels; the mask
# settings below (keyed by the names in ``LEVER_ENV_VARS``) compose another build mode on top of it (``served_mode`` /
# ``library.compose_mode``). Each setting is a mask: unset or empty = the level the base build names, 0 = the lever off,
# n > 0 = the lever at mask n; a combination that is not a build mode in ``library.MODES`` raises. ``library.MODES``
# holds no build of the served levels with nb, bh or rl changed, so the serving path only runs ``DEFAULT_MODE``.
#: nb, one-exchange MoE boundaries (``stages/moe_hops_layout.cuh`` ``MSO_MK_NB``: 1 MhH1, 2 MhF1). Numerics contract:
#: the second norm's statistic (inv2) comes from the expanded square A + 2 inv1 B + inv1^2 C of the unrounded FP32
#: gated residual instead of the mean square of its BF16 rounding (a rounding point moves; no precision is lowered:
#: every stored value keeps its type, inv1 and the residual stream stay bitwise). Every boundary output is the
#: float64 evaluation of the expansion rounded to BF16 (within the tests' 1 ulp bound), so an output can differ from
#: the stock one by 1 ulp where the rounding point moved. Details at the macro. The served build's level is 3.
NB_ENV_VAR = "MUSESPARK_OPEN_MEGAKERNEL_NB"
#: bh, two-term pool / router consumers at H1 (``MSO_MK_BH``: 1 the branch term and the combine, 2 the residual term,
#: 4 the pool's lo planes; needs nb bit 1). Numerics contract: the pool / router GEMVs run on the two BF16 terms
#: bf16((alpha r)(w + 1)) and bf16((beta x)(w + 1)) and combine inv2 (L(rA) + inv1 L(xB)) in FP32, instead of one GEMV
#: on bf16(x_ffn) (the BF16 input rounding points and the sum order move; operands stay BF16 x BF16 with FP32
#: accumulation; inv1 / inv2 and the residual stream are nb's bits). With bits 1 and 2 the pool and logit values are
#: closer to the unrounded two-term value than on the nb x_ffn path. Details at the macro.
#: Bit 8 = pool-only consumers (no router code; an rl build served with MUSESPARK_OPEN_MEGAKERNEL_RL=all, which
#: ``ServedMegakernel`` enforces): the same values as the build without bit 8 at RL=all, only dead code leaves.
BH_ENV_VAR = "MUSESPARK_OPEN_MEGAKERNEL_BH"
#: fz, zero-exchange F1 (``MSO_MK_FZ`` = 1; needs nb bit 2). Numerics contract: none beyond nb's; MhF1 rebuilds nb's
#: row statistics from the unpool units' stock leaves in nb's tree, and every output is bitwise the fz = 0 build's
#: (the 8-rank chain test). Slower than the fz = 0 build, so the served build leaves it off.
FZ_ENV_VAR = "MUSESPARK_OPEN_MEGAKERNEL_FZ"
# The value-identical levers of the served build (``DEFAULT_MODE``), part of the ``megakernel`` fast path:
# * pb 15, B-fragment input polls with one 256-bit load per fragment (``msoMegakernel.cu`` ``MSO_MK_PB``: 1 MhUnpool's
#   m, 2 MhPool's x_ffn (with bh: the pool's term planes), 4 QkvRun's early x, 8 AhOproj's input). Value-identical: the
#   same words, lanes, registers and load strength.
# * rp 1, the experts routing state computed once per rank by CTA kCtas - 1 and polled by the others
#   (``stages/experts_defs.cuh`` ``MSO_MK_RP`` = 1). Value-identical: the same route() / place() code on the same
#   logits and a bit copy of its tables. The rl builds need it.
# * ah 903 (bits 1, 2, 4, 128, 256 and 512), the sliding-window attention front (``msoMegakernel.cu`` ``MSO_MK_AH``:
#   1, the merging SwaAttn units sum the gate from the gate runs' partials and the plan holds no gate QkvFin unit; 2,
#   the q / k / v QkvFin units run first on early CTAs whose QKV run is pure gate, host plan only; 4, with 2 only,
#   those CTAs run the fin and the run before the dependency wait; 8, with 4 only, those CTAs hold no QKV run and
#   their pure-gate runs are second runs of plain early CTAs; 16, each merging SwaAttn unit runs its merge's code
#   once dry before the real pass, in the CTA's early-x wait, or with 32 in the unit's Q wait; 128, with 1 and
#   without 16, bit 1's gate over warps 4-7 of the merging unit, one gate value per thread; 256, the SWA block pass's
#   key-half staging rows padded against bank conflicts; 512, with 128, the merging unit's key half 1 runs the whole
#   merge while half 0 publishes; 1024, with 128 and without 512, half 1 enters the merge first and polls half 0's
#   slots for it). Value-identical: bit 1 runs the gate fin's statements on the same partials in the same order (bit
#   128: per gate value on another thread), bits 2 / 8 move units between CTAs, bit 4 moves those CTAs' wait, bit 16
#   adds a pass with no global side effect, bit 256 moves shared-memory staging addresses, bit 512 moves the merge's
#   statements between threads (the same words, sums in the same order), bit 1024 moves polls between threads (the
#   same words through shared memory).
# * xp 40 (8 | 32), the experts front's X path (``stages/experts_defs.cuh`` ``MSO_MK_XP``: 1 row owners, i.e. 8 CTAs
#   poll and quantize whole pooled rows and the others poll none; 2 two owners per row; 4 the CTAs without a group poll
#   the routing state first and request their early weight units at the cut, needs rp; 8 the rp publisher runs route() /
#   place() once on zero logits before its logit poll, so its real pass runs warm code; 32 the stock queue cut of every
#   slot count computed at the stage entry and read back after the state, 64 the early requests' tile 0 from that table
#   (needs 32), 128 the hop-1 slots emptied by the next launch's MhClear instead of by MhH1, 256 the same inside
#   MhClear's one byte range, 512 row-aligned quantization shares, 1024 no share on the MhPool CTAs). Value-identical:
#   bits 32 / 64 compute the same integers earlier, bit 128 moves when a slot is emptied; a group's codes and scales are
#   the same arithmetic on the same row whichever CTA runs it (the row's sum of squares in the block-per-thread order);
#   only the CTA that runs it and the order of the requests / state poll / quantization move; the warm-up pass's words
#   are written again or cleared before any reader.
# * km 8, FaAttn's MMA issue (``stages/fullAttentionConfig.cuh`` ``MSO_MK_KM``: 8 PV(t) of a full tile and its two
#   commits, 16 QK(t) and its commit, each as one warp-collective statement with one code copy for every ring slot, and
#   qkTile's patch check as one compare per tile). Value-identical: the same MMAs, operands, accumulate flags and
#   commits by the same elected lane in the same order, and the same pages patched.
# * lm 2, the full layer's merge hand-off (``stages/fullAttentionConfig.cuh`` ``MSO_MK_LM``: 1 FaMerge's row is the
#   CTA's arrival ticket, so the 128 merges run on the CTAs whose FaAttn ends first (the host plan puts a FaMerge
#   candidate after every CTA's FaAttn, ``layer_mk.lm_ticket_plan``); 2 the gate's eight run slots are polled by the
#   merge's eight warps together with their partials and summed by warp 0, so warp 0 does not poll the gate before
#   its partials). Value-identical: the same merge code on the same row's slots in the same order whichever CTA runs it;
#   the same gate words from the same addresses and the same FP32 additions in ascending slot order from +0 (an absent
#   slot adds +0.0f, exact), the same BF16 rounding and emptying stores.
# * cf (not in the served build: its builds have no lm), the experts stage's h loader code
#   (``stages/experts_defs.cuh`` ``MSO_MK_CF``: 1 the four in-flight k-blocks of a group stored and released by one
#   rolled body, 2 the next copy's workspace clear as a rolled loop, 4 with the stock queue the slot scan in closed
#   form, every FC2 tile of the stock queue being one slot of k-blocks 0 .. 15). Value-identical: the same h words
#   polled from the same addresses until full, stored unchanged at the same shared-memory addresses before the same
#   HReady arrivals in the same order; only the code form and which integers are recomputed change.
#: The mask settings' names by ``library.LEVERS`` name (rl's is ``RL_ENV_VAR`` below: ``swa`` / ``all`` / ``1`` = level
#: 1, ``0`` = 0, an integer mask with bit 1 (``rl_mask``) = that level, unset or empty = the level the base build
#: names).
LEVER_ENV_VARS = {
    "nb": NB_ENV_VAR,
    "bh": BH_ENV_VAR,
    "fz": FZ_ENV_VAR,
    "rl": "MUSESPARK_OPEN_MEGAKERNEL_RL",  # = RL_ENV_VAR
}


def lever_levels(environ) -> Dict[str, Optional[int]]:
    """The numerics knobs' masks in ``environ`` (``LEVER_ENV_VARS``; None when a knob is unset or empty). rl's knob
    takes its run-time values: ``swa`` / ``all`` / ``1`` are level 1 (the build's rl code), ``0`` is level 0, a build
    mask with bit 1 (``rl_mask``, e.g. ``17``) is that level."""
    levels: Dict[str, Optional[int]] = {}
    for lever, var in LEVER_ENV_VARS.items():
        text = str(environ.get(var, "")).strip()
        if lever == "rl":
            if text in RL_OPTIONS:
                levels[lever] = None if not text else (1 if RL_OPTIONS[text] else 0)
            elif rl_mask(text):
                levels[lever] = rl_mask(text)
            else:
                raise ValueError(
                    f"[musespark_open megakernel] {var}={text!r}: one of {sorted(RL_OPTIONS)} "
                    "or a build mask with bit 1"
                )
            continue
        try:
            levels[lever] = int(text, 0) if text else None
        except ValueError:
            raise ValueError(
                f"[musespark_open megakernel] {var}={text!r}: an integer mask"
            ) from None
    return levels


def served_mode(base: str, environ) -> str:
    """The build mode the launches use: ``base`` (``DEFAULT_MODE`` at install) with the numerics knobs of ``environ``
    applied (``library.compose_mode``). Raises ``ValueError`` when the knobs name a combination that has no build mode.
    rl on (``swa`` / ``all``) keeps a base whose rl mask already has bit 1 (e.g. the served build's rl 1)."""
    from . import library as _library

    levels = lever_levels(environ)
    if levels.get("rl") == 1 and base in _library.MODES and _library.lever_level(base, "rl") & 1:
        levels["rl"] = None
    if all(level is None for level in levels.values()):
        return base
    try:
        return _library.compose_mode(base, levels)
    except ValueError as error:
        knobs = " ".join(
            f"{LEVER_ENV_VARS[k]}={str(environ.get(LEVER_ENV_VARS[k], '')).strip()}"
            for k, v in levels.items()
            if v is not None
        )
        raise ValueError(f"[musespark_open megakernel] mode {base} {knobs}: {error}") from None


# ---- the router logits by linearity through o_proj (``rl.py``; a build with ``library.rl_level`` bit 1) ----
#: ``swa``: the sliding-window NVFP4 layers form their router logits from o_proj partials (``rl.IFLAG_RL_ON``; FP32
#: composites, the experts stage in ``rl.experts_order()`` on those layers unless the build has the MSO_MK_RP
#: publisher); ``all`` (or ``1``): every NVFP4 layer of the kernel (full layers too); ``0``: off; unset or empty: the
#: build's rl, i.e. every NVFP4 layer when the build has ``library.rl_level`` bit 1 (the served build), else off.
#: The knob is also the rl lever of the build mode (``LEVER_ENV_VARS``): ``swa`` / ``all`` / ``1`` compose rl level 1
#: into the base build (the composed mode must exist, so rl needs the MSO_MK_RP publisher); ``0`` removes rl from the
#: build (rp stays as named); unset keeps the build as named. An integer mask with bit 1 (``rl_mask``, e.g. ``17``:
#: bit 16 = router-free, ``library.rl_level``) composes that rl level and means ``all`` at run time.
#: NUMERICS RULE: the router input differs from stock's (FP32 composites, no BF16 roundings of the o_proj sum, the
#: gated residual and x_ffn), so near-tied top-8 choices can differ; validate with GSM8K / GPQA before serving. No
#: precision is lowered (FP32 instead of BF16-rounded router inputs; every stored tensor keeps its type); the tests
#: check the logits against the FP64 reference (``rl.reference``, ``rl.top8_stats``). Details at the macro
#: (msoMegakernel.cu).
RL_ENV_VAR = "MUSESPARK_OPEN_MEGAKERNEL_RL"
RL_OPTIONS = {"0": "", "": "", "swa": "swa", "1": "all", "all": "all"}


def rl_mask(text: str) -> int:
    """An rl build mask given as ``RL_ENV_VAR``'s value: an integer above 1 with bit 1 (e.g. ``17``), else 0. Such a
    value composes that rl level into the build mode and means ``all`` at run time."""
    try:
        mask = int(str(text).strip(), 0)
    except ValueError:
        return 0
    return mask if mask > 1 and mask & 1 else 0


#: The experts stage front order of an rl layer in a build without the MSO_MK_RP publisher (``rl.experts_order``:
#: ``iac``, ``lpre`` or ``qf``). The served build has the publisher, so its order stays the build's.
RL_ORDER = "iac"
# ---- end router logits by linearity ----

ROWS = 8
HIDDEN = 8192
TP = 8
#: Distinct (metadata, positions, capture buffer) bindings with parameter blocks; a further one runs stock.
MAX_BINDINGS = 4
#: The attention half's query format on the whole-layer launches (``AttnHalf.make_layer(q_fp8=)``: the FP8 query).
Q_FP8 = True
#: The sliding-window layers' o_proj canary (``attn_half.OPROJ_CANARY``, ``kPAhFlags`` bit 0: o_proj's lane 0 of warp 0
#: waits on one word of its gated input before the unit polls all of it). Value-identical: the full data-as-flag poll of
#: every word is unchanged with or without the canary, so the same words reach the same registers and only when the
#: polling starts moves (a scheduling change). Without it the unit polls its whole input at once. The full layers'
#: o_proj never takes it (``full_half``: ``AhFlags`` 0).
OPROJ_CANARY = True


def _log(msg: str, rank: int = 0, all_ranks: bool = False) -> None:
    if all_ranks or rank == 0:
        print(f"[musespark_open megakernel] {msg}", flush=True)


def workspace_copies(launches: int) -> int:
    """Copies of a workspace whose forward makes ``launches`` launches on it: a divisor of ``launches`` (a captured
    graph must end on the copy it starts on) and at least 3 (the hops' clear rule: ``layer_mk.HopWorkspace``), 4
    preferred, then 3, then 5, then the smallest larger divisor (62 launches -> 31)."""
    for copies in (4, 3, 5):
        if launches % copies == 0:
            return copies
    for copies in range(6, launches + 1):  # the smallest larger divisor
        if launches % copies == 0:
            return copies
    raise ValueError(f"{launches} launches per forward: no workspace copy count >= 3 divides it")


# ---- a cyclic copy plan for a launch count that 4 does not divide ----
#: Physical copies of the main workspace (``ServedMegakernel.copies``; the served launches rotate over 4).
PLAN_COPIES = 4
#: The experts descriptor bits of ``experts.moe_cap_flags`` (early gather requests, producer units in flight), added to
#: every layer's experts flags: 3 early requests and at most 6 units requested and not landed per CTA. Part of the fixed
#: experts word of the served build.
DEFAULT_CAP_MOE = hex((3 << 52) | (6 << 56))  # = experts.MOE_CAP_DEFAULT


def copy_plan(launches: int, copies: int = PLAN_COPIES) -> List[Tuple[int, int]]:
    """Per launch of a forward ``(copy, num_copies)``, cyclic over the forward.

    The kernel's clear rule needs every three consecutive launches (cyclically) on distinct copies: launch n clears
    the next launch's copy ``(copy + 1) % num_copies`` and the previous one's ``(copy - 1) % num_copies``. With
    ``launches % copies == 0`` this is the ``n % copies`` rotation. Otherwise (``copies == 4``) the cycle is ``a``
    blocks ``0 1 2 3`` then ``b`` blocks ``0 1 2`` (4 a + 3 b = launches, fewest 3-blocks); each launch's
    ``num_copies`` (3 or 4) is the one whose ``+- 1`` neighbours are the plan's.
    """
    if launches % copies == 0:
        return [(p % copies, copies) for p in range(launches)]
    if copies != 4:
        raise ValueError(
            f"{launches} launches on {copies} copies: no copy plan (4 copies or a divisor)"
        )
    for b in range(1, launches // 3 + 1):
        if (launches - 3 * b) % 4 == 0 and launches - 3 * b >= 0:
            seq = [0, 1, 2, 3] * ((launches - 3 * b) // 4) + [0, 1, 2] * b
            break
    else:
        raise ValueError(f"{launches} launches: not 4 a + 3 b")
    plan = []
    for p, c in enumerate(seq):
        nxt, prv = seq[(p + 1) % launches], seq[(p - 1) % launches]
        assert len({prv, c, nxt}) == 3, (p, prv, c, nxt)
        n = next((n for n in (copies, 3) if (c + 1) % n == nxt and (c - 1) % n == prv), None)
        assert n is not None, (p, prv, c, nxt)
        plan.append((c, n))
    return plan


# ---- end cyclic copy plan ----


def early_x_flags(
    sliding: Sequence[bool], indices: Sequence[int], enabled: bool, full_one: bool = False
) -> List[Tuple[bool, bool]]:
    """Per replaced layer ``(x_in, x_out)``: a sliding-window layer after a replaced layer polls the early x, and a
    replaced layer whose successor is a replaced sliding-window layer publishes it. With ``full_one`` (one-launch
    full layers) a full-attention layer polls it too, so every replaced layer but the last publishes it."""
    out = []
    for pos, index in enumerate(indices):
        x_in = enabled and pos > 0 and (bool(sliding[index]) or full_one)
        x_out = enabled and pos + 1 < len(indices) and (bool(sliding[indices[pos + 1]]) or full_one)
        out.append((x_in, x_out))
    return out


def refusals(capability, mapping, layers) -> List[str]:
    """Why this model and mapping cannot take the kernel (empty: it can)."""
    reasons = []
    if capability != (10, 7):
        reasons.append(f"compute capability {capability} (the kernel is built for sm_107a)")
    if mapping is None:
        return reasons + ["no mapping"]
    if mapping.tp_size != TP or getattr(mapping, "moe_ep_size", TP) != TP or mapping.pp_size != 1:
        reasons.append(
            f"mapping tp {mapping.tp_size} ep {getattr(mapping, 'moe_ep_size', None)} pp {mapping.pp_size} "
            f"(the kernel runs TP{TP} EP{TP})"
        )
    if getattr(mapping, "enable_attention_dp", False):
        reasons.append("attention DP")
    has_cp = getattr(mapping, "has_cp_helix", None)
    if callable(has_cp) and has_cp():
        reasons.append("context parallelism")
    if len(layers) < 3:
        reasons.append(f"{len(layers)} decoder layers")
    for index in range(1, len(layers) - 1):
        ff = getattr(layers[index], "feed_forward", None)
        state = getattr(ff, "_replicated_experts", None)
        stacks = getattr(state, "stacks", None) if state is not None else None
        if not stacks or "w3_w1_weight" not in stacks:
            reasons.append(f"layer {index} has no replicated three-window expert stacks")
            break
    return reasons


# ---- edge layers ----
def edge_refusals(layers, final_norm) -> List[str]:
    """Why the edge layers (0 and ``n - 1``) cannot take the kernel (empty: they can)."""
    from . import bf16_experts

    reasons = []
    for index in (0, len(layers) - 1):
        ff = getattr(layers[index], "feed_forward", None)
        state = getattr(ff, "_replicated_experts", None)
        stacks = getattr(state, "stacks", None) if state is not None else None
        w13 = stacks.get("w3_w1_weight") if stacks else None
        if w13 is None or w13.dtype != torch.bfloat16 or w13.shape[0] != bf16_experts.WINDOW:
            reasons.append(f"layer {index} has no replicated two-window BF16 expert stack")
    if not layers[0].self_attn.is_sliding:
        reasons.append("layer 0 is not a sliding-window layer")
    if layers[-1].self_attn.is_sliding:
        reasons.append(f"layer {len(layers) - 1} is not a full-attention layer")
    eps = getattr(final_norm, "variance_epsilon", None)
    if final_norm is None or getattr(final_norm, "weight", None) is None or eps is None:
        reasons.append("no final norm")
    elif abs(float(eps) - FINAL_NORM_EPS) > 1e-12:
        reasons.append(f"final norm eps {eps} (the kernel's boundary norm uses {FINAL_NORM_EPS})")
    return reasons


def layer0_attn_input(layer, res: torch.Tensor) -> torch.Tensor:
    """The stock layer 0's attention input: its pre-QKV norm of the embedding residual (the stock layer's branch for
    ``precomputed_attn_input=None``)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import _rmsnorm_fp32_in

    if layer.add_attn._preqkv_norm_fp32:
        return _rmsnorm_fp32_in(res, layer.pre_qkv_proj_norm)
    return layer.pre_qkv_proj_norm(res.to(layer.pre_qkv_proj_norm.weight.dtype))


# ---- end edge layers ----


class ServedMegakernel:
    """The replaced layers' kernels on this rank and the per-forward decision (module docstring).

    Args:
        layers: the text model's decoder layers.
        mapping: the model's parallel mapping.
        comm: the TP group's mpi4py communicator (the MoE hop workspace and the build barrier).
        mode: the library build.
        early_x: chain the sliding-window layers with early x (the next sliding-window layer's QKV runs poll x_attn
            before the dependency wait).
        full_one: a full-attention layer is one launch (``LayerKernels.make_full_layer_one(v2=True)``), with early x in
            and out of every replaced layer; False is the two-launch full layer (launch Q + launch F).
        i2b_flags: kPIFlags bits ORed into every sliding-window layer (``DEFAULT_I2B_FLAGS``).
        final_norm: the model's final norm (with ``edges``, layer ``n - 1`` applies it).
        edges: layers 0 and ``n - 1`` on the kernel too.
        cap_moe: the experts descriptor bits of ``experts.moe_cap_flags`` (``DEFAULT_CAP_MOE``).
        w1_plan: the sliding-window layers' attention-half plan options (``DEFAULT_W1_PLAN``).
        fa_flags: kPIFlags bits of the FaAttn stage on every full-attention layer (``FA_FLAGS``); None =
            ``layer_mk.F3_DEFAULT_FLAGS``.
        fa_entry_pages: ring pages a FaAttn unit requests in its prologue (a multiple of 4, at most the ring; 0 = the
            whole ring).
        fa_gates_early: the full layers' 16 gate QkvSum units on early CTAs
            (``layer_mk.plan_full_one(gates_early=True)``); ``fa_late_first`` takes precedence.
        fa_o_replicas: copies (1 or 2) of the full layers' gated o_proj input that FaMerge publishes
            (``LayerKernels(f3_o_replicas=)``).
        fa_late_first: the full layers' late-CTA plan (``FA_LATE_FIRST``; 0 = off).
        fh_plan: plan options of the one-launch full layer with ``fa_late_first`` 3 (``FH_PLAN``).

    The defaults are the served configuration (``install``).
    """

    def __init__(
        self,
        layers,
        mapping,
        comm,
        mode: str = DEFAULT_MODE,
        early_x: bool = True,
        full_one: bool = True,
        i2b_flags: int = DEFAULT_I2B_FLAGS,
        final_norm=None,
        edges: bool = False,
        cap_moe: int = int(DEFAULT_CAP_MOE, 0),
        w1_plan: str = DEFAULT_W1_PLAN,
        fa_flags: Optional[int] = FA_FLAGS,
        fa_entry_pages: int = 0,
        fa_gates_early: bool = False,
        fa_o_replicas: int = 1,
        fa_late_first: int = FA_LATE_FIRST,
        fh_plan: int = FH_PLAN,
        rl_option: str = "",
    ) -> None:
        self.layers = layers
        self.mapping = mapping
        self.comm = comm
        self.rank = int(mapping.tp_rank)
        self.mode = mode
        self.early_x = bool(early_x)
        self.full_one = bool(full_one)
        self.i2b_flags = int(i2b_flags)
        self.edges = bool(edges)  # layers 0 and n - 1 on the kernel too
        self.final_norm = final_norm  # applied by layer n - 1 with edges
        self.cap_moe = int(cap_moe)  # the in-flight caps
        from . import experts as _experts  # MOE_CAP_MASK

        assert self.cap_moe & ~_experts.MOE_CAP_MASK == 0, (
            f"cap_moe: only bits 52-59 ({self.cap_moe:#x})"
        )
        self.w1_plan = tuple(sorted(set(filter(None, str(w1_plan).replace("+", ",").split(",")))))
        assert set(self.w1_plan) <= set(W1_PLAN_OPTIONS), (
            f"[musespark_open megakernel] w1_plan={w1_plan!r}"
        )
        self.fa_flags = None if fa_flags is None else int(fa_flags)
        self.fa_entry_pages = int(fa_entry_pages)
        self.fa_gates_early = bool(fa_gates_early)
        self.fa_late_first = int(fa_late_first)
        assert self.fa_late_first in (0, 1, 2, 3), "fa_late_first: 0 .. 3"
        self.fa_o_replicas = int(fa_o_replicas)
        self.fh_plan = int(fh_plan)
        assert 0 <= self.fh_plan <= 15, "fh_plan: 0 .. 15"  # bit 3: layer_mk.FT_PLAN_LATE64
        assert not self.fh_plan or self.fa_late_first == 3, "fh_plan needs fa_late_first=3"
        assert self.fa_o_replicas in (1, 2), "fa_o_replicas: 1 or 2"
        # ---- the rl knob ----
        rl_opt = str(rl_option)
        assert rl_opt in RL_OPTIONS or rl_mask(rl_opt), (
            f"{RL_ENV_VAR}: one of {sorted(RL_OPTIONS)} or a mask with bit 1"
        )
        if rl_opt:
            # a build mask runs rl on every layer
            self.rl = RL_OPTIONS[rl_opt] if rl_opt in RL_OPTIONS else "all"
        else:
            # Unset: the build's rl (every NVFP4 layer of an rl build, else off).
            from . import library as _library_rl

            self.rl = (
                "all"
                if self.mode in _library_rl.MODES and _library_rl.rl_level(self.mode) & 1
                else ""
            )
        self.rl_order = RL_ORDER
        assert self.rl_order in ("iac", "lpre", "qf"), "rl_order: iac, lpre or qf"
        self.rl_ws = None
        self.rl_slabs: Dict[int, object] = {}
        #: The sliding-window layers' o_proj canary (``OPROJ_CANARY``).
        self.oproj_canary = OPROJ_CANARY
        # bh bit 8 (pool-only consumers) and rl bit 16 (router-free) have no router code, so every router slot
        # must run rl's units
        from . import library as _library_lv

        if self.mode in _library_lv.MODES and (
            _library_lv.lever_level(self.mode, "bh") & 8
            or _library_lv.lever_level(self.mode, "rl") & 16
        ):
            if self.rl != "all" or self.edges:
                raise ValueError(
                    f"[musespark_open megakernel] mode {self.mode} (bh bit 8 / rl bit 16: no router code) needs "
                    f"{RL_ENV_VAR}=all and no edge layers (rl {self.rl or 'off'}, edges {self.edges})"
                )
        # ---- end rl knob ----
        assert self.fa_entry_pages % 4 == 0 and self.fa_entry_pages >= 0, (
            "fa_entry_pages: a multiple of 4"
        )
        if self.edges:
            assert self.full_one, (
                "[musespark_open megakernel] the edge layers need one-launch full layers"
            )
        # An MSO_MK_FT_GATE build sums the gate in FaMerge of the one-launch full layer only
        from . import library as _library  # ft_gate

        assert self.full_one or not _library.ft_gate(self.mode), (
            f"[musespark_open megakernel] {self.mode} needs one-launch full layers"
        )
        self.indices = (
            tuple(range(0, len(layers))) if self.edges else tuple(range(1, len(layers) - 1))
        )
        self.first = self.indices[0]
        #: With edges, the last replaced layer's x output is the final hidden state (``core.norm`` is not run).
        self.final_norm_done = self.edges
        self.sliding = [bool(layer.self_attn.is_sliding) for layer in layers]
        self.full_count = sum(1 for i in self.indices if not self.sliding[i])
        # The main workspace's copies and the forward's cyclic copy plan (60 launches: the p % 4 rotation; 62 with the
        # edge layers: 4 copies, 14 blocks 0123 + 2 blocks 012).
        self.copies = PLAN_COPIES
        self.plan = copy_plan(len(self.indices), self.copies)
        try:
            self.q_copies = workspace_copies(self.full_count) if self.full_count else 2
        except ValueError:
            # Launch Q's workspace serves only the two-launch full layers; one-launch layers never use it (the
            # two full layers of a test slice have no copy count >= 3 that divides 2).
            if not self.full_one:
                raise
            self.q_copies = 2
        self.flags = early_x_flags(self.sliding, self.indices, self.early_x, self.full_one)
        self.kernels = None
        self.bindings: Dict[tuple, Dict[int, object]] = {}
        self.current: Optional[Dict[int, object]] = None
        self.refused_bindings = 0
        self.forwards = 0
        self.joined_forks = 0
        self.memory_bytes = 0

    # ------------------------------------------------------------------ install
    def prepare(self) -> None:
        """Compile, build the workspaces and pack every replaced layer's weights (eager, collective)."""
        from . import full_half, layer_mk, library

        assert not torch.cuda.is_current_stream_capturing()
        agreed = self.comm.allgather(
            (
                self.mode,
                self.early_x,
                self.full_one,
                self.i2b_flags,
                len(self.layers),
                self.edges,
                self.copies,
                self.cap_moe,
                self.w1_plan,
                self.fa_flags,
                self.fa_entry_pages,
                self.fa_gates_early,
                self.fa_o_replicas,
                self.fa_late_first,
                self.fh_plan,
                self.rl,
                self.rl_order,
            )
        )
        if len(set(agreed)) != 1:
            raise RuntimeError(
                f"[musespark_open megakernel] the ranks disagree on the configuration: {agreed}"
            )
        if self.rank == 0:
            library.library(self.mode)
        self.comm.Barrier()
        torch.cuda.synchronize()
        free0, _ = torch.cuda.mem_get_info()
        allocated0 = torch.cuda.memory_allocated()
        local_rank = int(getattr(self.mapping, "local_rank", self.rank % 4))
        k = layer_mk.LayerKernels(
            self.mode,
            self.comm,
            self.rank,
            local_rank,
            copies=self.copies,
            i2b_flags=self.i2b_flags,
            bf16=self.edges,  # the BF16 experts stage of the edge layers
            w1_oproj_off_merge="offmerge" in self.w1_plan,
            f3_flags=self.fa_flags,
            fa_gates_early=self.fa_gates_early,
            f3_o_replicas=self.fa_o_replicas,
        )
        if self.fa_late_first:
            k.enable_fa_late_first(
                gates_after=self.fa_late_first == 2,
                sums_after=self.fa_late_first == 3,
                fh_plan=self.fh_plan,
            )
        k.ws.copy_plan = self.plan  # the cyclic copy plan
        # Every layer's experts flags: DEFAULT_MOE_FLAGS, the in-flight caps, the table-driven loops, the tile queue
        k.moe_flags_extra |= DEFAULT_MOE_FLAGS
        k.moe_flags_extra |= self.cap_moe
        k.moe_flags_extra |= LX_MOE_FLAGS
        k.moe_flags_extra |= PX_MOE_FLAGS[PX_LEVEL]
        # ---- the rl workspace, the per-layer weight slabs and (``all``) the experts order of every layer ----
        if self.rl:
            from . import rl

            if not rl.check_layout(k.lib) & 1:
                raise ValueError(
                    f"[musespark_open megakernel] {RL_ENV_VAR}={self.rl} needs an rl build, not {self.mode}"
                )
            if self.rl == "all" and not library.rp_level(
                self.mode
            ):  # with MSO_MK_RP the order stays the build's
                clear, add = rl.experts_order(self.rl_order)
                k.moe_flags_extra = (k.moe_flags_extra & ~clear) | add
            self.rl_ws = rl.RlWorkspace(self.comm, self.rank, local_rank, k.ws.copies)
            for index in self.indices:
                if self.edges and index in (0, len(self.layers) - 1):
                    continue  # the BF16-expert layers keep the router
                if self.rl == "all" or self.sliding[index]:
                    self.rl_slabs[index] = rl.LayerSlabs(self.layers[index], self.rank)
            order = "build (rp)" if library.rp_level(self.mode) else self.rl_order
            _log(  # the rl state on every rank
                f"{RL_ENV_VAR}={self.rl}: rl workspace {self.rl_ws.copies} x {rl.COPY_BYTES} B (multicast) on rank "
                f"{self.rank}, weight slabs on {len(self.rl_slabs)} layers ({min(self.rl_slabs, default=-1)}-"
                f"{max(self.rl_slabs, default=-1)}), order {order}",
                self.rank,
                all_ranks=True,
            )
        # ---- end rl workspace ----
        # ---- a build with the experts flags fixed at compile time traps on any other word ----
        fixed = library.fixed_moe_flags(self.mode)
        if fixed is not None and (layer_mk.MOE_FLAG_MH_F2 | k.moe_flags_extra) != fixed:
            raise ValueError(
                f"[musespark_open megakernel] mode {self.mode} fixes the experts flags to {fixed:#x}, the layers "
                f"would launch with {layer_mk.MOE_FLAG_MH_F2 | k.moe_flags_extra:#x}"
            )
        fa_fixed = library.fixed_fa_flags(self.mode)
        if fa_fixed is not None and self.full_count:
            # Every one-launch full layer sets FA_OUT_ARENA and QKV_ARENA; the other fixed bits come from the FA flags.
            full = layer_mk.IFLAG_FA_OUT_ARENA | layer_mk.IFLAG_QKV_ARENA | k.f3_flags
            zero, one = fa_fixed
            if not self.full_one or (full & one) != one or full & zero:
                raise ValueError(
                    f"[musespark_open megakernel] mode {self.mode} fixes the full-attention bits (zero {zero:#x}, one "
                    f"{one:#x}); the full layers would launch with {full:#x} (full one {self.full_one})"
                )
        # ---- end fixed flag words ----
        # Launch Q's workspace rotates over the full layers' launches of a forward.
        k.full = full_half.FullHalf(self.mode, copies=self.q_copies)
        # The FaAttn units' prologue ring pages (``fa_entry_pages``; 0 = the whole ring).
        if self.fa_entry_pages:
            k.full.entry_pages = self.fa_entry_pages
        self.kernels = k
        dev = k.ws.arena.device
        n = len(self.indices)
        self.x_out = [torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(n)]
        self.res_out = [
            torch.zeros(ROWS, HIDDEN, dtype=torch.float32, device=dev) for _ in range(n)
        ]
        self.dummy_slot = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
        # Packed once per layer; every binding's parameter blocks reuse them.
        from . import attn_half

        self.moe: Dict[int, object] = {}
        self.packed_qkv: Dict[int, torch.Tensor] = {}
        last = len(self.layers) - 1
        bf16_turn = 0
        for pos, index in enumerate(self.indices):
            layer = self.layers[index]
            # ---- the edge layers take the BF16 experts stage (copies of its workspace by turn); layer n - 1's
            # next norm is the final norm ----
            nxt = self.final_norm if index == last else self.layers[index + 1].pre_qkv_proj_norm
            if self.edges and index in (0, last):
                self.moe[index] = layer_mk.moe_inputs_from_bf16_layer(
                    layer, nxt.weight.data, self.rank, bf16_turn
                )
                bf16_turn += 1
            else:
                self.moe[index] = layer_mk.moe_inputs_from_layer(layer, nxt.weight.data, self.rank)
            # ---- end edge layers ----
            self.moe[index].hop.pack(k.moe_hop.row_units)
            w_qkv = layer.self_attn.qkv_proj.weight.data
            if self.sliding[index]:
                self.packed_qkv[index] = attn_half.pack_qkv_heads_first(w_qkv)
        # The o_proj and full-layer QKV packs the stage objects make inside make_layer, made here (before the KV
        # cache is sized) and handed back by _packed_once.
        from . import qkv

        self._packs: Dict[tuple, torch.Tensor] = {}
        self._pack_sources: Dict[str, Dict[int, torch.Tensor]] = {"bf16": {}, "qkv": {}}
        slices = k.full.qkv.slices
        for index in self.indices:
            attn = self.layers[index].self_attn
            w_o = attn.o_proj.weight.data
            self._pack_sources["bf16"][w_o.data_ptr()] = w_o
            self._packs[("bf16", w_o.data_ptr(), ())] = qkv.pack_bf16(w_o.contiguous())
            if not self.sliding[index] and self.full_one:
                # The one-launch full layer's heads-first QKV pack (handed to make_full_layer_one).
                hf = layer_mk.heads_first_qkv(attn.qkv_proj.weight.data)
                self.packed_qkv[index] = qkv.pack_qkv(hf, k.full.lib.qkv_slices)
                del hf
            elif not self.sliding[index]:
                w_qkv = attn.qkv_proj.weight.data
                self._pack_sources["qkv"][w_qkv.data_ptr()] = w_qkv
                self._packs[("qkv", w_qkv.data_ptr(), (slices,))] = qkv.pack_qkv(
                    w_qkv.contiguous(), slices
                )
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        # Tensors through the caching allocator (which may reuse cached blocks: the device-free delta undercounts
        # them), plus what the device-free delta sees outside it (the multicast buffer).
        self.memory_bytes = torch.cuda.memory_allocated() - allocated0
        self.device_free_delta = free0 - free1
        self.comm.Barrier()
        _log(
            f"megakernel: layers {self.indices[0]}-{self.indices[-1]} ({n}: {n - self.full_count} sliding-window, "
            f"{self.full_count} full) as whole-layer launches on decode forwards of one sequence and {ROWS} rows; "
            f"mode {self.mode} {k.lib.attributes()}, early x {self.early_x}, full one {self.full_one}, "
            f"i2b_flags {self.i2b_flags:#x}, t3_flags {k.t3_flags:#x}, f3_flags {k.f3_flags:#x}, experts flags "
            f"{k.moe_flags_extra:#x}, edges {self.edges}, w1_plan {','.join(self.w1_plan) or '-'}, "
            f"fa_entry_pages {k.full.entry_pages}, fa_gates_early {self.fa_gates_early}, "
            f"fa_late_first {self.fa_late_first}, fa_o_replicas {k.f3_o_replicas}, fh_plan {self.fh_plan}, "
            f"rl {self.rl or 'off'} ({len(self.rl_slabs)} layers), "
            f"oproj canary {'on' if self.oproj_canary else 'off'}, "
            f"copies {self.copies} (launch Q "
            f"{self.q_copies}); workspaces and packed weights "
            f"{self.memory_bytes / 2**30:.2f} GiB of tensors on this GPU "
            f"(device free memory {self.device_free_delta / 2**30:.2f} GiB lower)",
            self.rank,
        )

    # ------------------------------------------------------------------ bindings
    @contextlib.contextmanager
    def _packed_once(self):
        """``qkv.pack_bf16`` / ``qkv.pack_qkv`` of a replaced layer's o_proj / full-layer QKV weight return the copy
        packed for the first binding (the stage objects pack inside ``make_layer``)."""
        from . import qkv

        pack_bf16, pack_qkv = qkv.pack_bf16, qkv.pack_qkv
        packs = self._packs

        def once(fn, name):
            sources = self._pack_sources[name]

            def packed(w, *args):
                src = sources.get(w.data_ptr())
                if src is None or tuple(src.shape) != tuple(w.shape) or src.dtype != w.dtype:
                    return fn(w, *args)
                key = (name, w.data_ptr(), args)
                if key not in packs:
                    packs[key] = fn(w, *args)
                return packs[key]

            return packed

        qkv.pack_bf16, qkv.pack_qkv = once(pack_bf16, "bf16"), once(pack_qkv, "qkv")
        try:
            yield
        finally:
            qkv.pack_bf16, qkv.pack_qkv = pack_bf16, pack_qkv

    def _layer_pool(self, layer, md) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer's FP8 pool view ``[blocks, 2, 32, 64]`` and block table ``[2, max_blocks]`` as the stock decode
        kernels bind them (``kernels/swa_block/attention.py``)."""
        from tensorrt_llm.bindings.internal import thop

        from ...attention import trtllm_gen_fmha

        attention = layer.self_attn
        attn = attention.attn
        fmha = trtllm_gen_fmha(attn)
        if fmha is None:
            raise RuntimeError(
                f"[musespark_open megakernel] layer {attention.layer_idx}: no trtllm-gen FMHA bound"
            )
        pool, tables, _ = thop.build_trtllm_gen_kv_cache_metadata(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            attn.get_local_layer_idx(md),
            attention.num_key_value_heads,
            md.tokens_per_block,
            attention.head_dim,
            2,
            fmha._get_total_num_blocks(md),
            attn.quant_mode,
            0,
            1,
            torch.bfloat16,
        )
        return pool.view(torch.uint8), tables[0]

    def _build(self, md, positions: torch.Tensor, spec_metadata) -> Dict[int, object]:
        from . import full_attention, library

        k = self.kernels
        seq = md.kv_lens_cuda_runtime[:1]
        cap_buf = (
            getattr(spec_metadata, "captured_hidden_states", None)
            if spec_metadata is not None
            else None
        )
        cap_idx = getattr(spec_metadata, "_layer_to_idx", {}) if spec_metadata is not None else {}
        blocks: Dict[int, object] = {}
        last = len(self.layers) - 1
        with self._packed_once():
            for pos, index in enumerate(self.indices):
                # Every forward runs the whole cyclic copy plan, so the layer at position pos always runs on the
                # plan's copy pos: only that copy's block is built (with the plan's NumCopies).
                k.ws.build_copy = self.plan[pos]
                # with edges, layer n - 1 applies the final norm
                final = self.edges and index == last
                layer = self.layers[index]
                attn = layer.self_attn
                kv_scale = attn.attn.kv_scale_orig_quant
                kv_scale = (
                    float(kv_scale.view(-1)[0]) if isinstance(kv_scale, torch.Tensor) else 1.0
                )
                pool, table = self._layer_pool(layer, md)
                capture = None
                if cap_buf is not None and index in cap_idx:
                    c0 = cap_idx[index] * HIDDEN
                    capture = cap_buf[:ROWS, c0 : c0 + HIDDEN]
                x_in, x_out = self.flags[pos]
                if self.sliding[index]:
                    args = (
                        attn.qkv_proj.weight.data,
                        attn.o_proj.weight.data,
                        pool,
                        table,
                        seq,
                        positions,
                        attn.rotary_emb.rotary_cos_sin,
                        self.dummy_slot,
                    )
                    kwargs = dict(
                        kv_scale=kv_scale,
                        q_scale=float(attn.scale_query_by),
                        prep_eps=float(attn.q_norm.variance_epsilon),
                        gate_eps=float(attn.o_head_norm.variance_epsilon),
                        window=int(attn.attention_window_size),
                        q_fp8=Q_FP8,
                        oproj_canary=self.oproj_canary,  # OPROJ_CANARY
                        packed_qkv=self.packed_qkv[index],
                    )
                    saved_moe = k.moe_flags_extra  # the experts order of an rl layer (``swa``)
                    if (
                        self.rl == "swa"
                        and index in self.rl_slabs
                        and not library.rp_level(self.mode)
                    ):
                        from . import rl

                        clear, add = rl.experts_order(self.rl_order)
                        k.moe_flags_extra = (saved_moe & ~clear) | add
                    try:
                        blocks[index] = k.make_swa_layer(
                            args,
                            kwargs,
                            self.moe[index],
                            self.x_out[pos],
                            self.res_out[pos],
                            capture=capture,
                            early_x_in=x_in,
                            early_x_out=x_out,
                            final_norm=final,  # layer n - 1 with edges
                        )
                    finally:
                        k.moe_flags_extra = saved_moe
                    self._rl_apply(index, blocks[index])
                else:
                    f_args = (attn.o_proj.weight.data, pool, table, seq)
                    f_kwargs = dict(
                        kv_scale=kv_scale,
                        q_scale=float(getattr(attn, "scale_query_by", full_attention.Q_SCALE)),
                        eps=float(attn.q_norm.variance_epsilon),
                    )
                    if self.full_one:  # one launch (the ``v2`` tables)
                        blocks[index] = k.make_full_layer_one(
                            attn.qkv_proj.weight.data,
                            f_args,
                            f_kwargs,
                            self.moe[index],
                            self.x_out[pos],
                            self.res_out[pos],
                            capture=capture,
                            early_x_in=x_in,
                            early_x_out=x_out,
                            v2=True,
                            packed_qkv=self.packed_qkv[index],
                            final_norm=final,  # layer n - 1 with edges
                        )
                        self._rl_apply(index, blocks[index])
                        continue
                    blocks[index] = k.make_full_layer(
                        attn.qkv_proj.weight.data,
                        f_args,
                        f_kwargs,
                        self.moe[index],
                        self.x_out[pos],
                        self.res_out[pos],
                        capture=capture,
                        early_x_out=x_out,
                    )
        k.ws.build_copy = None  # every copy's block again
        torch.cuda.synchronize()
        return blocks

    # ---- rl layers ----
    def _rl_apply(self, index: int, entry) -> None:
        """The rl fields and ``rl.IFLAG_RL_ON`` on an rl layer's blocks (its weight tables filled again)."""
        if index not in self.rl_slabs:
            return
        from . import rl

        k = self.kernels
        if isinstance(entry, dict):
            mk = entry["mk"]
            blocks = entry.get("one", entry.get("swa"))
        else:
            mk, blocks = k.swa_mk, entry
        rl.apply(mk, blocks, rl.layer_fields(self.rl_ws, self.rl_slabs[index]), rl.IFLAG_RL_ON)

    # ---- end rl layers ----

    # ------------------------------------------------------------------ per forward
    def begin(self, hidden_states, position_ids, attn_metadata, spec_metadata) -> bool:
        """Whether this forward runs the replaced layers as the kernel (all of them; else every layer is stock)."""
        if self.kernels is None:
            return False
        if not isinstance(hidden_states, torch.Tensor) or tuple(hidden_states.shape) != (
            ROWS,
            HIDDEN,
        ):
            return False
        if attn_metadata is None or attn_metadata.num_contexts != 0 or attn_metadata.num_seqs != 1:
            return False
        if attn_metadata.kv_cache_manager is None:
            return False
        if (
            position_ids is None
            or position_ids.numel() != ROWS
            or position_ids.dtype != torch.int32
        ):
            return False
        positions = position_ids.view(-1)
        if not positions.is_contiguous():
            return False
        cap = (
            getattr(spec_metadata, "captured_hidden_states", None)
            if spec_metadata is not None
            else None
        )
        key = (
            id(attn_metadata),
            attn_metadata.kv_cache_block_offsets.data_ptr(),
            attn_metadata.kv_lens_cuda_runtime.data_ptr(),
            positions.data_ptr(),
            None if cap is None else cap.data_ptr(),
            tuple(sorted(getattr(spec_metadata, "_layer_to_idx", {}).items()))
            if spec_metadata is not None
            else (),
        )
        blocks = self.bindings.get(key)
        if blocks is None:
            if torch.cuda.is_current_stream_capturing() or len(self.bindings) >= MAX_BINDINGS:
                self.refused_bindings += 1
                if self.refused_bindings == 1:
                    _log(
                        f"a decode binding {'first seen inside a capture' if torch.cuda.is_current_stream_capturing() else 'past the binding limit'} "  # noqa: E501
                        "runs the stock layers",
                        self.rank,
                        all_ranks=True,
                    )
                return False
            blocks = self._build(attn_metadata, positions, spec_metadata)
            self.bindings[key] = blocks
            # The metadata object stays alive with its blocks (its id is part of the key).
            blocks["_md"] = attn_metadata
            _log(
                f"parameter blocks for binding {len(self.bindings)} ({type(attn_metadata).__name__})",
                self.rank,
            )
        self.current = blocks
        return True

    @contextlib.contextmanager
    def scope(self):
        """The forward's layer loop: the layer before the first replaced one forks no next-QKV L2 prefetch."""
        # edges
        moe = getattr(self.layers[self.first - 1], "feed_forward", None) if self.first > 0 else None
        saved = moe.__dict__.get("_qkv_prefetch_after_experts", None) if moe is not None else None
        if saved is not None:
            moe._qkv_prefetch_after_experts = None
        try:
            yield
        finally:
            if saved is not None:
                moe._qkv_prefetch_after_experts = saved
            self.current = None

    def replaces(self, index: int) -> bool:
        return self.first <= index <= self.indices[-1]

    def run(
        self, core, hidden_states: torch.Tensor, x_attn: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every replaced layer: ``(residual after the last one, the next layer's normed attention input)``. With
        ``edges``, ``x_attn`` is None (layer 0's input is made here) and the second item is the final hidden
        state (the final norm of the last layer's output)."""
        blocks = self.current
        assert blocks is not None, "begin() first"
        plan = getattr(core, "_dense_gemm_plan", None)
        if plan is not None:
            pending = bool(getattr(plan, "pending", False)) or any(
                getattr(f, "pending", False) for f in getattr(plan, "named_forks", {}).values()
            )
            if pending:
                # A fork still in flight would run beside the persistent kernel; it waits on nothing the kernel makes.
                self.joined_forks += 1
                if getattr(plan, "named_forks", None):
                    plan.join_all()
                else:
                    plan.join()
        k = self.kernels
        assert hidden_states.dtype == torch.float32 and tuple(hidden_states.shape) == (ROWS, HIDDEN)
        res = hidden_states.contiguous()
        if self.edges:  # layer 0's attention input (the stock op)
            assert x_attn is None
            x_attn = layer0_attn_input(self.layers[0], res)
        assert x_attn is not None and x_attn.dtype == torch.bfloat16
        x = x_attn.contiguous()
        for pos, index in enumerate(self.indices):
            if self.sliding[index]:
                k.run_swa(blocks[index], x, res)
            else:
                k.run_full(blocks[index], x, res)
            x, res = self.x_out[pos], self.res_out[pos]
        self.forwards += 1
        return res, x


def install(core, mapping, capability=None, comm=None) -> Optional[ServedMegakernel]:
    """Build the kernel for ``core`` (the adopted text model) when the model and mapping are the kernel's on every
    rank of the TP group (else None, with the agreed reason logged); idempotent. Collective over the TP group. The
    build is ``DEFAULT_MODE``, the served build (the ``megakernel`` lever and its numerics-budget levers)."""
    existing = getattr(core, "_megakernel", None)
    if existing is not None:
        return existing
    from tensorrt_llm._utils import mpi_disabled, prefer_pinned

    from ... import dispatch as _tp_agreement

    rank = int(getattr(mapping, "tp_rank", 0)) if mapping is not None else 0
    if capability is None and torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
    layers = list(core.layers)
    reasons = refusals(capability, mapping, layers)
    if not reasons and not prefer_pinned():
        reasons.append("no pinned host memory (the kernel writes its watchdog record there)")
    # ---- edge layers (off in the served configuration) ----
    edges = False
    if edges and not reasons:
        edge_reasons = edge_refusals(layers, getattr(core, "norm", None))
        if edge_reasons:
            _log(
                f"edge layers ignored (layers 0 and {len(layers) - 1} stay stock): "
                + "; ".join(edge_reasons),
                rank,
            )
            edges = False
    # ---- end edge layers ----
    if comm is None and not reasons:
        if mpi_disabled():
            reasons.append("no MPI communicator (the MoE hop workspace is built on one)")
        else:
            from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

            comm = _get_mnnvl_workspace_comm(mapping)
    refusal = _tp_agreement.agreed_refusal(mapping, "; ".join(reasons) or None)
    if refusal is not None:
        _log("megakernel not installed: " + refusal, rank)
        return None
    runner = ServedMegakernel(
        layers,
        mapping,
        comm,
        mode=DEFAULT_EDGES_MODE if edges else DEFAULT_MODE,
        early_x=True,
        full_one=True,
        i2b_flags=DEFAULT_I2B_FLAGS,
        final_norm=getattr(core, "norm", None),  # applied by layer n - 1 with edges
        edges=edges,
        cap_moe=int(DEFAULT_CAP_MOE, 0),
        w1_plan=DEFAULT_W1_PLAN,
        fa_flags=FA_FLAGS,
        fa_entry_pages=0,
        fa_gates_early=False,
        fa_o_replicas=1,
        fa_late_first=FA_LATE_FIRST,
        fh_plan=FH_PLAN,
    )
    runner.prepare()
    core._megakernel = runner
    return runner
