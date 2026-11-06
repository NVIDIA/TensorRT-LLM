# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A whole decoder layer per megakernel launch on one rank of the TEP8 group: both halves' stages in one plan.

Sliding-window layer, one launch: the attention half (``attn_half``: QKV runs, prep + KV append, SWA attention, merge
+ norm-gate, o_proj) publishes the o_proj partial into the MoE hop workspace's hop1 area; then the MoE half
(``moe_hops`` hop1 consumer, router, pool -> F2', ``experts`` routing + placement + quant + NVFP4 experts + post-norm
weighted sum -> finalize area, finalize all-reduce, unpool, F1 consumer -> the residual r'' and the next layer's
normed input x_attn, the DFlash capture).

Full-attention layer, two launches: launch Q (``full_half``: the QKV projection into a plain buffer) and launch F (the
full-attention stages + o_proj, then the same MoE half). One CTA cannot hold the QKV run and the attention unit's ring
at once (see ``schedule_full``).

Per CTA, the MoE half follows the attention units (``plan_moe_half``):

* CTAs ``[0, 64)``: ``MhPool`` (weights requested once the CTA's o_proj unit ends), ``MoeExperts`` (unit ``c + 1``:
  a final unit);
* ``[64, 128)``: ``MoeExperts``, ``MhUnpool`` (weights requested when the experts end);
* ``[128, 160)``: ``MhRouter``, ``MoeExperts``;
* ``[160, 192)``: ``MhH1``, ``MoeExperts``, ``MhFinal``, ``MhF1`` (row units ``u = c - 160``; U = 4 boundary units per
  row, so every row CTA has both boundaries);
* ``[192, 212)``: ``MhClear``, ``MoeExperts``.

``MoeExperts`` owns the CTA's whole unit region, so MhH1's r' goes through global memory (``kIFlagMhRes1Global``).

Every launch of a forward uses one library workspace and one MoE hop workspace with the same copy rotation; the
launches of one forward on it must be a multiple of ``copies`` (C >= 3; a slice of 6 SWA + 2 full layers makes 8, so
C = 4).

The stage objects (``attn_half``, ``full_half``, ``moe_hops``, ``experts``) build their parameter fields
(``_capture_fields``); this module merges them per layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import (
    attn_half,
    experts,
    full_half,
    library,
    moe_hops,
    qkv,
    schedule,
    schedule_attn,
    schedule_full,
    schedule_moe_hops,
)

#: ``kPIFlags`` bits (``msoMegakernel.cu``).
IFLAG_FA_OUT_ARENA = 1
IFLAG_OPROJ_TO_MH = 2
IFLAG_MH_RES1_GLOBAL = 4
IFLAG_X_OUT = 8
IFLAG_X_IN = 16
# ---- one-launch full layer ----
#: ``kIFlagQkvArena``: QkvSum publishes the raw projection into the arena; FaAttn / FaMerge poll it.
IFLAG_QKV_ARENA = 32
#: ``kIFlagFaRevUnits``: the FaAttn slot s < eff computes unit eff - 1 - s (the step's rows on the first CTA).
IFLAG_FA_REV_UNITS = 64
#: ``kIFlagPreWaitAttn``: with early x the attention half (QKV .. o_proj) runs before the dependency wait.
IFLAG_PRE_WAIT_ATTN = 128
#: ``kIFlagQkvHeadsFirst``: the QKV weight packed heads-first; QkvSum stores at the checkpoint columns.
IFLAG_QKV_HEADS_FIRST = 256
#: ``kIFlagFaLateLight``: the last 64 FaAttn slots get 13 / 16 of an early slot's pages (all units keyed).
IFLAG_FA_LATE_LIGHT = 512
#: Checkpoint 64-row block of each heads-first block (q_0 .. q_15 | k_0 k_1 | v_0 v_1 | g_0 .. g_15).
HEADS_FIRST_BLOCKS = [2 * h for h in range(16)] + [32, 33, 34, 35] + [2 * h + 1 for h in range(16)]
# ---- end one-launch full layer ----
# ---- kPIFlags bits of the pre-wait, entry and issue options (msoMegakernel.cu kIFlagI2b*) ----
#: With early x in: the CTA's leading attention-half units run before griddepcontrol.wait too.
IFLAG_I2B_PRE_WAIT = 1 << 20
#: No entry hook for the gate QkvFin blocks; a QkvFin block's entry loads in one round trip.
IFLAG_I2B_ENTRY_LITE = 1 << 21
#: The weight requests that follow a unit come from lane 0 of warp 7, not thread 0.
IFLAG_I2B_ISSUE_WARP7 = 1 << 22
# ---- end kIFlagI2b* bits ----
# ---- kPIFlags bits of the multi-reader polls (msoMegakernel.cu kIFlagT3*): canary-gated polls, weights first ----
#: MhPool / MhRouter wait on one word of each MhH1 unit they read before polling x_ffn.
IFLAG_T3_CANARY_X = 1 << 40
#: MhUnpool waits on one word of each MhFinal unit before polling m.
IFLAG_T3_CANARY_M = 1 << 41
#: QkvRun (early x in) waits on one word of each MhF1 unit of its K slice before polling x_attn.
IFLAG_T3_CANARY_XIN = 1 << 42
#: The experts' coalesced logit poll (route()) waits on 32 of the 512 logit vectors (4 per source rank) first.
IFLAG_T3_CANARY_LOGITS = 1 << 43
IFLAG_T3_ALL = IFLAG_T3_CANARY_X | IFLAG_T3_CANARY_M | IFLAG_T3_CANARY_XIN | IFLAG_T3_CANARY_LOGITS
#: The boundaries' row-norm exchange with 16-byte slot vectors (not in IFLAG_T3_ALL).
IFLAG_T3_XCHG_VEC = 1 << 44
#: MhH1 / MhPool / MhUnpool / MhF1 wait for their unit weights before polling their inputs.
IFLAG_T3_WAIT_FIRST = 1 << 45
#: MhF1 loads r' before its input poll.
IFLAG_T3_EARLY_RES1 = 1 << 46
#: MhFinal on all 256 threads (4 columns each).
IFLAG_T3_FINAL256 = 1 << 47
IFLAG_T3_R2 = IFLAG_T3_WAIT_FIRST | IFLAG_T3_EARLY_RES1 | IFLAG_T3_FINAL256
#: Weights before the input poll for AhOproj (the hop-1 producer) / for the early-x QKV runs.
IFLAG_T3_WAIT_FIRST_OPROJ = 1 << 48
IFLAG_T3_WAIT_FIRST_QKV = 1 << 49
IFLAG_T3_R3 = IFLAG_T3_R2 | IFLAG_T3_WAIT_FIRST_OPROJ | IFLAG_T3_WAIT_FIRST_QKV
#: The bits every layer gets unless ``LayerKernels(t3_flags=...)`` says otherwise (bitwise the layer without them).
T3_DEFAULT_FLAGS = IFLAG_T3_R3
# ---- end kIFlagT3* bits ----
# ---- kPIFlags bits of the FaAttn warpgroups (stages/fullAttentionConfig.cuh kIFlagF3*; bits 28 .. 31) ----
#: The two FaAttn warpgroups' softmaxes serialized in tile order.
IFLAG_F3_FA_PING_PONG = 1 << 28
#: The FaAttn ring refills armed and issued by the three non-issuer warps of each warpgroup.
IFLAG_F3_FA_REFILL_SIB = 1 << 29
#: kIFlagFaLateLight's late weight is 13 - ((IFlags >> 30) & 3) of 16 (bits 30, 31).
F3_FA_LATE_W_SHIFT = 30
#: With the ping-pong order: only the exp part of the softmax is ordered (bit 32).
IFLAG_F3_FA_PP_LATE = 1 << 32
#: Each Q-prep thread polls only its half row; the norm halves are exchanged in shared memory (bit 33).
IFLAG_F3_FA_Q_HALF = 1 << 33
#: The bits every one-launch full layer gets unless ``LayerKernels(f3_flags=...)`` says otherwise (ping-pong + refill
#: off the issuer: ordering-only, value-identical to the unflagged layer).
F3_DEFAULT_FLAGS = IFLAG_F3_FA_PING_PONG | IFLAG_F3_FA_REFILL_SIB
# ---- end kIFlagF3* bits ----
# ---- kPIFlags bits of the FaAttn issue and Q hand-off (stages/fullAttentionConfig.cuh kIFlagFa*; bits 34 .. 39) ----
#: The two FaAttn issuer warps issue their MMA chains in tile order (named barriers 9 and 10).
IFLAG_FA_ISSUE_ORDER = 1 << 34
#: With FaEntryPages below the ring, warp 0 requests the rest of the ring once its Q words landed.
IFLAG_FA_ENTRY_AT_Q = 1 << 35
#: QkvSum writes the q columns once more (arena kArAhQ); FaAttn units on odd CTAs read that copy.
IFLAG_FA_Q_REP = 1 << 36
#: A FaAttn unit after the CTA's QKV run gets its geometry and page ids at kernel entry into its own shared-memory meta
#: area (``kSmemFaMeta``), so its deferred prologue reads shared memory.
IFLAG_FA_EARLY_META = 1 << 37
#: QkvSum runs the FaAttn Q prep of each q block and publishes Q_hi / Q_lo at kArAhQ (excludes ``IFLAG_FA_Q_REP``).
IFLAG_FA_Q_PREP = 1 << 38
#: A deferred QKV run's weights (late_first's second runs) are prefetched into L2 at kernel entry.
IFLAG_FA_RUN_L2 = 1 << 39
# ---- end kIFlagFa* bits ----
# ---- plan options of the one-launch full layer with late_first (``fh_full_plan``; host tables only) ----
#: FaAttn slots: FaAttn-first CTAs first, two-run CTAs last.
FH_PLAN_SLOTS = 1
#: FaMerge units on the CTAs that end FaAttn first.
FH_PLAN_MERGES = 2
#: The other bits apply only to full layers that take the previous launch's early x; a full layer without it (the
#: first launch after a stock layer) keeps the default tables.
FH_PLAN_STEADY = 4
# ---- end plan options ----
# ---- FaAttn slot order with every FaAttn-first CTA on the low slots (``fh_full_plan(late64=True)``, with
# ``FH_PLAN_SLOTS``): at <= 64 keyed units (contexts up to 8k) no CTA that runs a QKV run first holds keys, and the
# last slot (the step's own rows at full occupancy) sits on a one-run CTA. Host tables only. ----
FT_PLAN_LATE64 = 8
# ---- end late64 slot order ----
# ---- kPIFlags bits of the warm passes and early weight requests (msoMegakernel.cu kIFlagW1*; bits 56 .. 61) ----
#: A QKV run with early x in whose x is not yet published (sampled in the prologue) runs a warm pass first: the same
#: code with every polled load, global store, weight wait and stamp off, so the real pass runs from a warm i-cache.
IFLAG_W1_WARM_QKV = 1 << 56
#: The same for AhOproj (its canary's first load decides).
IFLAG_W1_WARM_OPROJ = 1 << 57
#: AhOproj requests its own weights once attention Q is published, when they would otherwise follow the QKV run.
IFLAG_W1_OPROJ_AFTER_Q = 1 << 58
#: AhSwaAttn converts its pages to the BF16 tiles before its Q poll (only the step's rows after Q).
IFLAG_W1_SWA_PRE_CONVERT = 1 << 60
# ---- end kIFlagW1* bits ----
#: The experts stage's descriptor flag (``stages/experts_defs.cuh`` ``kMoeFlagMhF2``).
MOE_FLAG_MH_F2 = 32768
# ---- BF16-expert layers: ``kIFlagZ1FinalNorm``: MhF1's next norm is the model's final RMSNorm (gain w): layer 61 ----
IFLAG_Z1_FINAL_NORM = 1 << 26
#: The BF16 layers (0, 61) of a forward take turns on two copies of the BF16 experts workspace.
BF16_COPIES = 2
# ---- end BF16-expert layers ----

#: The experts flags every layer gets by default (``LayerKernels.moe_flags_extra``): the coalesced logit poll and the
#: quantization before the routing.
MOE_FLAGS_I2A_DEFAULT = experts.MOE_FLAG_LOGITS_COOP | experts.MOE_FLAG_QUANT_FIRST
# ---- the FC1 -> h -> FC2 hand-off flags every layer gets by default (FC2 units issued in pairs) ----
MOE_FLAGS_E3_DEFAULT = experts.MOE_FLAG_E3_FC2_PAIRS
#: Every ``MOE_FLAG_E3_*`` bit (a caller that picks its own hand-off flags replaces these bits of ``moe_flags_extra``).
MOE_FLAGS_E3_ALL = (
    experts.MOE_FLAG_E3_H_WIDE
    | experts.MOE_FLAG_E3_FC2_PAIRS
    | experts.MOE_FLAG_E3_RING10
    | experts.MOE_FLAG_E3_FC1_QUADS
    | experts.MOE_FLAG_E3_FC2_QUADS
    | experts.MOE_FLAG_E3_EPI_PRE
    | experts.MOE_FLAG_E3_XSF_PRE
)
# ---- end hand-off flags ----

#: ``kIClearXUnits``.
CLEAR_X_UNITS = 8

#: Dynamic regions of every stage of the merged plans.
DYNAMIC_SWA = dict(schedule_attn.DYNAMIC, MoeExperts=(0, schedule.SMEM_MAIN))


def plan_moe_half(ctas: int = 212) -> schedule.Plan:
    """The MoE half's units per CTA (module docstring)."""
    m = schedule_moe_hops
    assert ctas == 212 and m.DEFAULT_ROW_UNITS == 4
    plan: schedule.Plan = [[] for _ in range(ctas)]
    for c in range(ctas):
        if c < m.UNPOOL_CTA0:
            plan[c] += [("MhPool", c - m.POOL_CTA0), ("MoeExperts", c + 1)]
        elif c < m.ROUTER_CTA0:
            plan[c] += [("MoeExperts", 0), ("MhUnpool", c - m.UNPOOL_CTA0)]
        elif c < m.ROW_CTA0:
            plan[c] += [("MhRouter", c - m.ROUTER_CTA0), ("MoeExperts", 0)]
        elif c < m.CLEAR_CTA0:
            u = c - m.ROW_CTA0
            plan[c] += [("MhH1", u), ("MoeExperts", 0), ("MhFinal", u), ("MhF1", u)]
        else:
            u = c - m.CLEAR_CTA0
            # The early-x buffer copy of the previous launch (IClearX, 8 units of 16 KiB), then MhClear.
            plan[c] += ([("IClearX", u)] if u < CLEAR_X_UNITS else []) + [
                ("MhClear", u),
                ("MoeExperts", 0),
            ]
    return plan


# ---- MoE-front plan variants ----
#: ``MhRouter`` unit code of a row-split unit (``stages/moe_hops.cuh`` ``stageMhRouter``): expert row e, row half h.
def router_unit(e: int, half: int = 0, split: bool = False) -> int:
    return e + 32 * half + 256 if split else e


#: ``MhRouter`` unit code of a rank-local unit (``stageMhRouterLocal``): experts 4 q .. 4 q + 3 of all 256, row half h.
ROUTER_LOCAL_BIT = 512
ROUTER_LOCAL_EXPERTS = 4


def router_local_unit(q: int, half: int) -> int:
    return ROUTER_LOCAL_BIT + 64 * half + q


def plan_moe_half_i2a(
    ctas: int = 212, router_split: bool = False, router_local: bool = False
) -> schedule.Plan:
    """``plan_moe_half`` with the router options. ``router_split``: every expert row is two units of 4 rows (half 0 on
    the router CTAs [128, 160), half 1 on the unpool CTAs [64, 96), before their experts). ``router_local``: every
    rank computes all 256 logits, 128 units of 4 experts x 4 rows on CTAs [64, 192) (after ``MhH1`` on [160, 192))."""
    m = schedule_moe_hops
    plan = plan_moe_half(ctas)
    if router_local:
        for k in range(128):
            c = m.UNPOOL_CTA0 + k
            unit = ("MhRouter", router_local_unit(k % 64, k // 64))
            names = [st for st, _ in plan[c]]
            if "MhRouter" in names:
                plan[c][names.index("MhRouter")] = unit
            else:
                plan[c].insert(names.index("MoeExperts"), unit)
        return plan
    if not router_split:
        return plan
    for c in range(m.ROUTER_CTA0, m.ROW_CTA0):
        e = c - m.ROUTER_CTA0
        assert plan[c][0] == ("MhRouter", e)
        plan[c][0] = ("MhRouter", router_unit(e, 0, True))
    for e in range(32):
        c = m.UNPOOL_CTA0 + e
        assert plan[c][0][0] == "MoeExperts"
        plan[c].insert(0, ("MhRouter", router_unit(e, 1, True)))
    return plan


# ---- end MoE-front plan variants ----


def _unit_bytes(stage: str, unit: int) -> int:
    if stage == "MhRouter" and unit & ROUTER_LOCAL_BIT:  # a rank-local router unit
        return ROUTER_LOCAL_EXPERTS * schedule_moe_hops.ROUTER_UNIT_BYTES
    if stage in ("QkvRun", "AhOproj"):
        return schedule_attn.unit_bytes_attention(stage, unit, 212)
    return schedule_moe_hops.unit_bytes_moe_hops(stage, unit)


def swa_layer_tables(
    ctas: int = 212,
    late_ctas: int = schedule_attn.LATE_CTAS,
    router_split: bool = False,
    router_local: bool = False,
    oproj_off_merge: bool = False,
    ah: int = 0,
) -> schedule.Tables:
    """The sliding-window layer's role tables: ``schedule_attn``'s attention plan, then the MoE half (``router_split``,
    ``router_local``; ``oproj_off_merge``; ``ah``: the build's ``MSO_MK_AH`` mask, bit 1 ``gate_in_merge``, bit 2
    ``fin_first``, bit 8 ``gate_runs_off_fin``; bits 4 and 16 / 32 are kernel only; see
    ``schedule_attn.plan_attention``)."""
    attn = schedule_attn.plan_attention(
        ctas,
        late_ctas=late_ctas,
        oproj_off_merge=oproj_off_merge,  # o_proj off the merge CTAs
        gate_in_merge=bool(ah & 1),  # MSO_MK_AH bit 1
        fin_first=bool(ah & 2),  # MSO_MK_AH bit 2
        gate_runs_off_fin=bool(ah & 8),  # MSO_MK_AH bit 8
    )
    moe = plan_moe_half_i2a(ctas, router_split, router_local)  # the router options
    plan = [a + m for a, m in zip(attn, moe)]
    return schedule.build_tables(plan, _unit_bytes, dynamic=DYNAMIC_SWA)


# ---- ghost units (msoMegakernel.cu stageI2bGhost) on CTAs that are idle after their experts unit ----
#: (CTA, unit): QkvFin blocks (unit < 64: q 0-7, k 16-17, v 18-19, gate 20-27) on the 20 clear CTAs and SwaAttn blocks
#: 0-31 of KV head 0 (unit 64 + b) on the 32 router CTAs, so the units run on SMs all over the GPU.
GHOST_PLAN = [
    (schedule_moe_hops.CLEAR_CTA0 + i, u)
    for i, u in enumerate(list(range(8)) + list(range(16, 28)))
] + [(schedule_moe_hops.ROUTER_CTA0 + b, 64 + b) for b in range(32)]
#: L2-warm units (stageI2bWarm, kI2bWarmUnits) on the first pool CTAs.
WARM_UNITS = 4
WARM_CTA0 = schedule_moe_hops.POOL_CTA0
# ---- end ghost units ----


def full_layer_tables(
    ctas: int,
    units: int,
    region_bytes: int,
    router_split: bool = False,
    router_local: bool = False,
    ghost: bool = False,
    warm: bool = False,
) -> schedule.Tables:
    """Launch F of a full-attention layer: ``schedule_full``'s attention plan, then the MoE half
    (``router_split``, ``router_local``; ``ghost``: then the ghost units of ``GHOST_PLAN``, ``warm``: the L2-warm
    units; each on a CTA whose last MoE unit is the experts)."""
    attn = schedule_full.plan_full_attention(ctas, units)
    moe = plan_moe_half_i2a(ctas, router_split, router_local)  # the router options
    plan = [a + m for a, m in zip(attn, moe)]
    if ghost:  # ghost units
        for c, u in GHOST_PLAN:
            assert plan[c][-1][0] == "MoeExperts"
            plan[c].append(("I2bGhost", u))
    if warm:  # L2-warm units
        for u in range(WARM_UNITS):
            assert plan[WARM_CTA0 + u][-1][0] == "MoeExperts"
            plan[WARM_CTA0 + u].append(("I2bWarm", u))
    dyn = {
        "FaAttn": (0, int(region_bytes)),
        "MoeExperts": (0, schedule.SMEM_MAIN),
        "I2bGhost": (0, schedule.SMEM_MAIN),  # ghost units
        "I2bWarm": (0, 0),  # L2-warm units
    }
    return schedule.build_tables(plan, _unit_bytes, dynamic=dyn)


# ---- one-launch full layer ----
def heads_first_qkv(w_qkv: torch.Tensor) -> torch.Tensor:
    """The checkpoint-order QKV weight ``[2304, 8192]`` with its 64-row blocks in the heads-first order."""
    v = w_qkv.contiguous().view(schedule.QKV_BLOCKS, schedule.QKV_BLOCK_ROWS, -1)
    return v[HEADS_FIRST_BLOCKS].reshape(w_qkv.shape).contiguous()


def plan_full_one(
    ctas: int,
    units: int,
    late_ctas: int = 0,
    gates_after_fa: bool = False,
    gates_early: bool = False,
    late_first: bool = False,
    late_gates_after: bool = False,
    late_sums_after: bool = False,
    fh_slots: bool = False,
    fh_merges: bool = False,
    ft_gate: bool = False,
    ft_late64: bool = False,
    lm_ticket: bool = False,
) -> schedule.Plan:
    """The attention half of a full-attention layer as one launch: per CTA its ``QkvRun`` (run = CTA, checkpoint row
    order, weights at entry; with early x before the dependency wait), the 36 ``QkvSum`` units on the CTAs holding
    their block's last chunk (the projection as Lamport words into the arena, ``kIFlagQkvArena``), then
    ``schedule_full``'s units (``FaAttn``: its prologue and first ring tiles follow the CTA's run, ``FaMerge``,
    ``Clear``, ``AhOproj``). ``gates_after_fa`` (with ``late_ctas``): the 16 gate blocks' QkvSum units go to early merge
    CTAs after their FaAttn unit (only FaMerge reads the gate), so no late CTA runs a QkvSum before its FaAttn.
    ``gates_early`` (with ``late_ctas``): the 16 gate blocks' QkvSum units on early CTAs that hold no q / k / v sum,
    between their QKV run and their FaAttn unit (an early CTA waits there for its Q anyway), so no late CTA runs a
    QkvSum before its FaAttn and the gates are summed long before FaMerge reads them. ``late_first`` (with
    ``late_ctas``): the late CTAs run no QKV run (their FaAttn unit is first, its prologue and ring at kernel entry);
    their pure-gate runs are second runs of as many early CTAs that hold no QkvSum, and the gate sums sit on early CTAs
    as with ``gates_early`` (the sums read the runs' partials by run, so the values do not change).
    ``late_gates_after`` (with ``late_first``): the late CTAs run their pure-gate runs, and 16 of them the gate sums,
    after their FaAttn unit (they finish it first), so no early CTA runs a second run or a gate sum before its FaAttn.
    ``late_sums_after`` (with ``late_first``): the gate runs stay second runs of early CTAs, the 16 gate sums move to
    late CTAs after their FaAttn unit. ``fh_slots`` / ``fh_merges``: ``fh_full_plan``'s FaAttn slot order / FaMerge
    placement. ``ft_gate`` (a ``library.ft_gate`` build): no gate QkvSum unit (FaMerge sums its row's gate from the
    gate runs' partials itself), wherever the plan put them. ``ft_late64`` (with ``fh_slots``): ``fh_full_plan``'s
    ``late64`` slot order. ``lm_ticket`` (an ``MSO_MK_LM`` bit-1 build): ``lm_ticket_plan``'s FaMerge candidates
    on every CTA, applied last."""
    attn = schedule_full.plan_full_attention(ctas, units)
    second: Dict[int, int] = {}  # late_first's second runs
    post: Dict[int, list] = {}  # late_gates_after's units after FaAttn
    after: Dict[int, int] = {}
    plan: schedule.Plan = []
    if late_ctas == 0:
        owners = {schedule.qkv_block_owner(b, ctas, 4): b for b in range(schedule.QKV_BLOCKS)}
        run_of = {c: c for c in range(ctas)}
    else:
        # Heads-first weight (kIFlagQkvHeadsFirst): the last late_ctas CTAs (resident last, behind the previous
        # launch's tail) take pure-gate runs and the gate blocks' QkvSum units; the q / k / v blocks, which FaAttn
        # waits for, are summed on early CTAs (the sliding-window placement, schedule_attn.plan_attention).
        sa = schedule_attn
        early = list(range(ctas - late_ctas))
        late = list(range(ctas - late_ctas, ctas))
        late_runs = sa._spread(sa.pure_gate_runs(ctas), late_ctas)
        early_runs = [r for r in range(ctas) if r not in set(late_runs)]
        run_of = dict(zip(early + late, early_runs + late_runs))
        fin_qkv = sa._spread(early, sa.FIN_QKV)
        if late_first and late_gates_after:  # the late CTAs' runs and gate sums after their FaAttn
            owners = {c: b for b, c in enumerate(fin_qkv)}
            post = {c: [("QkvRun", r)] for c, r in zip(late, late_runs)}
            for k, c in enumerate(sa._spread(late, schedule.QKV_BLOCKS - sa.FIN_QKV)):
                post[c].append(("QkvSum", sa.FIN_QKV + k))
            for c in late:
                del run_of[c]
        elif late_first:  # the late CTAs' runs as second runs of early CTAs
            rest = [c for c in early if c not in set(fin_qkv)]
            fin_g = [] if late_sums_after else sa._spread(rest, schedule.QKV_BLOCKS - sa.FIN_QKV)
            owners = {c: b for b, c in enumerate(fin_qkv + fin_g)}
            if late_sums_after:
                for k, c in enumerate(sa._spread(late, schedule.QKV_BLOCKS - sa.FIN_QKV)):
                    post[c] = [("QkvSum", sa.FIN_QKV + k)]
            free = [c for c in rest if c not in set(fin_g)]
            second = dict(zip(sa._spread(free, late_ctas), late_runs))
            for c in late:
                del run_of[c]
        elif gates_early:  # the gate sums on early CTAs
            rest = [c for c in early if c not in set(fin_qkv)]
            fin = fin_qkv + sa._spread(rest, schedule.QKV_BLOCKS - sa.FIN_QKV)
            owners = {c: b for b, c in enumerate(fin)}
        elif gates_after_fa:
            merges = [c for c in range(min(ctas - late_ctas, 128)) if c not in set(fin_qkv)]
            fin_g = sa._spread(merges, schedule.QKV_BLOCKS - sa.FIN_QKV)
            owners = {c: b for b, c in enumerate(fin_qkv)}
            after = {c: sa.FIN_QKV + k for k, c in enumerate(fin_g)}
        else:
            fin = fin_qkv + sa._spread(late, schedule.QKV_BLOCKS - sa.FIN_QKV)
            owners = {c: b for b, c in enumerate(fin)}
    for c in range(ctas):
        p = [("QkvRun", run_of[c])] if c in run_of else []  # late_first: none on the late CTAs
        p += [("QkvRun", second[c])] if c in second else []  # late_first's second runs
        p += [("QkvSum", owners[c])] if c in owners else []
        a = list(attn[c])
        if c in post:  # late_gates_after / late_sums_after
            assert a[0][0] == "FaAttn", (c, a)
            a[1:1] = post[c]
        if c in after:
            assert a[0][0] == "FaAttn" and a[1][0] == "FaMerge", (c, a)
            a.insert(1, ("QkvSum", after[c]))
        plan.append(p + a)
    if fh_slots or fh_merges:  # FaAttn slot order / FaMerge placement
        plan = fh_full_plan(plan, slots=fh_slots, merges=fh_merges, late64=ft_late64)
    if ft_gate:  # no gate QkvSum units
        plan = ft_drop_gate_sums(plan, heads_first=late_ctas > 0)
    if lm_ticket:  # FaMerge by arrival ticket
        plan = lm_ticket_plan(plan)
    return plan


# ---- the one-launch full layer without gate QkvSum units (MSO_MK_FT_GATE builds) ----
def ft_gate_blocks(heads_first: bool) -> List[int]:
    """The QkvSum unit (= weight block) of each gate head h: ``20 + h`` in the heads-first weight, else ``2 h + 1``."""
    return (
        [schedule_attn.FIN_QKV + h for h in range(16)]
        if heads_first
        else [2 * h + 1 for h in range(16)]
    )


def ft_drop_gate_sums(plan: schedule.Plan, heads_first: bool) -> schedule.Plan:
    """``plan`` without its 16 gate QkvSum units (FaMerge of an ``MSO_MK_FT_GATE`` build sums its row's gate from the
    gate runs' partials and empties them, so a gate QkvSum would wait for words no run writes again). Every other unit
    keeps its CTA and order."""
    gates = set(ft_gate_blocks(heads_first))
    out = [[(st, u) for st, u in units if not (st == "QkvSum" and u in gates)] for units in plan]
    dropped = sum(len(a) - len(b) for a, b in zip(plan, out))
    assert dropped == len(gates), (
        f"found {dropped} gate QkvSum units in the plan, want {len(gates)}"
    )
    return out


# ---- end MSO_MK_FT_GATE plan ----


# ---- FaMerge candidates on every CTA (an MSO_MK_LM bit-1 build: the row is the CTA's arrival ticket) ----
#: FaMerge rows (8 tokens x 16 query heads); an ``MSO_MK_LM`` bit-1 build merges the first LM_ROWS tickets.
LM_ROWS = 128


def lm_ticket_plan(plan: schedule.Plan) -> schedule.Plan:
    """``plan`` (a one-launch full layer's attention half) with its ``FaMerge`` units replaced by one candidate
    ``("FaMerge", c)`` right after the ``FaAttn`` unit of every CTA c that runs one. In an ``MSO_MK_LM`` bit-1 build
    FaMerge ignores the unit and merges the row of its arrival ticket (the first ``LM_ROWS`` CTAs to enter it), so the
    merges run on the CTAs whose FaAttn ends first. Every other unit keeps its CTA and order; FaMerge holds no weight
    region, so the weight issue points do not move."""
    rows = sum(1 for p in plan for st, _ in p if st == "FaMerge")
    assert rows == LM_ROWS, f"found {rows} FaMerge units in the plan, want {LM_ROWS}"
    out = []
    for c, units in enumerate(plan):
        a = [(st, u) for st, u in units if st != "FaMerge"]
        names = [st for st, _ in a]
        assert names.count("FaAttn") <= 1, (c, units)
        if "FaAttn" in names:
            a.insert(names.index("FaAttn") + 1, ("FaMerge", c))
        out.append(a)
    cands = sum(1 for p in out for st, _ in p if st == "FaMerge")
    assert cands >= LM_ROWS, f"found {cands} FaMerge candidates, want >= {LM_ROWS}"
    return out


# ---- end FaMerge candidates ----


# ---- FaAttn slot order and FaMerge placement of the one-launch full layer (value-identical plan changes) ----
def fh_groups(plan: schedule.Plan) -> Dict[str, List[int]]:
    """The CTAs of a one-launch full-layer attention plan by what precedes their ``FaAttn`` unit: ``late`` (nothing:
    FaAttn first), ``double`` (two QKV runs), ``sums`` (a QkvSum), ``plain`` (one QKV run)."""
    out: Dict[str, List[int]] = {"late": [], "double": [], "sums": [], "plain": []}
    for c, units in enumerate(plan):
        names = [st for st, _ in units]
        fa = names.index("FaAttn")
        if fa == 0:
            out["late"].append(c)
        elif names[:fa].count("QkvRun") >= 2:
            out["double"].append(c)
        elif "QkvSum" in names[:fa]:
            out["sums"].append(c)
        else:
            out["plain"].append(c)
    return out


def fh_full_plan(
    plan: schedule.Plan, slots: bool = True, merges: bool = True, late64: bool = False
) -> schedule.Plan:
    """``plan`` (``plan_full_one``'s attention half) with

    * ``slots``: the FaAttn slots (= unit indices: slot s computes pages ``[s P / eff, (s + 1) P / eff)``, the merge
      reads partial slot s) in the order late CTAs (all but the last), plain, sums, double, the last late CTA. At short
      contexts only the first ``eff`` slots hold keys, so they land on the CTAs that run FaAttn first; at full occupancy
      the last slot (the unit holding the step's own rows) stays on a FaAttn-first CTA. Every slot computes the same
      range as before, so the outputs do not change.
    * ``late64`` (with ``slots``): the order late CTAs (all), plain (all but the last), sums, double, the last plain
      CTA, so the first 64 slots are FaAttn-first CTAs.
    * ``merges``: the 128 FaMerge units on the CTAs that end FaAttn first (plain and late), after their FaAttn and its
      trailing QkvSum; the ``Clear`` units and the three-tile o_proj units on the others (the two-run and sum CTAs, then
      plain CTAs if more are needed). A merge reads its row's partials by slot wherever it runs.
    """
    g = fh_groups(plan)
    ctas = len(plan)
    out = [list(u) for u in plan]
    if slots:
        assert len(g["late"]) >= 1
        if late64:  # late64
            order = g["late"] + g["plain"][:-1] + g["sums"] + g["double"] + g["plain"][-1:]
        else:
            order = g["late"][:-1] + g["plain"] + g["sums"] + g["double"] + g["late"][-1:]
        assert sorted(order) == list(range(ctas)), "every CTA holds one FaAttn slot"
        slot_of = {c: s for s, c in enumerate(order)}
        for c in range(ctas):
            out[c] = [("FaAttn", slot_of[c]) if st == "FaAttn" else (st, u) for st, u in out[c]]
    if merges:
        rows = sum(1 for p in plan for st, _ in p if st == "FaMerge")
        clears = sum(1 for p in plan for st, _ in p if st == "Clear")
        tiles = sum(
            len(schedule_attn.oproj_tiles(u)) for p in plan for st, u in p if st == "AhOproj"
        )
        hosts = sorted(g["plain"] + g["late"])
        assert len(hosts) >= rows, (len(hosts), rows)
        hosts = hosts[:rows]
        others = sorted(set(range(ctas)) - set(hosts))
        for c in range(ctas):
            out[c] = [(st, u) for st, u in out[c] if st not in ("FaMerge", "Clear", "AhOproj")]
        for m, c in enumerate(hosts):
            out[c].append(("FaMerge", m))
        for k, c in enumerate(others[:clears]):
            out[c].append(("Clear", k))
        n3 = tiles - 2 * ctas
        three = (
            others[:n3]
            if n3 <= len(others)
            else others
            + schedule_attn._spread(
                [c for c in range(ctas) if c not in set(others)], n3 - len(others)
            )
        )
        three = set(three)
        tile = 0
        for c in range(ctas):
            n = 3 if c in three else 2
            out[c].append(("AhOproj", schedule_attn.oproj_unit(tile, n)))
            tile += n
        assert tile == tiles, (tile, tiles)
    fa = sorted(u for p in out for st, u in p if st == "FaAttn")
    assert fa == list(range(ctas)), "FaAttn slots"
    assert sorted(u for p in out for st, u in p if st == "FaMerge") == sorted(
        u for p in plan for st, u in p if st == "FaMerge"
    ), "FaMerge rows"
    return out


# ---- end FaAttn slot order and FaMerge placement ----


def full_one_layer_tables(
    ctas: int,
    units: int,
    region_bytes: int,
    late_ctas: int = 0,
    gates_after_fa: bool = False,
    router_split: bool = False,
    router_local: bool = False,
    gates_early: bool = False,
    late_first: bool = False,
    late_gates_after: bool = False,
    late_sums_after: bool = False,
    fh_slots: bool = False,
    fh_merges: bool = False,
    ft_gate: bool = False,
    ft_late64: bool = False,
    lm_ticket: bool = False,
) -> schedule.Tables:
    """A full-attention layer as one launch: ``plan_full_one``, then the MoE half (with the router options
    ``router_split``, ``router_local``, as ``full_layer_tables``)."""
    attn = plan_full_one(
        ctas,
        units,
        late_ctas,
        gates_after_fa,
        gates_early,
        late_first,
        late_gates_after,
        late_sums_after,
        fh_slots,
        fh_merges,
        ft_gate,
        ft_late64,
        lm_ticket,
    )  # plan_full_one's options in its positional order
    # the router options, as the other layers
    moe = plan_moe_half_i2a(ctas, router_split, router_local)
    plan = [a + m for a, m in zip(attn, moe)]
    dyn = {"FaAttn": (0, int(region_bytes)), "MoeExperts": (0, schedule.SMEM_MAIN)}
    tables = schedule.build_tables(plan, _unit_bytes, dynamic=dyn)
    # faAfterQkvRun writes the FaAttn region once the unit before it ends: before FaAttn only the CTA's QKV runs (with
    # late_first: none, or two) and region-less units after them; a FaAttn unit first runs its prologue at entry.
    for c, units_c in enumerate(plan):
        names = [st for st, _ in units_c]
        fa = names.index("FaAttn")
        runs = 0
        while runs < fa and names[runs] == "QkvRun":
            runs += 1
        assert (fa == 0 or runs >= 1) and all(
            _unit_bytes(st, u) == 0 and st not in dyn for st, u in units_c[runs:fa]
        ), (c, units_c[:fa])
    return tables


# ---- end one-launch full layer ----


# ---- the BF16-expert layers (0 and 61) ----
def bf16_tables(t: schedule.Tables) -> schedule.Tables:
    """``t`` with every ``MoeExperts`` unit as ``MoeExpertsBf16`` (same CTAs, same region: the two stages own the
    whole unit region alike)."""
    code_n, code_b = schedule.STAGES["MoeExperts"], schedule.STAGES["MoeExpertsBf16"]
    roles = t.roles.clone()
    roles = torch.where((roles >> 16) == code_n, (roles & 0xFFFF) | (code_b << 16), roles)
    plan = [
        [("MoeExpertsBf16" if st == "MoeExperts" else st, u) for st, u in units] for units in t.plan
    ]
    return schedule.Tables(roles, t.smem.clone(), t.issue.clone(), plan)


class Bf16MoeInputs:
    """One BF16-expert layer's MoE-half inputs (``moe_inputs_from_bf16_layer``); ``copy``: its turn on the BF16
    experts workspace (``BF16_COPIES``)."""

    def __init__(self, hop, window, bias, w_pre, w_post, copy: int, router_full=None):
        self.hop = hop
        self.window = window
        self.bias = bias
        self.w_pre = w_pre
        self.w_post = w_post
        self.copy = int(copy)
        self.router_full = router_full


# ---- end BF16-expert layers ----


class _Capture:
    """Stands in for a stage object's ``Megakernel`` to collect the parameter fields its ``make_layer`` builds."""

    def make_layer(self, fields, flags=0, stamps=False, keep=()):
        return dict(fields), int(flags), tuple(keep)


def _capture_fields(stage, *args, **kwargs) -> Tuple[Dict[str, int], int, tuple]:
    mk = stage.mk
    stage.mk = _Capture()
    try:
        return stage.make_layer(*args, **kwargs)
    finally:
        stage.mk = mk


def _merge(*parts: Tuple[Dict[str, int], int, tuple]) -> Tuple[Dict[str, int], int, tuple]:
    fields: Dict[str, int] = {}
    flags = 0
    keep: tuple = ()
    for f, fl, k in parts:
        for name, v in f.items():
            assert name not in fields or fields[name] == v, f"conflicting parameter field {name}"
            fields[name] = v
        flags |= fl
        keep += k
    return fields, flags, keep


class HopWorkspace(moe_hops.MoeHopWorkspace):
    """``moe_hops.MoeHopWorkspace`` with ``copies`` other than 3 (the clear rule holds for any C >= 3: ``MhClear``
    of launch n empties copy n - 1, next written by launch n + C - 1 >= n + 2)."""

    def __init__(self, lib, comm, rank: int, local_rank: int, copies: int = 4):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert copies >= 3 and comm.Get_size() == moe_hops.RANKS
        self.arena_bytes = moe_hops.arena_bytes(lib)  # the arena copy of the build's ``nb`` level
        # the build's ``fz`` block per copy, after the copies
        self.fz_bytes = moe_hops.fz_bytes(lib)
        self.copies = copies
        self.rank = int(rank)
        assert not self.fz_bytes or copies <= moe_hops.FZ_MAX_COPIES, (
            "the fz blocks follow FZ_MAX_COPIES copies"
        )
        if self.fz_bytes:  # the fz blocks follow the room of FZ_MAX_COPIES copies
            total = moe_hops.FZ_MAX_COPIES * moe_hops.COPY_BYTES + copies * self.fz_bytes
        else:
            total = copies * moe_hops.COPY_BYTES
        self.total = total
        self.mcast = McastGPUBuffer(
            total, moe_hops.RANKS, self.rank, int(local_rank), True, comm.py2f()
        )
        self.uc = self.mcast.get_uc_buffer(self.rank, (total // 4,), torch.float32, 0)
        self.mc = self.mcast.get_mc_buffer((total // 4,), torch.float32, 0)
        self.uc.fill_(-0.0)
        dev = self.uc.device
        self.arena = torch.full(
            (copies * self.arena_bytes // 4,), -0.0, dtype=torch.float32, device=dev
        )
        self.peers = torch.tensor(
            [
                self.mcast.get_uc_buffer(r, (4,), torch.float32, 0).data_ptr()
                for r in range(moe_hops.RANKS)
            ],
            dtype=torch.int64,
            device=dev,
        )
        torch.cuda.synchronize()
        comm.Barrier()


class MoeInputs:
    """One layer's MoE-half inputs that are not the stages' weight objects."""

    def __init__(
        self,
        hop: moe_hops.LayerWeights,
        window: experts.Window,
        bias: torch.Tensor,
        gain_pre: torch.Tensor,
        global_scale: torch.Tensor,
        w_post: torch.Tensor,
        router_full: Optional[torch.Tensor] = None,
    ):
        self.router_full = router_full  # FP32 [256, 8192] (the rank-local router)
        self.hop = hop
        self.window = window
        self.bias = bias
        self.gain_pre = gain_pre
        self.global_scale = global_scale
        self.w_post = w_post


class LayerKernels:
    """The whole-layer launches of one rank (one library mode, one workspace pair).

    Args:
        mode: a K-slice compile mode (``lean_s4`` timed, ``check_s4`` checks / stamps).
        comm, rank, local_rank: the TEP8 group (MPI communicator) for the MoE hop workspace.
        copies: workspace copies (>= 3; the launches of one forward must be a multiple of it).
        full_mode: the build of the full layers' launches (default ``mode``): the sliding-window launches can then
            run a build without the full-attention code (``MSO_MK_LANES``) while sharing the workspaces.
        router_split: the router as 64 row-half units (``plan_moe_half_i2a``; default on).
        router_local: every rank computes all 256 logits (``stageMhRouterLocal``; needs ``MoeInputs.router_full``;
            no faster than ``router_split``).
    """

    def __init__(
        self,
        mode: str,
        comm,
        rank: int,
        local_rank: int,
        copies: int = 4,
        timeout_ms: int = 20000,
        full_mode: Optional[str] = None,
        router_split: bool = True,
        router_local: bool = False,
        i2b_flags: int = 0,
        i2b_ghost: bool = False,
        i2b_warm: bool = False,
        t3_flags: Optional[int] = None,
        f3_flags: Optional[int] = None,
        f3_o_replicas: int = 1,
        bf16: bool = False,
        w1_oproj_off_merge: bool = False,
        fa_gates_early: bool = False,
    ):
        self.lib = library.library(mode)
        # kPIFlags bits added to every layer (sliding-window and full); None = T3_DEFAULT_FLAGS
        self.t3_flags = T3_DEFAULT_FLAGS if t3_flags is None else int(t3_flags)
        # kPIFlags bits added to every one-launch full-attention layer; None = F3_DEFAULT_FLAGS
        self.f3_flags = F3_DEFAULT_FLAGS if f3_flags is None else int(f3_flags)
        # Copies of the gated o_proj input FaMerge publishes in a one-launch full layer (AhOproj on CTA c reads copy
        # c % n; the arena holds two before kArFaQkv): fewer CTAs poll the same words.
        assert f3_o_replicas in (1, 2), "kArFaQkv follows two kArAhO replicas"
        self.f3_o_replicas = int(f3_o_replicas)
        self.full_lib = library.library(full_mode or mode)
        # The full layers' launches share the hop workspace, so both builds must have the same arena copy and fz blocks
        if full_mode is not None and full_mode != mode:
            assert moe_hops.arena_bytes(self.full_lib) == moe_hops.arena_bytes(self.lib), (
                "mixed hop arena layouts"
            )
            assert moe_hops.fz_bytes(self.full_lib) == moe_hops.fz_bytes(self.lib), (
                "mixed hop fz layouts"
            )
        self.mode = mode
        self.i2b_flags = int(i2b_flags)  # kPIFlags bits added to every sliding-window layer
        self.ws = library.Workspace(self.lib, copies)
        self.hops = HopWorkspace(self.lib, comm, rank, local_rank, copies)
        self.rank = int(rank)
        self.attn = attn_half.AttnHalf(mode, workspace=self.ws, o_replicas=4)
        self.full = full_half.FullHalf(full_mode or mode)
        self.moe_hop = moe_hops.MoeHopStage.__new__(moe_hops.MoeHopStage)
        self.moe_hop.lib = self.lib
        self.moe_hop.fma = moe_hops.check_layout(self.lib)
        self.moe_hop.row_units = moe_hops.row_units(self.lib)
        self.moe_hop.ws = self.ws
        self.moe_hop.hops = self.hops
        self.moe_hop.mk = None
        self.experts = experts.ExpertsStage(mode, workspace=self.ws)
        self.router_local = bool(router_local)  # the rank-local router
        router_split = router_split and not router_local
        # o_proj off the merge CTAs (schedule_attn.plan_attention oproj_off_merge)
        self.w1_oproj_off_merge = bool(w1_oproj_off_merge)
        # The sliding-window plan follows the build's MSO_MK_AH mask (the bit-1 plan holds no gate QkvFin units, which a
        # build without bit 1 would wait for; bit 2 is a plan change only)
        self.ah = library.lever_level(mode, "ah")
        self.swa_tables = swa_layer_tables(
            self.lib.ctas,
            router_split=router_split,
            router_local=router_local,
            oproj_off_merge=self.w1_oproj_off_merge,
            ah=self.ah,
        )
        self.full_tables = full_layer_tables(
            self.lib.ctas,
            self.full.units,
            self.full.region_bytes,
            router_split=router_split,
            router_local=router_local,
            ghost=i2b_ghost,  # ghost units
            warm=i2b_warm,  # L2-warm units
        )
        self.swa_mk = library.Megakernel(self.lib, self.ws, self.swa_tables, timeout_ms=timeout_ms)
        self.full_mk = library.Megakernel(
            self.full_lib, self.ws, self.full_tables, timeout_ms=timeout_ms
        )
        # ---- one-launch full layer ----
        # The one-launch tables take the same MoE-front plan (the router options) as the other layers.
        rt = dict(router_split=router_split, router_local=router_local)
        # An MSO_MK_FT_GATE build's one-launch full layers hold no gate QkvSum units (FaMerge sums the gate)
        self.ft_gate = library.ft_gate(full_mode or mode)
        rt["ft_gate"] = self.ft_gate
        # An MSO_MK_LM bit-1 build's one-launch full layers hold a FaMerge candidate on every CTA (lm_ticket_plan)
        self.lm_ticket = bool(library.lm_level(full_mode or mode) & 1)
        rt["lm_ticket"] = self.lm_ticket
        self.full1_tables = full_one_layer_tables(
            self.lib.ctas, self.full.units, self.full.region_bytes, **rt
        )
        self.full1_mk = library.Megakernel(
            self.full_lib, self.ws, self.full1_tables, timeout_ms=timeout_ms
        )
        # ``v2``: heads-first QKV with pure-gate runs on the late CTAs (run before the wait with kIFlagPreWaitAttn)
        self.full2_tables = full_one_layer_tables(
            self.lib.ctas,
            self.full.units,
            self.full.region_bytes,
            late_ctas=schedule_attn.LATE_CTAS,
            **rt,
        )
        self.full2_mk = library.Megakernel(
            self.full_lib, self.ws, self.full2_tables, timeout_ms=timeout_ms
        )
        # ``v3``: the ``v2`` tables with the gate QkvSum units after FaAttn on early merge CTAs
        self.full3_tables = full_one_layer_tables(
            self.lib.ctas,
            self.full.units,
            self.full.region_bytes,
            late_ctas=schedule_attn.LATE_CTAS,
            gates_after_fa=True,
            **rt,
        )
        self.full3_mk = library.Megakernel(
            self.full_lib, self.ws, self.full3_tables, timeout_ms=timeout_ms
        )
        # ---- end one-launch full layer ----
        # ---- ``v2`` tables with the gate QkvSum units on early CTAs (``fa_gates_early``: ``v2`` layers use them) ----
        self._fa_rt = (rt, timeout_ms)
        self.full4_mk = None
        self.full5_mk = None
        self.fh_plan = 0  # the FH_PLAN_* bits (enable_fa_late_first)
        self.full5fh_mk = None  # FH_PLAN_STEADY's tables for the layers with early x in
        if fa_gates_early:
            self.enable_fa_gates_early()
        # ---- end gate QkvSum placement ----
        # ---- the BF16-expert layers' launches (a *_z1 build): the SWA and the ``v2`` one-launch full tables with
        # the BF16 experts stage in the NVFP4 stage's units ----
        self.bf16 = None
        if bf16:
            from . import bf16_experts

            self.bf16 = bf16_experts.Bf16ExpertsStage(mode, workspace=self.ws)
            self.swa_bf16_mk = library.Megakernel(
                self.lib, self.ws, bf16_tables(self.swa_tables), timeout_ms=timeout_ms
            )
            self.full2_bf16_mk = library.Megakernel(
                self.full_lib, self.ws, bf16_tables(self.full2_tables), timeout_ms=timeout_ms
            )
        # ---- end BF16-expert layers ----
        dev = self.ws.arena.device
        # MhH1 -> MhF1: r' (one buffer: launch n + 1's MhH1 runs after launch n completed, through x_attn).
        self.res1 = torch.zeros(8, moe_hops.HIDDEN, dtype=torch.float32, device=dev)
        # Early x: [copies][8][8192] BF16 Lamport words (kEmpty = -0.0 as FP32 words).
        self.xbuf = torch.full(
            (copies * 8 * moe_hops.HIDDEN // 2,), -0.0, dtype=torch.float32, device=dev
        )
        self._zero_bf16 = torch.zeros(8, moe_hops.HIDDEN, dtype=torch.bfloat16, device=dev)
        self._zero_exp = torch.zeros(8, moe_hops.POOL_WIDTH, dtype=torch.bfloat16, device=dev)
        # ---- extra experts-descriptor flag bits of every layer; default: the coalesced logit poll and the
        # quantization before the routing ----
        # and FC2 units issued in pairs
        self.moe_flags_extra = MOE_FLAGS_I2A_DEFAULT | MOE_FLAGS_E3_DEFAULT
        # ---- end experts-descriptor flag bits ----
        # ---- the ghost units' scratch (zeros = non-empty Lamport words) and parameter block (filled from the
        # first sliding-window layer's block, ``_fill_ghost``) ----
        self.ghost = bool(i2b_ghost)
        self.ghost_params = None
        if self.ghost:
            lib = self.lib
            self._g = dict(
                arena=torch.zeros(lib.arena_bytes // 4, dtype=torch.float32, device=dev),
                qkv_part=torch.zeros(lib.qkv_part_bytes // 4, dtype=torch.float32, device=dev),
                ah_part=torch.zeros(lib.ah_part_bytes // 4, dtype=torch.float32, device=dev),
                pool=torch.zeros(4096, dtype=torch.uint8, device=dev),
                bt=torch.zeros(2 * 256, dtype=torch.int32, device=dev),
                seq=torch.full((1,), 4096, dtype=torch.int32, device=dev),
                pos=torch.arange(8, dtype=torch.int32, device=dev),
                cs=torch.zeros(8 * 64, dtype=torch.float32, device=dev),
            )
            self.ghost_params = torch.zeros(
                len(library.PARAM_FIELDS), dtype=torch.int64, device=dev
            )
            self._ghost_filled = False
        self.warm = bool(i2b_warm)
        self.warm_table = (
            torch.zeros(1 + 2 * 32, dtype=torch.int64, device=dev) if self.warm else None
        )
        self._warm_filled = False
        # ---- end ghost units ----

    # ---- one-launch full-layer table variants ----
    def enable_fa_gates_early(self) -> None:
        """Build the ``full4`` one-launch full-layer tables (the ``v2`` tables with the gate QkvSum units on early
        CTAs); the ``v2`` layers made afterwards use them."""
        rt, timeout_ms = self._fa_rt
        self.full4_tables = full_one_layer_tables(
            self.lib.ctas,
            self.full.units,
            self.full.region_bytes,
            late_ctas=schedule_attn.LATE_CTAS,
            gates_early=True,
            **rt,
        )
        self.full4_mk = library.Megakernel(
            self.full_lib, self.ws, self.full4_tables, timeout_ms=timeout_ms
        )

    def enable_fa_late_first(
        self, gates_after: bool = False, sums_after: bool = False, fh_plan: int = 0
    ) -> None:
        """Build the ``full5`` one-launch full-layer tables (``plan_full_one(late_first=True)``: no QKV run on the late
        CTAs before their FaAttn; ``gates_after``: their gate runs and the gate sums after it, else as second runs of
        early CTAs with the gate sums early, or with ``sums_after`` on late CTAs after their FaAttn); the ``v2``
        layers made afterwards use them. ``fh_plan``: bit 0 ``FH_PLAN_SLOTS``, bit 1 ``FH_PLAN_MERGES``
        (``fh_full_plan``); with bit 2 ``FH_PLAN_STEADY`` they apply only to the layers made with ``early_x_in``."""
        rt, timeout_ms = self._fa_rt
        self.fh_plan = int(fh_plan)  # the FH_PLAN_* bits

        def tables(bits: int):
            return full_one_layer_tables(
                self.lib.ctas,
                self.full.units,
                self.full.region_bytes,
                late_ctas=schedule_attn.LATE_CTAS,
                late_first=True,
                late_gates_after=bool(gates_after),
                late_sums_after=bool(sums_after),
                fh_slots=bool(bits & FH_PLAN_SLOTS),  # FaAttn slot order
                fh_merges=bool(bits & FH_PLAN_MERGES),  # FaMerge placement
                ft_late64=bool(bits & FT_PLAN_LATE64),  # late64 slot order
                **rt,
            )

        steady = bool(self.fh_plan & FH_PLAN_STEADY)  # the plan bits only for early-x-in layers
        self.full5_tables = tables(0 if steady else self.fh_plan)
        self.full5_mk = library.Megakernel(
            self.full_lib, self.ws, self.full5_tables, timeout_ms=timeout_ms
        )
        self.full5fh_mk = None
        if steady and self.fh_plan & (
            FH_PLAN_SLOTS | FH_PLAN_MERGES
        ):  # a second set for early-x-in layers
            self.full5fh_tables = tables(self.fh_plan)
            self.full5fh_mk = library.Megakernel(
                self.full_lib, self.ws, self.full5fh_tables, timeout_ms=timeout_ms
            )

    # ---- end one-launch full-layer table variants ----

    # ---- the MoE half's fields ----
    def _moe_parts(
        self, moe: MoeInputs, x_out: torch.Tensor, res_out: torch.Tensor, capture, check_out
    ):
        outputs = dict(MhOutRes1=self.res1, MhOutXattn=x_out)
        outputs.update(check_out or {})
        hop = _capture_fields(
            self.moe_hop, moe.hop, self._zero_bf16, self._zero_exp, outputs=outputs, capture=capture
        )
        if isinstance(moe, Bf16MoeInputs):  # a BF16-expert layer
            assert self.bf16 is not None, "LayerKernels(bf16=True) builds the BF16-expert layers"
            io = self.bf16.inputs()
            ex_fields, ex_flags, ex_keep = _capture_fields(
                self.bf16,
                moe.window,
                moe.bias,
                moe.w_pre,
                moe.w_post,
                self.rank,
                io,
                own_copy=(moe.copy, BF16_COPIES),
                nvfp4_ws=self.experts.moe_ws,
            )
        else:
            io = self.experts.inputs()
            ex_fields, ex_flags, ex_keep = _capture_fields(
                self.experts,
                moe.window,
                moe.bias,
                moe.gain_pre,
                moe.global_scale,
                moe.w_post,
                self.rank,
                io,
            )
        # ---- the rank-local router reads the full [256, 8192] FP32 table ----
        if self.router_local:
            assert moe.router_full is not None and tuple(moe.router_full.shape) == (
                moe_hops.EXPERTS,
                moe_hops.HIDDEN,
            )
            hop[0]["MhWRouter"] = moe.router_full.data_ptr()
            hop = (hop[0], hop[1], tuple(hop[2]) + (moe.router_full,))
        # ---- end rank-local router ----
        desc = ex_keep[0]
        # plus the default experts flags
        desc[experts.DESC["Flags"]] |= MOE_FLAG_MH_F2 | self.moe_flags_extra
        desc[experts.DESC["Out"]] = 0
        mine = dict(IResOut=res_out.data_ptr(), IXBuf=self.xbuf.data_ptr())
        return [hop, (ex_fields, ex_flags, ex_keep), (mine, 0, (x_out, res_out))]

    def make_swa_layer(
        self,
        attn_args: Sequence[object],
        attn_kwargs: Dict[str, object],
        moe: MoeInputs,
        x_out: torch.Tensor,
        res_out: torch.Tensor,
        capture: Optional[torch.Tensor] = None,
        check_out: Optional[Dict[str, torch.Tensor]] = None,
        stamps: bool = False,
        early_x_in: bool = False,
        early_x_out: bool = False,
        pre_wait_attn: bool = False,
        final_norm: bool = False,
    ):
        """A sliding-window layer's blocks. ``attn_args`` / ``attn_kwargs``: ``AttnHalf.make_layer``'s arguments
        (the slot argument is unused: o_proj publishes into the hop workspace). ``early_x_in``: the QKV runs poll the
        previous launch's early x (which must have been made with ``early_x_out``) before the dependency wait."""
        a = _capture_fields(self.attn, *attn_args, **attn_kwargs)
        parts = [a] + self._moe_parts(moe, x_out, res_out, capture, check_out)
        fields, flags, keep = _merge(*parts)
        fields["IFlags"] = (
            IFLAG_OPROJ_TO_MH
            | IFLAG_MH_RES1_GLOBAL
            | (IFLAG_X_IN if early_x_in else 0)
            | (IFLAG_X_OUT if early_x_out else 0)
            # opt-in for SWA layers
            | (IFLAG_PRE_WAIT_ATTN if (pre_wait_attn and early_x_in) else 0)
            | self.i2b_flags  # kIFlagI2b* bits
            | self.t3_flags  # kIFlagT3* bits
            | (IFLAG_Z1_FINAL_NORM if final_norm else 0)  # layer 61
        )
        if isinstance(
            moe, Bf16MoeInputs
        ):  # BF16 experts: its own tables; the launch takes the Megakernel with it
            mk = self.swa_bf16_mk
            return dict(swa=mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep), mk=mk)
        blocks = self.swa_mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep)
        if self.ghost and not self._ghost_filled:  # ghost units
            self._fill_ghost(blocks[0].tensor)
        if self.warm and not self._warm_filled:  # L2-warm units
            self._fill_warm(blocks[0].tensor, res_out)
        return blocks

    def _fill_warm(self, block: torch.Tensor, res_out: torch.Tensor) -> None:
        """The L2-warm table: the shared workspaces (every copy) and the first sliding-window layer's own buffers."""
        P = library.PARAM
        lib = self.lib
        ranges = [
            (self.ws.qkv_part.data_ptr(), lib.qkv_part_bytes),
            (self.ws.arena.data_ptr(), self.ws.copies * lib.arena_bytes),
            (int(block[P["AhPart"]]), lib.ah_part_bytes),
            (self.xbuf.data_ptr(), self.xbuf.numel() * 4),
            (self.res1.data_ptr(), self.res1.numel() * 4),
            (self.hops.uc.data_ptr(), self.hops.total),
            (self.hops.arena.data_ptr(), self.hops.arena.numel() * 4),
            (res_out.data_ptr(), res_out.numel() * res_out.element_size()),
        ]
        host = torch.zeros(self.warm_table.numel(), dtype=torch.int64)
        host[0] = len(ranges)
        for i, (a, n) in enumerate(ranges):
            host[1 + 2 * i] = int(a)
            host[2 + 2 * i] = int(n)
        self.warm_table.copy_(host)
        torch.cuda.current_stream().synchronize()
        self._warm_filled = True

    # ---- ghost units ----
    def _fill_ghost(self, block: torch.Tensor) -> None:
        """The ghost parameter block: a sliding-window layer's scalars and flags, every buffer a scratch one."""
        P = library.PARAM
        g = self._g
        vals = block[: len(library.PARAM_FIELDS)].clone()
        over = dict(
            Arena=g["arena"].data_ptr(),
            Copy=0,
            NumCopies=1,
            Stamps=0,
            QkvPart=g["qkv_part"].data_ptr(),
            AhPart=g["ah_part"].data_ptr(),
            AhPool=g["pool"].data_ptr(),
            AhBlockTable=g["bt"].data_ptr(),
            AhBtStride=256,
            AhSeqLen=g["seq"].data_ptr(),
            AhPositions=g["pos"].data_ptr(),
            AhCosSin=g["cs"].data_ptr(),
            AhAttnOut=0,
            AhOReplicas=1,
            I2bGhost=0,
        )
        for name, v in over.items():
            vals[P[name]] = int(v)
        self.ghost_params.copy_(vals)
        torch.cuda.current_stream().synchronize()
        self._ghost_filled = True

    # ---- end ghost units ----

    def make_full_layer(
        self,
        w_qkv: torch.Tensor,
        f_args: Sequence[object],
        f_kwargs: Dict[str, object],
        moe: MoeInputs,
        x_out: torch.Tensor,
        res_out: torch.Tensor,
        capture: Optional[torch.Tensor] = None,
        check_out: Optional[Dict[str, torch.Tensor]] = None,
        stamps: bool = False,
        early_x_out: bool = False,
    ):
        """A full-attention layer's blocks: ``dict(q=launch Q, f=launch F)``; ``f_args`` / ``f_kwargs``:
        ``FullHalf.f_fields``'s arguments."""
        q = self.full.q_layer(w_qkv, stamps=stamps)
        f = self.full.f_fields(*f_args, **f_kwargs)
        parts = [f] + self._moe_parts(moe, x_out, res_out, capture, check_out)
        fields, flags, keep = _merge(*parts)
        fields["IFlags"] = (
            IFLAG_FA_OUT_ARENA
            | IFLAG_OPROJ_TO_MH
            | IFLAG_MH_RES1_GLOBAL
            | (IFLAG_X_OUT if early_x_out else 0)
            | self.t3_flags  # kIFlagT3* bits
        )
        if self.ghost:  # ghost units
            fields["I2bGhost"] = self.ghost_params.data_ptr()
        if self.warm:  # L2-warm units
            fields["I2bWarm"] = self.warm_table.data_ptr()
        return dict(q=q, f=self.full_mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep))

    # ---- one-launch full layer ----
    def make_full_layer_one(
        self,
        w_qkv: torch.Tensor,
        f_args: Sequence[object],
        f_kwargs: Dict[str, object],
        moe: MoeInputs,
        x_out: torch.Tensor,
        res_out: torch.Tensor,
        capture: Optional[torch.Tensor] = None,
        check_out: Optional[Dict[str, torch.Tensor]] = None,
        stamps: bool = False,
        early_x_in: bool = False,
        early_x_out: bool = False,
        fa_rev_units: bool = False,
        v2: bool = False,
        fa_late_light: bool = False,
        v3: bool = False,
        packed_qkv: Optional[torch.Tensor] = None,
        final_norm: bool = False,
    ):
        """A full-attention layer as one launch (``full_one_layer_tables``): ``dict(one=blocks)``. ``early_x_in``:
        the QKV runs poll the previous launch's early x before the dependency wait. ``packed_qkv`` (serving): the
        ``qkv.pack_qkv`` of the (heads-first with ``v2``) QKV weight, packed once by the caller."""
        v2 = v2 or v3
        if packed_qkv is not None:  # packed once by the caller
            packed = packed_qkv
        else:
            w = heads_first_qkv(w_qkv) if v2 else w_qkv.contiguous()
            packed = qkv.pack_qkv(w, self.full.lib.qkv_slices)
        q = (qkv.qkv_fields(packed), 0, (packed,))
        f = self.full.f_fields(*f_args, **f_kwargs)
        f[0]["AhOReplicas"] = self.f3_o_replicas  # o_proj input copies
        assert f[0]["AhOReplicas"] <= 2, "kArFaQkv follows two kArAhO replicas"
        parts = [q, f] + self._moe_parts(moe, x_out, res_out, capture, check_out)
        fields, flags, keep = _merge(*parts)
        fields["IFlags"] = (
            IFLAG_FA_OUT_ARENA
            | IFLAG_OPROJ_TO_MH
            | IFLAG_MH_RES1_GLOBAL
            | IFLAG_QKV_ARENA
            | (IFLAG_FA_REV_UNITS if fa_rev_units else 0)
            | (IFLAG_FA_LATE_LIGHT if fa_late_light else 0)
            | (IFLAG_QKV_HEADS_FIRST | (IFLAG_PRE_WAIT_ATTN if early_x_in else 0) if v2 else 0)
            | (IFLAG_X_IN if early_x_in else 0)
            | (IFLAG_X_OUT if early_x_out else 0)
            | self.t3_flags  # kIFlagT3* bits
            | self.f3_flags  # kIFlagF3* bits
            | (IFLAG_Z1_FINAL_NORM if final_norm else 0)  # layer 61
        )
        assert not (fields["IFlags"] & IFLAG_FA_Q_PREP and fields["IFlags"] & IFLAG_FA_Q_REP), (
            "both flags use kArAhQ"
        )
        mk = self.full3_mk if v3 else self.full2_mk if v2 else self.full1_mk
        if v2 and not v3 and self.full4_mk is not None:  # the gate sums on early CTAs
            mk = self.full4_mk
        if v2 and not v3 and self.full5_mk is not None:  # no QKV run on the late CTAs
            mk = self.full5_mk
            if early_x_in and self.full5fh_mk is not None:  # FH_PLAN_STEADY
                mk = self.full5fh_mk
        if isinstance(moe, Bf16MoeInputs):  # a BF16-expert layer: the ``v2`` one-launch tables only
            assert v2 and not v3, "the BF16 full layer has the v2 one-launch tables"
            mk = self.full2_bf16_mk
        return dict(one=mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep), mk=mk)

    # ---- end one-launch full layer ----

    def run_swa(self, layer, x: torch.Tensor, res: torch.Tensor, pdl: bool = True):
        if isinstance(layer, dict):  # a BF16-expert layer
            return layer["mk"].launch(layer["swa"], library.KernelIo(x=x, res=res), pdl=pdl)
        return self.swa_mk.launch(layer, library.KernelIo(x=x, res=res), pdl=pdl)

    def run_full(self, layer, x: torch.Tensor, res: torch.Tensor, pdl: bool = True):
        if "one" in layer:  # a one-launch full layer
            return layer["mk"].launch(layer["one"], library.KernelIo(x=x, res=res), pdl=pdl)
        self.full.qkv(layer["q"], x=x, out=self.full.qkv_buf, pdl=pdl)
        return self.full_mk.launch(
            layer["f"], library.KernelIo(res=res, qkv=self.full.qkv_buf), pdl=pdl
        )


def moe_inputs_from_layer(layer, next_norm_weight: torch.Tensor, rank: int) -> MoeInputs:
    """The MoE half's inputs of a served decoder layer (``core.layers[i]`` of the text model) on ``rank``: the router
    rows, the pool and unpool slices, the two boundaries' coefficients, the three-window expert stack the stock path
    holds."""
    ff = layer.feed_forward
    r = int(rank)
    router = ff.router.weight.data.float()
    router_full = router.contiguous() if router.shape[0] == 256 else None  # the rank-local router
    router = (
        router[32 * r : 32 * r + 32].contiguous() if router.shape[0] == 256 else router.contiguous()
    )
    pool = ff.pooler.weight.data
    pool = (
        pool[512 * r : 512 * r + 512].contiguous() if pool.shape[0] == 4096 else pool.contiguous()
    )
    unpool = ff.unpooler.weight.data
    unpool = (
        unpool[1024 * r : 1024 * r + 1024].contiguous()
        if unpool.shape[0] == 8192
        else unpool.contiguous()
    )
    a1, b1 = layer.add_attn.coefficients()
    a2, b2 = layer.add_ffn.coefficients()
    hop = moe_hops.LayerWeights(
        router=router,
        pool=pool,
        unpool=unpool,
        alpha1=a1.float().contiguous(),
        beta1=b1.float().contiguous(),
        w_preffn=layer.pre_ffn_norm.weight.data.to(torch.bfloat16).contiguous(),
        gain_pf=layer.post_ffn_norm.gain().to(torch.bfloat16).contiguous(),
        alpha2=a2.float().contiguous(),
        beta2=b2.float().contiguous(),
        w_next=next_norm_weight.to(torch.bfloat16).contiguous(),
    )
    state = getattr(ff, "_replicated_experts", None)
    assert state is not None, "the layer has no replicated-experts stacks"
    st = state.stacks
    assert st["w3_w1_weight"].shape[0] == experts.WINDOW, (
        f"the layer's expert stack holds {st['w3_w1_weight'].shape[0]} experts, the stage needs the "
        f"three-window {experts.WINDOW}"
    )
    window = experts.Window(
        w13=st["w3_w1_weight"].view(torch.uint8),
        sf13=st["w3_w1_weight_scale"],
        w2=st["w2_weight"].view(torch.uint8),
        sf2=st["w2_weight_scale"],
        alpha1=st["fc31_alpha"].float().contiguous(),
        scale_c=st["fc31_scale_c"].float().contiguous(),
        alpha2=st["fc2_alpha"].float().contiguous(),
    )
    backend = getattr(ff.experts, "backend", ff.experts)
    gs = backend.fc31_input_scale.float().reshape(1).contiguous()
    gain_pre = (ff.pre_expert_norm.weight.data.float() + 1).to(torch.bfloat16).contiguous()
    return MoeInputs(
        hop=hop,
        window=window,
        bias=ff.expert_bias.data.float().contiguous(),
        gain_pre=gain_pre,
        global_scale=gs,
        w_post=ff.post_expert_norm_weight.data.to(torch.bfloat16).contiguous(),
        router_full=router_full,  # the rank-local router
    )


# ---- BF16-expert layers ----
def moe_inputs_from_bf16_layer(
    layer, next_norm_weight: torch.Tensor, rank: int, copy: int
) -> Bf16MoeInputs:
    """``moe_inputs_from_layer`` for a BF16-expert layer (0 or 61): the same hop weights, the two-window BF16 stack
    ``[own | r - 1]`` of ``replicated_experts`` (``ring_buddy``), the raw pre-expert norm weight (gain ``1 + w``).
    ``next_norm_weight``: the next layer's pre-QKV norm weight, or (layer 61) the final norm's."""
    from . import bf16_experts

    ff = layer.feed_forward
    r = int(rank)
    router = ff.router.weight.data.float()
    router_full = router.contiguous() if router.shape[0] == 256 else None
    router = (
        router[32 * r : 32 * r + 32].contiguous() if router.shape[0] == 256 else router.contiguous()
    )
    pool = ff.pooler.weight.data
    pool = (
        pool[512 * r : 512 * r + 512].contiguous() if pool.shape[0] == 4096 else pool.contiguous()
    )
    unpool = ff.unpooler.weight.data
    unpool = (
        unpool[1024 * r : 1024 * r + 1024].contiguous()
        if unpool.shape[0] == 8192
        else unpool.contiguous()
    )
    a1, b1 = layer.add_attn.coefficients()
    a2, b2 = layer.add_ffn.coefficients()
    hop = moe_hops.LayerWeights(
        router=router,
        pool=pool,
        unpool=unpool,
        alpha1=a1.float().contiguous(),
        beta1=b1.float().contiguous(),
        w_preffn=layer.pre_ffn_norm.weight.data.to(torch.bfloat16).contiguous(),
        gain_pf=layer.post_ffn_norm.gain().to(torch.bfloat16).contiguous(),
        alpha2=a2.float().contiguous(),
        beta2=b2.float().contiguous(),
        w_next=next_norm_weight.to(torch.bfloat16).contiguous(),
    )
    state = getattr(ff, "_replicated_experts", None)
    assert state is not None, "the layer has no replicated-experts stacks"
    st = state.stacks
    w13, w2 = st["w3_w1_weight"], st["w2_weight"]
    assert w13.dtype == torch.bfloat16 and w13.shape[0] == bf16_experts.WINDOW, (
        f"the BF16 layer's stack is {tuple(w13.shape)} {w13.dtype}; the stage needs the two-window BF16 stack"
    )
    window = bf16_experts.Bf16Window(
        w13=w13.view(bf16_experts.WINDOW, 64, 2 * bf16_experts.INTER, 64),
        w2=w2.view(bf16_experts.WINDOW, 64, bf16_experts.HIDDEN, 64),
    )
    return Bf16MoeInputs(
        hop=hop,
        window=window,
        bias=ff.expert_bias.data.float().contiguous(),
        w_pre=ff.pre_expert_norm.weight.data.to(torch.bfloat16).contiguous(),
        w_post=ff.post_expert_norm_weight.data.to(torch.bfloat16).contiguous(),
        copy=copy,
        router_full=router_full,
    )


# ---- end BF16-expert layers ----


def capture_layers() -> List[int]:
    """The DFlash capture layers of the target (``MhCapture``)."""
    return [1, 16, 30, 44, 59]
