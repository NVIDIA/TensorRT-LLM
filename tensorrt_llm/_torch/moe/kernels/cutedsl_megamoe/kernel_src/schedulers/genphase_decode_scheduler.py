# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Generation-phase decode FC12 scheduler: the gen-phase kernel's decode claim policy (``schedule_policy``
``("decode", L)``), opt-in for the die-blind build (``die_aware`` ``off``, beside grouped / phase_interleave)
and the only schedule of the die-aware forms (``expert_half``, ``on`` = n_split).

Claim order. One static source list per form; a claim that returns a past-the-end id moves to the next source, and
past the last source the scheduler returns Done. A source is (side, phase): "own" is this CTA's die
(``%locality_domain_id``), "other" the other die; the counter of (side, phase) is die(side) * 2 + phase, die(own) =
this CTA's die, die(other) = 1 - die (die-blind: die 0, counters 0 / 1).
    off            FC1, FC2
    expert_half    FC1 own, FC2 own, FC1 other, FC2 other
    on (n_split)   FC1 own, FC2 own, FC2 other                  (never FC1 other; with a lead no steal pass)
The steal pass (the other die's sources) starts with fresh forward-only cursors.

Optional static FC1 lead L (``decode_lead``; None = L infinite = exactly the order above): the FC1 and FC2 ids of a
die's stream share its FC1 counter, and the source list keeps the FC1 sources only (off: the one stream; expert_half:
own, then the other die's; n_split: own only, no steal pass). The layout is exact for the stream's K token blocks.
K is the sum over the stream's slots of ceil(tokens / cluster tile M): the per-slot formula and slot set of the
cursors (fc12_mapping ``_advance_phase_cursor``: tokens from the per-slot sizes or the prefix sum, slots below
expert_count; expert_half: only the slots ``_owner_owns_slot`` gives the owner). This module keeps its own copy of the
formula (``_lead_slot_token_blocks``, ``_lead_slot_owner``) so the mapping module, shared with the other schedulers,
carries no decode-specific code; the CPU tests pin K to the cursor walk. expert_half has one K per owner (the steal
pass decodes with the other owner's), n_split one K (both dies' streams cover every slot). Lead builds must call
``compute_stream_token_blocks`` on the scheduler warp (all lanes) once, after W3 and before the first
``gen_next_work``: lane-parallel plain loads and a warp shuffle reduction. With L' = min(L, K - 1),
P = (L' + 1) * b1 and M = K - L' - 1 (b1 / b2 = cluster tiles per token block of the FC1 / FC2 cursors):
    [0, P)                       FC1 ids of token blocks 0..L'
    [P, P + M * (b1 + b2))       segment s < M: the b2 FC2 ids of token block s, then the b1 FC1 ids of token
                                 block s + L' + 1
    [P + M * (b1 + b2), ...)     the FC2 ids of token blocks M..K-1 in order (FC2 local id M * b2 + the offset);
                                 an id from K * (b1 + b2) on is past the end of FC2 and ends the source
Every id below K * (b1 + b2) is a tile (no dummy claims; one past-the-end id per counter and cluster, as without a
lead), and every FC1 tile an FC2 tile needs from its own die has a smaller id of the same counter: FC1 of
token block s sits below P (s <= L') or in segment s - L' - 1 < min(s, M), FC2 of token block s in segment s or
after every segment. A stream with no token blocks (K = 0: L' = -1, P = M = 0) ends at its first id. L >= K - 1
runs every FC1 id of the counter before its FC2 ids. K must be exact: a K below the stream's hands out FC2 tiles
whose FC1 never comes (a hang on fc1_done; the placeholder K = 0 of a build that skips the call is the extreme case),
a K above it maps claims to no tile. Every CTA of the grid computes the same K per owner from the same final sizes
(the communication grid writes them before its ``sizes_ready`` release, and the K pass reads them only after W3), so
every cluster that decodes a counter, the other die's stealers included (expert_half steal pass), uses one layout:
exactly-once needs that across clusters, and the cluster's tile sequence (K1; the leader alone claims) within one.

Waits. The scheduler (gen_next_work and the mapping) never waits for in-kernel progress: no spin, no sleep, no choice
that depends on a peeked counter. The kernel's scheduler warp blocks only at W:
  W1  its queue slot (the consumers release a slot as they take its tile, K1);
  W2  the cluster claim channel (every CTA of the cluster consumes every broadcast);
  W3  the one-time wait for the communication grid's ``sizes_ready`` before the first claim (with a lead, the K
      computation follows it: plain loads of the expert sizes and a warp shuffle, no wait);
  W4  die-blind DLB builds only: the helper-weight READY gate in the publish loop (the external weight transport,
      never a tile of this launch);
  W5  the claim gate (kernel ``claim_gate_on_tma_a``, default on): after publishing its n-th unit the scheduler warp
      waits on a CTA-local mbarrier until this CTA's TMA-A has taken that unit (K7), then claims the next one; Done is
      published without it.
W1, W2 and W5 depend on in-kernel progress, but only on units this cluster claimed earlier. W1 (publishing unit j) waits
for every consumer warp of this CTA to have taken unit j - 2 (two-stage queue; each consumer warp takes a unit after its
part of the previous one, e.g. TMA-B after the previous unit's FC2 fc1_done spin, the MMA warp after the previous unit's
k-tiles). W2 waits for the leader's broadcast and, at the leader, for every CTA's scheduler warp to have consumed the
previous broadcast, i.e. for the cluster's own W1 / W5 on earlier units. W5 (after publishing unit j) waits for this
CTA's TMA-A to have taken j, which it does once it has issued every weight load of unit j - 1, i.e. once AB stages are
free; each weight stage is released by the MMAs of every MMA pair of the cluster, which can need an earlier FC2 unit's
TMA-B fc1_done wait (K3) and the TMEM accumulators the epilogue releases inside each unit. So W1, W2 and W5 are open
once the cluster has finished every unit claimed before the one being claimed or published; W3 waits only for the
communication grid, W4 only for the external weight transport.

Liveness relies on these kernel invariants; changing any of them needs a new argument:
  K1  Every consumer warp of a CTA takes tiles in queue order, queue order is claim order, and ``consume_work``
      releases the stage at once. All CTAs of a cluster see the same sequence (one broadcast id per claim; source
      switches and cursor resets are cluster-uniform).
  K2  An FC1 tile waits only for the communication grid (TMA-B: per-expert ``fc1_sf_ready``; the gather and epilogue
      warps wait once for ``token_data_ready`` / ``metadata_ready`` before their loops), never for another tile.
  K3  The only tile-to-tile wait is the FC2 TMA-B spin on its own fc1_done slot (the extension's peek only sets
      ``peek_ready_bit``); weight prefetch (TMA-A), the MMA, the gather and both epilogues never wait for a tile.
      TMA-A waits only for AB stages (and arrives on the W5 mbarrier, K7); the gather warp's L2 weight prefetch waits
      only for ``sizes_ready`` / ``metadata_ready``.
  K4  An FC1 tile's fc1_done increment is released after the epilogue has taken the NEXT tile from the queue; a flag
      batch holds only consecutive FC1 tiles and is flushed by a full batch, a phase change or Done, and once more
      after the loop. (With the standalone top-k reduce an FC2 unit drains only its reads; an FC1 unit still drains
      fully before its increment.)
  K5  The fc1_done slot of FC2 (owner, token block k) is completed by the FC1 tiles of token block k of: the single
      stream (off), the same owner (expert_half), both owners (n_split).
  K6  n_split only: each die runs at least one cluster that stays until that die's FC1 counter is exhausted (an
      n_split cluster claims its own die's FC1 first and leaves only at Done). Premise, which the kernel's host guard
      assumes and nothing checks at run time (green contexts, MPS): more CTAs than the larger die has SMs are resident
      at once, and a cluster lies within one die. It holds when this grid and the communication grid fit on the GPU
      (184 + 28 <= 212 SMs on the target GPU) and everything that can hold SMs when the launch starts (the previous
      call's tail stragglers and top-k reduce, the communication grid) finishes without waiting for this launch, so
      the whole grid becomes resident. off / expert_half do not need it: a die without clusters is drained by the
      other die's later sources.
  K7  TMA-A arrives on the W5 mbarrier once per unit it takes (Done included), right after ``consume_work`` and before
      any load of that unit; it never waits on it.
Liveness. K1 orders each cluster's consumers by claim order. FC1 has no tile dependency (K2), and its delayed/batched
increment can flush once the next tile is published and taken (K4). W1/W2 permit that lookahead once earlier cluster
units progress. W5 waits only until this CTA's TMA-A takes the unit just claimed and published, requiring weight stages
freed by MMAs of units this cluster claimed earlier; it adds no wait for a later claim or directly for another cluster's
tile. W3/W4 depend only on external producers.
  off / expert_half: FC2 depends only on its owner's FC1 tiles. Without a lead, that owner's FC1 counter is observed
  exhausted before FC2 is claimed; with a lead, the dependencies have smaller IDs on the same counter. The earliest
  unfinished claim can progress, including the lookahead needed to release its FC1 increment; induction gives
  completion.
  n_split without a lead: FC1 own is the first and only FC1 source. No FC1 claim or increment is held behind an FC2
  wait. With K6, both dies finish their FC1 halves, releasing every FC2 tile. Stealing FC1 would allow each die to queue
  the other's FC1 behind an FC2 waiting for it, creating a cycle.
  n_split with lead L: each cluster claims only its own die's counter; both dies decode with the same K, so the same
  L' = min(L, K-1) (K = 0: nothing to claim). FC1 block s can follow FC2 blocks only through s-L'-1. FC2 block s needs
  FC1 block s on both dies; its own-die dependencies have smaller counter IDs. W1/W2/W5 add dependencies only on this
  cluster's earlier claims. Blocks 0..L' have no FC2 predecessor, so induction over token blocks, using K6, gives
  completion. A steal pass would queue another counter's FC1 behind the stealer's FC2 and invalidate this order.
Review rule: any further consumer- or scheduler-side wait (beyond W and K3) before a CTA's last unit is done invalidates
this proof; the post-loop waits are covered under Tail.

Tail. Every form ends with the gen-phase kernel's last-arriver tail (``GenphaseTail``): the last CTA to arrive resets
the tail_reset span (these claim counters and fc1_done included), then advances the epoch and signals the peers.
``finish_claim_channel`` drains the claim channel because that tail has no grid sync; it relies on every CTA consuming
every broadcast, past-the-end ids included (all CTAs of a cluster run the same claim sequence), which W5 does not
change. After its loop every warp meets only CTA-local joins (sync_threads; the epilogue warps' tail barrier), its CTA's
tail arrival (which never waits; only the last arriver waits, in the cross-rank barrier, for the other ranks' last
arrivers) and finally a release/acquire exit cluster barrier (the scheduler warp included, after
``finish_claim_channel``), which keeps the CTA's shared memory alive until its cluster peers' remote arrives have
landed. All of them follow every unit of the CTA and its last fc1_done flush, and no tile, tail arrival, reset or
cross-rank barrier waits on anything after them, so they add no edge to the argument above. The kernel fires
``launch_dependents`` (in the tail; TMA-A once drained under the standalone top-k reduce) and the communication kernel
releases its dependent at entry, so the next gen-phase launch can start while this one's CTAs leave. It cannot claim
early: the communication kernel releases it only after it saw this rank's epoch advanced (after the reset), and the next
scheduler claims only after its ``sizes_ready`` (W3). No CTA touches a counter after its tail arrival. The source index
and the cursors are registers. The die id is 0 or 1 (two dies); it is clamped so a counter index never leaves the
registered 2 * die_count counters. A wrong die id costs locality, not results (any CTA can run any tile), but for
n_split every cluster reporting one die breaks K6.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from .base import SchedulerBase, SchedulerWorkTileBase
from .fc12_mapping import (
    BlockPhase,
    SwapAbFc12WorkTileInfo,
    create_phase_interleaved_fc12_mapping_state,
    make_fc12_done_tile,
    map_phase_interleaved_fc12_work_id,
    reset_phase_interleaved_cursors_for_owner,
)
from .non_clc_mixed_cga import NonClcMixedCgaConfig, NonClcMixedCgaSchedulerWorker

# The policy: (side, phase) per source; side 0 = this CTA's die (die-blind: the one stream), 1 = the other die;
# phase 0 = FC1, 1 = FC2. Safety-critical: n_split never claims the other die's FC1 (K6 and the argument above).
_SOURCES = {
    "off": ((0, 0), (0, 1)),
    "expert_half": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "on": ((0, 0), (0, 1), (1, 1)),
}


@cute.jit
def _decode_lead(
    work_id: Int32, token_blocks: Int32, blocks_fc1: int, blocks_fc2: int, lead: int
) -> Tuple[Int32, Int32]:
    """Exact lead layout of one die's counter over its ``token_blocks`` (K) token blocks (module docstring):
    ``(is_fc2 as 0 / 1, phase-local id)``. Signed Int32 math: a stream with K = 0 gives L' = -1, P = M = 0, so
    every id is an FC2 id past the end. K must be the stream's exact count (``compute_stream_token_blocks``)."""
    segment_ids = Int32(blocks_fc1 + blocks_fc2)
    effective_lead = cutlass.min(Int32(lead), token_blocks - Int32(1))  # L'
    prologue = (effective_lead + Int32(1)) * Int32(blocks_fc1)  # P
    segments = token_blocks - effective_lead - Int32(1)  # M >= 0
    epilogue = prologue + segments * segment_ids  # first id of the FC2-only tail
    is_fc2 = Int32(0)
    local_id = work_id
    if work_id >= epilogue:
        is_fc2 = Int32(1)
        local_id = segments * Int32(blocks_fc2) + work_id - epilogue
    elif work_id >= prologue:
        segment = (work_id - prologue) // segment_ids
        offset = work_id - prologue - segment * segment_ids
        if offset < Int32(blocks_fc2):
            is_fc2 = Int32(1)
            local_id = segment * Int32(blocks_fc2) + offset
        else:
            local_id = prologue + segment * Int32(blocks_fc1) + offset - Int32(blocks_fc2)
    return is_fc2, local_id


@cute.jit
def _lead_slot_token_blocks(mapping, slot: Int32) -> Int32:
    """Token blocks of one slot, the cursors' formula (fc12_mapping ``_advance_phase_cursor``): the slot's tokens
    from the per-slot sizes, else from the inclusive prefix sum, then ceil(tokens / mapping_cluster_tile_m). A copy,
    so the mapping module shared with the other schedulers carries no decode-specific code; the CPU tests pin K to
    the cursor walk."""
    token_count = Int32(0)
    if cutlass.const_expr(mapping.expert_token_sizes is not None):
        token_count = mapping.expert_token_sizes[slot]
    else:
        prefix_end = mapping.expert_token_prefix_sum[slot]
        prefix_begin = Int32(0)
        if slot > Int32(0):
            prefix_begin = mapping.expert_token_prefix_sum[slot - Int32(1)]
        token_count = prefix_end - prefix_begin
    tile_m = mapping.mapping_cluster_tile_m
    return (token_count + Int32(tile_m - 1)) // Int32(tile_m)


@cute.jit
def _lead_slot_owner(mapping, slot: Int32) -> Int32:
    """expert_half: the owner of a load-sorted slot, the cursors' rule (fc12_mapping ``_owner_owns_slot``): its
    expert's half ``[owner * E / 2, (owner + 1) * E / 2)``, the half the host placed on that die."""
    return (mapping.expert_remap[slot] * Int32(mapping.owner_count)) // Int32(mapping.expert_count)


@cute.jit
def _count_stream_token_blocks(mapping) -> Tuple[Int32, Int32]:
    """K of the stream(s), scheduler warp, all 32 lanes: lane l sums slots l, l + 32, ... below expert_count (the
    cursors' loop bound) with the cursors' per-slot formula, then a butterfly shuffle makes the sums warp-uniform.
    Returns (K of owner 0, K of owner 1) for expert_half (``_lead_slot_owner``), else (K, 0)."""
    lane = cute.arch.lane_idx()
    blocks0 = Int32(0)
    blocks1 = Int32(0)
    for chunk in cutlass.range_constexpr((mapping.expert_count + 31) // 32):
        slot = Int32(chunk * 32) + lane
        if slot < Int32(mapping.expert_count):
            blocks = _lead_slot_token_blocks(mapping, slot)
            if cutlass.const_expr(mapping.expert_half):
                owner = _lead_slot_owner(mapping, slot)
                if owner == Int32(0):
                    blocks0 = blocks0 + blocks
                elif owner == Int32(1):
                    blocks1 = blocks1 + blocks
            else:
                blocks0 = blocks0 + blocks
    for step in cutlass.range_constexpr(5):
        blocks0 = blocks0 + Int32(
            cute.arch.shuffle_sync_bfly(blocks0, offset=16 >> step, mask=0xFFFFFFFF, mask_and_clamp=31)
        )
        if cutlass.const_expr(mapping.expert_half):
            blocks1 = blocks1 + Int32(
                cute.arch.shuffle_sync_bfly(blocks1, offset=16 >> step, mask=0xFFFFFFFF, mask_and_clamp=31)
            )
    return blocks0, blocks1


class GenphaseDecodeFc12Scheduler(SchedulerBase):
    """Static per-form source list over per-(die, phase) atomic counters; see the module docstring."""

    pipeline_mbarriers_region = "fc12.genphase_decode.scheduler.pipeline_mbarriers"
    work_tiles_region = "fc12.genphase_decode.scheduler.work_tiles"
    work_tile_type = SwapAbFc12WorkTileInfo
    # Loop-carried device state; every other attribute is a host constant copied on rebuild. ``_lead_k_per_owner``
    # (the lead layout's K per owner, set by compute_stream_token_blocks) is () without a lead: no MLIR value.
    _device_fields = ("_work_id_worker", "_mapping", "_source", "_lead_k_per_owner")

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {"expert_count": int, "intermediate_gateup_size": int, "hidden_size": int}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "is_swap_ab": bool,
            "launch_cluster_count": int,
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "die_aware": OptionalRequirement(str),
            "decode_lead": OptionalRequirement(Optional[int]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        for name in ("mma_tiler_mnk", "cluster_shape_mn", "use_2cta_instrs", "token_padding_block",
                     "sf_padding_block", "work_id_mode", "is_swap_ab", "launch_cluster_count"):
            setattr(self, name, impl_desc[name])
        self.die_aware = impl_desc.get("die_aware") or "off"
        self.decode_lead = impl_desc.get("decode_lead")
        self._validate(impl_desc.get("fallback_cluster_shape_mn"))

        self.n_split = self.die_aware == "on"
        self.expert_half = self.die_aware == "expert_half"
        self.die_count = 1 if self.die_aware == "off" else 2
        sources = _SOURCES[self.die_aware]
        if self.decode_lead is not None:
            sources = tuple(source for source in sources if source[1] == 0)  # one counter per die, both phases
        self.sources = sources
        self._fc2_source_mask = sum(1 << index for index, (_, phase) in enumerate(sources) if phase == 1)
        self._first_other_side_source = next(
            (index for index, (side, _) in enumerate(sources) if side == 1), None
        )

        # swap-AB: the mapping's M is the token tile, its N the weight-row (feature) tile
        cta_tile_m = self.mma_tiler_mnk[0] // (2 if self.use_2cta_instrs else 1)
        self.mapping_cta_tile_shape_mnk = (self.mma_tiler_mnk[1], cta_tile_m, self.mma_tiler_mnk[2])
        self.mapping_cluster_shape_mn = (self.cluster_shape_mn[1], self.cluster_shape_mn[0])
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=NonClcMixedCgaConfig(
                preferred_cluster_shape=self.cluster_shape_mn,
                fallback_cluster_shape=None,
                launch_cluster_count=self.launch_cluster_count,
                preferred_cluster_count=None,
                fallback_cluster_count=None,
            ),
            work_id_mode="atomic_counter",
            stream_count=2 * self.die_count,
        )
        self._mapping = None
        self._source = None
        self._lead_k_per_owner = ()

    def _validate(self, fallback_cluster_shape_mn) -> None:
        """The policy's rules live here only; the kernel builds this scheduler in its constructor."""
        if self.die_aware not in _SOURCES:
            raise ValueError(f"die_aware must be off, on or expert_half, got {self.die_aware!r}.")
        lead = self.decode_lead
        if lead is not None and (type(lead) is not int or lead < 0):
            raise ValueError(f"decode_lead must be None (L infinite) or an int >= 0, got {lead!r}.")
        if lead is not None and lead >= 1 << 16:
            raise ValueError(
                "decode_lead must be below 65536 "
                "(a lead longer than the stream orders like None; use None)"
            )
        if self.work_id_mode != "atomic_counter":
            raise ValueError("The decode scheduler claims from atomic counters (work_id_mode='atomic_counter').")
        if not self.is_swap_ab:
            raise ValueError("The decode scheduler maps swap-AB tiles only.")
        if fallback_cluster_shape_mn not in (None, self.cluster_shape_mn):
            raise ValueError("The decode scheduler launches a single cluster shape (no mixed-CGA fallback).")
        if len(self.mma_tiler_mnk) != 3 or len(self.cluster_shape_mn) != 2:
            raise ValueError("mma_tiler_mnk must have three dimensions and cluster_shape_mn two.")
        if self.mma_tiler_mnk[0] % (2 if self.use_2cta_instrs else 1):
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
        for name in ("expert_count", "intermediate_gateup_size", "hidden_size", "launch_cluster_count",
                     "token_padding_block", "sf_padding_block"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < (1 << 31):
                raise ValueError(f"{name} must be a positive Python int below 2**31, got {value!r}.")
        # weight rows of one cluster tile (swap-AB: the CTA tile's M times the cluster's M)
        feature_tile = self.mma_tiler_mnk[0] // (2 if self.use_2cta_instrs else 1) * self.cluster_shape_mn[0]
        if self.die_aware == "on" and (
            self.intermediate_gateup_size % (2 * feature_tile) or self.hidden_size % (2 * feature_tile)
        ):
            # each die's half must be whole cluster tiles, so a block never straddles the dies
            raise ValueError(
                f"n_split halves every expert's weight rows between the dies: intermediate "
                f"{self.intermediate_gateup_size} and hidden {self.hidden_size} must be multiples of two cluster "
                f"feature tiles ({2 * feature_tile} rows)."
            )
        if self.die_aware == "expert_half" and self.expert_count % 2:
            raise ValueError("die_aware expert_half needs an even local expert count (each die owns an expert half).")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the work transport and the cluster claim-broadcast channel."""
        super().register_smem_regions(smem_workspace)
        self._work_id_worker.register_smem_regions(smem_workspace)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register the 2 * die_count claim counters (tail_reset)."""
        self._work_id_worker.register_device_workspace(device_workspace)

    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc=None) -> Tuple[int, int, int]:
        """The persistent grid: launch_cluster_count clusters (the kernel's tail counts promised_launchable_sm_count
        arrivals; n_split's K6)."""
        if max_active_clusters is not None and max_active_clusters < self.launch_cluster_count:
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least launch_cluster_count "
                f"({self.launch_cluster_count})."
            )
        return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], self.launch_cluster_count)

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: cute.Tensor,
        block_idx: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
        expert_remap: Optional[cute.Tensor] = None,
        die: Optional[Int32] = None,
    ) -> None:
        """Bind the transport, the claim worker, the phase cursors and the source index (0). Lead builds: K is a
        placeholder 0 here (a fixed MLIR structure for the device state) and the scheduler warp must call
        ``compute_stream_token_blocks`` before its first ``gen_next_work``; without it every id decodes as FC2 and
        the CTA hangs on fc1_done.

        ``expert_token_sizes``: tokens per slot. ``expert_remap`` (slot -> expert): expert_half only, tells which die
        owns a slot. ``die``: this CTA's die, die-aware forms only, uniform within a cluster; clamped to {0, 1}.
        """
        if cutlass.const_expr(self.expert_half and expert_remap is None):
            raise ValueError("die_aware expert_half needs the slot -> expert remap.")
        if cutlass.const_expr((die is None) != (self.die_count == 1)):
            raise ValueError("The die-aware forms need the CTA's die; the die-blind form takes none.")
        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=None,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )
        owner_index = 0
        if cutlass.const_expr(self.die_count > 1):
            owner_index = cutlass.max(cutlass.min(Int32(die), Int32(1)), Int32(0))
        self._mapping = create_phase_interleaved_fc12_mapping_state(
            expert_count=self.expert_count,
            intermediate_gateup_size=self.intermediate_gateup_size,
            hidden_size=self.hidden_size,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=True,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=None,
            owner_index=owner_index,
            owner_count=self.die_count,
            expert_half=self.expert_half,
            expert_remap=expert_remap if self.expert_half else None,
        )
        self._source = Int32(0)
        if cutlass.const_expr(self.decode_lead is not None):
            # placeholder: K per owner comes from the scheduler warp's compute_stream_token_blocks (mandatory)
            self._lead_k_per_owner = (Int32(0), Int32(0)) if self.expert_half else (Int32(0),)

    @cute.jit
    def compute_stream_token_blocks(self) -> None:
        """Scheduler warp (all 32 lanes), once after ``sizes_ready`` (W3) and before the first ``gen_next_work``: K of
        the lead layout per owner (module docstring). Mandatory in lead builds (``assign_device_members`` leaves a
        placeholder 0 that hangs FC2); a no-op without a lead. Plain loads, no wait."""
        if cutlass.const_expr(self.decode_lead is not None):
            blocks0, blocks1 = _count_stream_token_blocks(self._mapping)
            self._lead_k_per_owner = (blocks0, blocks1) if self.expert_half else (blocks0,)

    @cute.jit
    def finish_claim_channel(self) -> None:
        """Scheduler warp, once after the last claim (last-arriver tail): see the worker's docstring."""
        self._work_id_worker.finish_claim_channel()

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim from the current source until an id maps to a tile; past-the-end moves to the next source, past the
        last source returns Done. Never waits (module docstring). Lead builds: only after
        ``compute_stream_token_blocks`` (the exact layout's K)."""
        work_tile = make_fc12_done_tile(True)
        worker = self._work_id_worker
        mapping = self._mapping
        source = self._source
        coord = worker.cta_coord_in_preferred_cluster
        resolved = Boolean(False)
        while not resolved:
            if source >= Int32(len(self.sources)):
                work_tile = make_fc12_done_tile(True)  # the only Done: every source returned past-the-end
                resolved = Boolean(True)
            else:
                is_fc2 = (Int32(self._fc2_source_mask) >> source) & Int32(1)
                work_id = worker.claim_next_work(Int32(mapping.owner_index) * Int32(2) + is_fc2)
                ends_source = Boolean(True)
                if cutlass.const_expr(self.decode_lead is not None):
                    token_blocks = self._lead_k_per_owner[0]
                    if cutlass.const_expr(self.expert_half):
                        # the counter's owner (flipped for the steal pass): owner 0 or 1
                        token_blocks = token_blocks + (self._lead_k_per_owner[1] - token_blocks) * Int32(
                            mapping.owner_index
                        )
                    is_fc2, work_id = _decode_lead(
                        work_id,
                        token_blocks,
                        mapping.fc1_cursor.blocks_per_token_block,
                        mapping.fc2_cursor.blocks_per_token_block,
                        self.decode_lead,
                    )
                    ends_source = is_fc2 == Int32(1)  # FC1 local ids stay below K * b1; FC2 past the end ends it
                work_tile, has_work, mapping = map_phase_interleaved_fc12_work_id(
                    work_id, Int32(BlockPhase.Linear1) + is_fc2, (coord[1], coord[0], coord[2]), mapping
                )  # Linear2 = Linear1 + 1; swap-AB mapping coordinate
                if has_work:
                    resolved = Boolean(True)
                elif ends_source:
                    source = source + Int32(1)
                    if cutlass.const_expr(self._first_other_side_source is not None):
                        if source == Int32(self._first_other_side_source):
                            # the other die's ids: fresh forward-only cursors, owner flipped
                            mapping = reset_phase_interleaved_cursors_for_owner(
                                mapping, Int32(1) - Int32(mapping.owner_index)
                            )
        self._work_id_worker = worker
        self._mapping = mapping
        self._source = source
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for name in self._device_fields:
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self, values: list) -> "GenphaseDecodeFc12Scheduler":
        index = len(super().__extract_mlir_values__())
        result = super().__new_from_mlir_values__(values[:index])
        for name, value in vars(self).items():  # host constants (the base rebuilt the transport)
            if name not in self._device_fields and not hasattr(result, name):
                setattr(result, name, value)
        for name in self._device_fields:
            state = getattr(self, name)
            count = len(extract_mlir_values(state))
            setattr(result, name, new_from_mlir_values(state, values[index : index + count]))
            index += count
        if index != len(values):
            raise ValueError(f"GenphaseDecodeFc12Scheduler MLIR value count mismatch: {index} vs {len(values)}.")
        return result


__all__ = ["GenphaseDecodeFc12Scheduler"]
