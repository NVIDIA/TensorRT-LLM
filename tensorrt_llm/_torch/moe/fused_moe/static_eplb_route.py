# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-off DEP glue for GPT-OSS (TLLM_GPTOSS_DEP3_GLUE) and its static-EPLB slot-routing fast path.

``TLLM_GPTOSS_DEP3_GLUE`` selects parts: ``1`` or ``all`` = every part, else a comma list of

- ``route``: inside CUDA-graph capture, a static EPLB placement (``layer_updates_per_iter: 0``) with redundant slots
  routes expert ids to slot ids with one Triton kernel instead of ``moeComputeRouteKernel``
  (``moeLoadBalanceKernels.cu``), whose warp 0 arbitrates every 32 elements serially. The result is the same slot
  for every element: the replica is chosen by the same rule (below), so dispatch, the per-rank partial sums and the
  combine see the same rows in the same order. Each layer compares the fast path against
  ``torch.ops.trtllm.moe_load_balance_routing`` once, eagerly, before it may be captured; a mismatch keeps the C++
  kernel for that layer. Eager calls always take the C++ kernel.
- ``mxfp8``: GPT-OSS under attention DP with a post-quant one-sided all-to-all takes its MoE input from the fused
  post-attention add+RMSNorm+MXFP8 kernel (``cute_dsl_kernels/gptoss_glue_fusion.py``), which writes the bytes
  ``mxfp8_quantize`` would; see ``gate_norm_mxfp8_dep3`` there.

Unset (the default), importing this module only parses the flag; no kernel is built and nothing changes.

The replica rule of ``moeComputeRouteKernel`` (1024 elements per CTA, flat index f = token * top_k + k): for a valid
expert e with ``count = expertReplicaCount[e] > 1``, the slot is ``globalSlotIds[expertReplicaStartOffset[e] + (n +
blockIdx + epRank') % count]`` where n is the number of elements with expert e before f in f's 1024-element block and
epRank' is the EP rank when routing is offset by rank (attention DP), else 0. A single-replica expert takes its one
slot; an id outside [0, num_experts) takes ``ep_size * slots_per_rank``. Replicas are ordered by global slot id
(``prepareGpuPlacementInfo``).
"""

import functools
import os
import sys
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

DEP3_GLUE_ENV = "TLLM_GPTOSS_DEP3_GLUE"
_PARTS = ("route", "mxfp8")

# moeComputeRouteDevice: kThreadCount * kEltPerThread elements per CTA. The replica rule depends on it.
ROUTE_BLOCK = 1024
# Replicated experts handled per layer (one scan each); more keeps the C++ kernel.
MAX_REPLICATED_EXPERTS = 16


def _parse_parts(value: str) -> FrozenSet[str]:
    value = value.strip().lower()
    if value in ("", "0", "off", "false"):
        return frozenset()
    if value in ("1", "all", "on", "true"):
        return frozenset(_PARTS)
    parts = frozenset(p.strip() for p in value.split(",") if p.strip())
    unknown = sorted(parts - set(_PARTS))
    if unknown:
        print(f"[TRT-LLM] [W] {DEP3_GLUE_ENV}: unknown part(s) {unknown} ignored; known: {list(_PARTS)}",
              file=sys.stderr, flush=True)
    return parts & frozenset(_PARTS)


DEP3_GLUE_PARTS: FrozenSet[str] = _parse_parts(os.environ.get(DEP3_GLUE_ENV, ""))
STATIC_ROUTE_ENABLED = "route" in DEP3_GLUE_PARTS
DEP3_NORM_MXFP8_ENABLED = "mxfp8" in DEP3_GLUE_PARTS

_LOGGED: set = set()


def log_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[TRT-LLM] [I] {msg}", file=sys.stderr, flush=True)


# =============================================================================
# Placement tables (host)
# =============================================================================


def placement_from_assignments(
    assignments: Sequence[int], num_experts: int, ep_size: int, slots_per_rank: int
) -> Tuple[List[int], List[int], List[int]]:
    """(expertReplicaCount, expertReplicaStartOffset, globalSlotIds) as ``setInitialWeightAssignments`` +
    ``prepareGpuPlacementInfo`` build them: replicas in increasing global slot id."""
    total = ep_size * slots_per_rank
    if len(assignments) != total:
        raise ValueError(f"{len(assignments)} assignments for {ep_size} x {slots_per_rank} slots")
    count = [0] * num_experts
    for e in assignments:
        if not 0 <= e < num_experts:
            raise ValueError(f"expert id {e} outside [0, {num_experts})")
        count[e] += 1
    start, offset = [], 0
    for e in range(num_experts):
        start.append(offset)
        offset += count[e]
    global_slot_ids = [total] * total
    placed = [0] * num_experts
    for slot, e in enumerate(assignments):  # rank-major, local slot minor = global slot id order
        global_slot_ids[start[e] + placed[e]] = slot
        placed[e] += 1
    return count, start, global_slot_ids


class StaticRouteTables:
    """The device tables of one layer's static placement."""

    def __init__(self, assignments: Sequence[int], num_experts: int, ep_size: int, ep_rank: int,
                 slots_per_rank: int, device: torch.device):
        count, start, gsi = placement_from_assignments(assignments, num_experts, ep_size, slots_per_rank)
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.invalid_slot = ep_size * slots_per_rank
        replicated = [e for e in range(num_experts) if count[e] > 1]
        if not replicated:
            raise ValueError("no redundant slot: the C++ no-redundant kernel is already a table lookup")
        if len(replicated) > MAX_REPLICATED_EXPERTS:
            raise ValueError(f"{len(replicated)} replicated experts > {MAX_REPLICATED_EXPERTS}")
        self.max_replicas = max(count[e] for e in replicated)
        # direct[e]: the first (only, for single-replica experts) slot; direct[num_experts]: the invalid slot.
        direct = [gsi[start[e]] for e in range(num_experts)] + [self.invalid_slot]
        rep_slots = []
        for e in replicated:
            row = gsi[start[e]:start[e] + count[e]]
            rep_slots.extend(row + [self.invalid_slot] * (self.max_replicas - len(row)))
        i32 = dict(dtype=torch.int32, device=device)
        self.direct = torch.tensor(direct, **i32)
        self.rep_experts = torch.tensor(replicated, **i32)
        self.rep_counts = torch.tensor([count[e] for e in replicated], **i32)
        self.rep_slots = torch.tensor(rep_slots, **i32)
        self.num_replicated = len(replicated)
        self.host = (count, start, gsi)


# =============================================================================
# Kernel
# =============================================================================

_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    # No value or alignment specialization of the runtime arguments: one compile per (NUM_REP, MAX_REPLICAS,
    # USE_PDL), made during the eager check, so a CUDA-graph capture never compiles or loads a module.
    @triton.jit(do_not_specialize=["experts_ptr", "out_ptr", "n_elts", "direct_ptr", "rep_experts_ptr",
                                   "rep_counts_ptr", "rep_slots_ptr", "rank_offset", "num_experts"])
    def _static_eplb_route_kernel(
        experts_ptr,
        out_ptr,
        n_elts,
        direct_ptr,
        rep_experts_ptr,
        rep_counts_ptr,
        rep_slots_ptr,
        rank_offset,
        num_experts,
        BLOCK: tl.constexpr,
        NUM_REP: tl.constexpr,
        MAX_REPLICAS: tl.constexpr,
        USE_PDL: tl.constexpr,
    ):
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n_elts
        e = tl.load(experts_ptr + offs, mask=valid, other=num_experts)
        e = tl.where((e < 0) | (e >= num_experts), num_experts, e)
        slot = tl.load(direct_ptr + e)
        base = pid + rank_offset
        for r in tl.static_range(NUM_REP):
            rep_e = tl.load(rep_experts_ptr + r)
            rep_c = tl.load(rep_counts_ptr + r)
            hit = e == rep_e
            hit_i = hit.to(tl.int32)
            before = tl.cumsum(hit_i, axis=0) - hit_i  # same-expert elements earlier in this block
            idx = (before + base) % rep_c
            s = tl.load(rep_slots_ptr + r * MAX_REPLICAS + idx, mask=hit, other=0)
            slot = tl.where(hit, s, slot)
        tl.store(out_ptr + offs, slot, mask=valid)

    _KERNEL = _static_eplb_route_kernel
    return _KERNEL


@functools.lru_cache(maxsize=None)
def _use_pdl(device: torch.device) -> bool:
    """launchWithPdlWhenEnabled's rule: TRTLLM_ENABLE_PDL (default on) and SM >= 90."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    major, _ = torch.cuda.get_device_capability(device)
    return bool(get_env_enable_pdl() and major >= 9)


def static_eplb_route(token_selected_experts: torch.Tensor, tables: StaticRouteTables,
                      offset_by_ep_rank: bool) -> torch.Tensor:
    """Slot ids for ``token_selected_experts`` ([T, top_k] int32, contiguous) under ``tables``."""
    out = torch.empty_like(token_selected_experts)
    n = token_selected_experts.numel()
    if n == 0:
        return out
    grid = ((n + ROUTE_BLOCK - 1) // ROUTE_BLOCK,)
    use_pdl = _use_pdl(token_selected_experts.device)
    _get_kernel()[grid](
        token_selected_experts,
        out,
        n,
        tables.direct,
        tables.rep_experts,
        tables.rep_counts,
        tables.rep_slots,
        tables.ep_rank if offset_by_ep_rank else 0,
        tables.num_experts,
        BLOCK=ROUTE_BLOCK,
        NUM_REP=tables.num_replicated,
        MAX_REPLICAS=tables.max_replicas,
        USE_PDL=use_pdl,
        num_warps=8,
        launch_pdl=use_pdl,
    )
    return out


# =============================================================================
# Per-layer state: gate, one eager verification, then capture-only use
# =============================================================================


class StaticRouteState:
    """None-like until verified; ``tables`` once the fast path is proven equal for this layer."""

    def __init__(self):
        self.tables: Optional[StaticRouteTables] = None
        self.refused: Optional[str] = None
        self.verified = False


_STATS: Dict[str, int] = {"verified": 0, "refused": 0, "capture_unverified": 0}


def _refuse(state: StaticRouteState, moe, reason: str) -> None:
    state.refused = reason
    _STATS["refused"] += 1
    log_once(f"route-refused:{reason}",
             f"{DEP3_GLUE_ENV} route: C++ moeComputeRoute kept (layer {getattr(moe, 'layer_idx', '?')} and any "
             f"other with this reason): {reason}")


def _build(moe) -> StaticRouteTables:
    lb = moe.layer_load_balancer
    if not lb.is_static_routing():
        raise ValueError("dynamic EPLB (layer_updates_per_iter > 0): the placement moves")
    comm = getattr(moe, "comm", None)
    if comm is not None and getattr(comm, "_rank_mask_enabled", False):
        raise ValueError("EP rank masking (fault tolerance) can reconfigure the placement")
    if getattr(moe, "num_slots", 0) == getattr(moe, "num_experts", 0):
        raise ValueError("no redundant slot")
    return StaticRouteTables(moe.initial_global_assignments, moe.num_experts, moe.ep_size, moe.ep_rank,
                             moe.expert_size_per_partition, torch.device("cuda", torch.cuda.current_device()))


def _verification_inputs(experts: torch.Tensor, tables: StaticRouteTables) -> List[torch.Tensor]:
    """The live input, plus a synthetic multi-block one that hits every replicated expert often and has invalid ids."""
    top_k = experts.shape[1]
    g = torch.Generator().manual_seed(1234 + tables.ep_rank)
    t = 2 * ROUTE_BLOCK // top_k + 37  # 3 CTAs, the last one partial
    syn = torch.randint(0, tables.num_experts, (t, top_k), generator=g, dtype=torch.int32)
    reps = tables.rep_experts.cpu()
    hot = torch.rand((t, top_k), generator=g) < 0.4
    syn[hot] = reps[torch.randint(0, len(reps), (int(hot.sum()),), generator=g)]
    syn[5, 1] = -1
    syn[7, 2] = tables.num_experts
    return [experts, syn.to(experts.device)]


def maybe_static_route(moe, token_selected_experts: torch.Tensor, offset_by_ep_rank: bool) -> Optional[torch.Tensor]:
    """The fast-path slot ids, or None to run ``layer_load_balancer.route``. Called only with the flag set."""
    state = getattr(moe, "_static_route_state", None)
    if state is None:
        state = StaticRouteState()
        moe._static_route_state = state
        try:
            state.tables = _build(moe)
        except Exception as e:  # any miss keeps the C++ kernel for this layer
            _refuse(state, moe, repr(e))
    if state.refused is not None:
        return None
    x = token_selected_experts
    if x.dtype != torch.int32 or x.dim() != 2 or not x.is_contiguous() or not x.is_cuda:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if not state.verified:
        if capturing:
            _STATS["capture_unverified"] += 1
            log_once("route-capture-unverified",
                     f"{DEP3_GLUE_ENV} route: a graph was captured before its layer's eager check; that graph keeps "
                     "the C++ kernel")
            return None
        _verify(state, moe, x, offset_by_ep_rank)
        return None  # eager: the C++ kernel, always
    return static_eplb_route(x, state.tables, offset_by_ep_rank) if capturing else None


def _verify(state: StaticRouteState, moe, x: torch.Tensor, offset_by_ep_rank: bool) -> None:
    lb = moe.layer_load_balancer
    ptr = lb.single_layer_load_balancer_ptr
    try:
        for inp in _verification_inputs(x, state.tables):
            for offset in (False, True):
                ref = torch.ops.trtllm.moe_load_balance_routing(inp, offset, ptr)
                got = static_eplb_route(inp, state.tables, offset)
                if not torch.equal(ref, got):
                    bad = int((ref != got).sum())
                    raise ValueError(f"{bad}/{ref.numel()} slots differ from moe_load_balance_routing "
                                     f"(offset_by_ep_rank={offset}, rows={inp.shape[0]})")
    except Exception as e:
        _refuse(state, moe, f"verification failed: {e!r}")
        return
    state.verified = True
    _STATS["verified"] += 1
    t = state.tables
    log_once("route-verified",
             f"{DEP3_GLUE_ENV} route active: static-EPLB slot routing by one Triton kernel inside CUDA graphs, "
             f"bit-equal to moe_load_balance_routing (first layer {getattr(moe, 'layer_idx', '?')}: "
             f"{t.num_replicated} replicated expert(s), up to {t.max_replicas} replicas, {t.invalid_slot} slots, "
             f"ep_rank {t.ep_rank}); moeComputeRouteKernel should be absent from graphed steps")


def route_stats() -> Dict[str, int]:
    return dict(_STATS)
