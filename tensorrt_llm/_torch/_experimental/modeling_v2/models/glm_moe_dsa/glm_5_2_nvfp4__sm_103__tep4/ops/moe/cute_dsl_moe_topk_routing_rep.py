# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's decode sigmoid top-k with replicated hot experts: the push top-k's selection
and weights, and tile maps over this rank's 64 home slots plus `nrep` replica slots
(`64 + h` holds hot expert `hot_ids[h]`), each active hot expert assigned to the EP rank
with the fewest active experts. One CuTe DSL kernel per decode row count (9..64)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cutlass
import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_decode_topk_routing_rep as _rep
from .cute_dsl_moe_routing import LOCAL_EXPERTS, MAP_KEYS, TILE, TOP_K
from .cute_dsl_moe_topk_routing import EXPERTS, _sigmoid_topk

#: The decode rows the replica kernel serves: the push top-k's cluster widths.
uses_rep_path = _rep.uses_rep_path
#: The most replica slots a handle takes.
MAX_NREP = _rep.MAX_NREP
OUTPUTS = _rep.OUTPUT_ORDER


def allocate_outputs(rows: int, nrep: int, device) -> dict[str, torch.Tensor]:
    """One output set for exactly `rows` tokens: the weights and ids, and the six maps with
    `min(64 + nrep, 8 * rows)` tiles."""
    return _rep.allocate_outputs(int(rows), int(nrep), device)


@dataclass(frozen=True)
class CompiledTopkRoutingRep:
    """The replica top-k compiled for exactly `rows` rows, `logit_splits` K partials and
    `nrep` replica slots, PDL roles compiled in. `launch(logits, bias, hot_ids,
    local_expert_offset, routed_scale, outputs, expert_hist)` is the kernel call."""

    rows: int
    logit_splits: int
    nrep: int
    signal_gg1: bool
    launch: Callable[..., None]
    expert_hist: bool = False


def build_decode(
    rows: int, logit_splits: int, nrep: int, device, *, signal_gg1: bool, expert_hist: bool = False
) -> CompiledTopkRoutingRep:
    """Compile the replica top-k for exactly `rows` rows (9..64) reading `logit_splits`
    partials, with `nrep` replica slots. Call at load; nothing is launched. It always runs
    as the router's PDL dependent; `signal_gg1` releases FC1 early (powers of two only, as
    the push top-k). `expert_hist`: also writes the step's per-home-expert token counts
    into the int32 `[64]` row the call passes (the L2 prefetch's ranking)."""
    rows, nrep = int(rows), int(nrep)
    if not uses_rep_path(rows):
        raise ValueError(f"the replica top-k serves the push widths, not {rows} rows")
    shape = (rows, EXPERTS) if logit_splits == 1 else (rows, int(logit_splits), EXPERTS)
    logits = torch.zeros(shape, dtype=torch.float32, device=device)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    hot = torch.full((nrep,), -1, dtype=torch.int32, device=device)
    outputs = allocate_outputs(rows, nrep, device)
    signal = bool(_USE_PDL and signal_gg1)
    hist = {"expert_hist": torch.zeros(64, dtype=torch.int32, device=device)} if expert_hist else {}
    as_t = _rep._base._as_tensor
    with torch.cuda.device(device):
        compiled = _rep.compile_fused_topk_routing_rep(
            logits,
            bias,
            hot,
            outputs,
            nrep=nrep,
            wait_for_router_pdl=_USE_PDL,
            signal_gg1_pdl=signal,
            **hist,
        )

    def launch(logits, bias, hot_ids, local_expert_offset, routed_scale, outputs, expert_hist=None):
        compiled(
            as_t(logits.reshape(-1), assumed_align=16),
            as_t(bias),
            as_t(hot_ids, assumed_align=16),
            *(as_t(outputs[name]) for name in OUTPUTS),
            cutlass.Int32(int(logits.shape[0])),
            cutlass.Int32(local_expert_offset),
            cutlass.Float32(routed_scale),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
        )

    return CompiledTopkRoutingRep(rows, int(logit_splits), nrep, signal, launch, bool(expert_hist))


def reference_assignment(topk_ids: torch.Tensor, hot_ids) -> list[int]:
    """The EP rank serving each replica slot (-1: unused or inactive): every active hot
    expert, in slot order, to the rank with the fewest active non-hot-or-assigned experts
    so far (ties: the lower rank)."""
    return _rep.reference_assignment(topk_ids.reshape(-1).cpu(), [int(h) for h in hot_ids])


def slot_ids(topk_ids: torch.Tensor, hot_ids, local_expert_offset: int) -> torch.Tensor:
    """Per expanded index (`token * 8 + k`), the local slot it maps to on this rank:
    `e - offset` for a home expert that is not hot, `64 + h` for hot expert `h` assigned
    here, -1 otherwise."""
    ids = topk_ids.reshape(-1).long().cpu()
    hot = [int(h) for h in hot_ids]
    assign = _rep.reference_assignment(ids, hot)
    ep = int(local_expert_offset) // LOCAL_EXPERTS
    out = torch.full_like(ids, -1)
    for x, e in enumerate(ids.tolist()):
        if e in hot:
            h = hot.index(e)
            if assign[h] == ep:
                out[x] = LOCAL_EXPERTS + h
        elif e // LOCAL_EXPERTS == ep:
            out[x] = e - int(local_expert_offset)
    return out.to(topk_ids.device)


def reference_slot_maps(slots: torch.Tensor, nslots: int) -> dict[str, torch.Tensor]:
    """`reference_maps`' canonical form over `nslots` slots (one tile per active slot at
    decode widths): slots ascending, rows by ascending expanded index."""
    s = slots.reshape(-1).long()
    dev, n = s.device, s.numel()
    own = s >= 0
    counts = torch.bincount(s[own], minlength=nslots)
    tiles_per = (counts + TILE - 1) // TILE
    tile_slot = torch.repeat_interleave(torch.arange(nslots, device=dev), tiles_per)
    tiles = int(tile_slot.numel())
    first_tile = torch.cumsum(tiles_per, 0) - tiles_per
    j = torch.arange(tiles, device=dev) - first_tile[tile_slot]
    mn = torch.arange(tiles, device=dev) * TILE + torch.clamp(
        counts[tile_slot] - j * TILE, max=TILE
    )
    idx = torch.nonzero(own).reshape(-1)
    e = s[idx]
    order = torch.argsort(e * n + idx)
    idx, e = idx[order], e[order]
    start = torch.cumsum(counts, 0) - counts
    row = first_tile[e] * TILE + torch.arange(idx.numel(), device=dev) - start[e]
    e2p = torch.full((n,), -1, dtype=torch.long, device=dev)
    e2p[idx] = row
    p2e = torch.full((tiles * TILE,), -1, dtype=torch.long, device=dev)
    p2e[row] = idx
    i32 = torch.int32
    return {
        "tile_idx_to_expert_idx": tile_slot.to(i32),
        "tile_idx_to_mn_limit": mn.to(i32),
        "expanded_idx_to_permuted_idx": e2p.to(i32),
        "permuted_idx_to_expanded_idx": p2e.to(i32),
        "total_num_padded_tokens": torch.tensor([tiles * TILE], dtype=i32, device=dev),
        "num_non_exiting_tiles": torch.tensor([tiles], dtype=i32, device=dev),
    }


def canonical_slot_maps(
    maps: Mapping[str, torch.Tensor], num_expanded: int, nslots: int
) -> dict[str, torch.Tensor]:
    """A kernel's maps in `reference_slot_maps`' canonical form: within a slot the row
    order is the atomic claim order, so each slot's rows are re-sorted by expanded index
    and both index maps rebuilt from the forward map."""
    tiles = int(maps["num_non_exiting_tiles"].reshape(-1)[0].item())
    dev = maps["tile_idx_to_expert_idx"].device
    tile_slot = maps["tile_idx_to_expert_idx"][:tiles].long()
    mn = maps["tile_idx_to_mn_limit"][:tiles].long()
    rows = [torch.arange(t * TILE, int(mn[t].item()), device=dev) for t in range(tiles)]
    valid = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=dev)
    expanded = maps["permuted_idx_to_expanded_idx"].long()[valid]
    slot = tile_slot[valid // TILE]
    order = torch.argsort(slot * int(num_expanded) + expanded)
    expanded, slot = expanded[order], slot[order]
    first_tile = torch.full((nslots,), -1, dtype=torch.long, device=dev)
    for t in reversed(range(tiles)):
        first_tile[tile_slot[t]] = t
    counts = torch.bincount(slot, minlength=nslots)
    start = torch.cumsum(counts, 0) - counts
    row = first_tile[slot] * TILE + torch.arange(expanded.numel(), device=dev) - start[slot]
    e2p = torch.full((int(num_expanded),), -1, dtype=torch.long, device=dev)
    e2p[expanded] = row
    p2e = torch.full((tiles * TILE,), -1, dtype=torch.long, device=dev)
    p2e[row] = expanded
    i32 = torch.int32
    return {
        "tile_idx_to_expert_idx": tile_slot.to(i32),
        "tile_idx_to_mn_limit": mn.to(i32),
        "expanded_idx_to_permuted_idx": e2p.to(i32),
        "permuted_idx_to_expanded_idx": p2e.to(i32),
        "total_num_padded_tokens": maps["total_num_padded_tokens"].reshape(-1)[:1].to(i32),
        "num_non_exiting_tiles": maps["num_non_exiting_tiles"].reshape(-1)[:1].to(i32),
    }


class CuteDslMoeTopkRoutingRep(OpWrapper):
    """Biased sigmoid top-8 of 256 experts and this rank's tile maps over its home slots
    and `nrep` replica slots.

    Returns `(weights [rows, 8] fp32, ids [rows, 8] int32, maps)`: views into `outputs`.
    Selection and weights are `cute_dsl_moe_topk_routing`'s. `hot_ids` is int32 `[nrep]`:
    slot `64 + h` holds global expert `hot_ids[h]` on every rank (-1 = unused slot). The
    maps' `tile_idx_to_expert_idx` holds local slots: `e - local_expert_offset` for a home
    expert that is not hot, `64 + h` for a hot expert this rank was assigned. Every rank
    computes the same assignment from the same ids, so each (token, k) is mapped on
    exactly one EP rank. `outputs` is `allocate_outputs(rows, nrep)`. `expert_hist`, on a
    handle built with it: the int32 `[64]` row of this step's home-expert token counts.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The push widths' two cluster sizes, the tensor-core router's partials and the SIMT
    # router's plane, a hot list that matches the batch (worst skew), a random one, an
    # all-unused one (home mapping), and the counts for the L2 prefetch.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="64 rows, 8 replicas of this batch's most selected experts, rank 0",
            spec=dict(rows=64, splits=4, nrep=8, hot="freq", offset=0, signal_gg1=True),
        ),
        Cell(
            why="64 rows, 8 frequency replicas, the last rank, writing the prefetch counts",
            spec=dict(
                rows=64, splits=4, nrep=8, hot="freq", offset=192, signal_gg1=True, expert_hist=True
            ),  # fmt: skip
        ),
        Cell(
            why="32 rows, 8 random replicas (some inactive), rank 1",
            spec=dict(rows=32, splits=4, nrep=8, hot="rand", offset=64, signal_gg1=True),
        ),
        Cell(
            why="12 rows on the 2-CTA cluster, 8 frequency replicas, rank 2, no early release",
            spec=dict(rows=12, splits=4, nrep=8, hot="freq", offset=128, signal_gg1=False),
        ),
        Cell(
            why="48 rows, every replica slot unused: the home mapping",
            spec=dict(rows=48, splits=4, nrep=8, hot="none", offset=64, signal_gg1=False),
        ),
        Cell(
            why="64 rows, 16 frequency replicas",
            spec=dict(rows=64, splits=4, nrep=16, hot="freq", offset=128, signal_gg1=True),
        ),
    )

    note = """
    The trunk builds one handle per push width (9..64 rows) when replicas are on, and the
    FC1/FC2 stacks and per-expert alphas carry `64 + nrep` slots in the same order. The MTP
    layer, which has no replicas, passes an all `-1` hot list and gets the home mapping.
    Narrower decode widths keep `cute_dsl_moe_topk_routing`'s block kernel and the home
    mapping on every rank alike, so the assignment stays consistent across ranks.

    Not bitwise against the kernel without replicas end to end: a hot expert's partial is
    computed on another rank, so the EP all-reduce adds the bf16 partials in another
    grouping. Selection, weights and each rank's maps are exact.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        hot_ids: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        compiled: CompiledTopkRoutingRep,
        expert_hist: torch.Tensor | None = None,
    ):
        compiled.launch(
            logits, bias, hot_ids, local_expert_offset, routed_scale, outputs, expert_hist
        )
        rows = int(logits.shape[0])
        n = rows * TOP_K
        weights = outputs["topk_weights"].reshape(-1)[:n].view(rows, TOP_K)
        ids = outputs["topk_indices"].reshape(-1)[:n].view(rows, TOP_K)
        return weights, ids, {k: outputs[k] for k in MAP_KEYS}

    def reference(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        hot_ids: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        compiled: CompiledTopkRoutingRep,
        expert_hist: torch.Tensor | None = None,
    ):
        """`noaux_tc_op`'s arithmetic, then the greedy assignment and `reference_slot_maps`
        of the ids. Fresh tensors; `outputs`, `compiled` and `expert_hist` are not read."""
        weights, ids, _ = _sigmoid_topk(logits, bias, routed_scale)
        slots = slot_ids(ids, hot_ids.tolist(), local_expert_offset)
        return weights, ids, reference_slot_maps(slots, LOCAL_EXPERTS + int(hot_ids.numel()))

    def is_valid(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        hot_ids: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        compiled: CompiledTopkRoutingRep,
        expert_hist: torch.Tensor | None = None,
    ) -> None:
        # The compiled handle checks nothing: a wrong hot list length is read past its end,
        # and an output set sized for fewer tiles is written past its end.
        rows = int(logits.shape[0])
        splits = 1 if logits.dim() == 2 else int(logits.shape[1])
        assert logits.dtype == torch.float32 and logits.is_contiguous()
        assert logits.shape[-1] == EXPERTS and logits.data_ptr() % 16 == 0
        assert tuple(bias.shape) == (EXPERTS,) and bias.dtype == torch.float32
        assert local_expert_offset in (0, 64, 128, 192), (
            f"local_expert_offset selects one EP4 window, got {local_expert_offset}"
        )
        assert rows == compiled.rows, f"compiled for {compiled.rows} rows, got {rows}"
        assert splits == compiled.logit_splits, (
            f"compiled for {compiled.logit_splits} logit plane(s), got {splits}"
        )
        assert (
            hot_ids.dtype == torch.int32
            and tuple(hot_ids.shape) == (compiled.nrep,)
            and hot_ids.is_contiguous()
            and hot_ids.data_ptr() % 16 == 0
        ), f"hot_ids must be a contiguous 16-byte aligned int32 [{compiled.nrep}]"
        ref = allocate_outputs(rows, compiled.nrep, "meta")
        for name in OUTPUTS:
            t = outputs[name]
            assert tuple(t.shape) == tuple(ref[name].shape) and t.dtype == ref[name].dtype, (
                f"{name} must be {ref[name].dtype} {tuple(ref[name].shape)} for {rows} rows "
                f"and {compiled.nrep} replica slots"
            )
            assert t.is_contiguous(), f"{name} must be contiguous"
        if compiled.expert_hist:
            assert (
                expert_hist is not None
                and expert_hist.dtype == torch.int32
                and tuple(expert_hist.shape) == (64,)
                and expert_hist.is_contiguous()
                and expert_hist.data_ptr() % 16 == 0
            ), "this handle writes expert_hist: a contiguous 16-byte aligned int32 [64]"
        else:
            assert expert_hist is None, "this handle writes no expert counts"


cute_dsl_moe_topk_routing_rep = CuteDslMoeTopkRoutingRep()
