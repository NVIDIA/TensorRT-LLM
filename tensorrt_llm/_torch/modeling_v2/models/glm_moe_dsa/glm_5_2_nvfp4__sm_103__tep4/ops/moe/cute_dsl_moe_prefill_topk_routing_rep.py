# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's context sigmoid top-k with replicated hot experts split across the EP ranks:
the prefill top-k's selection and weights, and tile maps over this rank's 64 home slots plus
`nrep` replica slots (`64 + h` holds hot expert `hot_ids[h]`). A hot expert's rows are cut
by producer CTA and water-filled onto the ranks with the lightest tile loads, so every rank
derives the same assignment without communication. One CuTe DSL kernel (histogram, prefix,
scatter) per histogram width, 129..8192 rows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe import moe_prefill_frontend_schedule as _sched
from ...kernels.moe import moe_prefill_topk_routing_rep as _rep
from .cute_dsl_moe_routing import LOCAL_EXPERTS, MAP_KEYS, TILE, TOP_K, allocate_maps
from .cute_dsl_moe_topk_routing import (
    EXPERTS,
    PREFILL_MAX_ROWS,
    PREFILL_PRODUCER_CTAS,
    _sigmoid_topk,
)
from .cute_dsl_moe_topk_routing_rep import MAX_NREP, reference_slot_maps

#: The histogram widths the replica kernel is built for; the prefill top-k's 4-warp width
#: (65..128 rows) keeps the home mapping on every rank alike.
REP_WARPS = (8, 16)
#: What an `outputs` set holds: the top-k pair, the histogram scratch, the maps.
OUTPUTS = (
    "topk_weights",
    "topk_indices",
    "partial_counts",
    "slot_offsets",
    "expert_bases",
    *MAP_KEYS,
)


def serves(rows: int) -> bool:
    """Whether the replica top-k serves a context step of `rows` rows (129..8192)."""
    return (
        LOCAL_EXPERTS < rows <= PREFILL_MAX_ROWS
        and _sched.propose_schedule(rows).histogram_warps in REP_WARPS
    )


def max_tiles(rows: int, nrep: int) -> int:
    """The tile records the kernel may write for `rows` rows: every slot can end in one
    partial tile."""
    return _rep.max_tiles(int(rows), int(nrep))


def allocate_scratch(nrep: int, device) -> dict[str, torch.Tensor]:
    """The histogram scratch for one `outputs` set: the per-producer-CTA count slab over all
    256 experts, the per-CTA slot offsets and the `64 + nrep` slot bases. The caller adds the
    top-k pair and a map set sized `allocate_maps(capacity, replicas=nrep)`."""
    i32 = {"dtype": torch.int32, "device": device}
    slots = LOCAL_EXPERTS + int(nrep)
    return {
        "partial_counts": torch.zeros(PREFILL_PRODUCER_CTAS * EXPERTS, **i32),
        "slot_offsets": torch.zeros(PREFILL_PRODUCER_CTAS * slots, **i32),
        "expert_bases": torch.zeros(slots, **i32),
    }


@dataclass(frozen=True)
class CompiledPrefillTopkRoutingRep:
    """The replica top-k at each width of `REP_WARPS`, `nrep` replica slots, one logit plane,
    PDL as the engine sets it. `launch(logits, bias, hot_ids, local_expert_offset,
    routed_scale, outputs)` picks the width by row count."""

    nrep: int
    launch: Callable[..., None]


def _launch(handles: dict[int, Callable[..., None]], nrep: int) -> Callable[..., None]:
    def launch(logits, bias, hot_ids, local_expert_offset, routed_scale, outputs):
        rows = int(logits.shape[0])
        n = rows * TOP_K
        q = max_tiles(rows, nrep)
        handles[_sched.propose_schedule(rows).histogram_warps](
            logits.reshape(-1),
            bias,
            hot_ids,
            outputs["topk_weights"].reshape(-1)[:n],
            outputs["topk_indices"].reshape(-1)[:n],
            outputs["partial_counts"],
            outputs["slot_offsets"],
            outputs["expert_bases"],
            outputs["tile_idx_to_expert_idx"][:q],
            outputs["tile_idx_to_mn_limit"][:q],
            outputs["expanded_idx_to_permuted_idx"][:n],
            outputs["permuted_idx_to_expanded_idx"][: q * TILE],
            outputs["total_num_padded_tokens"],
            outputs["num_non_exiting_tiles"],
            rows,
            local_expert_offset,
            float(routed_scale),
        )

    return launch


def build_prefill(nrep: int, device) -> CompiledPrefillTopkRoutingRep:
    """Compile the replica top-k once per width of `REP_WARPS` with `nrep` replica slots and
    launch each once on random logits and an all -1 hot list (the home mapping): a compiled
    handle still loads its module on its first launch. Call at load. Raises if a compile
    fails: the target then keeps the prefill top-k's home mapping on context steps."""
    nrep = int(nrep)
    if not 1 <= nrep <= MAX_NREP:
        raise ValueError(f"nrep must be in 1..{MAX_NREP}, got {nrep}")
    first: dict[int, int] = {}
    for rows in range(LOCAL_EXPERTS + 1, PREFILL_MAX_ROWS + 1):
        if serves(rows):
            first.setdefault(_sched.propose_schedule(rows).histogram_warps, rows)
    handles: dict[int, Callable[..., None]] = {}
    launch = _launch(handles, nrep)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    hot = torch.full((nrep,), -1, dtype=torch.int32, device=device)
    with torch.cuda.device(device):
        for warps, rows in first.items():
            handles[warps] = _rep.compile_topk_routing_rep(
                nrep=nrep, logit_splits=1, warps=warps, use_pdl=_USE_PDL
            )
            outputs = {
                "topk_weights": torch.zeros(rows * TOP_K, dtype=torch.float32, device=device),
                "topk_indices": torch.zeros(rows * TOP_K, dtype=torch.int32, device=device),
                **allocate_scratch(nrep, device),
                **allocate_maps(PREFILL_MAX_ROWS, device, replicas=nrep),
            }
            logits = torch.randn(rows, 1, EXPERTS, dtype=torch.float32, device=device)
            launch(logits, bias, hot, 0, 2.5, outputs)
    return CompiledPrefillTopkRoutingRep(nrep, launch)


def slot_ids(topk_ids: torch.Tensor, hot_ids, local_expert_offset: int) -> torch.Tensor:
    """Per expanded index (`token * 8 + k`), the local slot it maps to on this rank: `e -
    offset` for a home expert that is not hot, `64 + h` for the rows of hot expert `h` whose
    producer CTA's chunk this rank owns, -1 otherwise (the kernel module's host reference)."""
    rows = int(topk_ids.shape[0])
    hot = [int(h) for h in hot_ids]
    warps = _sched.propose_schedule(rows).histogram_warps
    owner = _rep.reference_owner(topk_ids.reshape(rows, TOP_K), rows, warps, hot).reshape(-1)
    ids = topk_ids.reshape(-1).long().cpu()
    ep = int(local_expert_offset) // LOCAL_EXPERTS
    hot_index = {e: h for h, e in enumerate(hot) if e >= 0}
    slot = torch.where(
        torch.tensor([e in hot_index for e in ids.tolist()]),
        torch.tensor([LOCAL_EXPERTS + hot_index.get(e, 0) for e in ids.tolist()]),
        ids - int(local_expert_offset),
    )
    return torch.where(owner == ep, slot, torch.full_like(slot, -1)).to(topk_ids.device)


class CuteDslMoePrefillTopkRoutingRep(OpWrapper):
    """Biased sigmoid top-8 of 256 experts on a context step and this rank's tile maps over
    its home slots and `nrep` replica slots, balanced across the four EP ranks.

    Returns `(weights [rows, 8] fp32, ids [rows, 8] int32, maps)`: views into `outputs`.
    Selection and weights are `cute_dsl_moe_topk_routing`'s prefill kernel's, bit for bit.
    `hot_ids` is int32 `[nrep]`: slot `64 + h` holds global expert `hot_ids[h]` on every rank
    (-1 = unused slot). Each rank's load is its home experts that are not hot, in 128-row
    tiles; the hot experts' rows, in `hot_ids` order and within one expert in producer-CTA
    order, are cut into four consecutive segments that level the loads, and a producer CTA's
    rows of a hot expert go to the rank whose segment holds its first row. The maps'
    `tile_idx_to_expert_idx` holds local slots: `e - local_expert_offset` for a home expert
    that is not hot, `64 + h` for a hot expert's owned rows. Every (token, k) is mapped on
    exactly one EP rank. `outputs`: the top-k pair (>= rows * 8), `allocate_scratch(nrep)`
    and a map set from `allocate_maps(capacity, replicas=nrep)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Each histogram width at its edges, a served context step with the hot list matching
    # its busiest experts (the skew the kernel levels), a random hot list with an unused
    # slot, an all-unused list (the home mapping), 16 replicas, and the token budget.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="129 rows, the 8-warp histogram's first, 8 frequency replicas, rank 0",
            spec=dict(rows=129, nrep=8, hot="freq", offset=0),
        ),
        Cell(
            why="512 rows, the 8-warp histogram's last, 8 random replicas, rank 1",
            spec=dict(rows=512, nrep=8, hot="rand", offset=64),
        ),
        Cell(
            why="513 rows, the 16-warp histogram's first, 8 frequency replicas, rank 2",
            spec=dict(rows=513, nrep=8, hot="freq", offset=128),
        ),
        Cell(
            why="7310 rows, a served context step skewed onto rank 1, 8 frequency replicas",
            spec=dict(rows=7310, nrep=8, hot="freq", offset=64, skew=True),
        ),
        Cell(
            why="2048 rows, every replica slot unused: the home mapping",
            spec=dict(rows=2048, nrep=8, hot="none", offset=192),
        ),
        Cell(
            why="8192 rows, the token budget, 16 frequency replicas, the last rank",
            spec=dict(rows=8192, nrep=16, hot="freq", offset=192, skew=True),
        ),
    )

    note = """
    The trunk builds one handle when context replicas are on (`TRTLLM_GLM52_CTX_EXPERT_
    REPLICAS=R`, R <= the decode replicas, whose loader fills the replica slots) and passes
    each trunk MoE layer's first R hot ids, the decode replicas' list. The FC1 / FC2 stacks
    and per-expert alphas carry `64 + R` slots in the same order and the maps then run on
    the prefill pair compiled for `64 + R` experts. Row counts the 4-warp histogram selects
    (65..128) keep the prefill top-k's home mapping on every rank alike; so do the MTP
    layer and layers without a hot list.

    Not bitwise against the home mapping end to end: a hot expert's rows are computed on
    another rank, so the EP all-reduce adds the bf16 partials in another grouping.
    Selection, weights and each rank's maps are exact. The row order within one slot is
    the histogram's atomic claim order; `reference` returns the canonical order.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        hot_ids: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        compiled: CompiledPrefillTopkRoutingRep,
    ):
        compiled.launch(logits, bias, hot_ids, local_expert_offset, routed_scale, outputs)
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
        compiled: CompiledPrefillTopkRoutingRep,
    ):
        """`noaux_tc_op`'s arithmetic, then the host water fill (`slot_ids`) and
        `reference_slot_maps` of the ids. Fresh tensors; `outputs` and `compiled` are not
        read."""
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
        compiled: CompiledPrefillTopkRoutingRep,
    ) -> None:
        # The compiled handles check nothing: a hot list of another length is read past its
        # end, and maps sized without the replica slots' tiles are written past their end.
        rows = int(logits.shape[0])
        assert serves(rows), f"the replica top-k serves 129..{PREFILL_MAX_ROWS} rows, got {rows}"
        assert logits.dtype == torch.float32 and logits.is_contiguous()
        assert logits.shape[-1] == EXPERTS and logits.numel() == rows * EXPERTS, (
            f"logits must be one fp32 plane of {EXPERTS} experts per row"
        )
        assert tuple(bias.shape) == (EXPERTS,) and bias.dtype == torch.float32
        assert bias.is_contiguous(), "bias must be contiguous fp32 [256]"
        assert local_expert_offset in (0, 64, 128, 192), (
            f"local_expert_offset selects one EP4 window, got {local_expert_offset}"
        )
        nrep = compiled.nrep
        assert (
            hot_ids.dtype == torch.int32
            and tuple(hot_ids.shape) == (nrep,)
            and hot_ids.is_contiguous()
            and hot_ids.data_ptr() % 16 == 0
        ), f"hot_ids must be a contiguous 16-byte aligned int32 [{nrep}]"
        n = rows * TOP_K
        q = max_tiles(rows, nrep)
        slots = LOCAL_EXPERTS + nrep
        need = {
            "topk_weights": (n, torch.float32),
            "topk_indices": (n, torch.int32),
            "partial_counts": (PREFILL_PRODUCER_CTAS * EXPERTS, torch.int32),
            "slot_offsets": (PREFILL_PRODUCER_CTAS * slots, torch.int32),
            "expert_bases": (slots, torch.int32),
            "tile_idx_to_expert_idx": (q, torch.int32),
            "tile_idx_to_mn_limit": (q, torch.int32),
            "expanded_idx_to_permuted_idx": (n, torch.int32),
            "permuted_idx_to_expanded_idx": (q * TILE, torch.int32),
            "total_num_padded_tokens": (1, torch.int32),
            "num_non_exiting_tiles": (1, torch.int32),
        }
        for name, (size, dtype) in need.items():
            t = outputs[name]
            assert t.dtype == dtype and t.is_contiguous(), f"{name} must be contiguous {dtype}"
            assert t.numel() >= size, (
                f"{name} holds {t.numel()}; {rows} rows and {nrep} replica slots write {size}"
            )
            assert t.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned"


cute_dsl_moe_prefill_topk_routing_rep = CuteDslMoePrefillTopkRoutingRep()
