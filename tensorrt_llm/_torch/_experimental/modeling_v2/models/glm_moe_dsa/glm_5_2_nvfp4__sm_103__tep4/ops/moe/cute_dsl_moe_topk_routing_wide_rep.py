# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's wide decode sigmoid top-k (MTP3 / MTP5 decode rows, 65..256) with replicated hot
experts: the wide top-k's selection and weights, and tile maps over this rank's 64 home slots
plus `nrep` replica slots (`64 + h` holds hot expert `hot_ids[h]`), each active hot expert
assigned whole to the EP rank with the fewest active experts. One CuTe DSL kernel per wide row
range (65..128, 129..192, 193..256)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_topk_routing_push_wide_rep as _wrep
from .cute_dsl_moe_routing import LOCAL_EXPERTS, MAP_KEYS, TOP_K
from .cute_dsl_moe_topk_routing import EXPERTS, _sigmoid_topk, wide_range
from .cute_dsl_moe_topk_routing_rep import reference_slot_maps, slot_ids

#: The most replica slots a handle takes.
MAX_NREP = _wrep.MAX_NREP
OUTPUTS = _wrep.OUTPUT_ORDER
max_tiles = _wrep.max_tiles


def allocate_outputs(rows: int, nrep: int, device) -> dict[str, torch.Tensor]:
    """One output set for up to `rows` tokens (any count of a wide range): the weights and ids,
    and the six maps with `max_tiles(rows, nrep)` tiles."""
    rows, tiles = int(rows), max_tiles(int(rows), int(nrep))
    i32 = {"dtype": torch.int32, "device": device}
    return {
        "topk_weights": torch.empty(rows * TOP_K, dtype=torch.float32, device=device),
        "topk_indices": torch.empty(rows * TOP_K, **i32),
        "tile_idx_to_expert_idx": torch.empty(tiles, **i32),
        "tile_idx_to_mn_limit": torch.empty(tiles, **i32),
        "expanded_idx_to_permuted_idx": torch.empty(rows * TOP_K, **i32),
        "permuted_idx_to_expanded_idx": torch.empty(tiles * 128, **i32),
        "total_num_padded_tokens": torch.empty(1, **i32),
        "num_non_exiting_tiles": torch.empty(1, **i32),
    }


@dataclass(frozen=True)
class CompiledTopkRoutingWideRep:
    """The wide replica top-k compiled for one row range (`rows` = its last count, any count
    of the range runs on it), `logit_splits` K partials and `nrep` replica slots, waiting for
    the router as its PDL dependent. `launch(logits, bias, hot_ids, local_expert_offset,
    routed_scale, outputs, expert_hist)` is the kernel call."""

    rows: int
    logit_splits: int
    nrep: int
    launch: Callable[..., None]
    expert_hist: bool = False
    cluster: int = 8
    trim: bool = False


def build_wide_rep(
    rows: int,
    logit_splits: int,
    nrep: int,
    device,
    *,
    expert_hist: bool = False,
    cluster: int = 8,
    trim: bool = False,
) -> CompiledTopkRoutingWideRep:
    """Compile the wide replica top-k for the row range holding `rows` (`wide_range`) reading
    `logit_splits` planes, with `nrep` replica slots, on a `cluster`-CTA cluster (8, or 16 as
    the wide top-k's `build_wide`), `trim`: with its trimmed winner lookup. Call at load;
    nothing is launched. No early FC1 release
    (as the wide top-k the trunk builds). `expert_hist`: also write the step's per-home-expert
    token counts into the int32 `[64]` row the call passes."""
    _, hi = wide_range(rows)
    nrep = int(nrep)
    shape = (hi, EXPERTS) if logit_splits == 1 else (hi, int(logit_splits), EXPERTS)
    logits = torch.zeros(shape, dtype=torch.float32, device=device)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    hot = torch.full((nrep,), -1, dtype=torch.int32, device=device)
    outputs = allocate_outputs(hi, nrep, device)
    hist = torch.zeros(64, dtype=torch.int32, device=device) if expert_hist else None
    as_t = _wrep._base._as_tensor
    with torch.cuda.device(device):
        compiled = _wrep.compile_fused_topk_routing_wide_rep(
            logits,
            bias,
            hot,
            outputs,
            nrep=nrep,
            wait_for_router_pdl=_USE_PDL,
            signal_gg1_pdl=False,
            expert_hist=hist,
            cluster=int(cluster),
            trim=bool(trim),
        )

    def launch(logits, bias, hot_ids, local_expert_offset, routed_scale, outputs, expert_hist=None):
        # Flat views: the kernel indexes every output linearly, and the slot's weight buffer
        # is 2-D.
        compiled(
            as_t(logits.reshape(-1), assumed_align=16),
            as_t(bias),
            as_t(hot_ids, assumed_align=16),
            *(as_t(outputs[name].reshape(-1)) for name in OUTPUTS),
            cutlass.Int32(int(logits.shape[0])),
            cutlass.Int32(local_expert_offset),
            cutlass.Float32(routed_scale),
            cutlass_torch.current_stream(),
            *((as_t(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
        )

    return CompiledTopkRoutingWideRep(
        hi, int(logit_splits), nrep, launch, bool(expert_hist), int(cluster), bool(trim)
    )


class CuteDslMoeTopkRoutingWideRep(OpWrapper):
    """Biased sigmoid top-8 of 256 experts at 65..256 rows and this rank's tile maps over its
    home slots and `nrep` replica slots.

    Returns `(weights [rows, 8] fp32, ids [rows, 8] int32, maps)`: views into `outputs`.
    Selection and weights are the wide top-k's (`cute_dsl_moe_topk_routing`, `build_wide`).
    `hot_ids` is int32 `[nrep]`: slot `64 + h` holds global expert `hot_ids[h]` on every rank
    (-1 = unused slot). The maps' `tile_idx_to_expert_idx` holds local slots: `e - offset`
    for a home expert that is not hot, `64 + h` for a hot expert this rank was assigned; a
    slot past 128 rows takes two adjacent tiles. Every rank computes the same assignment
    from the same ids, so each (token, k) is mapped on exactly one EP rank. `outputs` holds
    at least `allocate_outputs(rows, nrep)`. `expert_hist`, on a handle built with it: the
    int32 `[64]` row of this step's home-expert token counts.
    """

    ARCHS = frozenset({Arch.SM_103})

    # MTP3's and MTP5's verify rows, an odd count per range, the router's partials and the
    # prefill router's single plane, a hot list that matches the batch (worst skew), a
    # random one, an all-unused one (home mapping), 16 slots, and the prefetch counts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="128 rows (MTP3), 8 replicas of this batch's most selected experts, rank 0",
            spec=dict(rows=128, splits=4, nrep=8, hot="freq", offset=0),
        ),
        Cell(
            why="192 rows (MTP5), 8 frequency replicas, the last rank, writing the counts",
            spec=dict(rows=192, splits=4, nrep=8, hot="freq", offset=192, expert_hist=True),
        ),
        Cell(
            why="97 rows, 8 random replicas (some inactive), rank 1",
            spec=dict(rows=97, splits=4, nrep=8, hot="rand", offset=64),
        ),
        Cell(
            why="150 rows, one plane (the prefill router's logits), 8 frequency replicas, rank 2",
            spec=dict(rows=150, splits=1, nrep=8, hot="freq", offset=128),
        ),
        Cell(
            why="144 rows, every replica slot unused: the home mapping",
            spec=dict(rows=144, splits=4, nrep=8, hot="none", offset=64),
        ),
        Cell(
            why="256 rows (32 token-warps), 16 frequency replicas",
            spec=dict(rows=256, splits=4, nrep=16, hot="freq", offset=128),
        ),
    )

    note = """
    The trunk builds one handle per wide row range when MTP3 / MTP5 and the context
    replicas are on (`_build_moe_wide_rep`): it runs on the context replicas' `64 + R` deep
    prefill pair and hot ids, and the MTP layer, which has no hot ids, keeps the wide top-k.

    Not bitwise against the wide top-k end to end: a hot expert's partial is computed on
    another rank, so the EP all-reduce adds the bf16 partials in another grouping.
    Selection, weights and each rank's maps are exact.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
        hot_ids: torch.Tensor,
        local_expert_offset: int,
        routed_scale: float,
        outputs: dict[str, torch.Tensor],
        compiled: CompiledTopkRoutingWideRep,
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
        compiled: CompiledTopkRoutingWideRep,
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
        compiled: CompiledTopkRoutingWideRep,
        expert_hist: torch.Tensor | None = None,
    ) -> None:
        # The compiled handle checks nothing: a row count of another range runs with too few
        # token-warps, a wrong hot list length is read past its end, and an output set sized
        # for fewer tiles is written past its end.
        rows = int(logits.shape[0])
        splits = 1 if logits.dim() == 2 else int(logits.shape[1])
        assert logits.dtype == torch.float32 and logits.is_contiguous()
        assert logits.shape[-1] == EXPERTS and logits.data_ptr() % 16 == 0
        assert tuple(bias.shape) == (EXPERTS,) and bias.dtype == torch.float32
        assert local_expert_offset in (0, 64, 128, 192), (
            f"local_expert_offset selects one EP4 window, got {local_expert_offset}"
        )
        _, hi = wide_range(rows)
        assert hi == compiled.rows, f"compiled for the range ending at {compiled.rows}, got {rows}"
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
            assert t.numel() >= ref[name].numel() and t.dtype == ref[name].dtype, (
                f"{name} must hold {ref[name].numel()} {ref[name].dtype} for {rows} rows "
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


cute_dsl_moe_topk_routing_wide_rep = CuteDslMoeTopkRoutingWideRep()
