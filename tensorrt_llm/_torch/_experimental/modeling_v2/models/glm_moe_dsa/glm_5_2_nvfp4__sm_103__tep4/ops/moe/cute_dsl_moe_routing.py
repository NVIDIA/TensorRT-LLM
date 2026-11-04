# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's general MoE routing: the grouped GEMMs' six tile maps from expert ids that
another op selected, on this rank's 64-expert window, in CuTe DSL.

Also the plain-torch statement of what those maps mean (`reference_maps`,
`canonical_maps`, `check_maps`), which the fused top-k entry and the expert GEMMs share.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe import mtp_moe_routing as _kernel

#: This rank's expert window at EP4, the grouped GEMMs' M tile, and the routing width.
LOCAL_EXPERTS = 64
TILE = 128
TOP_K = 8

#: The six maps, in the order the kernel takes them.
MAP_KEYS = (
    "tile_idx_to_expert_idx",
    "tile_idx_to_mn_limit",
    "expanded_idx_to_permuted_idx",
    "permuted_idx_to_expanded_idx",
    "total_num_padded_tokens",
    "num_non_exiting_tiles",
)

#: One row count per schedule bucket a step can reach. A decode step pads to a power of
#: two up to 64 rows. A context step pads to 128, 256 or 512 rows below 513 and runs the
#: exact count above (buckets 513-1024, 1025-2048, 2049-4096, 4097-8192); the MTP layer's
#: context step pads every count to a power of two, which lands in the same buckets.
ROW_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def max_tiles(rows: int) -> int:
    """The most 128-row tiles `rows` tokens can occupy: one token per expert first, which
    maximises the tile count, then the rest into one expert. `ceil(rows * 8 / 128)`
    under-counts. 575 at 8192 rows."""
    remaining = int(rows) * TOP_K
    filled = min(LOCAL_EXPERTS, remaining)
    return filled + max(0, remaining - filled) // TILE


def pad_expert_id(local_expert_offset: int) -> int:
    """The expert id padding rows route to: one this rank does not own, so every routing
    schedule drops them (no tile, `expanded_idx_to_permuted_idx = -1`). Their weight is 0.

    Not expert 0: all 8 slots of every padding row would land on it, and the single-CTA
    schedule (rows <= 128) gives each active expert exactly one 128-row tile. Past 128
    rows expert 0's entries spill into the next experts' tiles and race with theirs, and
    the losing tokens silently drop that expert's contribution.
    """
    return local_expert_offset - 1 if local_expert_offset > 0 else LOCAL_EXPERTS


def allocate_maps(capacity_rows: int, device) -> dict[str, torch.Tensor]:
    """One set of the six maps for up to `capacity_rows` tokens. Sized by capacity, never
    by a call: the grouped GEMMs are told the tile bound once and read the live tile count
    from `num_non_exiting_tiles`."""
    n = max_tiles(capacity_rows)
    i32 = {"dtype": torch.int32, "device": device}
    return {
        "tile_idx_to_expert_idx": torch.empty(n, **i32),
        "tile_idx_to_mn_limit": torch.empty(n, **i32),
        "expanded_idx_to_permuted_idx": torch.empty(int(capacity_rows) * TOP_K, **i32),
        "permuted_idx_to_expanded_idx": torch.empty(n * TILE, **i32),
        "total_num_padded_tokens": torch.empty(1, **i32),
        "num_non_exiting_tiles": torch.empty(1, **i32),
    }


def allocate_workspace(device) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's two scratch tensors: `int32[64]` expert counts and `int32[65]` barrier."""
    i32 = {"dtype": torch.int32, "device": device}
    return torch.empty(LOCAL_EXPERTS, **i32), torch.empty(LOCAL_EXPERTS + 1, **i32)


def schedule_bucket(rows: int) -> int:
    """The compile key `rows` tokens select."""
    return _kernel.routing_schedule_bucket(int(rows) * TOP_K)


def _ct(x: torch.Tensor) -> cute.Tensor:
    return from_dlpack(x, assumed_align=4).mark_layout_dynamic(leading_dim=0)


@dataclass(frozen=True)
class CompiledRouting:
    """The routing kernel compiled per schedule bucket (`schedule_bucket(rows)`)."""

    fns: Mapping[int, Callable[..., None]]

    def covers(self, rows: int) -> bool:
        return schedule_bucket(rows) in self.fns


def build(row_counts, device) -> CompiledRouting:
    """Compile the kernel once per distinct schedule bucket of `row_counts`. Call at load.

    The schedule is a `Constexpr` keyed by `schedule_bucket(rows)`; tensor extents, the
    expanded count and the expert offset stay runtime operands, so one compiled function
    serves every row count in its bucket, every map set and every rank. Compiles against
    scratch maps; nothing is launched. Each bucket costs one CuTe DSL compile (minutes of
    CPU for the large ones), which is why `ROW_COUNTS` names one row count per reachable
    bucket and no more. Raises if a compile fails.
    """
    rows_list = [int(r) for r in row_counts]
    maps = allocate_maps(max(rows_list), device)
    counts, barrier = allocate_workspace(device)
    fns: dict[int, Callable[..., None]] = {}
    with torch.cuda.device(device):
        for rows in rows_list:
            bucket = schedule_bucket(rows)
            if bucket in fns:
                continue
            sel = torch.zeros(rows * TOP_K, dtype=torch.int32, device=device)
            fns[bucket] = cute.compile(
                _kernel.routing_host_with_workspace,
                _ct(sel),
                *[_ct(maps[k]) for k in MAP_KEYS],
                _ct(counts),
                _ct(barrier),
                cutlass.Int32(rows * TOP_K),
                cutlass.Int32(0),
                LOCAL_EXPERTS,
                TILE,
                bucket,
                cutlass_torch.current_stream(),
            )
    return CompiledRouting(fns=fns)


# ---- what the maps mean, in plain torch ---------------------------------------------


def reference_maps(topk_ids: torch.Tensor, local_expert_offset: int) -> dict[str, torch.Tensor]:
    """The six maps for `topk_ids` (global ids, `[rows, 8]` or flat), in canonical form.

    Local experts in ascending order, `ceil(count / 128)` tiles each; within an expert,
    rows in ascending expanded index (`token * 8 + slot`). Ids outside
    `[local_expert_offset, local_expert_offset + 64)` get no row and
    `expanded_idx_to_permuted_idx = -1`. Padding rows of a tile have
    `permuted_idx_to_expanded_idx = -1`. Each tensor is exactly as long as its content
    (`num_non_exiting_tiles` tiles), int32.
    """
    ids = topk_ids.reshape(-1).long()
    dev = ids.device
    n = ids.numel()
    local = ids - int(local_expert_offset)
    own = (local >= 0) & (local < LOCAL_EXPERTS)
    counts = torch.bincount(local[own], minlength=LOCAL_EXPERTS)
    tiles_per = (counts + TILE - 1) // TILE
    tile_expert = torch.repeat_interleave(torch.arange(LOCAL_EXPERTS, device=dev), tiles_per)
    tiles = int(tile_expert.numel())
    first_tile = torch.cumsum(tiles_per, 0) - tiles_per
    j = torch.arange(tiles, device=dev) - first_tile[tile_expert]
    mn_limit = torch.arange(tiles, device=dev) * TILE + torch.clamp(
        counts[tile_expert] - j * TILE, max=TILE
    )
    idx = torch.nonzero(own).reshape(-1)
    e = local[idx]
    order = torch.argsort(e * n + idx)
    idx, e = idx[order], e[order]
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(idx.numel(), device=dev) - start[e]
    row = first_tile[e] * TILE + rank
    e2p = torch.full((n,), -1, dtype=torch.long, device=dev)
    e2p[idx] = row
    p2e = torch.full((tiles * TILE,), -1, dtype=torch.long, device=dev)
    p2e[row] = idx
    i32 = torch.int32
    return {
        "tile_idx_to_expert_idx": tile_expert.to(i32),
        "tile_idx_to_mn_limit": mn_limit.to(i32),
        "expanded_idx_to_permuted_idx": e2p.to(i32),
        "permuted_idx_to_expanded_idx": p2e.to(i32),
        "total_num_padded_tokens": torch.tensor([tiles * TILE], dtype=i32, device=dev),
        "num_non_exiting_tiles": torch.tensor([tiles], dtype=i32, device=dev),
    }


def canonical_maps(maps: Mapping[str, torch.Tensor], num_expanded: int) -> dict[str, torch.Tensor]:
    """A kernel's maps rewritten in `reference_maps`' canonical form.

    The kernels claim rows within one expert by atomics, so the row order inside an
    expert is unspecified; this re-sorts each expert's rows by expanded index and rebuilds
    both index maps from the forward map (`permuted_idx_to_expanded_idx` on valid rows).
    The tile list, limits and counts pass through. `check_maps` verifies that the
    kernel's own inverse agrees with its forward map, which this does not read.
    """
    tiles = int(maps["num_non_exiting_tiles"].reshape(-1)[0].item())
    dev = maps["tile_idx_to_expert_idx"].device
    tile_expert = maps["tile_idx_to_expert_idx"][:tiles].long()
    mn = maps["tile_idx_to_mn_limit"][:tiles].long()
    rows = [
        torch.arange(t * TILE, int(mn[t].item()), device=dev, dtype=torch.long)
        for t in range(tiles)
    ]
    valid = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=dev)
    expanded = maps["permuted_idx_to_expanded_idx"].long()[valid]
    expert = tile_expert[valid // TILE]
    order = torch.argsort(expert * int(num_expanded) + expanded)
    expanded, expert = expanded[order], expert[order]
    first_tile = torch.full((LOCAL_EXPERTS,), -1, dtype=torch.long, device=dev)
    for t in reversed(range(tiles)):
        first_tile[tile_expert[t]] = t
    counts = torch.bincount(expert, minlength=LOCAL_EXPERTS)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(expanded.numel(), device=dev) - start[expert]
    row = first_tile[expert] * TILE + rank
    e2p = torch.full((int(num_expanded),), -1, dtype=torch.long, device=dev)
    e2p[expanded] = row
    p2e = torch.full((tiles * TILE,), -1, dtype=torch.long, device=dev)
    p2e[row] = expanded
    i32 = torch.int32
    return {
        "tile_idx_to_expert_idx": tile_expert.to(i32),
        "tile_idx_to_mn_limit": mn.to(i32),
        "expanded_idx_to_permuted_idx": e2p.to(i32),
        "permuted_idx_to_expanded_idx": p2e.to(i32),
        "total_num_padded_tokens": maps["total_num_padded_tokens"].reshape(-1)[:1].to(i32),
        "num_non_exiting_tiles": maps["num_non_exiting_tiles"].reshape(-1)[:1].to(i32),
    }


def check_maps(
    maps: Mapping[str, torch.Tensor], topk_ids: torch.Tensor, local_expert_offset: int
) -> None:
    """Assert what a consumer of the maps relies on beyond `canonical_maps`: each valid
    row's expanded index is a local winner routed to that row's expert, the inverse map
    points back at it, every non-local winner maps to -1, and no expanded index is used
    twice."""
    ids = topk_ids.reshape(-1).long()
    n = ids.numel()
    tiles = int(maps["num_non_exiting_tiles"].reshape(-1)[0].item())
    tile_expert = maps["tile_idx_to_expert_idx"][:tiles].long()
    mn = maps["tile_idx_to_mn_limit"][:tiles].long()
    p2e = maps["permuted_idx_to_expanded_idx"].long()
    e2p = maps["expanded_idx_to_permuted_idx"][:n].long()
    local = ids - int(local_expert_offset)
    own = (local >= 0) & (local < LOCAL_EXPERTS)
    seen = torch.zeros(n, dtype=torch.long, device=ids.device)
    for t in range(tiles):
        lo, hi = t * TILE, int(mn[t].item())
        assert lo < hi <= lo + TILE, f"tile {t}: limit {hi} outside ({lo}, {lo + TILE}]"
        exp = p2e[lo:hi]
        assert bool(((exp >= 0) & (exp < n)).all()), f"tile {t}: expanded index out of range"
        assert bool((local[exp] == tile_expert[t]).all()), f"tile {t}: row of another expert"
        assert bool((e2p[exp] == torch.arange(lo, hi, device=exp.device)).all()), (
            f"tile {t}: expanded_idx_to_permuted_idx does not invert the tile's rows"
        )
        seen.index_add_(0, exp, torch.ones_like(exp))
    assert bool((seen[own] == 1).all()), "a local winner has no row, or two"
    assert bool((seen[~own] == 0).all()), "a non-local winner has a row"
    assert bool((e2p[~own] == -1).all()), "a non-local winner's inverse is not -1"


class CuteDslMoeRouting(OpWrapper):
    """Write the six grouped-GEMM tile maps for `topk_ids` into `maps`. Returns `maps`.

    `topk_ids` is the int32 global expert ids, `[rows, 8]` or flat, contiguous. `maps`
    is a set from `allocate_maps` with capacity for at least `rows` tokens; `counts`
    and `barrier` the kernel's `int32[64]` / `int32[65]` scratch (`allocate_workspace`).
    Ids outside `[local_expert_offset, local_expert_offset + 64)` are dropped, which is
    how padding rows (`pad_expert_id`) vanish. `compiled` is `build`'s result and must
    cover `schedule_bucket(rows)`.

    The kernels launch with PDL and do not wait for their input: the ids must be
    visible before launch. A caller whose ids come from a PDL producer stages them
    with a copy first; the copy is a graph node no PDL edge crosses.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per schedule the target reaches: the 256-thread single CTA (buckets 1, 2,
    # 4), the larger single CTA (8, 16), the shared-memory cluster kernels above 512
    # rows, and the 8192-row ceiling. Two cells carry padding rows the way the target
    # does: a decode width rounded up to a power of two, and an MTP context step.
    CELLS: tuple[Cell, ...] = (
        Cell(why="4 rows, one CTA: a decode width", spec=dict(rows=4, offset=64)),
        Cell(
            why="3 rows padded to 4 with the pad expert, as a non power-of-two decode width",
            spec=dict(rows=4, pad=1, offset=0),
        ),
        Cell(why="the full decode batch, 64 rows", spec=dict(rows=64, offset=192)),
        Cell(why="128 rows: the last one-CTA register schedule", spec=dict(rows=128, offset=128)),
        Cell(
            why="512 rows: the two-warp prefix single CTA at its ceiling",
            spec=dict(rows=512, offset=64),
        ),
        Cell(
            why="1024 rows: the shared-memory cluster schedule, experts over several tiles",
            spec=dict(rows=1024, offset=0),
        ),
        Cell(
            why="an MTP context step: 3000 rows padded to 4096 with the pad expert",
            spec=dict(rows=4096, pad=1096, offset=128),
        ),
        Cell(why="8192 rows, the token budget", spec=dict(rows=8192, offset=192)),
    )

    note = """
    `build(ROW_COUNTS, device)` compiles one function per schedule bucket every step
    reaches, once per engine, shared by the trunk and the MTP layer; a row count whose
    bucket was not built is a KeyError, never a compile. The trunk calls it when the
    fused top-k did not write the maps for these exact rows (a padded decode width, a
    context step whose prefill top-k did not build, or any step whose top-k fell back
    to `noaux_tc_op`); the MTP layer calls it on every context step. Padded rows carry
    `pad_expert_id` at weight 0.

    The row order inside one expert is unspecified (atomic claims), so a test compares
    `canonical_maps` against `reference_maps` and checks the inverse with `check_maps`.
    Padding entries of `permuted_idx_to_expanded_idx` past a tile's limit are unspecified;
    consumers bound their reads by `tile_idx_to_mn_limit`.
    """

    def __call__(
        self,
        topk_ids: torch.Tensor,
        local_expert_offset: int,
        maps: dict[str, torch.Tensor],
        counts: torch.Tensor,
        barrier: torch.Tensor,
        compiled: CompiledRouting,
    ) -> dict[str, torch.Tensor]:
        ids = topk_ids.reshape(-1)
        n = int(ids.numel())
        fn = compiled.fns[_kernel.routing_schedule_bucket(n)]
        fn(
            _ct(ids),
            *[_ct(maps[k]) for k in MAP_KEYS],
            _ct(counts),
            _ct(barrier),
            cutlass.Int32(n),
            cutlass.Int32(local_expert_offset),
            cutlass_torch.current_stream(),
        )
        return maps

    def reference(
        self,
        topk_ids: torch.Tensor,
        local_expert_offset: int,
        maps: dict[str, torch.Tensor],
        counts: torch.Tensor,
        barrier: torch.Tensor,
        compiled: CompiledRouting,
    ) -> dict[str, torch.Tensor]:
        """`reference_maps(topk_ids, local_expert_offset)`: fresh tensors in canonical
        form. `maps`, `counts`, `barrier` and `compiled` are not read."""
        return reference_maps(topk_ids, local_expert_offset)

    def is_valid(
        self,
        topk_ids: torch.Tensor,
        local_expert_offset: int,
        maps: dict[str, torch.Tensor],
        counts: torch.Tensor,
        barrier: torch.Tensor,
        compiled: CompiledRouting,
    ) -> None:
        # The kernel takes no shape it can check against the row count: every extent is
        # runtime, so a map set too small for these rows is written past its end, and a
        # strided id view is read as dense.
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous(), (
            "topk_ids must be contiguous int32; a strided view is read as dense"
        )
        n = topk_ids.numel()
        assert n % TOP_K == 0, f"topk_ids must hold whole tokens of {TOP_K}, got {n}"
        rows = n // TOP_K
        assert local_expert_offset in (0, 64, 128, 192), (
            f"local_expert_offset selects one EP4 window, got {local_expert_offset}"
        )
        tiles = max_tiles(rows)
        need = {
            "tile_idx_to_expert_idx": tiles,
            "tile_idx_to_mn_limit": tiles,
            "expanded_idx_to_permuted_idx": n,
            "permuted_idx_to_expanded_idx": tiles * TILE,
            "total_num_padded_tokens": 1,
            "num_non_exiting_tiles": 1,
        }
        for k, size in need.items():
            t = maps[k]
            assert t.dtype == torch.int32 and t.is_contiguous(), f"{k} must be contiguous int32"
            assert t.numel() >= size, f"{k} holds {t.numel()} entries; {rows} rows can write {size}"
        assert counts.dtype == torch.int32 and counts.numel() == LOCAL_EXPERTS
        assert barrier.dtype == torch.int32 and barrier.numel() == LOCAL_EXPERTS + 1
        assert compiled.covers(rows), f"no compiled schedule for {rows} rows"


cute_dsl_moe_routing = CuteDslMoeRouting()
