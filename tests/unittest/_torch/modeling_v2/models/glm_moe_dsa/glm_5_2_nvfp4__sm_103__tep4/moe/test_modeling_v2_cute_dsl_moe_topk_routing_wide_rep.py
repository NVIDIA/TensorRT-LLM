# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_topk_routing_wide_rep entry (MTP3 / MTP5 wide
decode rows, 65..256, with replicated hot experts): selection and weights equal the wide top-k's
bit for bit, each rank's maps equal the slot reference (a slot past 128 rows on two tiles), the
four EP ranks map every (token, k) exactly once, an all-unused hot list is the wide top-k's
home mapping, and a CUDA-graph replay equals the eager call."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL top-k kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    TILE,
    TOP_K,
    allocate_maps,
    canonical_maps,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    WIDE_RANGES,
    WIDE_SCHEDULES,
    build_wide,
    cute_dsl_moe_topk_routing,
    wide_range,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_rep import (  # noqa: E402, E501
    canonical_slot_maps,
    reference_assignment,
    reference_slot_maps,
    slot_ids,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_wide_rep import (  # noqa: E402, E501
    allocate_outputs,
    build_wide_rep,
    max_tiles,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing_wide_rep import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing_wide_rep as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS = 256
ROUTED_SCALE = 2.5
#: The trunk's slot buffers are sized for the context token budget (`_MOE_MAX_T`).
SLOT_ROWS = 8192
OFFSETS = (0, 64, 128, 192)
_COMPILED: dict = {}


def _sched_id(schedule) -> str:
    return f"cl{schedule[0]}{'trim' if schedule[1] else ''}"


def _wide(rows: int, splits: int, hist: bool = False):
    key = ("wide", wide_range(rows), splits, hist)
    if key not in _COMPILED:
        _COMPILED[key] = build_wide(rows, splits, torch.cuda.current_device(), expert_hist=hist)
    return _COMPILED[key]


def _wrep(rows: int, splits: int, nrep: int, hist: bool = False, schedule=(8, False)):
    key = ("wrep", wide_range(rows), splits, nrep, hist, schedule)
    if key not in _COMPILED:
        _COMPILED[key] = build_wide_rep(
            rows,
            splits,
            nrep,
            torch.cuda.current_device(),
            expert_hist=hist,
            cluster=schedule[0],
            trim=schedule[1],
        )
    return _COMPILED[key]


def _slot_outputs(nrep: int) -> dict:
    """An output set as the trunk's slot holds it: 2-D weights, maps for the token budget
    with `nrep` replica tiles; every entry -7 until written."""
    out = {
        "topk_weights": torch.empty(SLOT_ROWS, TOP_K, device="cuda"),
        "topk_indices": torch.empty(SLOT_ROWS * TOP_K, dtype=torch.int32, device="cuda"),
        **allocate_maps(SLOT_ROWS, "cuda", replicas=nrep),
    }
    for t in out.values():
        t.fill_(-7)
    return out


def _home(logits, bias, offset, splits):
    """The wide top-k (no replicas) on the same logits: ids, weights, maps."""
    out = _slot_outputs(0)
    w, ids, maps = cute_dsl_moe_topk_routing(
        logits, bias, offset, ROUTED_SCALE, out, None, _wide(int(logits.shape[0]), splits)
    )
    torch.cuda.synchronize()
    return w.clone(), ids.clone(), {k: v.clone() for k, v in maps.items()}


def _rep(logits, bias, hot_t, offset, splits, nrep, hist=None, check=True, schedule=(8, False)):
    out = _slot_outputs(nrep)
    handle = _wrep(int(logits.shape[0]), splits, nrep, hist is not None, schedule)
    if check:
        with validating(op):
            w, ids, maps = op(
                logits, bias, hot_t, offset, ROUTED_SCALE, out, handle, expert_hist=hist
            )
    else:
        w, ids, maps = op(logits, bias, hot_t, offset, ROUTED_SCALE, out, handle, expert_hist=hist)
    torch.cuda.synchronize()
    return w, ids, maps


def _check_inverse(maps, ids, offset: int, rows: int) -> torch.Tensor:
    """The kernel's own two index maps agree: every valid permuted row maps to an expanded
    index whose forward map returns it; an expanded index of another rank maps to -1.
    Returns the expanded indices this rank holds."""
    tiles = int(maps["num_non_exiting_tiles"][0])
    assert int(maps["total_num_padded_tokens"][0]) == tiles * TILE
    limit = maps["tile_idx_to_mn_limit"][:tiles].tolist()
    p2e, e2p = maps["permuted_idx_to_expanded_idx"], maps["expanded_idx_to_permuted_idx"]
    held = []
    for t, hi in enumerate(limit):
        assert t * TILE < hi <= (t + 1) * TILE, f"tile {t} limit {hi}"
        rows_t = p2e[t * TILE : hi].long()
        perm = torch.arange(t * TILE, hi, device="cuda")
        assert torch.equal(e2p[rows_t].long(), perm), f"tile {t}: inverse map disagrees"
        held.append(rows_t)
    held = torch.cat(held) if held else torch.empty(0, dtype=torch.long, device="cuda")
    mine = torch.zeros(rows * TOP_K, dtype=torch.bool, device="cuda")
    mine[held] = True
    assert bool((e2p[: rows * TOP_K][~mine] == -1).all()), "another rank's selection is mapped"
    return held


def _check_rank(logits, bias, hot_t, offset, splits, nrep, hist=False, schedule=(8, False)):
    """One rank: ids / weights = the (8-CTA) wide top-k's bitwise, maps = the slot reference of
    its own ids, the inverse maps consistent, the counts row the home counts. Returns
    `(ids, held expanded indices, canonical maps)`."""
    rows = int(logits.shape[0])
    w0, ids0, maps0 = _home(logits, bias, offset, splits)
    h = torch.full((64,), -1, dtype=torch.int32, device="cuda") if hist else None
    w, ids, maps = _rep(logits, bias, hot_t, offset, splits, nrep, h, schedule=schedule)
    assert torch.equal(ids, ids0), "ids differ from the wide top-k's"
    assert torch.equal(w, w0), "weights differ from the wide top-k's"
    nslots = 64 + nrep
    got = canonical_slot_maps(maps, rows * TOP_K, nslots)
    want = reference_slot_maps(slot_ids(ids, hot_t.tolist(), offset), nslots)
    for k in MAP_KEYS:
        assert torch.equal(got[k], want[k]), f"{k} differs from the slot reference"
    held = _check_inverse(maps, ids, offset, rows)
    if hist:
        local = ids[(ids >= offset) & (ids < offset + 64)].long() - offset
        assert torch.equal(h, torch.bincount(local, minlength=64).to(torch.int32)), (
            "expert_hist is not the step's home-expert counts"
        )
    if all(x < 0 for x in hot_t.tolist()):
        # No replicas: the wide top-k's home mapping exactly.
        home = canonical_maps(maps0, rows * TOP_K)
        for k in MAP_KEYS:
            assert torch.equal(got[k], home[k]), f"{k}: an unused hot list is not the home mapping"
    return ids, held, got


def _cover(held_per_rank, rows: int) -> None:
    seen = torch.zeros(rows * TOP_K, dtype=torch.long, device="cuda")
    for held in held_per_rank:
        seen.index_add_(0, held, torch.ones_like(held))
    assert bool((seen == 1).all()), f"{int((seen != 1).sum())} (token, k) not routed exactly once"


def _cell_case(spec, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, splits, nrep = spec["rows"], spec["splits"], spec["nrep"]
    if splits == 1:
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda")
    else:
        logits = torch.randn(rows, splits, EXPERTS, generator=g, device="cuda") * 0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    _, ids, _ = _home(logits.contiguous(), bias, 0, splits)
    if spec["hot"] == "freq":
        cnt = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS)
        hot = torch.argsort(cnt, descending=True, stable=True)[:nrep].tolist()
    elif spec["hot"] == "rand":
        hot = torch.randperm(EXPERTS, generator=g, device="cuda")[:nrep].tolist()
    else:
        hot = [-1] * nrep
    return logits.contiguous(), bias, torch.tensor(hot, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    rows, splits, nrep = spec["rows"], spec["splits"], spec["nrep"]
    logits, bias, hot_t = _cell_case(spec, zlib.crc32(cell.why.encode()) % 2**31)
    _check_rank(logits, bias, hot_t, spec["offset"], splits, nrep, spec.get("expert_hist", False))
    held = [_check_rank(logits, bias, hot_t, off, splits, nrep)[1] for off in OFFSETS]
    _cover(held, rows)


#: Odd and edge row counts of the three ranges: MTP3 / MTP5 at padded batches 17..32, one tile
#: per slot (<= 128 rows) and two (a slot past 128 rows).
ODD_ROWS = (65, 68, 96, 126, 127, 128, 129, 144, 150, 191, 192, 200)
#: `_skewed_logits`' layout, relative to EP rank 1's window (the favoured one).
WIN = 64
HOT = WIN + 1
EMPTY = tuple(range(WIN + 10, WIN + 14))
TIED = tuple((WIN + 20 + 2 * p, WIN + 21 + 2 * p) for p in range(8))


def _skewed_logits(rows: int, splits: int, g: torch.Generator):
    """A hot expert every row takes (past 64 rows from 65 up, two tiles past 128), four
    experts no row takes, eight tied pairs (equal partials and bias: equal scores bit for
    bit), and rank 1's window favoured so it holds the most active experts."""
    planes = torch.randn(rows, splits, EXPERTS, generator=g, device="cuda") * (
        1.0 if splits == 1 else 0.5
    )
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    bias[WIN : WIN + 64] += 0.3
    bias[HOT] = 4.0
    bias[list(EMPTY)] = -4.0
    for a, b in TIED:
        planes[:, :, b] = planes[:, :, a]
        bias[b] = bias[a]
    logits = planes[:, 0] if splits == 1 else planes
    return logits.contiguous(), bias


@pytest.mark.parametrize("schedule", WIDE_SCHEDULES, ids=_sched_id)
@pytest.mark.parametrize("splits", (4, 1))
@pytest.mark.parametrize("rows", ODD_ROWS)
def test_skewed_router_odd_rows(rows: int, splits: int, schedule) -> None:
    """On a skewed router: every rank's selection, weights and maps exact, the hot expert
    moved whole off its (busiest) home rank onto one rank's replica slot -- two tiles past
    128 rows -- an inactive hot expert and an unused slot assigned nowhere, and every
    (token, k) routed on exactly one rank. The draw is checked not to be degenerate. On both
    wide schedules alike (8-CTA; 16-CTA with the trimmed winner lookup)."""
    g = torch.Generator(device="cuda").manual_seed(rows * 10 + splits)
    logits, bias = _skewed_logits(rows, splits, g)
    _, ids, _ = _home(logits, bias, 0, splits)
    cnt = torch.bincount(ids.reshape(-1).long(), minlength=EXPERTS)
    # Slot 0 the hot expert, slot 1 an expert no row takes, slots 2..6 the next most selected
    # experts, slot 7 unused.
    order = [e for e in torch.argsort(cnt, descending=True, stable=True).tolist() if e != HOT]
    hot = [HOT, EMPTY[0], *order[:5], -1]
    hot_t = torch.tensor(hot, dtype=torch.int32, device="cuda")
    assign = reference_assignment(ids, hot)
    # Not degenerate: the hot expert takes every row, the empty ones none, a tie straddles
    # the selection edge, the hot expert leaves its busiest home window.
    assert int(cnt[HOT]) == rows > 64, "the hot expert does not take every row"
    assert all(int(cnt[e]) == 0 for e in EMPTY), "an empty expert got rows"
    straddles = 0
    for a, b in TIED:
        ha, hb = (ids == a).any(1), (ids == b).any(1)
        assert not bool((hb & ~ha).any()), f"tie {a}/{b} broken toward the higher id"
        straddles += int((ha & ~hb).sum())
    assert straddles > 0, "no tied pair split at the selection edge in this draw"
    assert assign[0] not in (-1, HOT // 64), f"the hot expert stayed home: {assign}"
    assert assign[1] == -1 and assign[7] == -1, f"an inactive or unused slot was assigned: {assign}"
    held = []
    for off in OFFSETS:
        _, h, got = _check_rank(
            logits, bias, hot_t, off, splits, 8, hist=(off == WIN), schedule=schedule
        )
        held.append(h)
        tiles = got["tile_idx_to_expert_idx"].tolist()
        if off // 64 == assign[0]:
            assert tiles.count(64) == (rows + TILE - 1) // TILE, "hot slot tile count"
        else:
            assert 64 not in tiles
        if off == WIN:
            assert (HOT - WIN) not in tiles, "a hot expert ran from its home slot"
    _cover(held, rows)


@pytest.mark.parametrize("schedule", WIDE_SCHEDULES, ids=_sched_id)
@pytest.mark.parametrize("rows", (65, 128, 129, 192, 200))
def test_graph_replay_matches_eager(rows: int, schedule) -> None:
    """The launch captured once per EP rank, replayed on fresh logits and hot ids written into
    its static buffers: ids, weights and canonical maps equal the eager call's, and the four
    ranks' replays map every (token, k) exactly once, some through a replica slot."""
    splits, nrep = 4, 8
    g = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.zeros(rows, splits, EXPERTS, device="cuda")
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    hot_t = torch.full((nrep,), -1, dtype=torch.int32, device="cuda")
    handle = _wrep(rows, splits, nrep, schedule=schedule)
    graphs = {}
    for off in OFFSETS:
        out = _slot_outputs(nrep)
        op(logits, bias, hot_t, off, ROUTED_SCALE, out, handle)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = op(logits, bias, hot_t, off, ROUTED_SCALE, out, handle)
        graphs[off] = (graph, out, res)
    for rep in range(3):
        logits.copy_(torch.randn(rows, splits, EXPERTS, generator=g, device="cuda") * 0.5)
        cnt = torch.bincount(_home(logits, bias, 0, splits)[1].reshape(-1).long(), minlength=256)
        hot_t.copy_(torch.argsort(cnt, descending=True, stable=True)[:nrep].to(torch.int32))
        held, replica_tiles = [], 0
        for off, (graph, out, (gw, gids, gmaps)) in graphs.items():
            for t in out.values():
                t.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            got = canonical_slot_maps(gmaps, rows * TOP_K, 64 + nrep)
            w, ids, maps = _rep(logits, bias, hot_t, off, splits, nrep, schedule=schedule)
            want = canonical_slot_maps(maps, rows * TOP_K, 64 + nrep)
            assert torch.equal(gw, w) and torch.equal(gids, ids), f"replay {rep} rank {off}"
            for k in MAP_KEYS:
                assert torch.equal(got[k], want[k]), f"replay {rep} rank {off}: {k}"
            held.append(_check_inverse(gmaps, gids, off, rows))
            replica_tiles += sum(1 for t in got["tile_idx_to_expert_idx"].tolist() if t >= 64)
        _cover(held, rows)
        assert replica_tiles >= 1, "no replica slot on any rank"


def test_tile_capacity() -> None:
    """The bound covers a replica slot per hot expert and a second tile per slot past 128 rows,
    and the trunk's slot maps (`allocate_maps(_MOE_MAX_T, replicas=R)`) hold it."""
    assert max_tiles(128, 8) == 72 + (1024 - 72) // TILE
    assert max_tiles(256, 8) == 87
    for nrep in (1, 8, 16):
        cap = allocate_maps(SLOT_ROWS, "meta", replicas=nrep)["tile_idx_to_expert_idx"].numel()
        assert cap >= max_tiles(256, nrep)
    assert tuple(r[1] for r in WIDE_RANGES) == (128, 192, 256)


def test_guard_refuses_a_row_count_of_another_range() -> None:
    hot = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    logits = torch.zeros(129, 4, EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="compiled for the range"):
        with validating(op):
            op(logits, torch.zeros(EXPERTS, device="cuda"), hot, 0, 1.0, _slot_outputs(8),
               _wrep(128, 4, 8))  # fmt: skip


def test_guard_refuses_a_short_hot_list() -> None:
    logits = torch.zeros(128, 4, EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="hot_ids"):
        with validating(op):
            op(logits, torch.zeros(EXPERTS, device="cuda"),
               torch.full((4,), -1, dtype=torch.int32, device="cuda"), 0, 1.0, _slot_outputs(8),
               _wrep(128, 4, 8))  # fmt: skip


def test_guard_refuses_outputs_without_replica_tiles() -> None:
    """Maps sized for the home tiles alone are written past their end by a replica slot."""
    hot = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    logits = torch.zeros(192, 4, EXPERTS, device="cuda")
    small = allocate_outputs(192, 0 + 1, "cuda")
    small["tile_idx_to_expert_idx"] = torch.empty(64, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="replica slots"):
        with validating(op):
            op(logits, torch.zeros(EXPERTS, device="cuda"), hot, 0, 1.0, small, _wrep(192, 4, 8))
