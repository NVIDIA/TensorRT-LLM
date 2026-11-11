# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_topk_routing entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL top-k kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_routing import (  # noqa: E402, E501
    MAP_KEYS,
    TILE,
    TOP_K,
    allocate_maps,
    canonical_maps,
    check_maps,
    reference_maps,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    WIDE_RANGES,
    WIDE_SCHEDULES,
    CompiledTopkRouting,
    allocate_decode_outputs,
    allocate_prefill_scratch,
    build_decode,
    build_prefill,
    build_wide,
    selection_margin,
    wide_range,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe.cute_dsl_moe_topk_routing import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

EXPERTS, HIDDEN = 256, 6144
ROUTED_SCALE = 2.5
#: Rows whose 8th and 9th selection scores sit closer than this (relative) may pick
#: either expert by the last bit of the sigmoid; see `selection_margin`.
TIE_MARGIN = 2.0**-20
#: Weights: the kernel normalizes in fp32 (a five-step butterfly sum of eight values,
#: one multiply, one divide) where the reference does it in fp64, and both take the
#: sigmoid's tanh form in fp32. At most ~6 fp32 roundings separate them, 2**-21 relative;
#: the band is 2**-19. A wrong normalization or a biased weight misses by 1e-2.
WEIGHT_RTOL = 2.0**-19

_COMPILED: dict = {}


def _compiled(spec) -> CompiledTopkRouting:
    rows = spec["rows"]
    if spec.get("wide"):
        hist = spec.get("expert_hist", False)
        cluster, trim = spec.get("schedule", (8, False))
        key = ("wide", wide_range(rows), spec["splits"], hist, cluster, trim)
        if key not in _COMPILED:
            _COMPILED[key] = build_wide(
                rows,
                spec["splits"],
                torch.cuda.current_device(),
                expert_hist=hist,
                cluster=cluster,
                trim=trim,
            )
        return _COMPILED[key]
    if rows > 64:
        key = "prefill"
        if key not in _COMPILED:
            _COMPILED[key] = build_prefill(torch.cuda.current_device())
        return _COMPILED[key]
    hist = spec.get("expert_hist", False)
    key = (rows, spec["splits"], spec["signal_gg1"], spec.get("fuse_zero", False), hist)
    if key not in _COMPILED:
        _COMPILED[key] = build_decode(
            rows,
            spec["splits"],
            torch.cuda.current_device(),
            signal_gg1=spec["signal_gg1"],
            fuse_zero=spec.get("fuse_zero", False),
            expert_hist=hist,
        )
    return _COMPILED[key]


def _inputs(spec, g):
    """Logits as each router writes them (`[rows, 256]`, or four K partials), and a bias
    wide enough to skew selection so experts take several tiles at the large counts."""
    rows = spec["rows"]
    if rows > 64 and not spec.get("wide"):
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda")
    elif spec["splits"] == 1:
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda")
    else:
        logits = torch.randn(rows, spec["splits"], EXPERTS, generator=g, device="cuda") * 0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    if spec.get("hot"):
        # Every row selects this rank's second expert: past 128 rows it spans two tiles.
        bias[spec["offset"] + 1] += 4.0
    return logits.contiguous(), bias


def _outputs(rows: int):
    if rows <= 64:
        out = allocate_decode_outputs(rows, "cuda")
    else:
        # Sized for the token budget, as the target's slot buffers are.
        out = {
            "topk_weights": torch.empty(PREFILL_MAX_ROWS, TOP_K, device="cuda"),
            "topk_indices": torch.empty(PREFILL_MAX_ROWS * TOP_K, dtype=torch.int32, device="cuda"),
            **allocate_prefill_scratch("cuda"),
            **allocate_maps(PREFILL_MAX_ROWS, "cuda"),
        }
    for t in out.values():
        t.fill_(-7)  # an entry the kernel had to write and did not reads as -7
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Ids as sets per row (a row's eight winners in any order), weights matched by id
    within `WEIGHT_RTOL`, and the maps exact in canonical form.

    Selection is exact except where the reference's 8th and 9th scores are within
    `TIE_MARGIN`: there the kernel's sigmoid and torch's may differ in the last bit,
    which no reference fixes. Those rows are exempt and must be rare. The maps are
    checked against `reference_maps` of the kernel's own ids (and against the op's
    reference outright when no row was exempt), with `check_maps` on the inverse.
    """
    spec = cell.spec
    rows, offset = spec["rows"], spec["offset"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    logits, bias = _inputs(spec, g)
    outputs = _outputs(rows)
    zero = None
    if spec.get("fuse_zero"):
        zero = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    compiled = _compiled(spec)
    hist = None
    if spec.get("expert_hist"):
        hist = torch.full((64,), -1, dtype=torch.int32, device="cuda")
    with validating(op):
        weights, ids, maps = op(
            logits, bias, offset, ROUTED_SCALE, outputs, zero, compiled, expert_hist=hist
        )
    ref_w, ref_ids, ref_maps = op.reference(
        logits, bias, offset, ROUTED_SCALE, outputs, zero, compiled
    )

    exempt = selection_margin(logits, bias) < TIE_MARGIN
    assert int(exempt.sum()) <= max(1, rows // 1000), f"{int(exempt.sum())} near-tie rows"
    keep = ~exempt
    got_ids, order = torch.sort(ids, dim=-1)
    want_ids, want_order = torch.sort(ref_ids, dim=-1)
    assert torch.equal(got_ids[keep], want_ids[keep]), "selected experts differ"
    got_w = torch.gather(weights, 1, order)
    want_w = torch.gather(ref_w, 1, want_order)
    torch.testing.assert_close(got_w[keep], want_w[keep], rtol=WEIGHT_RTOL, atol=0.0)

    check_maps(maps, ids, offset)
    got = canonical_maps(maps, rows * TOP_K)
    own = reference_maps(ids, offset)
    for k in MAP_KEYS:
        assert torch.equal(got[k], own[k]), f"{k} does not describe the kernel's own ids"
        if not bool(exempt.any()):
            assert torch.equal(got[k], ref_maps[k]), f"{k} differs from the reference"
    if zero is not None:
        assert bool((zero == 0).all()), "the fused zero plane was not cleared"
    if hist is not None:
        # This step's token count per local expert, every expert written (zeros included).
        local = ids[(ids >= offset) & (ids < offset + 64)].long() - offset
        want = torch.bincount(local, minlength=64).to(torch.int32)
        assert torch.equal(hist, want), "expert_hist is not the step's per-expert counts"


#: The wide kernel's row counts under test: each token-warp range at its edges, one tile per
#: expert (<= 128 rows) and two, MTP3's 128 and MTP5's 192 verify rows.
WIDE_ROWS = (65, 127, 128, 129, 191, 192, 200, 256)
#: The skew of `_skewed_logits`, relative to EP rank 1's window.
SKEW_OFFSET = 64
HOT = SKEW_OFFSET + 1
EMPTY = tuple(range(SKEW_OFFSET + 10, SKEW_OFFSET + 14))
TIED = tuple((SKEW_OFFSET + 20 + 2 * p, SKEW_OFFSET + 21 + 2 * p) for p in range(8))


def _skewed_logits(rows: int, splits: int, g: torch.Generator):
    """Logits no degenerate draw satisfies: a hot local expert every row takes (two tiles past
    128 rows), four local experts no row takes, and eight tied pairs (equal partials and
    bias, so equal scores bit for bit), the window favoured so about half the winners are
    local. Returns `(logits as the kernel reads them, the same summed in split order, bias)`."""
    planes = torch.randn(rows, splits, EXPERTS, generator=g, device="cuda") * (
        1.0 if splits == 1 else 0.5
    )
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    bias[SKEW_OFFSET : SKEW_OFFSET + 64] += 0.3
    bias[HOT] = 4.0
    bias[list(EMPTY)] = -4.0
    for a, b in TIED:
        planes[:, :, b] = planes[:, :, a]
        bias[b] = bias[a]
    summed = planes[:, 0]
    for s in range(1, splits):
        summed = summed + planes[:, s]
    logits = planes[:, 0] if splits == 1 else planes
    return logits.contiguous(), summed.contiguous(), bias


@pytest.mark.parametrize("schedule", WIDE_SCHEDULES, ids=("cl8", "cl16trim"))
@pytest.mark.parametrize("splits", (1, 4))
@pytest.mark.parametrize("rows", WIDE_ROWS)
def test_wide_matches_the_prefill_top_k_bitwise(rows: int, splits: int, schedule) -> None:
    """The wide kernel takes the prefill top-k's place on MTP3 / MTP5 decode steps. On the same
    scores (the K partials summed in split order, as the kernel sums them) its ids and weights
    equal the prefill kernel's bit for bit -- both select with the decode arithmetic -- and its
    maps the same tile layout exactly (canonical form: rows within an expert are claimed by
    atomics), which is `reference_maps` of those ids. Its counts are the step's. Over the four
    EP windows every `(token, k)` is routed on exactly one rank. On both schedules alike: the
    8-CTA cluster, and the 16-CTA one (half the token-warps per CTA, 16 ranks' histograms) with
    the trimmed winner lookup.

    The draw is checked not to be degenerate: the hot expert takes every row (two tiles past
    128), the cold ones none, and some tied pair is split at the selection edge with the
    lower id taken."""
    g = torch.Generator(device="cuda").manual_seed(rows * 10 + splits)
    logits, summed, bias = _skewed_logits(rows, splits, g)
    wide = _compiled(dict(rows=rows, splits=splits, wide=True, expert_hist=True, schedule=schedule))
    prefill = _compiled(dict(rows=max(rows, 65)))
    seen = torch.zeros(rows * TOP_K, dtype=torch.long, device="cuda")
    for offset in (0, 64, 128, 192):
        out_p, out_w = _outputs(rows), _outputs(rows)
        hist = torch.full((64,), -1, dtype=torch.int32, device="cuda")
        w_p, ids_p, maps_p = op(summed, bias, offset, ROUTED_SCALE, out_p, None, prefill)
        with validating(op):
            w, ids, maps = op(
                logits, bias, offset, ROUTED_SCALE, out_w, None, wide, expert_hist=hist
            )
        assert torch.equal(ids, ids_p), "ids differ from the prefill top-k's"
        assert torch.equal(w, w_p), "weights differ from the prefill top-k's"
        check_maps(maps, ids, offset)
        got = canonical_maps(maps, rows * TOP_K)
        want = canonical_maps(maps_p, rows * TOP_K)
        ref = reference_maps(ids, offset)
        for k in MAP_KEYS:
            assert torch.equal(got[k], want[k]), f"{k} differs from the prefill top-k's"
            assert torch.equal(got[k], ref[k]), f"{k} differs from reference_maps"
        local = ids[(ids >= offset) & (ids < offset + 64)].long() - offset
        counts = torch.bincount(local, minlength=64)
        assert torch.equal(hist, counts.to(torch.int32)), "expert_hist is not the step's counts"
        tiles = int(maps["num_non_exiting_tiles"][0])
        limit = maps["tile_idx_to_mn_limit"][:tiles].tolist()
        p2e = maps["permuted_idx_to_expanded_idx"]
        for t, hi in enumerate(limit):
            rows_t = p2e[t * TILE : hi].long()
            seen.index_add_(0, rows_t, torch.ones_like(rows_t))
        if offset == SKEW_OFFSET:
            expert = got["tile_idx_to_expert_idx"]
            assert int(counts[HOT - offset]) == rows, "the hot expert does not take every row"
            assert int((expert == HOT - offset).sum()) == (rows + TILE - 1) // TILE
            assert all(int(counts[e - offset]) == 0 for e in EMPTY), "a cold expert has rows"
            straddles = 0
            for a, b in TIED:
                ha, hb = (ids == a).any(1), (ids == b).any(1)
                assert not bool((hb & ~ha).any()), f"tie {a}/{b} broken toward the higher id"
                straddles += int((ha & ~hb).sum())
            assert straddles > 0, "no tied pair split at the selection edge in this draw"
    assert bool((seen == 1).all()), "a (token, k) is routed on 0 or 2+ ranks"


def test_wide_ranges_cover_the_mtp_decode_widths() -> None:
    """The three handles tile 65..256 without overlap; MTP3's 128 and MTP5's 192 verify rows
    each sit in one."""
    assert WIDE_RANGES == ((65, 128), (129, 192), (193, 256))
    assert wide_range(128) == (65, 128) and wide_range(192) == (129, 192)


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _fake(kind: str, rows, splits: int = 1, fuse_zero: bool = False) -> CompiledTopkRouting:
    return CompiledTopkRouting(kind, rows, splits, False, fuse_zero, _unreachable)


def test_guard_refuses_another_row_count() -> None:
    """A decode handle is exact-row; its outputs are sized for those rows."""
    logits = torch.zeros(5, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="compiled for 4 rows"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(5, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_partials_for_a_one_plane_handle() -> None:
    """Four K partials read as one plane would sum a quarter of the product."""
    logits = torch.zeros(8, 4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="logit plane"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(8, "cuda"), None, _fake("decode", 8))


def test_guard_refuses_outputs_sized_for_other_rows() -> None:
    logits = torch.zeros(4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    with pytest.raises(AssertionError, match="for 4 rows"):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(2, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_a_bf16_bias() -> None:
    logits = torch.zeros(4, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError):
        with validating(op):
            op(logits, bias, 0, 1.0, allocate_decode_outputs(4, "cuda"), None, _fake("decode", 4))


def test_guard_refuses_a_zero_plane_for_the_prefill_kernel() -> None:
    logits = torch.zeros(65, EXPERTS, device="cuda")
    bias = torch.zeros(EXPERTS, device="cuda")
    zero = torch.zeros(65, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="no zero plane"):
        with validating(op):
            op(logits, bias, 0, 1.0, _outputs(65), zero, _fake("prefill", None))
