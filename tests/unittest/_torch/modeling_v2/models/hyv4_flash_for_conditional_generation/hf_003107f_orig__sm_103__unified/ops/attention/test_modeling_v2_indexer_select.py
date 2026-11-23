# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the hyv4_flash compressed-indexer selection entry `indexer_select`.

The operands are the checkpoint's. `query` and `group_keys` reach the entry BF16 after
`query_projection` (std 0.0216, [2048, 1024] at `model.layers.5`) and `key_norm` (RMS 1.055,
the norm weight's measured mean), so a query element is drawn at `0.0216 * sqrt(1024)` = 0.692
and a group key at 1.055. `head_weights` are `head_weight_projection`'s output (std 0.0500,
[16, 3072] at the same layer) on hidden states taken at unit RMS: `0.0500 * sqrt(3072)` = 2.77.
That last one is an assumption, not a measurement -- the residual stream's RMS needs a forward
pass -- and it only sets the logit spread, because a positive scale on every head weight
rescales every logit and leaves the selection unchanged.

Each cell is judged against the entry's own reference: `lengths` and the own-group tail
exactly, the selected groups' logits as a sorted multiset within `LOGIT_ATOL`, and the selected
groups exactly wherever the row's k-th and (k+1)-th reference logits are farther apart than
that band. The band is derived in `test_logit_band_from_a_float64_recompute` from the distance
between the fp32 reference and a float64 recompute of the same formula; both the kernel and the
reference are fp32 evaluations of the same exact products, so a boundary farther apart than
twice the fp32 error cannot be decided differently by the two.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.indexer_select import (  # noqa: E501
    _reference_logits,
    indexer_select,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "indexer_select requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the hyv4_flash target's entries are certified on sm_103"
)

DEVICE = "cuda"
H, D = 16, 128
RATIO, INDEX_TOPK = 4, 2048
SELECT_K = INDEX_TOPK // RATIO - 1
#: The checkpoint's indexer tensors at `model.layers.5.self_attn.compressed_indexer` (measured
#: from the safetensors index): query_projection std 0.02163 over q_lora_rank 1024, key_norm
#: weight mean 1.055, head_weight_projection std 0.05004 over hidden 3072.
QUERY_STD = 0.02163 * 1024**0.5
KEY_STD = 1.055
HEAD_STD = 0.05004 * 3072**0.5
#: Measured, see `test_logit_band_from_a_float64_recompute`: the fp32 reference sits at most
#: 9.72e-07 from a float64 recompute of the same logits on the 5120-token cell (1280 groups)
#: and 8.74e-07 on the 2049-token one (512 groups). Two fp32 evaluations of the same exact
#: products differ by at most twice that, so a boundary farther apart than 2 * LOGIT_ATOL
#: cannot be decided differently by the kernel and the reference; the constant is 10x the
#: measurement and the band 2 * LOGIT_ATOL = 2e-05.
LOGIT_ATOL = 1e-5


def _args(spec: dict, seed: int) -> tuple:
    """`(query, group_keys, head_weights, row_batch, positions)` for a cell.

    A cell is one row per request (``mode`` absent: ``kv - 1`` is every row's position, and
    ``inactive`` rows are padded at -1) or one prefill chunk (``mode="prefill"``: its rows are
    positions ``kv - rows .. kv - 1`` of a single request).
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    rows, kv = spec["rows"], spec["kv"]
    groups = kv // RATIO
    if spec.get("mode") == "prefill":
        batch = 1
        positions = torch.arange(kv - rows, kv, dtype=torch.int32, device=DEVICE)
        row_batch = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    else:
        batch = rows
        positions = torch.full((rows,), kv - 1, dtype=torch.int32, device=DEVICE)
        row_batch = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        inactive = spec.get("inactive", 0)
        if inactive:
            positions[-inactive:] = -1
    query = (torch.randn(rows, H, D, generator=g, device=DEVICE) * QUERY_STD).to(torch.bfloat16)
    group_keys = (torch.randn(batch, groups, D, generator=g, device=DEVICE) * KEY_STD).to(
        torch.bfloat16
    )
    head_weights = torch.randn(rows, H, generator=g, device=DEVICE) * HEAD_STD
    return query, group_keys, head_weights, row_batch, positions


def _logits(query, group_keys, head_weights, row_batch, positions) -> torch.Tensor:
    """The reference's logits in fp32, the way the entry's `reference` computes them.

    The operands go in as bf16 from the checkpoint's projections; handed to `einsum` at that
    dtype each head's score would round to bf16 before the weighted sum, which the kernel does
    not do.
    """
    return _reference_logits(
        query.float(),
        group_keys.float(),
        head_weights.float(),
        row_batch,
        positions,
        ratio=RATIO,
        scale=(H * D) ** -0.5,
    )


def _check(out: tuple, ref: tuple, logits: torch.Tensor, positions: torch.Tensor) -> dict:
    """Compare a cell's output with the reference's and return what was measured.

    `lengths` and `nonempty` exactly; the padding past each length -1; the own-group tail
    exactly; every selected group's logit (the reference's) as a sorted multiset within
    `LOGIT_ATOL`; and the selected group ids exactly on every row whose boundary gap -- the
    row's k-th minus (k+1)-th reference logit -- is wider than `2 * LOGIT_ATOL`, where no
    legitimate tie rule can differ. Rows whose boundary is inside a tie are counted, not
    compared id by id.
    """
    packed, lengths, nonempty = out
    rpacked, rlengths, rnonempty = ref
    rows = packed.shape[0]
    col = torch.arange(INDEX_TOPK, device=DEVICE)
    assert lengths.dtype == torch.int32 and lengths.shape == (rows,)
    assert torch.equal(lengths, rlengths), (
        f"{(lengths != rlengths).sum()} of {rows} lengths differ from the reference"
    )
    assert torch.equal(nonempty, rnonempty), "the nonempty flags differ from the reference"
    assert bool((packed[col[None, :] >= lengths[:, None].long()] == -1).all()), (
        "the padding past a row's length must be -1"
    )
    active = positions >= 0
    qg = torch.div(positions.long(), RATIO, rounding_mode="floor").clamp(min=0)
    tail = torch.where(active, positions.long() - qg * RATIO, torch.full_like(qg, -1))
    n_sel = (lengths.long() - tail - 1) // RATIO
    assert bool((n_sel >= 0).all()) and bool((n_sel <= SELECT_K).all()), (
        f"a row's selected-group count {n_sel.tolist()[:8]} is outside 0 .. {SELECT_K}"
    )
    # The history slots, one `ratio`-wide run per selected group, ascending. The first `n_sel`
    # runs are selected groups; past them the own group's tail begins (checked separately
    # below), so the selected runs are taken from the row's own count, not from a `>= 0`.
    got = packed[:, : SELECT_K * RATIO].reshape(rows, SELECT_K, RATIO).long()
    want = rpacked[:, : SELECT_K * RATIO].reshape(rows, SELECT_K, RATIO).long()
    got_ok = torch.arange(SELECT_K, device=DEVICE)[None, :] < n_sel[:, None]
    got_g = torch.where(got_ok, got[:, :, 0] // RATIO, torch.zeros_like(got[:, :, 0]))
    want_g = torch.where(got_ok, want[:, :, 0] // RATIO, torch.zeros_like(want[:, :, 0]))
    run = torch.arange(RATIO, device=DEVICE)
    assert bool((got[got_ok] == got_g[got_ok][:, None] * RATIO + run[None, :]).all()), (
        "a selected group must expand to its whole run of `ratio` slots"
    )
    assert bool((want[got_ok] == want_g[got_ok][:, None] * RATIO + run[None, :]).all()), (
        "the reference's expansion is not a whole run of `ratio` slots"
    )
    # Selected logits, sorted: every valid top-k shares this multiset.
    neg = torch.full(got_g.shape, float("-inf"), dtype=logits.dtype, device=DEVICE)
    a = torch.where(got_ok, torch.gather(logits, 1, got_g), neg)
    b = torch.where(got_ok, torch.gather(logits, 1, want_g), neg)
    a, b = a.sort(dim=1).values, b.sort(dim=1).values
    assert torch.equal(a.isinf(), b.isinf()), "a row's number of visible candidates differs"
    finite = ~a.isinf()
    worst = float((a - b).abs()[finite].max()) if bool(finite.any()) else 0.0
    assert worst <= 2 * LOGIT_ATOL, (
        f"the selected logits differ by {worst:.3e}, past the {2 * LOGIT_ATOL:.3e} tie band"
    )
    # The boundary gap of each row's own selection, and the ids wherever it is wide enough.
    best = logits.topk(min(SELECT_K + 1, logits.shape[1]), dim=1).values
    at = n_sel.clamp(min=1, max=SELECT_K) - 1
    kth = best.gather(1, at[:, None]).squeeze(1)
    nxt = best.gather(1, (at + 1).clamp(max=best.shape[1] - 1)[:, None]).squeeze(1)
    gap = torch.where(n_sel > 0, kth - nxt, torch.zeros_like(kth))
    untied = gap > 2 * LOGIT_ATOL
    assert torch.equal(got_g[untied], want_g[untied]), (
        f"{(got_g[untied] != want_g[untied]).any(dim=1).sum()} of {int(untied.sum())} untied "
        f"rows select a different group than the reference"
    )
    # The own group's slots, up to the row's position.
    tail_cols = (col[None, :] >= (n_sel * RATIO)[:, None]) & (
        col[None, :] < lengths[:, None].long()
    )
    expect = (qg[:, None] * RATIO + col[None, :] - (n_sel * RATIO)[:, None]).to(torch.int32)
    assert torch.equal(packed[tail_cols], expect[tail_cols]), (
        "the own group's tail is not the row's slots up to its position"
    )
    empty = ~nonempty
    if bool(empty.any()):
        assert bool((packed[empty, 0] == 0).all()) and bool((lengths[empty] == 1).all()), (
            "a padded row must carry one dummy id and length 1"
        )
    return dict(
        rows=rows,
        selected=float(n_sel.float().mean()),
        untied=int(untied.sum()),
        tied=int((n_sel > 0).sum()) - int(untied.sum()),
        min_gap=float(gap[untied].min()) if bool(untied.any()) else float("inf"),
        multiset=worst,
    )


@pytest.mark.parametrize(
    "cell", indexer_select.CELLS, ids=[c.why[:44] for c in indexer_select.CELLS]
)
def test_indexer_select_matches_reference(cell, certifying):  # noqa: F811
    """See `_check`: lengths and the own-group tail exactly, the selected logits' multiset
    within the tie band, and the selected groups exactly on every row whose boundary is
    farther apart than the band."""
    certifying(indexer_select)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    query, group_keys, head_weights, row_batch, positions = args
    out = indexer_select(query, group_keys, head_weights, row_batch, positions)
    ref = indexer_select.reference(query, group_keys, head_weights, row_batch, positions)
    assert out[0].shape == (cell.spec["rows"], INDEX_TOPK)
    logits = _logits(query, group_keys, head_weights, row_batch, positions)
    m = _check(out, ref, logits, positions)
    print(f"MEASURED {cell.why[:44]}: {m}")


def test_logit_band_from_a_float64_recompute():
    """The band `_check` uses, derived from the reference's own distance to float64.

    Both the kernel and the reference evaluate the same exact products (BF16 operands, so
    every product is exact in fp32) and differ only in accumulation order, so the kernel's
    logits are within `2 * e` of the reference's when the reference is within `e` of the true
    value. Measured here on the two ends of the envelope's pool sizes.
    """
    worst = 0.0
    for spec in (dict(rows=1, kv=5120), dict(rows=8, kv=2049)):
        args = _args(spec, zlib.crc32(repr(spec).encode()) % 2**31)
        query, group_keys, head_weights, row_batch, positions = args
        fp32 = _logits(query, group_keys, head_weights, row_batch, positions)
        fp64 = _reference_logits(
            query.double(),
            group_keys.double(),
            head_weights.double(),
            row_batch,
            positions,
            ratio=RATIO,
            scale=(H * D) ** -0.5,
        )
        finite = torch.isfinite(fp32)
        assert torch.equal(finite, torch.isfinite(fp64))
        err = float((fp32[finite].double() - fp64[finite]).abs().max())
        print(f"MEASURED {spec}: fp32 vs float64 max |dlogit| {err:.3e}")
        worst = max(worst, err)
    assert worst <= LOGIT_ATOL / 8, (
        f"the fp32 reference sits {worst:.3e} from float64, past the {LOGIT_ATOL / 8:.3e} the "
        f"{LOGIT_ATOL:.3e} band was derived from"
    )


def test_tied_group_scores_select_the_same_logits():
    """Ties at the top-k threshold: eight distinct group keys repeated across the pool make
    every logit take one of eight values, so the boundary falls inside a block of exact ties.
    The kernel prefers the lower group id where the reference takes torch's order, so the
    selected ids legitimately differ -- `_check` compares the logits' multiset there and
    reports how many rows were tied."""
    spec = dict(rows=1, kv=5120)
    args = _args(spec, zlib.crc32(b"tied group keys") % 2**31)
    query, group_keys, head_weights, row_batch, positions = args
    group_keys = group_keys[:, torch.arange(group_keys.shape[1], device=DEVICE) % 8].contiguous()
    out = indexer_select(query, group_keys, head_weights, row_batch, positions)
    ref = indexer_select.reference(query, group_keys, head_weights, row_batch, positions)
    logits = _logits(query, group_keys, head_weights, row_batch, positions)
    m = _check(out, ref, logits, positions)
    print(f"MEASURED tied cell: {m}")
    assert m["tied"] > 0, "the tied cell must put a row's boundary inside a block of ties"


def test_guard_refuses_a_strided_query():
    """The kernel walks a row at `D` per head: a query sliced out of a wider last dimension
    reads its neighbour's columns, and the callable accepts the shape."""
    args = _args(dict(rows=8, kv=5120), seed=0)
    _, group_keys, head_weights, row_batch, positions = args
    g = torch.Generator(device=DEVICE).manual_seed(1)
    wide = (torch.randn(8, H, 2 * D, generator=g, device=DEVICE) * QUERY_STD).to(torch.bfloat16)
    sliced = wide[..., :D]
    assert sliced.shape == (8, H, D) and tuple(sliced.stride()) == (H * 2 * D, 2 * D, 1)
    got = indexer_select(sliced, group_keys, head_weights, row_batch, positions)
    want = indexer_select.reference(sliced, group_keys, head_weights, row_batch, positions)
    assert not torch.equal(got[0], want[0]), "the unguarded strided query was supposed to be wrong"
    with pytest.raises(AssertionError, match="dense within a row"):
        with validating(indexer_select):
            indexer_select(sliced, group_keys, head_weights, row_batch, positions)


def test_guard_refuses_a_strided_group_keys():
    """The kernel walks a request's groups at `D` each: a pool sliced out of a wider last
    dimension -- a fused buffer whose other half holds something else -- puts its groups
    `2 * D` apart, and the callable accepts the shape."""
    args = _args(dict(rows=8, kv=5120), seed=0)
    query, group_keys, head_weights, row_batch, positions = args
    g = torch.Generator(device=DEVICE).manual_seed(2)
    batch, groups, _ = group_keys.shape
    wide = (torch.randn(batch, groups, 2 * D, generator=g, device=DEVICE) * KEY_STD).to(
        torch.bfloat16
    )
    sliced = wide[..., :D]
    assert sliced.shape == group_keys.shape
    assert tuple(sliced.stride()) == (groups * 2 * D, 2 * D, 1)
    got = indexer_select(query, sliced, head_weights, row_batch, positions)
    want = indexer_select.reference(query, sliced, head_weights, row_batch, positions)
    assert not torch.equal(got[0], want[0]), (
        "the unguarded last-dim pool slice was supposed to be wrong"
    )
    with pytest.raises(AssertionError, match="dense within a request"):
        with validating(indexer_select):
            indexer_select(query, sliced, head_weights, row_batch, positions)


def test_guard_refuses_a_transposed_head_weights():
    """`head_weights` is square when the rows equal the heads, so `head_weights.t()` has the
    shape the callable accepts; the kernel reads it row-major and weights the heads by the
    wrong column."""
    args = _args(dict(rows=H, kv=5120), seed=0)
    query, group_keys, head_weights, row_batch, positions = args
    transposed = head_weights.t()
    assert transposed.shape == head_weights.shape
    got = indexer_select(query, group_keys, transposed, row_batch, positions)
    want = indexer_select.reference(query, group_keys, transposed, row_batch, positions)
    assert not torch.equal(got[0], want[0]), (
        "the unguarded transposed head weights were supposed to be wrong"
    )
    with pytest.raises(AssertionError, match="dense \\[rows, heads\\]"):
        with validating(indexer_select):
            indexer_select(query, group_keys, transposed, row_batch, positions)


def test_guard_refuses_a_pool_short_of_the_rows_complete_groups():
    """A pool shorter than the row's complete groups silently drops the missing groups from
    the recall: the candidate count is clamped to the pool, so the same row over the same
    cache recalls from a smaller set without a sign."""
    args = _args(dict(rows=1, kv=5120), seed=0)
    query, group_keys, head_weights, row_batch, positions = args
    short = group_keys[:, :640].contiguous()
    got = indexer_select(query, short, head_weights, row_batch, positions)
    want = indexer_select.reference(query, group_keys, head_weights, row_batch, positions)
    assert not torch.equal(got[0], want[0]), "the unguarded short pool was supposed to be wrong"
    with pytest.raises(AssertionError, match="silently miss"):
        with validating(indexer_select):
            indexer_select(query, short, head_weights, row_batch, positions)
