# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_mqa_logits entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.fp8_mqa_logits import (
    fp8_mqa_logits as op,
)
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.fp8_mqa_logits import (
    mqa_logits_schedule,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "fp8_mqa_logits requires a CUDA device"

E4M3_MAX = 448.0

# Gate, in fp32 ulps. See `_compare`.
ULP = 2.0**-23
ELEMENT_ULP = 8
RMS_ULP = 8

#: The logits budget a tiling caller cuts its queries to: floor(2**24 / keys)
#: rows per call, 1820 at 9216 keys and 512 at 32768.
LOGITS_BUDGET = 2**24

#: The tep4, tp4 and tep8 deployments' budget at their 1M envelope: floor(2**28 / keys)
#: rows per call, 256 at 1048576 keys and 8192 at 32768, about 1 GiB of fp32
#: logits.
LOGITS_BUDGET_1M = 2**28

#: Tiles per wider call in `test_2_28_tiles_are_bitwise_rows_of_a_wider_call`:
#: four 2**28-element tiles, about 4 GiB of fp32 logits.
WIDE_TILES = 4

#: Query rows per reference evaluation in `_check`. See there.
REF_ROWS = 2048


def _magnitude(q, kv, weights) -> torch.Tensor:
    """`M[i, j] = k_scale[j] * sum_h |weights[i, h]| * sum_d |q[i, h, d] * k[j, d]|`.

    The size of the terms logit `[i, j]` sums, without the cancellation between
    them: what any evaluation's rounding error scales with. Every term is
    non-negative and every fp8 code is exact in fp32, so an fp32 matmul is
    exact up to its accumulation, ample for a scale.
    """
    k, k_scale = kv
    qa, ka, wa = q.float().abs(), k.float().abs(), weights.float().abs()
    m = torch.zeros(q.shape[0], k.shape[0], device=q.device)
    for h in range(q.shape[1]):
        m += (qa[:, h] @ ka.t()) * wa[:, h : h + 1]
    return m * k_scale.float().abs().unsqueeze(0)


def _errors(out: torch.Tensor, ref: torch.Tensor, args) -> tuple:
    """Assert the -inf pattern exactly; return the in-window error terms.

    `(max error in ulps of M, sum of squared errors, sum of squared reference
    values)`, so that `_check` can sum a call's blocks into one RMS.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    must_inf = torch.isneginf(ref)
    assert bool(torch.isneginf(out[must_inf]).all()), "an out-of-window entry is not -inf"
    live = torch.isfinite(ref)
    assert bool(torch.isfinite(out[live]).all()), "an in-window entry is not finite"
    o = torch.where(live, out, torch.zeros_like(out)).double()
    r = torch.where(live, ref, torch.zeros_like(ref)).double()
    m = _magnitude(*args[:3]).double().clamp_min(1e-30)
    err = float(((o - r).abs() / m).max()) / ULP if o.numel() else 0.0
    return err, float(((o - r) ** 2).sum()), float((r**2).sum())


def _assert_band(err: float, sq_err: float, sq_ref: float) -> None:
    assert err <= ELEMENT_ULP, (
        f"max error {err:.3g} ulp of the term magnitude exceeds {ELEMENT_ULP}"
    )
    rms = sq_err**0.5 / max(sq_ref**0.5, 1e-30) / ULP
    assert rms <= RMS_ULP, f"relative RMS {rms:.3g} ulp exceeds {RMS_ULP}"


def _compare(out: torch.Tensor, ref: torch.Tensor, args) -> None:
    """8 fp32 ulps of each logit's term magnitude `M`, and 8 ulps relative RMS.

    fp8 x fp8 products are exact in fp32, so kernel and fp64 reference differ
    only where the kernel rounds: its tensor-core fp32 sum of 128 products per
    head, its fp32 FMA chain over the 32 weighted heads (relu taken as
    (a + |a|) / 2, which is exact), and the final product with the key scale.
    Each rounding is bounded by the magnitude of what it sums, so the element
    gate is stated in ulps of `M` (`_magnitude`) rather than of the logit:
    a row whose window holds one key -- the first token of every request --
    has a logit that is one cancelling sum, and a gate relative to it is
    ill-conditioned (measured up to 25.8 ulp of the row's largest |logit| on
    the 16-request cells). Measured on sm_103 over every cell and every
    1820-row tile of the two 9216-key steps: at most 0.20 ulp of `M` per
    element, 0.57 ulp relative RMS (about what rounding the output to fp32
    costs); over the two 32768-key steps, whole and as 512-row tiles, at most
    0.23 ulp and 0.572 ulp; over the sixteen 1M-envelope cells (tiles at 32768
    to 1048576 keys, packed chunks at 2097152 and 2373713), at most 0.220 ulp
    and 0.604 ulp (the 2-row packed chunk), and over the twelve of them tep4
    certified (tiles at 32768 to 1048576 keys) at most 0.220 ulp and 0.582 ulp;
    over dep4's 1M envelope cells (32768 to 1966090 keys: the 2**28, 2**26 and
    2**24 tiles and the packed blocks) at most 0.245 ulp and 0.577 ulp, and
    0.250 and 0.590 at other seeds; over the fifteen tep8 1M cells (share tiles
    at 4096 to 1048576 keys) at most 0.276 ulp and 0.573 ulp. The gate is 29x
    and 13x above the largest of those (35x and 13x above tp4's, 35x and 14x
    above tep4's, 29x and 14x above tep8's). A wrong
    formula lands far outside: no key scale at 1.5e8 ulp of `M`, a
    neighbouring row's head weights at 1.3e6, relu after the head sum at
    8.4e5, the second kernel fp16 weights select at 458 (RMS 4000 ulp), and
    merely rounding the head weights to fp16 at 195 (RMS 1777 ulp).

    The -inf pattern outside the windows is exact: the top-k that follows
    treats those columns as absent, so a window off by one key fails here.
    NaN in `ref` means unspecified, -inf must be -inf, and a finite entry must
    sit within the band.
    """
    _assert_band(*_errors(out, ref, args))


def _check(out: torch.Tensor, args) -> None:
    """`_compare` of a whole call against `op.reference`, `REF_ROWS` query rows at a time.

    The reference is row-separable: row `i` reads `q[i]`, `weights[i]` and the
    keys of its own window, and is -inf everywhere else. So each block of rows
    is evaluated over its key span `[c0, c1)` -- the smallest window start to
    the largest window end, clipped to the keys -- with the block's windows
    shifted by `c0`; outside the span every window of the block is empty, the
    reference holds -inf there, and the output must too, exactly. The blocks'
    error terms are summed, so the RMS gate is the whole call's. This keeps the
    fp64 transients at `[REF_ROWS, c1 - c0]` and skips the keys no row of a
    block can see -- most of a causal step's upper triangle, all but the
    diagonal of a packed one. fp64 matmuls are slow on this GPU, so their
    work decides what the 32768-key cells cost: measured 4.87 s and
    3.06 GiB for the one-request call, against 8.73 s and 6.12 GiB over every
    key in 4096-row blocks; an unblocked reference holds several [T, N] fp64
    tensors, 8 GiB each at T = N = 32768. A 2**28-element tile costs about the
    same whatever its keys: [256, 1048576] in one block 2.2 s and 11.5 GiB
    above its inputs, [8192, 32768] in four 2.0 s and 2.9 GiB (measured).
    `test_the_blocked_gate_sees_what_it_skips` shows the skipped entries are
    still checked.
    """
    q, (k, k_scale), w, ks, ke = args
    n = k.shape[0]
    assert out.dtype == torch.float32 and tuple(out.shape) == (q.shape[0], n), out.shape
    err, sq_err, sq_ref = 0.0, 0.0, 0.0
    for lo in range(0, q.shape[0], REF_ROWS):
        rows = slice(lo, lo + REF_ROWS)
        c0 = max(0, min(int(ks[rows].min()), n))
        c1 = max(min(int(ke[rows].max()), n), c0)
        block = out[rows]
        outside = torch.isneginf(block[:, :c0]).all() & torch.isneginf(block[:, c1:]).all()
        assert bool(outside), "an out-of-window entry is not -inf"
        if c1 == c0:
            continue
        span = (q[rows], (k[c0:c1], k_scale[c0:c1]), w[rows], ks[rows] - c0, ke[rows] - c0)
        e, se, sr = _errors(block[:, c0:c1], op.reference(*span), span)
        err, sq_err, sq_ref = max(err, e), sq_err + se, sq_ref + sr
    _assert_band(err, sq_err, sq_ref)


def _quantize_rows(x: torch.Tensor):
    """Per-row e4m3 with an fp32 scale, as the indexer quantizes its keys."""
    scale = (x.float().abs().amax(dim=-1) / E4M3_MAX).clamp_min(1e-12)
    return (x.float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale.contiguous()


def _step(spec, seed):
    """`(q, (k, k_scale), weights, ks, ke)` for every query row of one context step.

    The weights carry the indexer's own scale, `D**-0.5 * H**-0.5`, so the
    logits sit at the magnitudes the target produces.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    heads, dim = spec["heads"], spec["head_dim"]
    starts, ends, col = [], [], 0
    for cached, new in spec["seqs"]:
        for t in range(new):
            starts.append(col)
            ends.append(col + cached + t + 1)
        col += cached + new
    rows = len(starts)
    q = torch.randn(rows, heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    k, k_scale = _quantize_rows(torch.randn(col, dim, generator=g, device="cuda"))
    w = torch.randn(rows, heads, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
    ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
    return q, (k, k_scale), w.contiguous(), ks, ke


def _build(spec, seed):
    """One call's arguments: the step's query tile `rows`, as views, or the whole step."""
    q, kv, w, ks, ke = _step(spec, seed)
    lo, hi = spec.get("rows", (0, q.shape[0]))
    return q[lo:hi], kv, w[lo:hi], ks[lo:hi], ke[lo:hi]


def _schedule(spec, args):
    """The split-KV schedule a `split_kv` cell's call takes (`mqa_logits_schedule` over its
    own windows and keys), else None: the grid-stride call."""
    if not spec.get("split_kv"):
        return None
    q, (k, _), _, ks, ke = args
    return mqa_logits_schedule(ks, ke, k.shape[0], q.shape[1])


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(*args, schedule_meta=_schedule(cell.spec, args))
    q, (k, _), *_ = args
    n = k.shape[0]
    assert tuple(out.shape) == (q.shape[0], n)
    # A view of rows padded to align(N + 256, 256): the prefill top-k reads it
    # at stride(0) as it stands.
    assert out.stride(1) == 1 and out.stride(0) == (n + 256 + 255) // 256 * 256, out.stride()
    _check(out, args)


#: The whole step of every cell that scores a query tile of one, at the 2**24
#: budget: steps whose untiled call is itself a cell. The `budget` cells' steps
#: (dep4's) are not: at 32768 to 1048576 keys a whole chunk is [32768, n_k], up
#: to 2**35 logits, which no caller computes in one call
#: (`test_budget_tiles_are_bitwise_rows_of_the_2_28_tile`); nor are the
#: "Mistral Large 4 1M" cells' (tp4's and tep4's, `_TILED_STEPS_1M`), nor the
#: `share` cells' (tep8's, `_TEP8_SHARES`). tp4's and tep4's files left out
#: their 1M cells by the why prefix, dep4's its 1M cells by the `budget` key,
#: tep8's its 1M cells by the why prefix and the `share` key; no tp4, tep4 or
#: tep8 cell carries `budget`, no dep4 why starts with "Mistral Large 4 1M" and no
#: tp4, tep4 or dep4 cell carries `share`, so each member's cells select here
#: exactly as in its own file (the base cells' four steps).
_TILED_STEPS = {
    f"{len(c.spec['seqs'])}-request-{sum(a + b for a, b in c.spec['seqs'])}-key": {
        k: v for k, v in c.spec.items() if k != "rows"
    }
    for c in op.CELLS
    if "rows" in c.spec
    and "budget" not in c.spec
    and "share" not in c.spec
    and not c.why.startswith("Mistral Large 4 1M")
}

#: The whole step of every 1M-envelope cell that scores a 2**28 tile of one:
#: the "Mistral Large 4 1M" cells (tp4's and tep4's). dep4's 1M cells, whose whys
#: start otherwise, are not here; `_TILES_2_28` takes its `budget` cells. Nor
#: are tep8's ("Mistral Large 4 tep8 1M"), whose tiles are cut from their rank's
#: share: `_TEP8_SHARES` takes those.
_TILED_STEPS_1M = {
    f"{len(c.spec['seqs'])}-request-{sum(a + b for a, b in c.spec['seqs'])}-key": {
        k: v for k, v in c.spec.items() if k != "rows"
    }
    for c in op.CELLS
    if "rows" in c.spec and c.why.startswith("Mistral Large 4 1M")
}


@pytest.mark.parametrize("spec", list(_TILED_STEPS.values()), ids=list(_TILED_STEPS))
def test_query_tiles_are_bitwise_rows_of_one_call(spec) -> None:
    """Tiling the queries changes no bit: a row's logits depend on its own window only.

    The untiled call of each tiled step is a certified cell, so this carries
    the certification to every tile a 2**24-element logits budget cuts (at
    9216 keys five of 1820 rows and the 116-row tail, at 32768 keys sixty-four
    of 512 rows), and shows a repeated call is bitwise the same.
    """
    q, kv, w, ks, ke = _build(spec, seed=13)
    whole = op(q, kv, w, ks, ke)
    assert torch.equal(op(q, kv, w, ks, ke), whole), "a repeated call is not bitwise the same"
    tile = LOGITS_BUDGET // kv[0].shape[0]
    for lo in range(0, q.shape[0], tile):
        hi = min(lo + tile, q.shape[0])
        out = op(q[lo:hi], kv, w[lo:hi], ks[lo:hi], ke[lo:hi])
        assert torch.equal(out, whole[lo:hi]), f"rows [{lo}, {hi}) differ from the untiled call"


def _keys(spec) -> int:
    return sum(a + b for a, b in spec["seqs"])


def _tile_rows(spec) -> int:
    lo, hi = spec.get("rows", (0, sum(b for _, b in spec["seqs"])))
    return hi - lo


#: Every cell that is one whole query tile of the 2**28 budget and that the
#: 2**24 budget cuts into more than one.
_TILES_2_28 = [
    c
    for c in op.CELLS
    if c.spec.get("budget") == 2**28 and _tile_rows(c.spec) > 2**24 // _keys(c.spec)
]


def test_a_packed_chunk_split_per_request_changes_no_bit() -> None:
    """Each request's own call is bitwise its rows and keys of the packed call.

    The engine packs requests into one indexer chunk while sum(new) x
    sum(cached + new) <= 2**28, so two 5-token last chunks over 983040 cached
    keys each gather 1966090 keys, past max_seq_len. Split per request, each
    call scores its own 983045 keys with windows rebased to key 0, as the
    split cell does; this shows the split changes no bit of either request's
    logits, so certifying the split call certifies what the target reads.
    Each request's scales are a buffer of their own, as a per-request gather
    allocates them: a view at key 983045 is not 16-byte aligned, which the op
    refuses (`test_a_k_scale_view_off_16_bytes_raises`).
    """
    q, (k, s), w, ks, ke = _build(dict(heads=32, head_dim=128, seqs=[(983040, 5)] * 2), seed=21)
    packed = op(q, (k, s), w, ks, ke)
    row, col = 0, 0
    for cached, new in [(983040, 5)] * 2:
        n, rows = cached + new, slice(row, row + new)
        kv = (k[col : col + n], s[col : col + n].clone())
        own = op(q[rows], kv, w[rows], ks[rows] - col, ke[rows] - col)
        assert torch.equal(own, packed[rows, col : col + n]), f"request at key {col} differs"
        row, col = row + new, col + n


def test_a_k_scale_view_off_16_bytes_raises() -> None:
    """A `k_scale` view at a key offset that is not a multiple of 4 raises.

    The op asserts a 16-byte-aligned `k_scale`; at offsets 4 and 8 the view
    scores bitwise like a copy. This is what a caller that splits a packed
    chunk by slicing its gathered scales meets. The `k` rows are 128 bytes, so
    their views are always aligned.
    """
    q, (k, s), w, ks, ke = _ok()
    for off in (1, 2, 3, 4, 8):
        kv = (k[off:], s[off:])
        args = (q, kv, w, (ks - off).clamp_min(0), (ke - off).clamp_min(0))
        if off % 4:
            with pytest.raises(RuntimeError, match=r"data_ptr\(\)\) % 16u == 0u"):
                op(*args)
        else:
            assert torch.equal(op(*args), op(q, (k[off:], s[off:].clone()), *args[2:]))


#: The step of every tep8 share a cell tiles in more than one call, keyed by its
#: requests and the share's rows.
_TEP8_SHARES = {
    "+".join(f"{a}:{b}" for a, b in c.spec["seqs"])
    + f"-rows-{c.spec['share'][0]}-{c.spec['share'][1]}": {
        k: v for k, v in c.spec.items() if k != "rows"
    }
    for c in op.CELLS
    if "share" in c.spec
    and c.spec["share"][1] - c.spec["share"][0] > LOGITS_BUDGET_1M // _keys(c.spec)
}


#: A 1048576-token prompt's last chunk behind its 1015808 cached tokens, and the
#: 1048559-token envelope prompt's: the steps the split test divides eight ways.
_SPLIT_CHUNKS_1M = {"32768-token": (1015808, 32768), "32751-token": (1015808, 32751)}


def _ok():
    """A cached prefix of 64 keys and 8 new tokens: every window starts at key 0."""
    return _build(dict(heads=32, head_dim=128, seqs=[(64, 8)]), seed=6)


def test_row_padded_weights_are_read_at_their_stride() -> None:
    """The op reads `weights` rows at stride(0), so a padded view needs no copy."""
    q, kv, w, ks, ke = _ok()
    padded = torch.zeros(w.shape[0], 64, device="cuda")[:, : w.shape[1]]
    padded.copy_(w)
    with validating(op):
        out = op(q, kv, padded, ks, ke)
    assert torch.equal(out, op(q, kv, w, ks, ke))


def test_a_window_past_the_keys_is_clamped_to_them() -> None:
    """A window that ends beyond the last key scores the keys there are."""
    q, kv, w, ks, ke = _ok()
    far = ke + 1000
    with validating(op):
        out = op(q, kv, w, ks, far)
    _compare(out, op.reference(q, kv, w, ks, far), (q, kv, w))


def test_guard_refuses_half_weights() -> None:
    """fp16 weights select a second kernel, which answers at half precision."""
    q, kv, w, ks, ke = _ok()
    half = w.half()
    out = op(q, kv, half, ks, ke)
    with pytest.raises(AssertionError, match="max error"):
        _compare(out, op.reference(q, kv, half.float(), ks, ke), (q, kv, half))
    with pytest.raises(AssertionError, match="weights must be fp32"):
        with validating(op):
            op(q, kv, half, ks, ke)


def test_guard_refuses_a_negative_window_start() -> None:
    """A start of -1 is not read as 0: every logit of the call comes back -inf."""
    q, kv, w, ks, ke = _ok()
    out = op(q, kv, w, ks - 1, ke)
    assert bool(torch.isneginf(out).all())
    with pytest.raises(AssertionError, match="in-window entry is not finite"):
        _compare(out, op.reference(q, kv, w, ks - 1, ke), (q, kv, w))
    with pytest.raises(AssertionError, match="starts before key 0"):
        with validating(op):
            op(q, kv, w, ks - 1, ke)


def test_guard_refuses_an_inverted_window() -> None:
    """An inverted window is not read as empty: its row comes back with finite logits."""
    q, kv, w, ks, ke = _ok()
    inverted = ks.clone()
    inverted[3] = ke[3] + 2  # still inside the 72 keys
    out = op(q, kv, w, inverted, ke)
    assert bool(torch.isfinite(out[3]).any())
    with pytest.raises(AssertionError, match="out-of-window entry is not -inf"):
        _compare(out, op.reference(q, kv, w, inverted, ke), (q, kv, w))
    with pytest.raises(AssertionError, match="ends before it starts"):
        with validating(op):
            op(q, kv, w, inverted, ke)


#: The dep4 deployment's split-KV calls.
_SPLIT_KV = [c for c in op.CELLS if c.spec.get("split_kv")]


@pytest.mark.parametrize("cell", _SPLIT_KV, ids=[c.why[:44] for c in _SPLIT_KV])
def test_split_kv_schedule_is_bitwise_the_grid_stride_call(cell) -> None:
    """The schedule moves no bit: with it each 4-row query block's 256-key splits are spread
    over several SMs, without it one SM scans them all, and every logit comes out the same.

    The kernel computes a logit inside one 256-key split of its query block: a tensor-core
    product over the 128 dims, the fixed-order sum over the 32 weighted heads and the key
    scale (`sm100_mqa_logits.cuh`), and the splits sit at the same keys either way (the
    block's smallest window start rounded down to 4, then every 256). The cleaner that
    writes -inf outside the windows runs grid-stride in both. So the whole [T, N] view,
    -inf entries and row padding stride included, is bitwise the schedule-less call's, and
    the dep4 deployment's 2**26 tiles keep the certification of the grid-stride path. A
    schedule is reused across calls: the target builds one per tile and step and scores
    every indexer layer's keys with it, so a second call with the same schedule and
    other keys of the same count is bitwise its own grid-stride call too.
    """
    q, kv, w, ks, ke = args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    sched = _schedule(cell.spec, args)
    plain = op(*args)
    with validating(op):
        split = op(*args, schedule_meta=sched)
    assert split.shape == plain.shape and split.stride() == plain.stride()
    assert torch.equal(split.view(torch.int32), plain.view(torch.int32)), (
        "the split-KV schedule changed a logit"
    )
    del plain, split
    g = torch.Generator(device="cuda").manual_seed(29)
    k2 = _quantize_rows(torch.randn(kv[0].shape[0], kv[0].shape[1], generator=g, device="cuda"))
    again = op(q, k2, w, ks, ke, schedule_meta=sched)
    assert torch.equal(again.view(torch.int32), op(q, k2, w, ks, ke).view(torch.int32)), (
        "a reused schedule changed a logit"
    )


def test_guard_refuses_a_schedule_built_for_other_windows() -> None:
    """The kernel trusts the schedule; one built for another call's windows scores the wrong
    key spans."""
    q, kv, w, ks, ke = _build(dict(heads=32, head_dim=128, seqs=[(4096, 64)]), seed=8)
    good = mqa_logits_schedule(ks, ke, kv[0].shape[0], q.shape[1])
    with validating(op):
        op(q, kv, w, ks, ke, schedule_meta=good)
    other = mqa_logits_schedule(ks, ke - 3000, kv[0].shape[0], q.shape[1])
    with pytest.raises(AssertionError, match="was not built for this call"):
        with validating(op):
            op(q, kv, w, ks, ke, schedule_meta=other)
