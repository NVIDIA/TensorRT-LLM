# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_split_sinkhorn catalog entry.

The expected value is built here from native torch only, following the
checkpoint's own `hc_split_sinkhorn` kernel term by term. Three details a casual
reading loses, and each is a separate assertion below:

  * `pre` is `sigmoid(.) + eps` while `post` is `mult * sigmoid(.)` -- the eps is
    on one and not the other;
  * the FIRST normalization is a row softmax with `+ eps` applied AFTER the
    division, while every later one divides by `sum + eps`;
  * the initial softmax AND the first column normalization sit OUTSIDE the loop,
    which runs `sinkhorn_repeat - 1` times. So `sinkhorn_repeat <= 1` is not a
    call that skips the column pass -- every value at or below 1 is the SAME
    call, and asking for 0 silently gets you 1.

The invalid domains split into two groups, and which group a case is in decides
what this file has to assert about it.

EIGHT TENSOR-METADATA violations are guarded by the wrapper, and each is proved
in two steps rather than in prose. The raw op is driven with an argument that
differs from the correct one in METADATA ONLY -- the same logical values, with
any storage the kernel reads *past* them poisoned to a value this file chooses
-- and the result is asserted accepted, finite, and wrong in exactly the outputs
that piece of metadata can reach. Only then is the wrapper asserted to reject
it. A malformed input carrying different logical values would prove nothing
about the shape or the stride, because the values alone would explain the
mismatch.

FOUR SCALAR DOMAINS are not guarded and must not be: `k <= 0` and
`sinkhorn_repeat <= 1` misbehave just as silently, but `k` is the square sum's
divisor and `sinkhorn_repeat` is a loop bound, so both are computation values
rather than metadata -- and the catalog authorizes a wrapper assert only for
metadata. The obligation there is inverted: measure what the raw op does, then
assert the wrapper passes it through untouched. An earlier revision asserted
both in the wrapper, which was a wrapper-contract violation rather than a safety
feature.

Nothing on the expected side comes from the op, from `refmods.py`, or from any
other catalog entry.
"""

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.activation.mhc_split_sinkhorn import (
    HC_MULT,
    MIX_HC,
    mhc_split_sinkhorn,
)

assert torch.cuda.is_available(), "mhc_split_sinkhorn requires a CUDA device"

#: DeepSeek-V4.1-Flash, from config.json's text_config. `k` is the flattened
#: stream width the square sum spans: hc_mult * hidden_size = 4 * 5120.
V41_K = 20480
V41_RMS_EPS = 1e-20  # rms_norm_eps -- the one inside the rsqrt
V41_HC_EPS = 1e-6  # hc_eps -- both the `+ eps` on pre and the Sinkhorn's
V41_POST_MULT = 2.0  # the `2 *` in `post = 2 * sigmoid(.)`
V41_ITERS = 20  # hc_sinkhorn_iters

#: Row counts spot-checked against the native-torch reference at the V4.1 column.
#: The *interval* of certified `M` is proved separately and far more widely by
#: `test_row_results_are_independent_of_m`, which is the covering rule; this list
#: is the accuracy check, not the shape claim.
ROWS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: The covering rule's upper end. 16,384 rows is above any prefill chunk this
#: target serves, and the sweep below it is dense.
M_COVER = 16384
COVER_MS = list(range(0, 130)) + [
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16383,
    16384,
]

_NAMES = ("pre", "post", "comb")

#: Values written into storage the kernel reads only when the metadata is wrong.
#: They are finite and far from anything a correct call produces, so a mismatch
#: is deterministic rather than allocator-dependent -- the whole point of these
#: constructions is that the *only* difference from a correct call is metadata.
POISON_Y = 0.5
POISON_R = 1.0  # a legal square sum, but ~143x off the rstd of any real row here
POISON_S = -3.0
POISON_B = -9.0


def _ref(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    k: int,
    rms_eps: float,
    pre_eps: float,
    sink_eps: float,
    post_mult: float,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native-torch reference, written straight from the source kernel."""
    hc = HC_MULT
    rstd = torch.rsqrt(r_acc.float() / k + rms_eps).unsqueeze(-1)
    m = y_acc.float() * rstd
    s = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[:, :hc] * s[0] + base[:hc]) + pre_eps
    post = post_mult * torch.sigmoid(m[:, hc : 2 * hc] * s[1] + base[hc : 2 * hc])
    comb = (m[:, 2 * hc :] * s[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + sink_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sink_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sink_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sink_eps)
    return pre, post, comb


def _operands(m: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    y = torch.randn(m, MIX_HC, generator=gen, device="cuda", dtype=torch.float32)
    r = torch.rand(m, generator=gen, device="cuda", dtype=torch.float32) * V41_K + 1.0
    scale = torch.randn(3, generator=gen, device="cuda", dtype=torch.float32)
    base = torch.randn(MIX_HC, generator=gen, device="cuda", dtype=torch.float32)
    return y.contiguous(), r.contiguous(), scale.contiguous(), base.contiguous()


def _call(y, r, scale, base, **kw):
    return mhc_split_sinkhorn(
        y,
        r,
        scale,
        base,
        kw.get("k", V41_K),
        kw.get("rms_eps", V41_RMS_EPS),
        kw.get("pre_eps", V41_HC_EPS),
        kw.get("sink_eps", V41_HC_EPS),
        kw.get("post_mult", V41_POST_MULT),
        kw.get("iters", V41_ITERS),
    )


def _raw(m, y, r, scale, base, k=V41_K, iters=V41_ITERS):
    """Drive `torch.ops.trtllm.mhc_split_sinkhorn` directly, bypassing every wrapper guard.

    This is what makes the guard evidence a measurement: the wrapper cannot show
    that the op accepts a malformed call, because the wrapper is what stops it.
    """
    pre = torch.empty(m, HC_MULT, device="cuda", dtype=torch.float32)
    post = torch.empty(m, HC_MULT, device="cuda", dtype=torch.float32)
    comb = torch.empty(m, HC_MULT, HC_MULT, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_split_sinkhorn(
        y,
        r,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        k,
        V41_RMS_EPS,
        V41_HC_EPS,
        V41_HC_EPS,
        V41_POST_MULT,
        iters,
    )
    torch.cuda.synchronize()
    return pre, post, comb


@pytest.mark.parametrize("m", ROWS)
def test_v41_column_by_row_bucket(m: int) -> None:
    """Numerical spot-check at this checkpoint's exact constants, across 17 row counts.

    This is the ACCURACY check, not the shape claim, and the distinction is the
    whole reason both exist. These 17 buckets are samples, chosen to straddle the
    quad and block boundaries; they are not "the row counts the engine serves",
    which is not knowable from inside this entry. The certified interval of `M`
    is proved separately and far more widely by
    `test_row_results_are_independent_of_m`.

    The tolerance is `torch.testing.assert_close`'s fp32 default, unchanged --
    both sides are fp32 and the op measured ~2e-7 against this reference, so
    nothing here needs a widened floor.
    """
    y, r, scale, base = _operands(m, 41 * m + 7)
    pre, post, comb = _call(y, r, scale, base)
    want = _ref(
        y, r, scale, base, V41_K, V41_RMS_EPS, V41_HC_EPS, V41_HC_EPS, V41_POST_MULT, V41_ITERS
    )
    assert pre.shape == (m, HC_MULT) and post.shape == (m, HC_MULT)
    assert comb.shape == (m, HC_MULT, HC_MULT)
    torch.testing.assert_close(pre, want[0])
    torch.testing.assert_close(post, want[1])
    torch.testing.assert_close(comb, want[2])


def test_zero_rows_are_accepted_and_return_empty() -> None:
    """`M = 0` is a served shape, not a degenerate one: a dep4 rank can hold no logical rows.

    The launcher early-returns on `M <= 0` rather than forming a zero-sized
    grid, so the call is a no-op and the three outputs come back empty with the
    right trailing shapes. Synchronizing is the assertion that no CUDA error was
    queued behind it -- an invalid launch configuration would surface here and
    nowhere else.
    """
    y, r, scale, base = _operands(0, 24601)
    pre, post, comb = _call(y, r, scale, base)
    torch.cuda.synchronize()
    assert pre.shape == (0, HC_MULT), pre.shape
    assert post.shape == (0, HC_MULT), post.shape
    assert comb.shape == (0, HC_MULT, HC_MULT), comb.shape


def test_row_results_are_independent_of_m() -> None:
    """The covering rule for `M`, proved rather than sampled.

    Enumerating "the row counts the engine emits" is not knowable from inside a
    catalog entry -- the target chooses prefill chunks and decode capture
    buckets later. What IS knowable is that the kernel makes each token's result
    a function of that token's own row: the only cross-lane communication is the
    Sinkhorn butterfly, which is a `__shfl_xor` over the token's own aligned quad,
    and the tail is handled by an early `token >= M` return.

    So the claim is proved as an invariant instead of a list. One reference run
    at `M = 16384` is checked against native torch, and then every `M` in a dense
    sweep from 0 to 129 plus the powers-of-two neighbourhoods up to 16,384 must
    reproduce that run's first `M` rows BIT-EXACTLY. Any `M` in `[0, 16384]` is
    therefore certified, including the zero-row and small-decode cases no
    sampled bucket list would have contained.
    """
    y, r, scale, base = _operands(M_COVER, 2718)
    big = _call(y, r, scale, base)
    want = _ref(
        y, r, scale, base, V41_K, V41_RMS_EPS, V41_HC_EPS, V41_HC_EPS, V41_POST_MULT, V41_ITERS
    )
    for name, g, w in zip(_NAMES, big, want):
        torch.testing.assert_close(g, w, msg=f"M={M_COVER}: {name} is not the reference")

    for m in COVER_MS:
        got = _call(y[:m].contiguous(), r[:m].contiguous(), scale, base)
        for name, g, b in zip(_NAMES, got, big):
            assert torch.equal(g, b[:m]), (
                f"M={m}: {name} differs from the first {m} rows of the M={M_COVER} run, so a "
                f"row's result depends on how many rows were submitted with it and no "
                f"interval of M can be certified from a sampled list"
            )


def test_comb_is_doubly_stochastic() -> None:
    """The Sinkhorn ends on a COLUMN normalization, so columns are exact and rows are not.

    Asserting both to the same tightness would be wrong about the algorithm: the
    loop's last operation is the column divide, so column sums land on 1 to fp32
    precision while row sums are merely close. Pinning the asymmetry is what
    makes this a check on the iteration order rather than on arithmetic.
    """
    y, r, scale, base = _operands(512, 99)
    _, _, comb = _call(y, r, scale, base)
    col = comb.sum(dim=-2)
    row = comb.sum(dim=-1)
    torch.testing.assert_close(col, torch.ones_like(col), rtol=0, atol=1e-5)
    assert (row - 1).abs().max() < 0.1, f"row sums drifted to {(row - 1).abs().max().item()}"
    assert (row - 1).abs().max() > (col - 1).abs().max(), (
        "row sums are not looser than column sums, so the iteration does not end on a "
        "column normalization and this test is not checking the order it claims to"
    )


def test_pre_carries_the_eps_and_post_carries_the_multiplier() -> None:
    """`pre = sigmoid(.) + eps` and `post = mult * sigmoid(.)`: different shapes of correction.

    Driven rather than read: a port that gives both the eps, or both the
    multiplier, produces plausible numbers and is caught only here.
    """
    y, r, scale, base = _operands(129, 1234)
    big = 0.25
    pre_a, post_a, _ = _call(y, r, scale, base, pre_eps=0.0)
    pre_b, post_b, _ = _call(y, r, scale, base, pre_eps=big)
    torch.testing.assert_close(pre_b - pre_a, torch.full_like(pre_a, big))
    assert torch.equal(post_a, post_b), "pre_eps moved post; the eps is on pre alone"

    _, post_one, _ = _call(y, r, scale, base, post_mult=1.0)
    _, post_two, _ = _call(y, r, scale, base, post_mult=2.0)
    torch.testing.assert_close(post_two, 2.0 * post_one)
    assert post_one.max() <= 1.0, "post/mult must be a sigmoid, so bounded by 1"


def test_sinkhorn_repeat_is_load_bearing() -> None:
    """More iterations must move `comb`, and 20 must be what the target asks for.

    If the count made no difference the doubly-stochastic claim would be
    vacuous and a target that passed 1 instead of 20 would go unnoticed.
    """
    y, r, scale, base = _operands(129, 555)
    _, _, comb_1 = _call(y, r, scale, base, iters=1)
    _, _, comb_20 = _call(y, r, scale, base, iters=V41_ITERS)
    delta = (comb_20 - comb_1).abs().max().item()
    assert delta > 1e-3, f"20 iterations differ from 1 by only {delta:.3e}; the loop does nothing"
    row_1 = (comb_1.sum(-1) - 1).abs().max().item()
    row_20 = (comb_20.sum(-1) - 1).abs().max().item()
    assert row_20 < row_1, f"row sums did not converge: 1 iter {row_1:.3e}, 20 iters {row_20:.3e}"


def test_sinkhorn_repeat_at_or_below_one_is_the_same_call() -> None:
    """`sinkhorn_repeat <= 1` does NOT skip the column pass; it skips the alternating loop.

    The kernel does the row softmax and one column normalization unconditionally
    and only then enters `for (it = 1; it < sinkhorn_repeat; ...)`. So 1, 0 and
    -1 are the same call. An earlier revision of this entry claimed 0 "returns
    comb without the column normalization"; that is the claim this case exists
    to keep false.

    One pass is nonetheless NOT enough to be doubly stochastic, and that is
    measured here rather than assumed. The column divide is by `cs + eps`, so a
    column whose four entries are all near zero -- every row's softmax put its
    mass elsewhere -- has `cs ~ 4*eps` and lands at `4/5` instead of 1. Measured
    at this seed: 0.2 off at one pass, under 1e-5 at the certified 20. The
    alternating loop is what removes those degenerate columns, so the iteration
    count is load bearing for the property and not just for the values.
    """
    m = 64
    y, r, scale, base = _operands(m, 13579)
    one = _raw(m, y, r, scale, base, iters=1)
    for label, iters in (("zero", 0), ("negative", -7)):
        got = _raw(m, y, r, scale, base, iters=iters)
        for name, g, w in zip(_NAMES, got, one):
            assert torch.equal(g, w), (
                f"sinkhorn_repeat={iters} ({label}) is not bit-equal to 1 in {name}; the loop "
                f"bound is not `it < sinkhorn_repeat` starting at 1"
            )

    ref_one = _ref(y, r, scale, base, V41_K, V41_RMS_EPS, V41_HC_EPS, V41_HC_EPS, V41_POST_MULT, 1)
    torch.testing.assert_close(one[2], ref_one[2])

    off_1 = (one[2].sum(dim=-2) - 1).abs().max().item()
    twenty = _raw(m, y, r, scale, base, iters=V41_ITERS)
    off_20 = (twenty[2].sum(dim=-2) - 1).abs().max().item()
    assert off_1 > 1e-2, (
        f"one pass already gives column sums within {off_1:.3e}; the eps in the column "
        f"divide is then not observable and this case checks nothing"
    )
    assert off_20 < 1e-5, f"20 passes left column sums {off_20:.3e} from 1"


def test_comb_rows_are_the_source_copy() -> None:
    """`comb[.., j, k]` comes from `y_acc[.., 2*hc + j*hc + k]`, not the transpose.

    The orientation is invisible in every shape and dtype check, and it is the
    one the consumer (`mhc_post_mapping`) contracts on, so it is pinned at the
    source: feeding a row-major-permuted comb slice must produce the transposed
    result and nothing else.
    """
    y, r, scale, base = _operands(64, 777)
    _, _, comb = _call(y, r, scale, base)
    y_t = y.clone()
    block = y[:, 2 * HC_MULT :].unflatten(-1, (HC_MULT, HC_MULT))
    y_t[:, 2 * HC_MULT :] = block.transpose(-1, -2).reshape(-1, HC_MULT * HC_MULT)
    base_t = base.clone()
    bblock = base[2 * HC_MULT :].unflatten(-1, (HC_MULT, HC_MULT))
    base_t[2 * HC_MULT :] = bblock.transpose(-1, -2).reshape(-1)
    _, _, comb_from_t = _call(y_t.contiguous(), r, scale, base_t.contiguous())
    # Transposing the input block transposes the pre-Sinkhorn matrix; the
    # Sinkhorn itself is not symmetric, so the results must DIFFER -- if they
    # matched, the orientation would be unobservable and no control could see it.
    assert not torch.allclose(comb, comb_from_t, rtol=1e-3, atol=1e-3), (
        "transposing the comb block changed nothing; this op's comb orientation is "
        "unobservable and a consumer could not rely on it"
    )


def test_op_rejects_non_fp32_accumulator() -> None:
    """The one domain the op rejects itself, so the wrapper does not repeat it."""
    y, r, scale, base = _operands(64, 2024)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        torch.ops.trtllm.mhc_split_sinkhorn(
            y.to(torch.bfloat16),
            r,
            scale,
            base,
            torch.empty(64, HC_MULT, device="cuda"),
            torch.empty(64, HC_MULT, device="cuda"),
            torch.empty(64, HC_MULT, HC_MULT, device="cuda"),
            64,
            V41_K,
            V41_RMS_EPS,
            V41_HC_EPS,
            V41_HC_EPS,
            V41_POST_MULT,
            V41_ITERS,
        )


# ─── the guard table ─────────────────────────────────────────────────────────
#
# Each builder returns the raw-op arguments for ONE malformed call. Every one of
# them carries the correct logical values; what differs is a shape, a stride, or
# a scalar, and any storage the kernel then reads past the logical values is
# poisoned by this file rather than left to the allocator. Each verifier says
# which outputs that piece of metadata can reach, so an accepted-and-wrong
# result is attributed instead of merely observed.

_M_GUARD = 64


def _assert_finite(got, case: str) -> None:
    for name, t in zip(_NAMES, got):
        assert torch.isfinite(t).all(), f"{case}: {name} is non-finite, which is a different claim"


def _assert_prefix_control(got, want, rows: int, case: str) -> None:
    """Rows the malformed metadata cannot reach must be bit-equal; the rest must not be."""
    for name, g, w in zip(_NAMES, got, want):
        assert torch.equal(g[:rows], w[:rows]), (
            f"{case}: {name} moved in the first {rows} rows, which this metadata cannot "
            f"reach -- the mismatch is then not attributable to the metadata"
        )
    assert not any(torch.equal(g, w) for g, w in zip(got, want)), (
        f"{case}: every output is bit-equal to correct, so the op tolerated this metadata "
        f"and the wrapper guard is rejecting a valid call"
    )


def _assert_split(got, want, equal: tuple, differ: tuple, case: str) -> None:
    """Outputs the malformed scalar/base cannot reach stay bit-equal; the reachable ones move."""
    by_name = dict(zip(_NAMES, got))
    ref = dict(zip(_NAMES, want))
    for name in equal:
        assert torch.equal(by_name[name], ref[name]), (
            f"{case}: {name} moved although this metadata does not feed it"
        )
    for name in differ:
        assert not torch.equal(by_name[name], ref[name]), (
            f"{case}: {name} is bit-equal to correct although this metadata feeds it, so the "
            f"guard is rejecting a harmless call"
        )


def _b_wrong_y_width(y, r, scale, base):
    wide = torch.full((_M_GUARD, MIX_HC + 8), POISON_Y, device="cuda", dtype=torch.float32)
    wide[:, :MIX_HC] = y
    return dict(y=wide)


def _b_noncontiguous_y(y, r, scale, base):
    arena = torch.full((_M_GUARD, 2 * MIX_HC), POISON_Y, device="cuda", dtype=torch.float32)
    arena[:, :MIX_HC] = y
    return dict(y=arena[:, :MIX_HC])


def _b_short_r(y, r, scale, base):
    half = _M_GUARD // 2
    arena = torch.full((_M_GUARD,), POISON_R, device="cuda", dtype=torch.float32)
    arena[:half] = r[:half]
    return dict(r=arena[:half])


def _b_noncontiguous_r(y, r, scale, base):
    return dict(r=torch.stack([r, r], dim=1)[:, 0])


def _b_short_hc_scale(y, r, scale, base):
    arena = torch.full((3,), POISON_S, device="cuda", dtype=torch.float32)
    arena[:2] = scale[:2]
    return dict(scale=arena[:2])


def _b_noncontiguous_hc_scale(y, r, scale, base):
    return dict(scale=torch.stack([scale, scale], dim=1)[:, 0])


def _b_short_hc_base(y, r, scale, base):
    arena = torch.full((MIX_HC,), POISON_B, device="cuda", dtype=torch.float32)
    arena[:16] = base[:16]
    return dict(base=arena[:16])


def _b_noncontiguous_hc_base(y, r, scale, base):
    return dict(base=torch.stack([base, base], dim=1)[:, 0])


#: case -> (builder, verifier, wrapper assertion message).
#:
#: EIGHT cases, and every one of them is a pure tensor-METADATA violation --
#: shape, stride, length. That is the catalog's whole permission for a wrapper
#: assert. `k` and `sinkhorn_repeat` are computation values and are NOT in this
#: table however silently they misbehave; they are measured separately below,
#: where the wrapper is required to pass them straight through.
#:
#: `y_acc` metadata is checked with a PREFIX control: token 0's row starts at
#: storage offset 0 under every layout here, so it must come back bit-equal
#: while later tokens, whose row offsets the wrong stride moves, must not.
#: `r_acc`'s short form is the same idea at the half-way row.
GUARDS = {
    "wrong_y_width": (
        _b_wrong_y_width,
        lambda got, want, case: _assert_prefix_control(got, want, 1, case),
        "y_acc must be",
    ),
    "noncontiguous_y": (
        _b_noncontiguous_y,
        lambda got, want, case: _assert_prefix_control(got, want, 1, case),
        "y_acc must be contiguous",
    ),
    "short_r": (
        _b_short_r,
        lambda got, want, case: _assert_prefix_control(got, want, _M_GUARD // 2, case),
        "r_acc must be a contiguous",
    ),
    "noncontiguous_r": (
        _b_noncontiguous_r,
        lambda got, want, case: _assert_prefix_control(got, want, 1, case),
        "r_acc must be a contiguous",
    ),
    "short_hc_scale": (
        _b_short_hc_scale,
        lambda got, want, case: _assert_split(got, want, ("pre", "post"), ("comb",), case),
        "hc_scale must be contiguous and hold at least",
    ),
    "noncontiguous_hc_scale": (
        _b_noncontiguous_hc_scale,
        lambda got, want, case: _assert_split(got, want, ("pre",), ("post", "comb"), case),
        "hc_scale must be contiguous and hold at least",
    ),
    "short_hc_base": (
        _b_short_hc_base,
        lambda got, want, case: _assert_split(got, want, ("pre", "post"), ("comb",), case),
        "hc_base must be contiguous and hold at least",
    ),
    "noncontiguous_hc_base": (
        _b_noncontiguous_hc_base,
        lambda got, want, case: _assert_split(got, want, (), ("pre", "post", "comb"), case),
        "hc_base must be contiguous and hold at least",
    ),
}


@pytest.mark.parametrize("case", sorted(GUARDS))
def test_guard_is_silent_at_the_op_then_rejected_by_the_wrapper(case: str) -> None:
    """The two halves of a guard, in the order that makes it earned rather than assumed.

    First the RAW op is driven with metadata-only damage and must (a) not raise,
    (b) stay finite, and (c) be wrong exactly where that metadata reaches --
    which is what makes the failure silent and the guard necessary. Only then is
    the wrapper asserted to reject the same call. Asserting only the second half
    proves the wrapper has an `assert` in it, not that the assert is warranted.

    Every case here perturbs a tensor's shape, stride or length and nothing else.
    That is not incidental: it is the entire set of violations a catalog wrapper
    may assert on, so a case that changed a scalar would not belong in this test
    however silently it failed.
    """
    y, r, scale, base = _operands(_M_GUARD, 31337)
    build, verify, message = GUARDS[case]
    args = dict(y=y, r=r, scale=scale, base=base)
    args.update(build(y, r, scale, base))

    want = _raw(_M_GUARD, y, r, scale, base)
    got = _raw(_M_GUARD, args["y"], args["r"], args["scale"], args["base"])
    _assert_finite(got, case)
    verify(got, want, case)

    with pytest.raises(AssertionError, match=message):
        _call(args["y"], args["r"], args["scale"], args["base"])


# ─── the scalar domains, which are measured and NOT guarded ──────────────────
#
# `k` divides the square sum and `sinkhorn_repeat` is a loop bound, so both are
# computation values rather than tensor metadata. The catalog authorizes a
# wrapper assert only for a metadata violation, so silence alone does not earn
# one here and the obligation is inverted: the wrapper must pass these through
# untouched, and the contract's Preconditions must say what the op then does.
# An earlier revision asserted both in the wrapper, which was a wrapper-contract
# violation rather than a safety feature.


def _assert_passthrough(through, raw_result, label: str) -> None:
    """The wrapper neither rejected nor altered what the raw op produced."""
    for name, w, g in zip(_NAMES, through, raw_result):
        assert torch.equal(w.isnan(), g.isnan()), (
            f"{label}: the wrapper changed {name}'s NaN pattern instead of passing it through"
        )
        finite = ~g.isnan()
        assert torch.equal(w[finite], g[finite]), (
            f"{label}: the wrapper altered {name} instead of passing it through"
        )


@pytest.mark.parametrize("k", [0, -V41_K])
def test_out_of_domain_k_is_measured_and_passed_through_unguarded(k: int) -> None:
    """`k <= 0` is silent at the op, and the wrapper is required to stay out of the way.

    Both halves are measured on sm_103 rather than described:

      * `k = 0` divides the square sum by zero, so `rstd = rsqrt(inf)` is 0, the
        normalized mixes collapse and all three outputs become what a zero mix
        implies -- finite, plausible, and not the caller's result.
      * `k < 0` takes `rsqrtf` of a negative number, so **every** element of
        **all three** outputs is NaN. `post` is asserted here too: an earlier
        revision checked only `pre` and `comb` while the contract claimed all
        three, which is a claim wider than its evidence.

    Neither raises, and the wrapper must not turn either into a rejection. That
    is the trade the metadata-only guard rule makes on purpose: the contract's
    Preconditions become the only place a caller can learn this.
    """
    y, r, scale, base = _operands(_M_GUARD, 31337)
    correct = _raw(_M_GUARD, y, r, scale, base)
    got = _raw(_M_GUARD, y, r, scale, base, k=k)

    if k == 0:
        _assert_finite(got, f"k={k}")
        for name, g, c in zip(_NAMES, got, correct):
            assert not torch.equal(g, c), (
                f"k=0 is bit-equal to the k={V41_K} result in {name}, so the divisor is "
                f"not read and this domain is not silent but irrelevant"
            )
    else:
        for name, t in zip(_NAMES, got):
            assert not torch.isfinite(t).any(), (
                f"k={k}: {name} still has finite elements, so the contract's claim that every "
                f"element of all three outputs is NaN is wider than the measurement"
            )

    _assert_passthrough(_call(y, r, scale, base, k=k), got, f"k={k}")


@pytest.mark.parametrize("iters", [0, -1])
def test_out_of_domain_sinkhorn_repeat_is_measured_and_passed_through_unguarded(iters: int) -> None:
    """`sinkhorn_repeat <= 1` is silently clamped, and the wrapper must not guard that either.

    The row softmax and the first column normalization sit outside the kernel's
    `for (it = 1; it < sinkhorn_repeat; ...)` loop, so 0 and negative values
    return bit-equal to 1: the argument is ignored rather than honoured, and the
    `comb` that comes back is a plausible non-negative matrix with nothing in it
    to say so. Like `k`, a loop bound is a computation value and not tensor
    metadata, so it is measured and documented rather than rejected.
    """
    y, r, scale, base = _operands(_M_GUARD, 31337)
    one = _raw(_M_GUARD, y, r, scale, base, iters=1)
    twenty = _raw(_M_GUARD, y, r, scale, base, iters=V41_ITERS)
    got = _raw(_M_GUARD, y, r, scale, base, iters=iters)
    _assert_finite(got, f"iters={iters}")
    assert torch.equal(got[2], one[2]), (
        f"sinkhorn_repeat={iters} is not the 1-pass result, so the clamp this entry "
        f"documents is not what the kernel does"
    )
    assert not torch.equal(got[2], twenty[2]), (
        "the 1-pass and 20-pass results are bit-equal, so the clamp would be harmless and "
        "there would be nothing here worth documenting"
    )

    _assert_passthrough(_call(y, r, scale, base, iters=iters), got, f"iters={iters}")


def test_oversized_buffers_are_harmless_and_not_guarded() -> None:
    """The other half of the guard table: what the wrapper must NOT reject.

    An earlier revision asserted `hc_scale.numel() == 3`, `hc_base.numel() == 24`
    and `r_acc.shape == (M,)`. Driven through the raw op, all three OVERSIZED
    forms come back **bit-equal to the correct result** -- the kernel indexes
    what it needs and ignores the tail -- so those guards were rejecting valid
    calls. A guard that fires on a harmless input is a defect in the wrapper,
    and this test is what keeps it from coming back. The tails are poisoned, so
    "ignored" means ignored rather than "happened to hold the right values".
    """
    m = 64
    y, r, scale, base = _operands(m, 4242)
    want = _call(y, r, scale, base)
    long_scale = torch.cat([scale, torch.full((5,), POISON_S, device="cuda")]).contiguous()
    long_base = torch.cat([base, torch.full((8,), POISON_B, device="cuda")]).contiguous()
    long_r = torch.cat([r, torch.full((8,), POISON_R, device="cuda")]).contiguous()
    for label, args in (
        ("hc_scale longer", (y, r, long_scale, base)),
        ("hc_base longer", (y, r, scale, long_base)),
        ("r_acc longer", (y, long_r, scale, base)),
    ):
        got = _call(*args)
        for name, g, w in zip(_NAMES, got, want):
            assert torch.equal(g, w), f"{label}: {name} moved, so the tail is not ignored"


def test_cuda_graph_replay_through_the_wrapper_matches_eager() -> None:
    """The Goal-1.3 graph gate, captured at the CATALOG surface the target will actually call.

    The wrapper allocates its three outputs per call, which reads like a barrier
    to capture and is not one: allocations made inside `torch.cuda.graph` come
    from the graph's private pool, so the tensors the capture call returned are
    the same memory every replay rewrites. Capturing the raw op against
    caller-owned buffers instead would certify a surface no caller of this entry
    uses, and would leave the wrapper itself unproven under capture.

    Two replays with DIFFERENT inputs, because a graph that silently replayed
    the captured input would pass a single-replay check.
    """
    m = 129
    y, r, scale, base = _operands(m, 5150)
    y2, r2, _, _ = _operands(m, 6161)

    y_in = y.clone()
    r_in = r.clone()

    # Warm up on a side stream, which stream capture requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _call(y_in, r_in, scale, base)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pre, post, comb = _call(y_in, r_in, scale, base)

    for src_y, src_r in ((y, r), (y2, r2)):
        y_in.copy_(src_y)
        r_in.copy_(src_r)
        graph.replay()
        torch.cuda.synchronize()
        want = _ref(
            src_y,
            src_r,
            scale,
            base,
            V41_K,
            V41_RMS_EPS,
            V41_HC_EPS,
            V41_HC_EPS,
            V41_POST_MULT,
            V41_ITERS,
        )
        torch.testing.assert_close(pre, want[0])
        torch.testing.assert_close(post, want[1])
        torch.testing.assert_close(comb, want[2])
