# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssmax_scaling catalog entry.

Every cell is gated twice against the entry:

* bit for bit against its `reference`. The specified computation is a chain of bf16
  roundings and the reference reproduces each link exactly (its docstring says why), so
  there is no band to derive: an implementation of the chain is bit-identical to it, and
  anything else is not. The one link that is not a correctly rounded operation is the
  device's fp32 log (CUDA logf, documented to 1 ulp), where the reference takes the
  correctly rounded one. Measured on sm_103: the two differ in the last fp32 bit at 935
  of the NVFP4 target's 32768 positions (266 of the FP8 target's 9216) and round to the same
  bf16 at every one of them, as they must: the exact log(pos + 1) of every position
  1..32767 lies at least 1.748 fp32 ulp from a bf16 rounding midpoint. The context cells
  drive every position, so a change on either side fails them. Above 32767 there is no
  such margin (38 positions below 2^20 lie within 1 fp32 ulp of a midpoint, measured), so
  the six sweep cells (three at 16 MLA heads or the indexer's 32, three at tep8's 8 MLA
  heads) and dep4's two 32-chunk cells drive every position 0..1048575 and hold the two
  logs' bf16 equal at each directly, beside the gates (`_position_sweep`, `_chunk_sweep`).
* against the closed form ``q * (s * log(pos + 1) + b)`` in fp64, within the chain's
  worst-case error derived at `_closed_form_ratio`. Bitwise agreement between the op and a
  reference written beside it cannot see a mistake the two share (``log(pos)`` for
  ``log(pos + 1)``, another rank's heads); this gate does.

`test_gates_reject_wrong_variants` measures what each gate rejects.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling import (
    FP8_MAX_SEQ_LEN,
    LONG_MAX_SEQ_LEN,
    MAX_NUM_TOKENS,
    MAX_SEQ_LEN,
    MLA_HEADS,
    MLA_PARAM_HEADS,
    MLA_TEP8_HEADS,
    NVFP4_MLA_LAYER0_RANK1,
    NVFP4_MLA_LAYER0_TEP8_RANK2,
    NVFP4_MLA_LAYER0_TEP8_RANK3,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling import (
    ssmax_scaling as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ssmax_scaling requires a CUDA device"

#: The extremes of s and b over both Mistral Large 4 checkpoints, every layer that carries them
#: (the NVFP4 checkpoint's read on disk as bf16, the FP8 checkpoint's rounded outward).
_CHECKPOINT_RANGES = {
    "mla": dict(s=(-0.078125, 0.574219), b=(0.902, 1.578)),
    "indexer": dict(s=(-0.0486, 0.0703), b=(0.973, 1.055)),
}

#: One rounding's relative error: to fp32 (torch's bf16 arithmetic computes in fp32),
#: then to bf16, whose unit roundoff is 2^-8.
_U = (1 + 2.0**-24) * (1 + 2.0**-8) - 1
#: The log's: CUDA logf's documented 1 ulp in fp32, then the bf16 rounding.
_U_LOG = (1 + 2.0**-23) * (1 + 2.0**-8) - 1
#: The last rounding's absolute error where q * scale is subnormal: half the bf16
#: subnormal spacing, after half the fp32 one.
_SUBNORMAL = 2.0**-134 + 2.0**-150
#: The smallest magnitude bf16 rounds to infinity: the largest finite bf16,
#: (2 - 2^-7) 2^127, plus half its ulp.
_OVERFLOW = 2.0**128 * (1 - 2.0**-9)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _queries(shape, kind: str, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    if kind == "normal":
        return (4 * torch.randn(shape, device="cuda", generator=g)).to(dtype)
    # Every bf16 binade from the smallest subnormal (2^-133) up, both signs; exponents stop
    # at 126 so that q itself stays finite.
    exponent = torch.randint(-133, 127, shape, device="cuda", generator=g).float()
    mantissa = 1 + torch.rand(shape, device="cuda", generator=g)
    sign = 2 * torch.randint(0, 2, shape, device="cuda", generator=g).float() - 1
    return (sign * mantissa * torch.exp2(exponent)).to(dtype)


def _positions(spec, g: torch.Generator) -> torch.Tensor:
    dtype = spec["positions_dtype"]
    cached = spec.get("cached", (0,) * len(spec["contexts"]))
    assert len(cached) == len(spec["contexts"]), (cached, spec["contexts"])
    parts = [
        torch.arange(c, c + n, dtype=dtype, device="cuda") for n, c in zip(spec["contexts"], cached)
    ]
    assert all(c + n <= spec["max_seq_len"] for n, c in zip(spec["contexts"], cached))
    if spec["decode_rows"]:
        top = spec["max_seq_len"]
        rows = torch.randint(1, top, (spec["decode_rows"],), device="cuda", generator=g)
        rows[0] = top - 1
        parts.append(rows.to(dtype))
    positions = torch.cat(parts)
    assert positions.numel() == spec["tokens"], (positions.numel(), spec["tokens"])
    return positions


def _build(spec, seed: int):
    """(q, positions, s, b) for one spec. s and b are drawn for all `param_heads` and
    sliced to rank `rank`'s `heads`, so they are views of the full parameter; with
    `corners`, the slice's heads (4 * rank) % heads .. + 3 hold the ranges' four (s, b)
    corners (4 * rank .. 4 * rank + 3 at 16 heads, where no two ranks' coincide), so another
    rank's slice differs from this one at every head. `slice_values` (s, b) then replaces the
    whole slice with those values (a checkpoint's own heads)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (spec["tokens"], spec["heads"], spec["head_dim"])
    q = _queries(shape, spec["q"], spec["dtype"], g)
    positions = _positions(spec, g)
    full = [
        torch.empty(spec["param_heads"], device="cuda", dtype=torch.float32)
        .uniform_(*spec[key], generator=g)
        .to(spec["dtype"])
        for key in ("s_range", "b_range")
    ]
    lo = spec["rank"] * spec["heads"]
    if spec["corners"]:
        at = lo + (4 * spec["rank"]) % spec["heads"]
        corners = [(s, b) for s in spec["s_range"] for b in spec["b_range"]]
        for i, values in enumerate(zip(*corners)):
            full[i][at : at + 4] = torch.tensor(values, device="cuda").to(spec["dtype"])
    for i, values in enumerate(spec.get("slice_values", ())):
        assert len(values) == spec["heads"], (len(values), spec["heads"])
        exact = torch.tensor(values, dtype=torch.float64)
        assert torch.equal(exact.to(spec["dtype"]).double(), exact), "values must be bf16"
        full[i][lo : lo + spec["heads"]] = exact.to(device="cuda", dtype=spec["dtype"])
    s, b = (t[lo : lo + spec["heads"]] for t in full)
    return q, positions, s, b


def _spec_of(site: str, context: int = MAX_SEQ_LEN, heads: int | None = None, **override):
    """The certified spec of `site`'s first `context`-row context cell at position 0 (the
    first at `heads` heads when given), with `override` applied."""
    base = next(
        c.spec
        for c in op.CELLS
        if c.spec["site"] == site
        and c.spec["contexts"] == (context,)
        and c.spec["q"] == "normal"
        and heads in (None, c.spec["heads"])
        and not any(c.spec.get("cached", ()))
        and "position_sweep" not in c.spec
        and "chunks" not in c.spec
    )
    return dict(base, **override)


def _bitwise(out: torch.Tensor, ref: torch.Tensor, what: str = "the reference") -> None:
    assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    differ = out.view(torch.int16) != ref.view(torch.int16)
    n = int(differ.sum())
    if n:
        at = tuple(int(i) for i in differ.nonzero()[0])
        raise AssertionError(
            f"{n} of {out.numel()} elements differ from {what} bit for bit; "
            f"first at {at}: {out[at].item()!r} vs {ref[at].item()!r}"
        )


def _closed_form_ratio(out, q, positions, s, b) -> float:
    """Worst |out - q * (s * log(pos + 1) + b)| as a fraction of the chain's error bound.

    With L = log(pos + 1) and C = L s + b, and each rounding's relative error at most u
    (`_U`; `_U_LOG` for the log):

        lp  = rnd(L)       = L (1 + a)        |a| <= u_log
        m   = rnd(lp s)                       |m - L s| <= |L s| e,  e = (1 + u_log)(1 + u) - 1
        C'  = rnd(m + b)                      |C' - C| <= |L s| e + (|C| + |L s| e) u  =: dC
        out = rnd(q C')                       |out - q C| <= |q| dC (1 + u) + |q C| u

    plus `_SUBNORMAL` where q C' is subnormal. An infinite output passes only where q C'
    can reach `_OVERFLOW`, with the closed form's sign; a nan fails. Inputs are finite.
    """
    ls = torch.log(positions.double() + 1)[:, None] * s.double()[None, :]
    c = ls + b.double()[None, :]
    e = (1 + _U_LOG) * (1 + _U) - 1
    dc = (ls.abs() * e + (c.abs() + ls.abs() * e) * _U)[..., None]
    qd = q.double()
    exact = qd * c[..., None]
    bound = qd.abs() * dc * (1 + _U) + exact.abs() * _U + _SUBNORMAL
    o = out.double()
    if bool(torch.isnan(o).any()):
        return math.inf
    inf = torch.isinf(o)
    if bool(inf.any()):
        reach = (exact.abs() + qd.abs() * dc)[inf] * (1 + 2.0**-24)
        if not bool((reach >= _OVERFLOW).all()):
            return math.inf
        if not torch.equal(torch.sign(o[inf]), torch.sign(exact[inf])):
            return math.inf
    finite = ~inf
    return ((o - exact).abs()[finite] / bound[finite]).max().item()


def _ulps_from_closed_form(out, q, positions, s, b) -> float:
    """Worst |out - q * (s * log(pos + 1) + b)| in bf16 ulps of the closed form (finite inputs)."""
    c = torch.log(positions.double() + 1)[:, None] * s.double()[None, :] + b.double()[None, :]
    exact = q.double() * c[..., None]
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs().clamp_min(2.0**-126))) - 7)
    return ((out.double() - exact).abs() / ulp).max().item()


def _sweep_counts(spec) -> list[int]:
    """The step sizes a sweep cell names (its `sweep_rows` ranges, and 2^k - 1, 2^k and
    2^k + 1 up to `tokens` with `sweep_pow2`), largest first."""
    counts = set()
    for lo, hi, step in spec["sweep_rows"]:
        counts.update(range(lo, hi + 1, step))
    if spec.get("sweep_pow2"):
        k = 1
        while k <= spec["tokens"]:
            counts.update(c for c in (k - 1, k, k + 1) if 1 <= c <= spec["tokens"])
            k *= 2
    return sorted(counts, reverse=True)


def _prefix_sweep(out, q, positions, s, b, counts) -> None:
    """Each step size T in `counts` as its own call, bitwise the leading T rows of `out`.

    Rows are independent, so what can move with T is each launch's grid and its last
    block's tail (the entry's note lists the launches). Compared as int64 words with the
    verdicts kept on the device and read once, largest T first, so the caching allocator
    reuses the largest call's blocks; the smallest failing T is re-run and counted element
    by element.
    """
    words = out.view(torch.int64)
    differ = torch.zeros(len(counts), dtype=torch.bool, device=out.device)
    for i, t in enumerate(counts):
        differ[i] = torch.ne(op(q[:t], positions[:t], s, b).view(torch.int64), words[:t]).any()
    bad = [counts[i] for i in differ.nonzero().flatten().tolist()]
    if bad:
        t = bad[-1]
        _bitwise(
            op(q[:t], positions[:t], s, b),
            out[:t],
            f"the leading {t} rows of the {out.shape[0]}-row call ({len(bad)} step sizes differ)",
        )


def _capture(q, positions, s, b):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(q, positions, s, b)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(q, positions, s, b)
    return graph, out


def _log_in_bf16(positions: torch.Tensor, where: str) -> int:
    """The device's fp32 log(pos + 1), the op's own first steps, against the correctly rounded
    fp32 log, both rounded to bf16 (the step the chain keeps): bitwise there, and the number
    of positions where the two differ in fp32."""
    device_log = torch.log(positions.to(torch.float32) + 1)
    rounded_log = torch.log(positions.double() + 1).to(torch.float32)
    _bitwise(
        device_log.to(torch.bfloat16),
        rounded_log.to(torch.bfloat16),
        f"the correctly rounded log in bf16 at {where}",
    )
    return int((device_log != rounded_log).sum())


def _position_sweep(spec, seed: int, s: torch.Tensor, b: torch.Tensor) -> tuple[float, str]:
    """`position_sweep` (lo, hi): every position lo .. hi - 1, `tokens` rows per call (call k
    at positions lo + k * tokens onward), each call on a fresh q and gated as a cell is:
    bitwise to the reference, the closed form within the chain's bound, its inputs untouched.

    Beside the gates it holds the device's log of every position to the correctly rounded
    one in bf16 (`_log_in_bf16`): the two differ in fp32 at some positions, and above 32767
    no margin keeps them together in bf16 (the module docstring), so it is measured at
    each. Returns the worst closed-form ratio and a summary line.
    """
    lo, hi = spec["position_sweep"]
    rows = spec["tokens"]
    assert (hi - lo) % rows == 0 and hi <= spec["max_seq_len"], (lo, hi, rows)
    shape = (rows, spec["heads"], spec["head_dim"])
    worst, fp32_differ, zero = 0.0, 0, 0
    for k in range((hi - lo) // rows):
        start = lo + k * rows
        where = f"positions {start}..{start + rows - 1}"
        g = torch.Generator(device="cuda").manual_seed(seed + 1 + k)
        q = _queries(shape, spec["q"], spec["dtype"], g)
        positions = torch.arange(start, start + rows, dtype=spec["positions_dtype"], device="cuda")
        q_before, positions_before = q.clone(), positions.clone()
        out = op(q, positions, s, b)
        assert torch.equal(q, q_before) and torch.equal(positions, positions_before), (
            "the entry must not mutate its inputs"
        )
        _bitwise(out, op.reference(q, positions, s, b), f"the reference at {where}")
        worst = max(worst, _closed_form_ratio(out, q, positions, s, b))
        fp32_differ += _log_in_bf16(positions, where)
        zero += int((out == 0).all(-1).sum())
    return worst, (
        f"{(hi - lo) // rows} calls over positions {lo}..{hi - 1}; the device log differs from "
        f"the correctly rounded one in fp32 at {fp32_differ} positions, in bf16 at none; "
        f"{zero} (position, head) queries zeroed"
    )


def _chunk_sweep(spec, seed: int, s: torch.Tensor, b: torch.Tensor) -> tuple[float, str]:
    """`chunks`: a whole prompt of chunks * tokens positions as `chunks` calls of `tokens` rows,
    chunk k at positions k * tokens onward (the chunk after a cached prefix of k * tokens),
    each on a fresh q and gated as a cell is: bitwise to the reference, the closed form within
    the chain's bound, its inputs untouched.

    Beside the gates it holds the device's log of every position to the correctly rounded
    one in bf16 (`_log_in_bf16`): the two differ in fp32 at some positions, and above 32767
    no margin keeps them together in bf16 (the module docstring), so it is measured at
    each. Returns the worst closed-form ratio and a summary line.
    """
    rows = spec["tokens"]
    top = spec["chunks"] * rows
    assert top <= spec["max_seq_len"], (spec["chunks"], rows, spec["max_seq_len"])
    shape = (rows, spec["heads"], spec["head_dim"])
    worst, fp32_differ, zero = 0.0, 0, 0
    for k in range(spec["chunks"]):
        start = k * rows
        where = f"positions {start}..{start + rows - 1}"
        g = torch.Generator(device="cuda").manual_seed(seed + 1 + k)
        q = _queries(shape, spec["q"], spec["dtype"], g)
        positions = torch.arange(start, start + rows, dtype=spec["positions_dtype"], device="cuda")
        q_before, positions_before = q.clone(), positions.clone()
        out = op(q, positions, s, b)
        assert torch.equal(q, q_before) and torch.equal(positions, positions_before), (
            "the entry must not mutate its inputs"
        )
        _bitwise(out, op.reference(q, positions, s, b), f"the reference at {where}")
        worst = max(worst, _closed_form_ratio(out, q, positions, s, b))
        fp32_differ += _log_in_bf16(positions, where)
        zero += int((out == 0).sum())
    return worst, (
        f"{spec['chunks']} chunk calls over positions 0..{top - 1}; {zero} zero; the device log "
        f"differs from the correctly rounded one in fp32 at {fp32_differ} positions, in bf16 at "
        "none"
    )


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec, seed = cell.spec, _seed(cell.why)
    q, positions, s, b = _build(spec, seed)
    s_before, b_before = s.clone(), b.clone()
    if "position_sweep" in spec:
        ratio, swept = _position_sweep(spec, seed, s, b)
        for now, then in ((s, s_before), (b, b_before)):
            assert torch.equal(now, then), "the entry must not mutate its inputs"
        assert ratio <= 1.0, f"{ratio:.3f} of the chain's error bound from the closed form"
        print(f"\n  {cell.why}: bitwise; closed form at {ratio:.3f} of the bound; {swept}")
        return
    if "chunks" in spec:
        ratio, swept = _chunk_sweep(spec, seed, s, b)
        for now, then in ((s, s_before), (b, b_before)):
            assert torch.equal(now, then), "the entry must not mutate its inputs"
        assert ratio <= 1.0, f"{ratio:.3f} of the chain's error bound from the closed form"
        print(f"\n  {cell.why}: bitwise; closed form at {ratio:.3f} of the bound; {swept}")
        return
    if spec["graph"]:
        graph, out = _capture(q, positions, s, b)
        fresh_q, fresh_positions, _, _ = _build(spec, seed + 1)
        q.copy_(fresh_q)
        positions.copy_(fresh_positions)
        q_before, positions_before = q.clone(), positions.clone()
        graph.replay()
        torch.cuda.synchronize()
        _bitwise(out, op(q, positions, s, b), "the eager call on the replayed inputs")
    else:
        q_before, positions_before = q.clone(), positions.clone()
        out = op(q, positions, s, b)

    assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
    assert out.data_ptr() != q.data_ptr(), "the entry is out-of-place"
    for now, then in ((q, q_before), (positions, positions_before), (s, s_before), (b, b_before)):
        assert torch.equal(now, then), "the entry must not mutate its inputs"
    _bitwise(out, op.reference(q, positions, s, b))
    ratio = _closed_form_ratio(out, q, positions, s, b)
    assert ratio <= 1.0, f"{ratio:.3f} of the chain's error bound from the closed form"
    swept = ""
    if "sweep_rows" in spec:
        counts = _sweep_counts(spec)
        _prefix_sweep(out, q, positions, s, b, counts)
        swept = f"; {len(counts)} step sizes bitwise the leading rows"
    print(
        f"\n  {cell.why}: bitwise; closed form at {ratio:.3f} of the bound; "
        f"{int(torch.isinf(out).sum())} overflowed, {int((out == 0).sum())} zero{swept}"
    )


def test_the_sweep_sees_one_flipped_bit() -> None:
    """The step-size sweep's control: one flipped bit in the full call's last head, last lane,
    fails exactly the step sizes whose leading rows hold it."""
    q, positions, s, b = _build(
        _spec_of("mla", heads=MLA_PARAM_HEADS, tokens=4096, contexts=(4096,)), 71
    )
    out = op(q, positions, s, b)
    _prefix_sweep(out, q, positions, s, b, [4096, 3001, 3000, 7])
    out.view(torch.int16)[3000, -1, -1] ^= 1
    with pytest.raises(AssertionError, match=r"leading 3001 rows .*\(2 step sizes differ\)"):
        _prefix_sweep(out, q, positions, s, b, [4096, 3001, 3000, 7])


def test_the_log_check_sees_the_first_parting_position() -> None:
    """The log check's control, for the position sweep and dep4's chunk sweep alike: past every
    envelope, at position 1695942, the device's log and the correctly rounded one round to
    different bf16 (measured): the check fails there and passes at its neighbours."""
    positions = torch.tensor([1695941, 1695942, 1695943], dtype=torch.int32, device="cuda")
    _log_in_bf16(positions[[0, 2]], "the neighbours")
    with pytest.raises(AssertionError, match=r"1 of 1 elements differ .* first at \(0,\)"):
        _log_in_bf16(positions[1:2], "position 1695942")


#: What the 16-head checkpoint-slice cell's two crossing heads compute, measured on sm_103
#: over positions 0..1048575: (local head, first and last position of the run where the
#: chain's factor is exactly 0, the factor at position 1048575).
_CROSSINGS = ((4, 517233, 550591, -0.046875), (10, 108418, 115409, -0.1796875))


def test_checkpoint_heads_cross_zero() -> None:
    """The 16-head checkpoint-slice cell's crossings, as the op computes them: with q = 1 the
    op returns its bf16 scale. On MLA layer 0's rank-1 slice, heads 4 and 10 hold a factor of
    exactly 0 over one run of positions around their closed-form zero and a negative factor
    after it; no head's factor takes the opposite sign to the closed form; the other 14
    heads stay positive at every position below 2^20."""
    spec = next(
        c.spec for c in op.CELLS if "slice_values" in c.spec and c.spec["heads"] == MLA_HEADS
    )
    assert spec["slice_values"] is NVFP4_MLA_LAYER0_RANK1 and spec["rank"] == 1
    _, _, s, b = _build(dict(spec, tokens=1, contexts=(1,)), 81)
    positions = torch.arange(LONG_MAX_SEQ_LEN, dtype=torch.int32, device="cuda")
    ones = torch.ones(LONG_MAX_SEQ_LEN, spec["heads"], 1, dtype=spec["dtype"], device="cuda")
    scale = op(ones, positions, s, b)[:, :, 0]
    _bitwise(scale, op.reference(ones, positions, s, b)[:, :, 0])
    closed = torch.log(positions.double() + 1)[:, None] * s.double() + b.double()
    opposite = (torch.sign(scale.double()) != torch.sign(closed)) & (scale != 0)
    assert not bool(opposite.any()), int(opposite.sum())
    crossing = [h for h, *_ in _CROSSINGS]
    others = [h for h in range(spec["heads"]) if h not in crossing]
    assert bool((scale[:, others] > 0).all()), "only heads 4 and 10 leave the positive range"
    for h, first, last, at_end in _CROSSINGS:
        zero = (scale[:, h] == 0).nonzero().flatten()
        assert (int(zero[0]), int(zero[-1]), zero.numel()) == (first, last, last - first + 1)
        cross = math.exp(-float(b[h]) / float(s[h])) - 1
        assert first <= cross <= last, (h, cross)
        assert bool((scale[:first, h] > 0).all()) and bool((scale[last + 1 :, h] < 0).all())
        assert scale[-1, h].item() == at_end, (h, scale[-1, h].item())
        print(
            f"\n  local head {h} (s {float(s[h])}, b {float(b[h])}): factor 0 at positions "
            f"{first}..{last} around the closed-form zero {cross:.1f}, {at_end} at 1048575"
        )


#: What the tep8 checkpoint-slice cells' crossing heads compute, measured on sm_103 over
#: positions 0..1048575: rank -> (local head, first and last position of the run where the
#: chain's factor is exactly 0, the factor at position 1048575).
_TEP8_CROSSINGS = {2: (4, 517233, 550591, -0.046875), 3: (2, 108418, 115409, -0.1796875)}


@pytest.mark.parametrize("rank", sorted(_TEP8_CROSSINGS))
def test_tep8_checkpoint_heads_cross_zero(rank: int) -> None:
    """The tep8 checkpoint-slice cells' crossings, as the op computes them: with q = 1 the op
    returns its bf16 scale. On MLA layer 0's rank-2 slice (heads 16..23) local head 4 (head
    20), and on its rank-3 slice (heads 24..31) local head 2 (head 26), hold a factor of
    exactly 0 over one run of positions around their closed-form zero and a negative factor
    after it; no head's factor takes the opposite sign to the closed form; the slice's other
    7 heads stay positive at every position below 2^20."""
    spec = next(
        c.spec
        for c in op.CELLS
        if "slice_values" in c.spec and c.spec["heads"] == MLA_TEP8_HEADS and c.spec["rank"] == rank
    )
    values = {2: NVFP4_MLA_LAYER0_TEP8_RANK2, 3: NVFP4_MLA_LAYER0_TEP8_RANK3}[rank]
    assert spec["slice_values"] is values
    _, _, s, b = _build(dict(spec, tokens=1, contexts=(1,)), 81)
    positions = torch.arange(LONG_MAX_SEQ_LEN, dtype=torch.int32, device="cuda")
    ones = torch.ones(LONG_MAX_SEQ_LEN, spec["heads"], 1, dtype=spec["dtype"], device="cuda")
    scale = op(ones, positions, s, b)[:, :, 0]
    _bitwise(scale, op.reference(ones, positions, s, b)[:, :, 0])
    closed = torch.log(positions.double() + 1)[:, None] * s.double() + b.double()
    opposite = (torch.sign(scale.double()) != torch.sign(closed)) & (scale != 0)
    assert not bool(opposite.any()), int(opposite.sum())
    h, first, last, at_end = _TEP8_CROSSINGS[rank]
    others = [i for i in range(spec["heads"]) if i != h]
    assert bool((scale[:, others] > 0).all()), f"only local head {h} leaves the positive range"
    zero = (scale[:, h] == 0).nonzero().flatten()
    assert (int(zero[0]), int(zero[-1]), zero.numel()) == (first, last, last - first + 1)
    cross = math.exp(-float(b[h]) / float(s[h])) - 1
    assert first <= cross <= last, (h, cross)
    assert bool((scale[:first, h] > 0).all()) and bool((scale[last + 1 :, h] < 0).all())
    assert scale[-1, h].item() == at_end, (h, scale[-1, h].item())
    print(
        f"\n  rank {rank} local head {h} (head {rank * spec['heads'] + h}; s {float(s[h])}, "
        f"b {float(b[h])}): factor 0 at positions {first}..{last} around the closed-form zero "
        f"{cross:.1f}, {at_end} at 1048575"
    )


def test_tep8_gates_at_the_1m_last_chunk() -> None:
    """What each gate rejects at far positions on a tep8 rank's 8 heads: the MLA's 1M last
    chunk on rank 1, positions 1015808..1048575, where a bf16 step of log(pos + 1) is 0.0625.

    The rounding variants still fail the bitwise gate, each moving at least 0.5% of the
    elements, and stay inside the closed form's bound; s and b swapped and rank 2's heads on
    rank 1 fail the closed form. A position off by one, either way, moves the chain's log at
    one row of the chunk only, the one whose log(pos + 1) and log(pos + 2) round to different
    bf16 (position 1028640), so the bitwise gate sees it at that row and at no other: an
    off-by-one in the positions a caller passes at far positions is not this entry's to see.
    """
    spec = next(
        c.spec
        for c in op.CELLS
        if c.spec["site"] == "mla"
        and c.spec["heads"] == MLA_TEP8_HEADS
        and c.spec.get("cached") == (LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,)
        and c.spec["q"] == "normal"
    )
    assert spec["rank"] == 1
    q, positions, s, b = _build(spec, 37)
    _, _, s_other, b_other = _build(dict(spec, rank=2), 37)
    assert int(positions[0]) == 1015808 and int(positions[-1]) == 1048575
    ref = op.reference(q, positions, s, b)
    bf16 = q.dtype
    lp32 = torch.log(positions.to(torch.float32) + 1)
    lp = lp32.to(bf16)
    rounding = {
        "fp32 scale, no intermediate rounding": (
            q.float() * (lp32[:, None] * s.float() + b.float())[..., None]
        ).to(bf16),
        "log not rounded to bf16": q * ((lp32[:, None] * s.float()).to(bf16) + b)[..., None],
        "scale as one fused multiply-add": q
        * torch.addcmul(b.float()[None, :], lp.float()[:, None], s.float()[None, :]).to(bf16)[
            ..., None
        ],
    }
    gross = {
        "s and b swapped": op(q, positions, b, s),
        "rank 2's heads on rank 1": op(q, positions, s_other, b_other),
    }
    off_by_one = {
        "positions off by one": (op(q, positions + 1, s, b), 1028640),
        "log(pos) for log(pos + 1)": (op(q, positions - 1, s, b), 1028641),
    }
    numel = ref.numel()
    for name, wrong in {**rounding, **gross, **{k: v for k, (v, _) in off_by_one.items()}}.items():
        differ = wrong.view(torch.int16) != ref.view(torch.int16)
        count = int(differ.sum())
        ratio = _closed_form_ratio(wrong, q, positions, s, b)
        rows = positions[differ.flatten(1).any(1)].tolist()
        print(
            f"\n  {name:38s} differs in {count:>9d} of {numel} ({100.0 * count / numel:7.3f}%), "
            f"{len(rows)} rows; closed form at {ratio:.4g} of the bound"
        )
        with pytest.raises(AssertionError):
            _bitwise(wrong, ref)
        if name in rounding:
            assert count >= numel // 200, (name, count)
            assert ratio <= 1.0, (name, ratio)
        elif name in gross:
            assert ratio > 1.0, (name, ratio)
        else:
            assert rows == [off_by_one[name][1]], (name, rows)


def test_draw_ranges_hold_the_checkpoints_parameters() -> None:
    """Every checkpoint-shaped cell draws s and b over a range holding both checkpoints' own."""
    checked = 0
    for cell in op.CELLS:
        spec = cell.spec
        if spec["q"] != "normal":
            continue
        for i, key in enumerate(("s", "b")):
            lo, hi = spec[f"{key}_range"]
            want_lo, want_hi = _CHECKPOINT_RANGES[spec["site"]][key]
            assert lo <= want_lo and want_hi <= hi, (cell.why, key, (lo, hi))
            for value in spec.get("slice_values", ((), ()))[i]:
                assert want_lo <= value <= want_hi, (cell.why, key, value)
        checked += 1
    assert checked == len(op.CELLS) - 5, "every cell but the five stress cells is checkpoint-shaped"


@pytest.mark.parametrize(
    "heads, context",
    [
        (MLA_HEADS, FP8_MAX_SEQ_LEN),
        (MLA_HEADS, MAX_SEQ_LEN),
        (MLA_PARAM_HEADS, MAX_SEQ_LEN),
        (MLA_TEP8_HEADS, MAX_SEQ_LEN),
    ],
)
def test_gates_reject_wrong_variants(heads: int, context: int) -> None:
    """Every plausible wrong chain fails the bitwise gate; the gross ones fail the closed form.

    The rounding variants, which only the bitwise gate can see, move at least 0.5% of the
    elements at every width. A gross variant fails the closed form however few elements it
    moves: a position off by one moves fewer at 32768 rows than at 9216 (log(pos) and
    log(pos + 1) round to the same bf16 at most large positions), and the closed form still
    sees the rows where they do not. With 16 heads (tep4) or 8 (tep8, tp8) the wrong heads are
    another rank's slice; with all 64 on the rank (dep4) they are the parameter rotated by 16
    heads, so each head meets another's s and b.
    """
    if heads in (MLA_HEADS, MLA_TEP8_HEADS):
        spec = _spec_of("mla", context, heads, rank=2)
        q, positions, s, b = _build(spec, 31)
        _, _, s_other, b_other = _build(dict(spec, rank=3), 31)
        wrong_heads = "rank 3's heads on rank 2"
    else:
        q, positions, s, b = _build(_spec_of("mla", context, heads), 31)
        s_other, b_other = s.roll(16), b.roll(16)
        wrong_heads = "the 64 heads rotated by 16"
    ref = op.reference(q, positions, s, b)
    bf16 = q.dtype
    lp32 = torch.log(positions.to(torch.float32) + 1)
    lp = lp32.to(bf16)
    fp32_scale = (q.float() * (lp32[:, None] * s.float() + b.float())[..., None]).to(bf16)
    rounding = {
        "fp32 scale, no intermediate rounding": fp32_scale,
        "log not rounded to bf16": q * ((lp32[:, None] * s.float()).to(bf16) + b)[..., None],
        "scale as one fused multiply-add": q
        * torch.addcmul(b.float()[None, :], lp.float()[:, None], s.float()[None, :]).to(bf16)[
            ..., None
        ],
    }
    gross = {
        "log(pos) for log(pos + 1)": op(q, positions - 1, s, b),
        "positions off by one": op(q, positions + 1, s, b),
        "s and b swapped": op(q, positions, b, s),
        wrong_heads: op(q, positions, s_other, b_other),
    }
    numel = ref.numel()
    for name, wrong in {**rounding, **gross}.items():
        differ = int((wrong.view(torch.int16) != ref.view(torch.int16)).sum())
        ratio = _closed_form_ratio(wrong, q, positions, s, b)
        print(
            f"\n  {name:38s} differs in {differ:>9d} of {numel} ({100.0 * differ / numel:7.3f}%);"
            f" closed form at {ratio:.4g} of the bound"
        )
        with pytest.raises(AssertionError):
            _bitwise(wrong, ref)
        if name in gross:
            assert ratio > 1.0, (name, ratio)
        else:
            assert differ >= numel // 200, (name, differ)
            assert ratio <= 1.0, (name, ratio)
    chain = _ulps_from_closed_form(ref, q, positions, s, b)
    single = _ulps_from_closed_form(fp32_scale, q, positions, s, b)
    print(
        f"\n  bf16 ulps from the closed form: the specified chain {chain:.3f}, "
        f"an fp32 scale rounded once {single:.3f}"
    )
    assert single < chain


def test_gates_at_the_1m_last_chunk() -> None:
    """What each gate rejects at far positions, with all 64 heads (dep4): the MLA's 1M last
    chunk, positions 1015808..1048575, where a bf16 step of log(pos + 1) is 0.0625.

    The rounding variants still fail the bitwise gate, each moving at least 0.5% of the
    elements, and stay inside the closed form's bound; s and b swapped and the 64 heads
    rotated by 16 fail the closed form. A position off by one, either way, moves the chain's
    log at one row of the chunk only, the one whose log(pos + 1) and log(pos + 2) round to
    different bf16 (position 1028640, measured), so the bitwise gate sees it at that row and
    at no other: an off-by-one in the positions a caller passes at far positions is not this
    entry's to see.
    """
    spec = next(
        c.spec
        for c in op.CELLS
        if c.spec["site"] == "mla"
        and c.spec["heads"] == MLA_PARAM_HEADS
        and c.spec.get("cached") == (LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,)
        and c.spec["contexts"] == (MAX_NUM_TOKENS,)
        and c.spec["q"] == "normal"
    )
    q, positions, s, b = _build(spec, 37)
    assert int(positions[0]) == 1015808 and int(positions[-1]) == 1048575
    ref = op.reference(q, positions, s, b)
    bf16 = q.dtype
    lp32 = torch.log(positions.to(torch.float32) + 1)
    lp = lp32.to(bf16)
    rounding = {
        "fp32 scale, no intermediate rounding": (
            q.float() * (lp32[:, None] * s.float() + b.float())[..., None]
        ).to(bf16),
        "log not rounded to bf16": q * ((lp32[:, None] * s.float()).to(bf16) + b)[..., None],
        "scale as one fused multiply-add": q
        * torch.addcmul(b.float()[None, :], lp.float()[:, None], s.float()[None, :]).to(bf16)[
            ..., None
        ],
    }
    gross = {
        "s and b swapped": op(q, positions, b, s),
        "the 64 heads rotated by 16": op(q, positions, s.roll(16), b.roll(16)),
    }
    off_by_one = {
        "positions off by one": (op(q, positions + 1, s, b), 1028640),
        "log(pos) for log(pos + 1)": (op(q, positions - 1, s, b), 1028641),
    }
    numel = ref.numel()
    for name, wrong in {**rounding, **gross, **{k: v for k, (v, _) in off_by_one.items()}}.items():
        differ = wrong.view(torch.int16) != ref.view(torch.int16)
        count = int(differ.sum())
        ratio = _closed_form_ratio(wrong, q, positions, s, b)
        rows = positions[differ.flatten(1).any(1)].tolist()
        print(
            f"\n  {name:38s} differs in {count:>9d} of {numel} ({100.0 * count / numel:7.3f}%), "
            f"{len(rows)} rows; closed form at {ratio:.4g} of the bound"
        )
        with pytest.raises(AssertionError):
            _bitwise(wrong, ref)
        if name in rounding:
            assert count >= numel // 200, (name, count)
            assert ratio <= 1.0, (name, ratio)
        elif name in gross:
            assert ratio > 1.0, (name, ratio)
        else:
            assert rows == [off_by_one[name][1]], (name, rows)


def _guard_inputs():
    return _build(
        _spec_of("mla", FP8_MAX_SEQ_LEN, rank=0, tokens=16, contexts=(), decode_rows=16), 41
    )


def _rowwise_scale(q, positions, s, b) -> torch.Tensor:
    """The chain with row t paired with head t's terms: what [H, 1] s and b compute at T == H."""
    lp = torch.log(positions.to(torch.float32) + 1).to(q.dtype)
    return q * (lp * s + b)[:, None, None]


_SILENT = {
    "s and b of length 1": lambda q, p, s, b: (
        (q, p, s[:1], b[:1]),
        op(q, p, s[:1].expand(s.numel()), b[:1].expand(b.numel())),
    ),
    "s and b as [H, 1] at T == H": lambda q, p, s, b: (
        (q, p, s[:, None], b[:, None]),
        _rowwise_scale(q, p, s, b),
    ),
    "positions of length 1": lambda q, p, s, b: (
        (q, p[:1], s, b),
        op(q, p[:1].expand(p.numel()), s, b),
    ),
    "one row of q, positions of length 16": lambda q, p, s, b: (
        (q[:1], p, s, b),
        op(q[:1].expand(p.numel(), -1, -1), p, s, b),
    ),
    "2-D q [T, H * D] at T == H": lambda q, p, s, b: ((q.reshape(q.shape[0], -1), p, s, b), None),
}


@pytest.mark.parametrize("case", list(_SILENT), ids=list(_SILENT))
def test_guard_refuses_a_silent_broadcast(case) -> None:
    q, positions, s, b = _guard_inputs()
    assert q.shape[0] == q.shape[1] == 16, "T == H is what the pairing cases need"
    args, wrong = _SILENT[case](q, positions, s, b)
    # Unguarded, the op accepts the arguments and answers with a plausible wrong tensor.
    got = op(*args)
    right = op(q, positions, s, b)
    if wrong is not None:
        assert got.shape == right.shape
        _bitwise(got, wrong, "the broadcast it silently computes")
        assert not torch.equal(got, right)
    else:
        assert got.shape == (16, 16, 2048), got.shape
    with validating(op):
        with pytest.raises(ValueError, match="ssmax_scaling: .* must be"):
            op(*args)
        _bitwise(op(q, positions, s, b), right, "the unguarded call")


def test_unabsorbable_mismatches_raise_from_torch() -> None:
    """What the note quotes: an unsliced [64] s and b against one rank's 16 heads, a 16-head
    slice against all 64 (a tep4 slice at dep4), and a short positions, raise in torch."""
    q, positions, s, b = _guard_inputs()
    s64, b64 = torch.cat([s] * 4), torch.cat([b] * 4)
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(16\) must match the size of tensor b \(64\) "
        r"at non-singleton dimension 1",
    ):
        op(q, positions, s64, b64)
    q_all, positions_all, s_all, b_all = _build(
        _spec_of("mla", heads=MLA_PARAM_HEADS, tokens=16, contexts=(), decode_rows=16), 42
    )
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(64\) must match the size of tensor b \(16\) "
        r"at non-singleton dimension 1",
    ):
        op(q_all, positions_all, s_all[:16], b_all[:16])
    with pytest.raises(
        RuntimeError,
        match=r"The size of tensor a \(16\) must match the size of tensor b \(9\) "
        r"at non-singleton dimension 0",
    ):
        op(q, positions[:9], s, b)
    with validating(op):
        with pytest.raises(ValueError, match="s must be"):
            op(q, positions, s64, b64)
        with pytest.raises(ValueError, match="s must be"):
            op(q_all, positions_all, s_all[:16], b_all[:16])


def test_tep8_slice_mismatches() -> None:
    """At a tep8 or tp8 rank's 8 heads: s and b as [8, 1] at decode batch 8, a captured batch
    where T == H, pair row t with head t's terms silently, and `is_valid` refuses them; the
    unsliced [64] parameter and a tep4 rank's 16-head slice raise from torch, as the note
    quotes."""
    q, positions, s, b = _build(
        _spec_of("mla", heads=MLA_TEP8_HEADS, rank=3, tokens=8, contexts=(), decode_rows=8), 43
    )
    assert q.shape[0] == q.shape[1] == MLA_TEP8_HEADS
    right = op(q, positions, s, b)
    got = op(q, positions, s[:, None], b[:, None])
    assert got.shape == right.shape
    _bitwise(got, _rowwise_scale(q, positions, s, b), "the broadcast it silently computes")
    assert not torch.equal(got, right)
    s64, b64 = torch.cat([s] * 8), torch.cat([b] * 8)
    s16, b16 = torch.cat([s] * 2), torch.cat([b] * 2)
    for wide_s, wide_b in ((s64, b64), (s16, b16)):
        with pytest.raises(
            RuntimeError,
            match=rf"The size of tensor a \(8\) must match the size of tensor b \({wide_s.numel()}\) "
            r"at non-singleton dimension 1",
        ):
            op(q, positions, wide_s, wide_b)
    with validating(op):
        for wrong_s, wrong_b in ((s[:, None], b[:, None]), (s64, b64), (s16, b16)):
            with pytest.raises(ValueError, match="s must be"):
                op(q, positions, wrong_s, wrong_b)
        _bitwise(op(q, positions, s, b), right, "the unguarded call")


def test_rows_are_independent() -> None:
    """A padded decode capture runs dummy rows beside the live ones."""
    q, positions, s, b = _build(
        _spec_of("indexer", FP8_MAX_SEQ_LEN, tokens=16, contexts=(), decode_rows=16), 51
    )
    live = op(q[:3].clone(), positions[:3].clone(), s, b)
    q[3:] = math.nan
    q[3:, :, ::2] = math.inf
    positions[3], positions[4], positions[5:] = -1, -2, 2**30
    _bitwise(op(q, positions, s, b)[:3], live, "the live rows alone")

    # A negative position: log 0 = -inf at -1, nan at -2, on that row and no other.
    q, positions, s, b = _build(
        _spec_of("indexer", FP8_MAX_SEQ_LEN, tokens=3, contexts=(), decode_rows=3), 52
    )
    positions.copy_(torch.tensor([-1, -2, 5], dtype=positions.dtype))
    out = op(q, positions, s, b)
    assert not bool(torch.isfinite(out[0]).any())
    assert bool(torch.isnan(out[1]).all())
    assert bool(torch.isfinite(out[2]).all())


def test_dtypes_that_leave_the_result_unchanged() -> None:
    """int64 positions, fp32 s and b holding bf16 values, and a bf16 default dtype, at every
    position of the NVFP4 target's envelope (MAX_SEQ_LEN)."""
    q, positions, s, b = _build(_spec_of("mla", rank=1), 61)
    want = op(q, positions, s, b)
    _bitwise(op(q, positions.long(), s, b), want, "the int32-positions result")
    _bitwise(op(q, positions, s.float(), b.float()), want, "the bf16-s-and-b result")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        got = op(q, positions, s, b)
    finally:
        torch.set_default_dtype(previous)
    _bitwise(got, want, "the result under an fp32 default dtype")
