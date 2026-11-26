# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssmax_scaling entry.

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

from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.ssmax_scaling import (
    MAX_SEQ_LEN,
)
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.ssmax_scaling import (
    ssmax_scaling as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ssmax_scaling requires a CUDA device"

#: The extremes of s and b over both Mistral Large 4 checkpoints, every layer that carries them
#: (the NVFP4 checkpoint's read on disk as bf16, the FP8 checkpoint's rounded outward).
# _CHECKPOINT_RANGES unused
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
    print(
        f"\n  {cell.why}: bitwise; closed form at {ratio:.3f} of the bound; "
        f"{int(torch.isinf(out).sum())} overflowed, {int((out == 0).sum())} zero"
    )


def _guard_inputs():
    return _build(_spec_of("mla", MAX_SEQ_LEN, rank=0, tokens=16, contexts=(), decode_rows=16), 41)


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


def test_rows_are_independent() -> None:
    """A padded decode capture runs dummy rows beside the live ones."""
    q, positions, s, b = _build(
        _spec_of("indexer", MAX_SEQ_LEN, tokens=16, contexts=(), decode_rows=16), 51
    )
    live = op(q[:3].clone(), positions[:3].clone(), s, b)
    q[3:] = math.nan
    q[3:, :, ::2] = math.inf
    positions[3], positions[4], positions[5:] = -1, -2, 2**30
    _bitwise(op(q, positions, s, b)[:3], live, "the live rows alone")

    # A negative position: log 0 = -inf at -1, nan at -2, on that row and no other.
    q, positions, s, b = _build(
        _spec_of("indexer", MAX_SEQ_LEN, tokens=3, contexts=(), decode_rows=3), 52
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
