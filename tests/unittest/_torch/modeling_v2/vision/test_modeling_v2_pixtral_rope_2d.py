# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Mistral Large 4 pixtral_rope_2d entry.

The cells drive the vision tower's image grids (one patch, a small image, the
largest 110x110 image, two images per call up to the envelope's two max-size
images) at the tower's head geometry, in its packed-qkv layout. The rest of
the file pins what a cell cannot show: which rotation this is (the
discrimination test) and the inputs `is_valid` refuses because the op takes
them and answers wrongly.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.vision.pixtral_rope_2d import (  # noqa: E501
    pixtral_rope_2d as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "pixtral_rope_2d requires a CUDA device"

HEADS = 16
HEAD_DIM = 128
THETA = 10000.0

#: Explicit mantissa bits of each output dtype: one ulp at |v| in [2^e, 2^(e+1))
#: is 2^(e - bits).
_MANTISSA_BITS = {torch.bfloat16: 7, torch.float32: 23}

# The band, per element, against the fp64 closed form rounded once to the
# output dtype (`op.reference`). Writing y for the exact rotated value, y' for
# the op's fp32 value before its final rounding, and r for the magnitude of
# the input pair the element belongs to (a rotation preserves it):
#
#   |out - ref| <= |out - y'| + |y' - y| + |y - ref|
#              <= ulp(out) / 2 + r * delta + ulp(ref) / 2
#
# (rounding to nearest never leaves the binade downward, so the half ulp at the
# rounded value bounds each rounding). delta is the fp32 math's own error,
# relative to r:
#   - the frequency `1 / theta ** (2i / D)`: fp32 pow then fp32 reciprocal,
#     budgeted at 2^-21 relative (four ulp at the bottom of a binade; measured
#     7.2e-8 = 2^-23.7 on sm_103 against fp64);
#   - the angle `fp32(position) * f`: one more rounding, 2^-24 relative; the
#     angle error is the position times both, at most p_max * (2^-21 + 2^-24)
#     radians (measured 6.3e-6 rad over positions 0..109);
#   - cos and sin of the fp32 angle: 2 ulp each of a value <= 1, 2^-22 each;
#   - the complex multiply of the fp32 pair (exact from bf16) by the rotor: two
#     products and a sum, three roundings of magnitude <= r, 3 * 2^-24.
# The bound is position-dependent through p_max, the largest row or column of
# the cell. Measured against it: see the printed ratio per cell.
_FREQ_REL = 2.0**-21
_ANGLE_ROUND = 2.0**-24
_COS_SIN = 2 * 2.0**-22
_ROTATE = 3 * 2.0**-24


def _delta(p_max: int) -> float:
    return p_max * (_FREQ_REL + _ANGLE_ROUND) + _COS_SIN + _ROTATE


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _grids(grid) -> list[tuple[int, int]]:
    """A cell's `grid`: one image's (rows, cols), or a tuple of them."""
    if isinstance(grid[0], int):
        return [tuple(grid)]
    return [tuple(g) for g in grid]


def _positions(grids: list[tuple[int, int]]) -> torch.Tensor:
    """int64 [N, 2] (row, col), each image row-major from (0, 0), concatenated."""
    per_image = [
        torch.stack(
            torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
        ).reshape(-1, 2)
        for h, w in grids
    ]
    return torch.cat(per_image).to("cuda")


def _qk(n: int, heads: int, head_dim: int, dtype: torch.dtype, layout: str, seed: int):
    """q and k as the tower holds them: head-reshaped column chunks of one
    [N, 3 * heads * head_dim] projection ("packed_qkv"), or two contiguous
    tensors ("separate")."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "packed_qkv":
        qkv = (1.5 * torch.randn(n, 3 * heads * head_dim, generator=g, device="cuda")).to(dtype)
        q, k, _ = qkv.chunk(3, dim=-1)
        return q.reshape(n, heads, head_dim), k.reshape(n, heads, head_dim)
    assert layout == "separate", layout
    q = (1.5 * torch.randn(n, heads, head_dim, generator=g, device="cuda")).to(dtype)
    k = (1.5 * torch.randn(n, heads, head_dim, generator=g, device="cuda")).to(dtype)
    return q, k


def _build(spec, seed: int):
    positions = _positions(_grids(spec["grid"]))
    q, k = _qk(
        positions.shape[0], spec["heads"], spec["head_dim"], spec["dtype"], spec["layout"], seed
    )
    return q, k, positions


def _ulp(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp64 spacing of `dtype` at each value of `v` (0 at 0)."""
    _, exponent = torch.frexp(v.double())
    spacing = torch.ldexp(
        torch.ones_like(v, dtype=torch.float64), exponent - 1 - _MANTISSA_BITS[dtype]
    )
    return torch.where(v == 0, torch.zeros_like(spacing), spacing)


def _pair_magnitude(x: torch.Tensor) -> torch.Tensor:
    """|(x[2j], x[2j+1])| in fp64, repeated onto both elements of the pair."""
    pairs = x.double().reshape(*x.shape[:-1], -1, 2)
    return pairs.norm(dim=-1).repeat_interleave(2, dim=-1)


def _bound(x: torch.Tensor, out: torch.Tensor, ref: torch.Tensor, p_max: int) -> torch.Tensor:
    return (
        0.5 * _ulp(out, out.dtype) + 0.5 * _ulp(ref, out.dtype) + _pair_magnitude(x) * _delta(p_max)
    )


def _ratio(x, out, candidate, p_max) -> float:
    """Largest |out - candidate| over the band, both sides in out's dtype."""
    bound = _bound(x, out, candidate, p_max)
    err = (out.double() - candidate.double()).abs()
    assert bool((err[bound == 0] == 0).all()), "a zero pair must come out zero"
    return (err / bound.clamp_min(torch.finfo(torch.float64).tiny)).max().item()


def _budget_share(x, out, ref, p_max) -> float:
    """Largest error beyond the two final roundings, over r * delta."""
    rounding = 0.5 * _ulp(out, out.dtype) + 0.5 * _ulp(ref, out.dtype)
    beyond = ((out.double() - ref.double()).abs() - rounding).clamp_min(0.0)
    budget = _pair_magnitude(x) * _delta(p_max)
    return (beyond / budget.clamp_min(torch.finfo(torch.float64).tiny)).max().item()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    q, k, positions = _build(spec, seed=_seed(cell.why))
    before = (q.clone(), k.clone(), positions.clone())

    q_out, k_out = op(q, k, positions, spec["theta"])

    for x, out in ((q, q_out), (k, k_out)):
        assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
        assert out.is_contiguous()
        assert out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    for was, now in zip(before, (q, k, positions)):
        assert torch.equal(was, now), "the entry must not mutate its inputs"

    q_ref, k_ref = op.reference(q, k, positions, spec["theta"])
    p_max = int(positions.max().item())
    for name, x, out, ref in (("q", q, q_out, q_ref), ("k", k, k_out, k_ref)):
        ratio = _ratio(x, out, ref, p_max)
        differ = int((out != ref).sum().item())
        # A ratio just under 1 at bf16 is a one-ulp rounding split (y' and y on
        # either side of a rounding boundary: |out - ref| is one ulp, the band
        # one ulp plus r * delta). What the fp32 math itself spends is the
        # error left after the two roundings, as a share of r * delta.
        spent = _budget_share(x, out, ref, p_max)
        print(
            f"\n  {cell.why} {name}: max |out-ref| / band {ratio:.3f}, fp32 budget spent "
            f"{spent:.3f}, {differ} of {out.numel()} elements not bitwise the closed form"
        )
        assert ratio <= 1.0, (name, ratio)

    # The patch at (0, 0) of every image has angle 0: the rotor is exactly 1.
    origin = (positions == 0).all(dim=-1)
    assert torch.equal(q_out[origin], q[origin]) and torch.equal(k_out[origin], k[origin])


def _closed_form(x, positions, theta, *, pairing="gptj", angles=None):
    """A 2-D rotation in fp64 (not rounded) with the pairing and the angle table
    as parameters, for the variants the discrimination test needs."""
    pos = positions.double()
    angle = angles(pos[:, 0:1], pos[:, 1:2], theta).unsqueeze(-2)
    cos, sin = torch.cos(angle), torch.sin(angle)
    x64 = x.double()
    half = x.shape[-1] // 2
    if pairing == "gptj":
        even, odd = x64[..., 0::2], x64[..., 1::2]
        out = torch.empty_like(x64)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out
    lo, hi = x64[..., :half], x64[..., half:]  # NeoX half-split: (i, i + D/2)
    return torch.cat((lo * cos - hi * sin, lo * sin + hi * cos), dim=-1)


def _freq(index, theta):
    return theta ** (-2.0 * index / HEAD_DIM)


def test_identity_is_the_pixtral_2d_rotation() -> None:
    """Which rotation the entry computes, at a non-square 40x60 image: every
    plausible wrong variant lands far outside the band, at bf16 (the cells'
    dtype) and at fp32 I/O, where the band is the fp32 math's own budget and
    the distance is not capped by the output's rounding; the closed form and
    the reference implementation's host-built table land inside it."""
    positions = _positions([(40, 60)])
    q, _ = _qk(positions.shape[0], HEADS, HEAD_DIM, torch.bfloat16, "packed_qkv", seed=41)
    q32 = q.float()
    out, _ = op(q, q, positions, THETA)
    out32, _ = op(q32, q32, positions, THETA)
    p_max = int(positions.max().item())

    def distances(candidate64):
        """(bf16 ratio, fraction of bf16 elements outside, fp32 ratio)."""
        cand = candidate64.to(torch.bfloat16)
        bound = _bound(q, out, cand, p_max)
        outside = ((out.double() - cand.double()).abs() > bound).double().mean().item()
        return (
            _ratio(q, out, cand, p_max),
            outside,
            _ratio(q32, out32, candidate64.to(torch.float32), p_max),
        )

    dev = q.device
    j = torch.arange(HEAD_DIM // 4, dtype=torch.float64, device=dev)
    even, odd = _freq(2 * j, THETA), _freq(2 * j + 1, THETA)
    lo, hi = _freq(j, THETA), _freq(j + HEAD_DIM // 4, THETA)
    every = _freq(torch.arange(HEAD_DIM // 2, dtype=torch.float64, device=dev), THETA)
    pixtral = dict(angles=lambda r, c, t: torch.cat((r * even, c * odd), -1))

    r16, outside, r32 = distances(_closed_form(q, positions, THETA, **pixtral))
    print(f"\n  closed form (correct): bf16 {r16:.3f}x, fp32 {r32:.3f}x band")
    assert r16 <= 1.0 and r32 <= 1.0 and outside == 0.0

    wrong = {
        "half-split (NeoX) pairing": dict(pixtral, pairing="neox"),
        "row and column swapped": dict(angles=lambda r, c, t: torch.cat((c * even, r * odd), -1)),
        "frequency halves, not interleaved": dict(
            angles=lambda r, c, t: torch.cat((r * lo, c * hi), -1)
        ),
        "the same frequencies on both axes": dict(
            angles=lambda r, c, t: torch.cat((r * even, c * even), -1)
        ),
        "axes alternating pair by pair": dict(
            angles=lambda r, c, t: torch.stack((r * even, c * odd), -1).flatten(-2)
        ),
        "rotation reversed": dict(angles=lambda r, c, t: -torch.cat((r * even, c * odd), -1)),
        "1-D raster index, standard rope": dict(angles=lambda r, c, t: (r * 60 + c) * every),
        "theta 1e6": dict(
            angles=lambda r, c, t: torch.cat((r * _freq(2 * j, 1e6), c * _freq(2 * j + 1, 1e6)), -1)
        ),
        "positions off by one": dict(
            angles=lambda r, c, t: torch.cat(((r + 1) * even, (c + 1) * odd), -1)
        ),
    }
    for name, variant in wrong.items():
        r16, outside, r32 = distances(_closed_form(q, positions, THETA, **variant))
        print(
            f"  {name:36s} bf16 {r16:8.1f}x band ({outside:6.1%} of elements outside), "
            f"fp32 {r32:10.1f}x band"
        )
        assert r16 > 50.0 and r32 > 1000.0, (name, r16, r32)

    # fp32 math, not the input dtype's: the same rotation with a bf16 rotor
    # (cos/sin rounded to bf16, as an implementation in the input dtype has it)
    # leaves the fp32 band. Its error is up to half a bf16 ulp of cos/sin,
    # 2^-9, against a band of about 3.2e-5 r at p_max 59 (about 60x), so the
    # margin asked of it is 10x rather than the structural variants' 1000x.
    angle = torch.cat((positions[:, 0:1].double() * even, positions[:, 1:2].double() * odd), -1)
    rotor = torch.complex(angle.cos().bfloat16().float(), angle.sin().bfloat16().float())
    bf16_rotor = torch.view_as_real(
        torch.view_as_complex(q32.reshape(*q32.shape[:-1], -1, 2)) * rotor.unsqueeze(-2)
    ).flatten(-2)
    times = _ratio(q32, out32, bf16_rotor, p_max)
    print(f"  {'bf16 rotor':36s} fp32 {times:10.1f}x band")
    assert times > 10.0, times

    # A correct alternative: the reference implementation's recipe, which builds
    # the frequency vector and the cos/sin table in fp32 on the host and moves
    # it to the device. Inside the band; not what the entry does (it builds
    # them on the device: a few fp32 ulp apart in the table).
    host_f = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    pos_h = positions.cpu()
    host_angle = torch.cat(
        (pos_h[:, 0:1].float() * host_f[0::2], pos_h[:, 1:2].float() * host_f[1::2]), -1
    )
    host_rotor = torch.polar(torch.ones_like(host_angle), host_angle).to(dev).unsqueeze(-2)
    for name, x, mine in (("bf16", q, out), ("fp32", q32, out32)):
        alt = (
            torch.view_as_real(
                torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * host_rotor
            )
            .flatten(-2)
            .to(x.dtype)
        )
        ratio = _ratio(x, mine, alt, p_max)
        differ = int((alt != mine).sum().item())
        print(
            f"  {'host-built table (correct) ' + name:36s} {ratio:8.3f}x band, "
            f"{differ} of {mine.numel()} elements differ from the entry"
        )
        assert ratio <= 1.0, (name, ratio)


# Guards: each input below is first shown, unguarded, to come back without an
# error and wrong; then `is_valid` refuses it. Every operand is allocated at the
# size the call reads, so the unguarded calls touch only memory the test owns.


def _small():
    positions = _positions([(2, 4)])
    q, k = _qk(positions.shape[0], HEADS, HEAD_DIM, torch.bfloat16, "separate", seed=44)
    return q, k, positions


def test_guard_positions_must_cover_every_token() -> None:
    q, k, positions = _small()
    q_out, _ = op(q, k, positions[:1], THETA)
    assert q_out.shape == q.shape, "accepted: one row of positions broadcasts"
    first_patch_everywhere, _ = op(q, k, positions[:1].expand_as(positions), THETA)
    assert torch.equal(q_out, first_patch_everywhere)
    assert not torch.equal(q_out, op(q, k, positions, THETA)[0])
    with validating(op), pytest.raises(AssertionError, match="positions must be"):
        op(q, k, positions[:1], THETA)


def test_guard_k_must_carry_the_tokens_of_q() -> None:
    q, k, positions = _small()
    _, k_out = op(q, k[:1], positions, THETA)
    assert k_out.shape == q.shape and k_out.shape != k[:1].shape, (
        "accepted: a one-token k broadcasts to every token's angles"
    )
    with validating(op), pytest.raises(AssertionError, match="same tokens"):
        op(q, k[:1], positions, THETA)


def test_guard_q_and_k_must_be_three_dimensional() -> None:
    q, k, positions = _small()
    flat_q, flat_k = q[:, 0], k[:, 0]
    q_out, _ = op(flat_q, flat_k, positions, THETA)
    n = positions.shape[0]
    assert q_out.shape == (n, n, HEAD_DIM), "accepted: [N, D] broadcasts to [N, N, D]"
    with validating(op), pytest.raises(AssertionError, match=r"\[tokens, heads, head_dim\]"):
        op(flat_q, flat_k, positions, THETA)
