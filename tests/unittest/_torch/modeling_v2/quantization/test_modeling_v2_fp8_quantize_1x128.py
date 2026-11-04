# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128 catalog entry.

The entry owns the contract. This file turns a cell into an input and holds
what a cell cannot carry: that every code disagreement is a rounding tie, the
`use_ue8m0` bit-exactness, the wrong-recipe discrimination, the alignment
guard, and the edge behaviour the entry's `note` records.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (  # noqa: E501
    AMAX_FLOOR,
    E4M3_MAX,
    SCALE_ULP_GATE,
    block_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128 as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp8_quantize_1x128 requires a CUDA device"

#: The least fraction of codes `_compare` wants bit-exact, in a whole call
#: and, through `_sweep`, in every leading-row prefix of one.
EXACT_FRACTION = 0.995


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude ordinal of an e4m3 byte: adjacent codes differ by 1."""
    b = codes.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b & 0x7F)


def _compare(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scales within 4 fp32 ulp; codes at most one e4m3 step apart, >=99.5% bit-exact.

    This build compiles the kernel's two fp32 divisions approximately
    (`--use_fast_math`), each within CUDA's documented 2 ulp, so neither
    output is reproducible to the last bit with `use_ue8m0=False`. Measured
    at Mistral Large 4's H 6144, T 9216, over randn, 1e-3-scaled, heavy-tailed
    and 150x-outlier-channel inputs: scales within 2 ulp (79-83% bit-exact),
    codes 99.90-99.94% bit-exact, the worst leading-row prefix 0.123% off,
    every miss a tie within 1.192e-07 of its midpoint.

    Wrong recipes at T=256, H=6144 (bytes agreeing / worst relative scale
    error): power-of-two 0.044 / 0.99, per-tensor 0.0001 / 2.2, per-row
    0.039 / 1.6, blocks down the token axis 0.19 / 1.07, blocks offset by 64
    0.48 / 0.92, e4m3 max as 240 0 / 0.87, signed max 0.58 / 0.58; the
    correct one 0.9995 / 1.4e-07.
    """
    data, scales = out
    ref_data, ref_scales = ref
    assert data.dtype == ref_data.dtype == torch.float8_e4m3fn, (data.dtype, ref_data.dtype)
    assert scales.dtype == ref_scales.dtype == torch.float32, (scales.dtype, ref_scales.dtype)
    assert data.shape == ref_data.shape, (data.shape, ref_data.shape)
    assert scales.shape == ref_scales.shape, (scales.shape, ref_scales.shape)
    if scales.numel():
        ulp = (
            scales.contiguous().view(torch.int32).long()
            - ref_scales.contiguous().view(torch.int32).long()
        )
        worst = int(ulp.abs().max())
        assert worst <= SCALE_ULP_GATE, f"scale {worst} fp32 ulp off > {SCALE_ULP_GATE}"
    if data.numel():
        step = (_ordinal(data) - _ordinal(ref_data)).abs()
        assert int(step.max()) <= 1, f"a code is {int(step.max())} e4m3 steps off"
        exact = 1.0 - (step != 0).float().mean().item()
        assert exact >= EXACT_FRACTION, f"only {100 * exact:.4f}% of codes bit-exact"


DEV = "cuda"
# e4m3: 3 mantissa bits, so a normal value carries at most a 2^-4 relative
# half-step; the smallest subnormal 2^-9 has a 2^-10 half-step, in scale units.
E4M3_REL_HALF_STEP = 2.0**-4
E4M3_SUBNORMAL_HALF_STEP = 2.0**-10
# A code can only disagree when the exact product sits within the first
# division's 2 ulp of an e4m3 rounding midpoint.
TIE_REL_GATE = 2.0 * 2.0**-23
#: An LLM hidden state's outlier channels: a dozen columns at 150x the rest,
#: so those columns set their blocks' amax and the others land in e4m3's
#: smallest codes.
OUTLIER_CHANNELS = 12
OUTLIER_GAIN = 150.0


def _expand(v: torch.Tensor, n: int) -> torch.Tensor:
    """`[T, nb]` block values -> `[T, n]` per-column values."""
    return v.repeat_interleave(128, dim=1)[:, :n].contiguous()


def _input(
    tokens: int, hidden: int, seed: int, magnitude: float = 3.0, dist: str = "randn"
) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(tokens, hidden, device=DEV, generator=g) * magnitude
    if dist == "outlier":
        cols = torch.randperm(hidden, device=DEV, generator=g)[:OUTLIER_CHANNELS]
        x[:, cols] *= OUTLIER_GAIN
    else:
        assert dist == "randn", dist
    return x.to(torch.bfloat16)


def _amax(x: torch.Tensor) -> torch.Tensor:
    """Exact per-block amax `[T, H/128]` fp32, floored as the kernel floors it."""
    t, n = x.shape
    return x.abs().reshape(t, n // 128, 128).amax(dim=2).float().clamp(min=AMAX_FLOOR)


def _ulp_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.contiguous().view(torch.int32).long() - b.contiguous().view(torch.int32).long()


def _assert_misses_are_ties(x: torch.Tensor, data: torch.Tensor, ref_data: torch.Tensor) -> float:
    """Every code disagreement is a rounding tie: the exact product sits within
    the scale's 2-ulp band of the midpoint between the two codes."""
    qs, _ = block_scales(x, False)
    prod = x.float() * _expand(qs, x.shape[1])
    diff = data.float() != ref_data.float()  # -0 and +0 are the same code here
    if not bool(diff.any()):
        return 0.0
    mid = ((data.float() + ref_data.float()) / 2).double()
    rel = ((prod.double()[diff] - mid[diff]).abs() / mid[diff].abs()).max().item()
    assert rel <= TIE_REL_GATE, f"a miss {rel:.3e} from its midpoint is not a tie"
    return rel


def _replays(
    x: torch.Tensor, use_ue8m0: bool, seed: int, replays: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture one call on `x`'s buffer and replay it on new data written into it.

    Each replay must be bitwise an eager call on the same data. Returns the
    graph's outputs for the last replay, whose input `x` then holds.
    """
    t, n = x.shape
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, use_ue8m0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_data, g_scales = op(x, use_ue8m0)
    for r in range(replays):
        x.copy_(_input(t, n, seed=seed + 1 + r, dist="outlier" if r % 2 else "randn"))
        graph.replay()
        e_data, e_scales = op(x, use_ue8m0)
        torch.cuda.synchronize()
        assert torch.equal(g_data.view(torch.uint8), e_data.view(torch.uint8)), (
            f"T={t}: graph replay {r} codes differ from eager"
        )
        assert torch.equal(g_scales.view(torch.int32), e_scales.view(torch.int32)), (
            f"T={t}: graph replay {r} scales differ from eager"
        )
    return g_data, g_scales


def _sweep(
    x: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
    ref_data: torch.Tensor,
    use_ue8m0: bool,
) -> None:
    """Every T 1..len(x) as its own call on the leading T rows of `x`.

    Each call's codes and scales must be bitwise the leading rows / columns of
    the full call `(data, scales)`, which `_compare` gated against the
    reference. Every prefix then has the full call's misses in its rows, so it
    must also keep `EXACT_FRACTION` on its own: a few rows of misses at the top
    would otherwise hide inside the whole call's average. Mismatches are
    counted on the device, so the sweep synchronizes once.
    """
    t_max, n = x.shape
    nb = (n + 127) // 128
    flags = torch.zeros(t_max + 1, dtype=torch.int64, device=DEV)
    for t in range(1, t_max + 1):
        d, s = op(x[:t], use_ue8m0)
        assert d.shape == (t, n) and d.stride() == (n, 1), (t, d.shape, d.stride())
        assert s.shape == (nb, t) and s.is_contiguous(), (t, s.shape, s.stride())
        assert d.untyped_storage().nbytes() == (t + 3) // 4 * 4 * n, t
        flags[t] = (d.view(torch.uint8) != data[:t].view(torch.uint8)).sum() + (
            s.view(torch.int32) != scales[:, :t].view(torch.int32)
        ).sum()
    bad = flags.nonzero().flatten().tolist()
    assert not bad, f"{len(bad)} row counts differ from the {t_max}-row call; first T={bad[0]}"
    misses = (_ordinal(data) != _ordinal(ref_data)).sum(dim=1).double().cumsum(0)
    rows = torch.arange(1, t_max + 1, device=DEV, dtype=torch.float64)
    worst = (misses / (rows * n)).max().item()
    assert worst <= 1.0 - EXACT_FRACTION, (
        f"a leading-row prefix has {100 * worst:.4f}% of its codes a step off"
    )


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    x = _input(spec["tokens"], spec["hidden"], seed=seed, dist=spec.get("dist", "randn"))
    if spec.get("captured"):
        data, scales = _replays(x, spec["use_ue8m0"], seed)
    else:
        data, scales = op(x, spec["use_ue8m0"])
    nb = (spec["hidden"] + 127) // 128
    assert data.shape == x.shape and data.is_contiguous() and data.device == x.device
    assert scales.shape == (nb, spec["tokens"]) and scales.is_contiguous()
    ref = op.reference(x, spec["use_ue8m0"])
    _compare((data, scales), ref)
    rel = _assert_misses_are_ties(x, data, ref[0]) if not spec["use_ue8m0"] else 0.0
    if spec.get("sweep"):
        _sweep(x, data, scales, ref[0], spec["use_ue8m0"])
    exact = (data.view(torch.uint8) == ref[0].view(torch.uint8)).float().mean().item()
    print(f"  {cell.why[:44]}: {100 * exact:.4f}% bit-exact, worst tie {rel:.3e}")


# ── the flag, the scale and the direction ─────────────────────────────────


@pytest.mark.parametrize("tokens,hidden", [(64, 512), (48, 6144)])
def test_use_ue8m0_is_bit_exact_and_a_different_quantization(tokens, hidden):
    """Once the dequant scale is a power of two its reciprocal is exact, so the
    approximate division cannot reach the result."""
    x = _input(tokens, hidden, seed=5 + tokens)
    data, sf = op(x, True)
    ref_data, ref_sf = op.reference(x, True)
    assert torch.equal(sf.view(torch.int32), ref_sf.view(torch.int32))
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert bool((torch.frexp(sf)[0] == 0.5).all())
    d0, sf0 = op(x, False)
    sf_changed = (sf != sf0).float().mean().item()
    d_changed = (data.view(torch.uint8) != d0.view(torch.uint8)).float().mean().item()
    print(f"  ue8m0 T={tokens} H={hidden}: scales changed {sf_changed:.5f}, codes {d_changed:.5f}")
    assert sf_changed > 0.9 and d_changed > 0.5
    assert bool((sf >= sf0 * (1 - 2**-20)).all()) and bool((sf <= sf0 * 2 * (1 + 2**-20)).all())


def test_scale_is_amax_over_448_either_spelling():
    """Within 4 ulp of both `amax / 448` and the kernel's `1 / (448 / amax)`."""
    x = _input(1024, 2560, seed=21)
    _, sf = op(x, False)
    amax = _amax(x)
    direct = (amax / E4M3_MAX).t()
    two_step = (1.0 / (E4M3_MAX / amax)).t()
    u_direct = _ulp_f32(sf, direct).abs().max().item()
    u_two_step = _ulp_f32(sf, two_step).abs().max().item()
    exact = int((sf.view(torch.int32) == two_step.contiguous().view(torch.int32)).sum())
    print(
        f"  max|ulp| vs amax/448 {u_direct}, vs 1/(448/amax) {u_two_step}; exact {exact}/{sf.numel()}"
    )
    assert u_direct <= SCALE_ULP_GATE and u_two_step <= SCALE_ULP_GATE
    # the amax itself is recoverable from the scale to within those ulps
    rec = sf.t() * E4M3_MAX
    assert bool(((rec - amax).abs() <= amax * (SCALE_ULP_GATE + 1) * 2**-23).all())


def test_scale_multiplies_the_code():
    """Round trip bounded by e4m3's own constants; the reciprocal reading is
    four orders of magnitude off."""
    x = _input(64, 2560, seed=2, magnitude=2.0)
    data, sf = op(x, False)
    cols = _expand(sf.t().contiguous(), 2560)
    deq = data.float() * cols
    bound = (E4M3_REL_HALF_STEP + (SCALE_ULP_GATE + 1) * 2**-23) * x.float().abs()
    bound = bound + E4M3_SUBNORMAL_HALF_STEP * cols
    assert bool(((deq - x.float()).abs() <= bound).all())

    def rel_rms(a):
        return ((a - x.float()).pow(2).mean().sqrt() / x.float().pow(2).mean().sqrt()).item()

    mul, div = rel_rms(deq), rel_rms(data.float() / cols)
    print(f"  multiply relRMS={mul:.5f}  divide relRMS={div:.5g}")
    assert mul < 0.05 and div > 100.0
    assert data.float().abs().max().item() == E4M3_MAX


def test_discrimination_against_wrong_recipes():
    """Each wrong reading of the amax map, scored against the same call."""
    t, n = 256, 6144
    x = _input(t, n, seed=99)
    data, sf = op(x, False)
    amax = _amax(x)

    def score(a_cols, amax_max=E4M3_MAX):
        qs_cols = amax_max / a_cols
        d = (x.float() * qs_cols).to(torch.float8_e4m3fn)
        agree = (d.view(torch.uint8) == data.view(torch.uint8)).float().mean().item()
        rel = (((1.0 / qs_cols)[:, ::128].t() - sf).abs() / sf).max().item()
        return agree, rel

    mant, exp = torch.frexp(amax / E4M3_MAX)
    pow2 = torch.ldexp(torch.ones_like(amax), torch.where(mant == 0.5, exp - 1, exp))
    variants = {
        "correct": (_expand(amax, n), E4M3_MAX),
        "ue8m0 power-of-two scale": (_expand(pow2 * E4M3_MAX, n), E4M3_MAX),
        "one scale per tensor": (_expand(amax.max().expand_as(amax), n), E4M3_MAX),
        "one scale per token row": (
            _expand(amax.amax(1, keepdim=True).expand_as(amax), n),
            E4M3_MAX,
        ),
        "blocks down the token axis": (
            _amax(x.t().contiguous()).t().repeat_interleave(128, dim=0)[:t].contiguous(),
            E4M3_MAX,
        ),
        "misaligned block boundaries": (
            torch.roll(_expand(_amax(torch.roll(x, -64, dims=1)), n), 64, dims=1).contiguous(),
            E4M3_MAX,
        ),
        "e4m3 max taken as 240": (_expand(amax, n), 240.0),
        "signed max instead of amax": (
            _expand(x.float().reshape(t, n // 128, 128).amax(dim=2).clamp(min=AMAX_FLOOR), n),
            E4M3_MAX,
        ),
    }
    for name, (a_cols, mx) in variants.items():
        agree, rel = score(a_cols, mx)
        print(f"  H={n} {name:34s} {agree:.5f}  {rel:.4e}")
        if name == "correct":
            assert agree >= EXACT_FRACTION and rel <= (SCALE_ULP_GATE + 1) * 2**-23
        else:
            assert agree < 0.9 and rel > 1e-3, name


def test_transposed_scale_buffer_is_undetectable_at_the_square_shape():
    tokens, hidden = 48, 6144
    x = _input(tokens, hidden, seed=4, magnitude=1.0)
    _, sf = op(x, False)
    tr = sf.t().contiguous()
    assert sf.shape == tr.shape == (tokens, tokens)
    assert tr.dtype == sf.dtype and tr.is_contiguous()
    spread = ((tr - sf).abs() / sf).max().item()
    print(f"  T={tokens} H={hidden}: worst relative difference to the transpose {spread:.4f}")
    assert spread > 0.1


def test_width_not_a_multiple_of_128_takes_its_own_tail_amax():
    x = torch.zeros(1, 144, dtype=torch.bfloat16, device=DEV)
    x[0, 0], x[0, 130] = 4.0, 0.25
    _, sf = op(x, False)
    assert abs(sf[0, 0].item() - 4.0 / E4M3_MAX) <= 4.0 / E4M3_MAX * 2**-20
    assert abs(sf[1, 0].item() - 0.25 / E4M3_MAX) <= 0.25 / E4M3_MAX * 2**-20


# ── the guard ─────────────────────────────────────────────────────────────


def test_a_base_off_4_byte_alignment_is_refused():
    """The kernel loads bf16 pairs, so the input's base must be 4-byte aligned.

    Measured on sm_103 in a child process: a contiguous `[16, 6144]` view one
    element (2 bytes) into its storage faults with `torch.AcceleratorError:
    CUDA error: misaligned address`, and the process's CUDA context is unusable
    afterwards. That is not driven here, since CI runs every catalog test in
    one process. Views 2 and 8 elements in are admitted and compute exactly
    what an aligned copy does.
    """
    flat = _input(1, 16 * 6144 + 64, seed=11).flatten()
    with validating(op):
        for off in (2, 8):
            x = flat[off : off + 16 * 6144].view(16, 6144)
            assert x.is_contiguous() and x.data_ptr() % 4 == 0
            d, s = op(x, False)
            d0, s0 = op(x.clone(), False)
            assert torch.equal(d.view(torch.uint8), d0.view(torch.uint8)), off
            assert torch.equal(s.view(torch.int32), s0.view(torch.int32)), off
        x = flat[1 : 1 + 16 * 6144].view(16, 6144)
        assert x.is_contiguous() and x.data_ptr() % 4 == 2
        with pytest.raises(ValueError, match="4-byte alignment"):
            op(x, False)


# ── edges ─────────────────────────────────────────────────────────────────


def test_zero_block_takes_the_amax_floor():
    x = torch.zeros(4, 256, dtype=torch.bfloat16, device=DEV)
    x[0, 0] = 1.0
    data, sf = op(x, False)
    floor = AMAX_FLOOR / E4M3_MAX
    assert abs(sf[0, 0].item() - 1.0 / E4M3_MAX) <= 1.0 / E4M3_MAX * 2**-20
    for b in range(2):
        for t in range(4):
            if (b, t) != (0, 0):
                assert abs(sf[b, t].item() - floor) <= floor * 2**-20, (b, t, sf[b, t].item())
    assert bool((data[1:].float() == 0).all()) and bool((data[0, 1:].float() == 0).all())


def test_nonfinite_input():
    """inf poisons its whole 128-block; NaN is confined to its own element."""
    x = torch.zeros(1, 128, dtype=torch.bfloat16, device=DEV)
    x[0, 0], x[0, 1] = float("inf"), 1.0
    data, sf = op(x, False)
    assert sf.flatten().tolist() == [float("inf")]
    assert torch.isnan(data[0, 0].float()) and data[0, 1].float().item() == 0.0

    x = torch.zeros(1, 128, dtype=torch.bfloat16, device=DEV)
    x[0, 0], x[0, 1] = float("nan"), 1.0
    data, sf = op(x, False)
    assert abs(sf[0, 0].item() - 1.0 / E4M3_MAX) <= 1.0 / E4M3_MAX * 2**-20
    assert torch.isnan(data[0, 0].float()) and data[0, 1].float().item() == E4M3_MAX
    assert bool((data[0, 2:].float() == 0).all())


def test_saturation_never_exceeds_the_block_max():
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device=DEV)
    x[0, 0], x[0, 1], x[0, 3], x[0, 128] = 1.0e4, 1.0, -1.0e4, 2.0**-12
    data, sf = op(x, False)
    assert data[0, 0].float().item() == E4M3_MAX and data[0, 3].float().item() == -E4M3_MAX
    assert data[0, 128].float().item() == E4M3_MAX
    assert bool(torch.isfinite(data.float()).all())
    # bf16 rounds 1e4 to 9984, and that is the magnitude the scale carries
    assert abs(sf[0, 0].item() - 9984.0 / E4M3_MAX) <= 9984.0 / E4M3_MAX * 2**-20


def test_storage_padding_and_density():
    """`data` views a 4-row-padded allocation; both outputs are dense."""
    for t in (1, 2, 3, 5, 20, 64):
        data, sf = op(_input(t, 2560, seed=t), False)
        assert data.untyped_storage().size() == (t + 3) // 4 * 4 * 2560, t
        assert data.is_contiguous() and data.stride() == (2560, 1)
        assert sf.is_contiguous() and sf.untyped_storage().size() >= sf.numel() * 4
        assert sf.stride() == (t, 1) or sf.shape[0] == 1
    _, sf = op(_input(1, 128, seed=0), False)
    assert sf.shape == (1, 1) and sf.is_contiguous()


def test_empty_input():
    data, sf = op(torch.empty(0, 2560, dtype=torch.bfloat16, device=DEV), False)
    assert data.shape == (0, 2560) and data.dtype == torch.float8_e4m3fn
    assert sf.shape == (20, 0) and sf.dtype == torch.float32


def test_registered_fake_matches_eager_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    def meta(*tensors):
        return tuple((tuple(a.shape), a.dtype, a.stride()) for a in tensors)

    cells = [(1, 6144), (48, 6144), (9216, 6144)]
    eager = {(t, n): meta(*op(_input(t, n, seed=0))) for t, n in cells}
    with FakeTensorMode():
        for t, n in cells:
            got = meta(*op(torch.empty(t, n, dtype=torch.bfloat16, device=DEV)))
            assert got == eager[(t, n)], (t, n, got, eager[(t, n)])


def test_cuda_graph_capture_and_replay():
    """A replay reproduces a fresh eager call bitwise; so does a side stream."""
    x = _input(64, 2560, seed=17, magnitude=1.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_data, g_sf = op(x)
    x.copy_(_input(64, 2560, seed=18, magnitude=1.0))
    e_data, e_sf = op(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_data.view(torch.uint8), e_data.view(torch.uint8))
    assert torch.equal(g_sf.view(torch.int32), e_sf.view(torch.int32))
    alt = torch.cuda.Stream()
    with torch.cuda.stream(alt):
        a_data, a_sf = op(x)
    alt.synchronize()
    assert torch.equal(a_data.view(torch.uint8), e_data.view(torch.uint8))
    assert torch.equal(a_sf.view(torch.int32), e_sf.view(torch.int32))


def test_input_untouched_and_deterministic():
    x = _input(128, 2560, seed=3)
    before = x.clone()
    d1, s1 = op(x, False)
    d2, s2 = op(x, False)
    assert torch.equal(x, before)
    assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8))
    assert torch.equal(s1.view(torch.int32), s2.view(torch.int32))


def test_rejects_unsupported():
    """Every precondition but the base alignment is the op's own and loud."""
    bf = dict(dtype=torch.bfloat16, device=DEV)
    cases = {
        "float16 input": lambda: op(torch.randn(4, 128, dtype=torch.float16, device=DEV)),
        "float32 input": lambda: op(torch.randn(4, 128, device=DEV)),
        "float8 input": lambda: op(torch.randn(4, 128, **bf).to(torch.float8_e4m3fn)),
        "cpu input": lambda: op(torch.randn(4, 128, dtype=torch.bfloat16)),
        "1-D input": lambda: op(torch.randn(128, **bf)),
        "3-D input": lambda: op(torch.randn(2, 4, 128, **bf)),
        "width 120": lambda: op(torch.randn(4, 120, **bf)),
        "width 2568": lambda: op(torch.randn(4, 2568, **bf)),
        "transposed": lambda: op(torch.randn(128, 8, **bf).t()),
        "column slice": lambda: op(torch.randn(8, 5120, **bf)[:, :2560]),
    }
    for name, fn in cases.items():
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn()
