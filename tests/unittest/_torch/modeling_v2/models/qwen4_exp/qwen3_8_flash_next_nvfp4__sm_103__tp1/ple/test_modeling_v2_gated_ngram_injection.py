# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_ngram_injection entry.

Every mis-reading of this formula keeps the shapes and stays bounded by
`|value|`, so beyond the cells one test measures how far outside the entry's
band each wrong reading lands, and two pin the floor and the zero-dot gate.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.gated_ngram_injection import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.gated_ngram_injection import (  # noqa: E501
    gated_ngram_injection as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gated_ngram_injection requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16-bit: `rtol = eps`, `atol` one subnormal spacing (`gate_ratio <= 1`)
    -- half torch's bf16 `rtol`. Nothing cancels (each element is a sigmoid
    times one `value` element), so the budget is the final rounding; with
    the reference also rounded once, op and reference agree or sit one ulp
    apart. One ulp is at most `eps * |ref|`, so this band always admits it
    and never two. Measured at 2048 tokens: 0.992 (a one-ulp element just
    above a binade boundary), eight wrong readings at 105x..1115x. The
    `atol` is for Gaussian `value` elements that land subnormal (measured
    on fp16).

    fp32: torch's default band. The 2560-term signed dot is accumulated at
    the width it is reported in, so its conditioning, not the cast, sets
    the error; a custom pair would be fitted to a result.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    if ref.dtype == torch.float32:
        torch.testing.assert_close(out, ref)
        return
    ratio = gate_ratio(out, ref)
    assert ratio <= 1.0, f"{ratio:.4g}x the band"


DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
WIDTH = HC_COUNT * HIDDEN
RMS_NORM_EPS = 1e-6
CLAMP_MIN = 1e-6

# This checkpoint's PLE tensors: norm_key / norm_query gammas (element RMS
# 0.1358, extremes 0.863 and -0.326), and the value's scale derived as
# sqrt(2560) * 0.015295 (value_proj RMS) * 0.00747 (dequantized table RMS).
# The entry is exactly linear in `value`, so its absolute scale decides nothing.
CKPT_NORM_GAMMA_RMS = 0.1358
CKPT_NORM_GAMMA_MAX = 0.863281
CKPT_NORM_GAMMA_MIN = -0.326172
CKPT_VALUE_RMS = 0.0058


def _gemma_grouped_rmsnorm(x, gamma, hc_count):
    """Where key/query come from -- here only for their distribution."""
    hidden = x.shape[-1] // hc_count
    xf = x.float().reshape(*x.shape[:-1], hc_count, hidden)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS)
    return (xf.flatten(-2) * (1.0 + gamma.float())).to(x.dtype)


def _make_operands(seed, dtype, leading, hc_count=HC_COUNT, value_dtype=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = hc_count * HIDDEN

    def normal(*dims, scale, dt=dtype):
        return (torch.randn(*dims, generator=g, device=DEV) * scale).to(dt)

    gammas = []
    for _ in range(2):
        gamma = normal(width, scale=CKPT_NORM_GAMMA_RMS)
        gamma[0] = CKPT_NORM_GAMMA_MAX
        gamma[width // 2] = CKPT_NORM_GAMMA_MIN
        gammas.append(gamma)
    key = _gemma_grouped_rmsnorm(normal(*leading, width, scale=1.0), gammas[0], hc_count)
    query = _gemma_grouped_rmsnorm(normal(*leading, width, scale=1.0), gammas[1], hc_count)
    value = normal(*leading, HIDDEN, scale=CKPT_VALUE_RMS, dt=value_dtype or dtype)
    return key, query, value


def _single_channel_operands(dots, value_dtype):
    """One nonzero channel per stream: each dot is exactly `dots[s]`."""
    hc_count = len(dots)
    key = torch.zeros(8, hc_count * HIDDEN, dtype=torch.bfloat16, device=DEV)
    query = torch.zeros_like(key)
    for stream, dot in enumerate(dots):
        key[:, stream * HIDDEN] = 1.0
        query[:, stream * HIDDEN] = dot
    value = torch.full((8, HIDDEN), 0.5, dtype=value_dtype, device=DEV)
    return key, query, value


def _build(spec, seed):
    if spec.get("kind") == "single_channel":
        dots = spec["dots"]
        key, query, value = _single_channel_operands(dots, spec["value_dtype"])
        return key, query, value, len(dots), spec["clamp_min"]
    hc_count = spec.get("hc_count", HC_COUNT)
    dtype = spec.get("dtype", torch.bfloat16)
    key, query, value = _make_operands(
        seed, dtype, spec["leading"], hc_count, spec.get("value_dtype")
    )
    return key, query, value, hc_count, spec.get("clamp_min", CLAMP_MIN)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = [a.clone() for a in args[:3]]
    out = op(*args)
    key, _, value, hc_count, _ = args
    assert out.shape == (*key.shape[:-1], hc_count * value.shape[-1])
    assert out.dtype == value.dtype
    for a, b in zip(args[:3], before):
        assert torch.equal(a, b), "the operands are read-only"
    _compare(out, op.reference(*args))


def test_flattened_call_matches_batched() -> None:
    key, query, value = _make_operands(11, torch.bfloat16, (3, 17))
    batched = op(key, query, value, HC_COUNT)
    flat = op(key.reshape(-1, WIDTH), query.reshape(-1, WIDTH), value.reshape(-1, HIDDEN), HC_COUNT)
    assert torch.equal(flat, batched.reshape(-1, WIDTH))


def test_output_is_bounded_by_the_value_it_gates() -> None:
    """sigmoid is in (0, 1): no stream amplifies its value, to 4096x the operands."""
    key, query, value = _make_operands(40, torch.bfloat16, (256,))
    for scale in (1.0, 8.0, 64.0, 512.0, 4096.0):
        got = op(
            (key.float() * scale).to(torch.bfloat16),
            (query.float() * scale).to(torch.bfloat16),
            value,
            HC_COUNT,
        )
        assert torch.isfinite(got.float()).all(), scale
        bound = value.float().abs().max().item() * (1.0 + torch.finfo(torch.bfloat16).eps)
        assert got.float().abs().max().item() <= bound, scale


def test_clamp_floor_moves_fp32_and_not_bf16() -> None:
    """Sub-floor dots move by > 1e-4 relative in fp32; bf16 is bitwise unmoved."""
    dots = (0.0, 5e-8, -5e-8, 0.05)
    key, query, value32 = _single_channel_operands(dots, torch.float32)
    floored = op(key, query, value32, len(dots), CLAMP_MIN)
    unfloored = op(key, query, value32, len(dots), 0.0)
    apart = (floored - unfloored).abs().unflatten(-1, (len(dots), HIDDEN))
    relative = (apart / floored.abs().unflatten(-1, (len(dots), HIDDEN))).amax(dim=(0, 2))
    print(f"\n  fp32 relative movement per stream: {relative.tolist()}")
    assert relative[0].item() == 0.0 and relative[3].item() == 0.0
    assert relative[1].item() > 1e-4 and relative[2].item() > 1e-4

    _, _, value16 = _single_channel_operands(dots, torch.bfloat16)
    assert torch.equal(
        op(key, query, value16, len(dots), CLAMP_MIN), op(key, query, value16, len(dots), 0.0)
    )


def test_exact_zero_dot_gives_exactly_half_the_value() -> None:
    key, query, value = _single_channel_operands((0.0,) * HC_COUNT, torch.float32)
    assert torch.equal(op(key, query, value, HC_COUNT), value.repeat(1, HC_COUNT) * 0.5)


def _signed_sqrt(x):
    return x.abs().clamp_min(CLAMP_MIN).sqrt() * x.sign()


def test_wrong_variants_land_outside_the_band() -> None:
    """Each reading stays shaped and bounded; each must land > 5x outside."""
    key, query, value = _make_operands(50, torch.bfloat16, (2048,))
    want = op.reference(key, query, value, HC_COUNT)
    correct = gate_ratio(op(key, query, value, HC_COUNT), want)
    print(f"\n  correct {correct:.4g}x the band")
    assert correct <= 1.0

    keys = key.double().unflatten(-1, (HC_COUNT, HIDDEN))
    queries = query.double().unflatten(-1, (HC_COUNT, HIDDEN))
    values = value.double().unsqueeze(-2)
    dot = (keys * queries).sum(dim=-1, keepdim=True)
    scaled = dot / math.sqrt(HIDDEN)

    def interleaved():
        k = key.double().unflatten(-1, (HIDDEN, HC_COUNT)).transpose(-1, -2)
        q = query.double().unflatten(-1, (HIDDEN, HC_COUNT)).transpose(-1, -2)
        g = (k * q).sum(dim=-1, keepdim=True) / math.sqrt(HIDDEN)
        return torch.sigmoid(_signed_sqrt(g)) * values

    variants = (
        ("no square root", lambda: torch.sigmoid(scaled) * values),
        (
            "unsigned square root",
            lambda: torch.sigmoid(scaled.abs().clamp_min(CLAMP_MIN).sqrt()) * values,
        ),
        ("no 1/sqrt(D) scale", lambda: torch.sigmoid(_signed_sqrt(dot)) * values),
        ("divided by D", lambda: torch.sigmoid(_signed_sqrt(dot / HIDDEN)) * values),
        (
            "scale after the root",
            lambda: torch.sigmoid(_signed_sqrt(dot) / math.sqrt(HIDDEN)) * values,
        ),
        ("sigmoid before the root", lambda: _signed_sqrt(torch.sigmoid(scaled)) * values),
        ("tanh instead of sigmoid", lambda: torch.tanh(_signed_sqrt(scaled)) * values),
        ("streams read interleaved", interleaved),
    )
    for name, build in variants:
        wrong = build().flatten(-2).to(torch.bfloat16)
        ratio = gate_ratio(wrong, want)
        print(f"  {name:<26} {ratio:10.4g}x the band")
        assert ratio > 5.0, f"the band does not separate {name!r} ({ratio:.4g}x)"
        with pytest.raises(AssertionError):
            _compare(wrong, want)


def test_fp32_chain_differs_from_the_bf16_chain() -> None:
    """The literal bf16 chain falls outside the band; the entry does not."""
    key, query, value = _make_operands(90, torch.bfloat16, (2048,))
    want = op.reference(key, query, value, HC_COUNT)
    got = op(key, query, value, HC_COUNT)
    native = (key.unflatten(-1, (HC_COUNT, HIDDEN)) * query.unflatten(-1, (HC_COUNT, HIDDEN))).sum(
        dim=-1, keepdim=True
    ) / math.sqrt(HIDDEN)
    native = (torch.sigmoid(_signed_sqrt(native)) * value.unsqueeze(-2)).flatten(-2)
    differing = (got != native).float().mean().item()
    print(
        f"\n  fp32 chain {gate_ratio(got, want):.4g}x, bf16 chain {gate_ratio(native, want):.4g}x"
    )
    print(f"  output elements whose bits differ: {differing:.4%}")
    assert gate_ratio(got, want) <= 1.0
    assert gate_ratio(native, want) > 1.0
    assert differing > 0.0


def test_guard_refuses_unequal_leading_dims() -> None:
    """Unguarded, one query (or value) row is broadcast over every key row."""
    key, query, value = _make_operands(60, torch.bfloat16, (64,))
    unguarded = op(key, query[:1], value, HC_COUNT)
    assert unguarded.shape == (64, WIDTH), "the hazard this guard covers is gone"
    assert not torch.equal(unguarded, op.reference(key, query, value, HC_COUNT))
    assert op(key, query, value[:1], HC_COUNT).shape == (64, WIDTH)

    with pytest.raises(AssertionError, match="broadcasts silently"):
        with validating(op):
            op(key, query[:1], value, HC_COUNT)
    with pytest.raises(AssertionError, match="broadcasts silently"):
        with validating(op):
            op(key, query, value[:1], HC_COUNT)


def test_malformed_widths_raise_rather_than_return() -> None:
    """Why the width is not guarded: both mistakes are loud."""
    key, query, value = _make_operands(70, torch.bfloat16, (32,))
    with pytest.raises(RuntimeError):
        op(key[:, : WIDTH - 8], query[:, : WIDTH - 8], value, HC_COUNT)
    with pytest.raises(RuntimeError):
        op(key, query, value, 3)


def test_cuda_graph_capture_and_replay() -> None:
    key, query, value = _make_operands(100, torch.bfloat16, (8,))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(key, query, value, HC_COUNT, CLAMP_MIN)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(key, query, value, HC_COUNT, CLAMP_MIN)

    fresh = _make_operands(101, torch.bfloat16, (8,))
    for static, new in zip((key, query, value), fresh):
        static.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(*fresh, HC_COUNT, CLAMP_MIN))
