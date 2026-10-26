# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the csa_region_compress entry: the cells, the wrong variants
the band rejects, both guards and CUDA-graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.csa_region_compress import (  # noqa: E501
    E4M3_MAX,
    PROXY_DIM,
    REGION_SIZE,
    RegionSummaryExpectation,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.attention.csa_region_compress import (  # noqa: E501
    csa_region_compress as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "csa_region_compress requires a CUDA device"

DEV = "cuda"

#: Relative bound on an fp32 evaluation's error, in units of `magnitude`.
#: With u = 2^-24: each weight is exp of an exact bf16 difference (CUDA expf,
#: <= 2 ulp = 4u), the product with k adds u and the 8-term sum (n - 1) u = 7u,
#: so the numerator is within 12u of the exact sum of |w k|; the denominator
#: within 4u + 7u = 11u; the divide u. 24u in all; 2^-18 = 64u leaves 2.7x.
REL_BOUND = 2.0**-18
#: Absolute slack for subnormal weights (a channel more than 87 nats below the
#: region maximum): at most 8 * 2^-149 * |k| / 1e-20, below 2^-70 at |k| <= 448,
#: far under half the smallest e4m3 subnormal (2^-10).
ABS_BOUND = 2.0**-70

#: Checkpoint scale, measured on the indexer of sparse layers 3 and 7 over the
#: oracle's 6101-token gsm8k prompt: k (LayerNorm + RoPE) RMS 0.99..1.01, |k|
#: max 8.4; z RMS 1.3 (layer 7) .. 2.9 (layer 3), range -8.9..14.3; a channel's
#: best logit at most 15.3 nats below its region maximum.
K_RMS = 1.0
Z_NOISE = 2.0
Z_CHANNEL = 2.0


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """The rounding chain, written out with an explicit saturation: bf16, clamp
    to +-448, e4m3fn; returned as fp32 for ordering."""
    return x.to(torch.bfloat16).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _bracket(exp: RegionSummaryExpectation) -> tuple[torch.Tensor, torch.Tensor]:
    """The e4m3 codes an fp32 evaluation may land on: the rounding chain is
    monotone, so every value within the error bound of the exact summary rounds
    into [chain(s - eps), chain(s + eps)]."""
    eps = REL_BOUND * exp.magnitude + ABS_BOUND
    return _e4m3(exp.unrounded - eps), _e4m3(exp.unrounded + eps)


def _violations(out: torch.Tensor, exp: RegionSummaryExpectation) -> int:
    lo, hi = _bracket(exp)
    value = out.float()
    return int((~((value >= lo) & (value <= hi))).sum().item())


def _compare(out: torch.Tensor, exp: RegionSummaryExpectation, out_dtype: torch.dtype) -> None:
    """Every output code lies between the codes of the exact summary moved by
    the fp32 error bound (see `REL_BOUND`); where both ends agree -- all but a
    handful of elements -- that is bit equality with the exact answer. Outputs
    are e4m3 values, so this is tighter than any ulp band on the output dtype:
    a code one e4m3 step off is outside unless the exact value sits within
    `eps` of the step's rounding boundary. NaN is never inside."""
    assert out.dtype == out_dtype, (out.dtype, out_dtype)
    assert out.shape == exp.summary.shape, (out.shape, exp.summary.shape)
    assert out.is_contiguous()
    lo, hi = _bracket(exp)
    bad = _violations(out, exp)
    ambiguous = int((lo != hi).sum().item())
    assert bad == 0, f"{bad} of {out.numel()} outside the band ({ambiguous} ambiguous codes)"


def _tie_points(gen: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    """Midpoints between adjacent finite e4m3 values (normal binades 2^-6..2^8
    below the 448 ceiling, and the subnormal range), random sign. Each has at
    most 5 significant bits, so it and its bf16 neighbours are exact in bf16."""
    mids = [(2 * j + 1) * 2.0**-10 for j in range(8)]
    for e in range(-6, 9):
        for m in range(8):
            mid = (1.0 + (2 * m + 1) / 16.0) * 2.0**e
            if mid < E4M3_MAX:
                mids.append(mid)
    table = torch.tensor(mids, device=DEV)
    pick = torch.randint(0, len(mids), shape, generator=gen, device=DEV)
    sign = torch.where(torch.rand(shape, generator=gen, device=DEV) < 0.5, -1.0, 1.0)
    return table[pick] * sign


def _build(spec: dict, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    regions = spec["regions"]
    shape = (regions, REGION_SIZE, PROXY_DIM)
    dist = spec.get("dist", "ckpt")
    if dist == "ckpt":
        k = torch.randn(shape, generator=gen, device=DEV) * K_RMS
        z = (
            torch.randn(shape, generator=gen, device=DEV) * Z_NOISE
            + torch.randn((1, 1, PROXY_DIM), generator=gen, device=DEV) * Z_CHANNEL
        )
    elif dist == "spread":
        # Channel d's logits sit ~d * 120/255 nats below the region's maximum:
        # unaffected below ~46, shrunk by the 1e-20 floor up to ~104, zero beyond.
        k = torch.randn(shape, generator=gen, device=DEV) * K_RMS
        gaps = torch.linspace(0.0, 120.0, PROXY_DIM, device=DEV)
        z = torch.randn(shape, generator=gen, device=DEV) - gaps
    elif dist == "midpoint":
        # Constant logits per region (every weight exactly 1, the denominator
        # exactly 8). Seven tokens at a tie point t, one at t + ulp_bf16(t): the
        # fp32 mean t + ulp/8 is exact, bf16 rounds it onto t, and e4m3 then
        # breaks the tie to even -- where fp32 -> e4m3 directly rounds away.
        t = _tie_points(gen, (regions, 1, PROXY_DIM))
        ulp = torch.ldexp(torch.ones_like(t), torch.frexp(t.abs()).exponent - 8)
        k = t.expand(shape).clone()
        k[:, -1:, :] = t + torch.sign(t) * ulp
        z = torch.randn((regions, 1, 1), generator=gen, device=DEV).expand(shape)
    elif dist == "saturate":
        mag = torch.rand(shape, generator=gen, device=DEV) * 400.0 + 300.0
        flip = torch.rand((regions, 1, PROXY_DIM), generator=gen, device=DEV) < 0.5
        sign = torch.where(flip, -1.0, 1.0)
        k = mag * sign
        z = torch.randn(shape, generator=gen, device=DEV)
    else:
        raise AssertionError(f"unknown dist {dist!r}")
    return (
        k.to(torch.bfloat16).contiguous(),
        z.to(torch.bfloat16).contiguous(),
        spec.get("out_dtype", torch.float8_e4m3fn),
    )


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    k, z, out_dtype = _build(cell.spec, _seed(cell.why))
    k_before, z_before = k.clone(), z.clone()
    out = op(k, z, out_dtype)
    assert torch.equal(k, k_before) and torch.equal(z, z_before), "inputs were modified"
    _compare(out, op.reference(k, z, out_dtype), out_dtype)


def _cell(dist: str):
    return next(c for c in op.CELLS if c.spec.get("dist", "ckpt") == dist)


def test_cells_reach_their_regimes():
    """Each regime cell exercises what its `why` claims: the spread cell has
    floored and zeroed channels, the midpoint cell has codes where rounding
    fp32 straight to e4m3 differs, the saturate cell has codes at +-448."""
    k, z, _ = _build(_cell("spread").spec, _seed(_cell("spread").why))
    exp = op.reference(k, z)
    floored = (exp.unrounded.abs() < 0.01 * k.float().abs().mean(1)).sum().item()
    assert floored > 0 and (exp.summary.float() == 0).sum().item() > 0

    k, z, _ = _build(_cell("midpoint").spec, _seed(_cell("midpoint").why))
    exp = op.reference(k, z)
    direct = exp.unrounded.float().to(torch.float8_e4m3fn).float()
    assert (direct != exp.summary.float()).sum().item() > 0

    k, z, _ = _build(_cell("saturate").spec, _seed(_cell("saturate").why))
    exp = op.reference(k, z)
    assert (exp.unrounded.abs() > 464).sum().item() > 0
    assert (exp.summary.float().abs() == E4M3_MAX).sum().item() > 0


def test_wrong_variants_land_outside_the_band():
    """Plausible misreadings of the compressor, each run on the cell built to
    expose it; every one must leave the band somewhere."""

    def chain(s: torch.Tensor) -> torch.Tensor:
        return s.to(torch.bfloat16).to(torch.float8_e4m3fn)

    def weighted(k, w):
        return (w * k.float()).sum(1) / w.sum(1)

    found = {}
    k, z, _ = _build(_cell("midpoint").spec, _seed(_cell("midpoint").why))
    w = torch.exp(z.float() - z.float().amax(dim=(1, 2), keepdim=True))
    found["fp32 straight to e4m3"] = _violations(
        weighted(k, w).to(torch.float8_e4m3fn), op.reference(k, z)
    )

    k, z, _ = _build(_cell("spread").spec, _seed(_cell("spread").why))
    w = torch.exp(z.float() - z.float().amax(dim=1, keepdim=True))
    found["per-channel shift"] = _violations(chain(weighted(k, w)), op.reference(k, z))

    k, z, _ = _build({"regions": 128}, 7)
    exp = op.reference(k, z)
    found["softmax over channels"] = _violations(
        chain((torch.softmax(z.float(), dim=2) * k.float()).sum(1)), exp
    )
    found["mean ignoring z"] = _violations(chain(k.float().mean(1)), exp)
    found["softmax over tokens of k, weighting z"] = _violations(
        chain((torch.softmax(k.float(), dim=1) * z.float()).sum(1)), exp
    )

    inside = [name for name, bad in found.items() if bad == 0]
    assert not inside, f"variants inside the band: {inside}; counts {found}"


def test_guard_refuses_broadcast_logits():
    """A `z` of `[R, 8, 1]` (or `[1, 8, D]`) beside a full `k` is accepted and
    returns `[R, D]`, computed from logits that are not the region's."""
    k, z, _ = _build({"regions": 32}, 11)
    narrow = z[:, :, :1].contiguous()
    out = op(k, narrow)
    assert out.shape == (32, PROXY_DIM)
    assert _violations(out, op.reference(k, z)) > 0, "the broadcast answer should be wrong"
    for bad in (narrow, z[:1].contiguous()):
        with pytest.raises(AssertionError, match="broadcasts silently"):
            with validating(op):
                op(k, bad)


def test_guard_refuses_lossy_out_dtype():
    """float8_e5m2 is accepted and silently re-rounds the e4m3 summary."""
    k, z, _ = _build({"regions": 32}, 12)
    lossy = op(k, z, torch.float8_e5m2)
    exact = op(k, z, torch.float8_e4m3fn)
    assert (lossy.float() != exact.float()).sum().item() > 0
    with pytest.raises(AssertionError, match="does not hold every e4m3 value exactly"):
        with validating(op):
            op(k, z, torch.float8_e5m2)


def test_out_dtypes_carry_the_same_values():
    """bf16, fp16 and fp32 outputs hold exactly the fp8 output's values."""
    k, z, _ = _build({"regions": 64}, 13)
    fp8 = op(k, z, torch.float8_e4m3fn).float()
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert torch.equal(op(k, z, dtype).float(), fp8), dtype


def test_empty_call_returns_empty():
    """No region completed this step: `[0, 8, 256]` in, `[0, 256]` out."""
    empty = torch.empty((0, REGION_SIZE, PROXY_DIM), dtype=torch.bfloat16, device=DEV)
    out = op(empty, empty)
    assert out.shape == (0, PROXY_DIM) and out.dtype == torch.float8_e4m3fn


def test_cuda_graph_capture_and_replay():
    """Captured at the decode width (128 regions), replayed on new keys and
    logits; bit-equal to an eager call on the same inputs."""
    k, z, _ = _build({"regions": 128}, 21)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(k, z)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(k, z)

    k_new, z_new, _ = _build({"regions": 128}, 22)
    k.copy_(k_new)
    z.copy_(z_new)
    graph.replay()
    torch.cuda.synchronize()
    eager = op(k, z)
    assert torch.equal(out.view(torch.uint8), eager.view(torch.uint8))
    _compare(out, op.reference(k, z), torch.float8_e4m3fn)
