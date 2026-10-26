# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the step5 clamped_swiglu entry.

The entry's identity is *where* the clamp sits. The engine's own clamped SwiGLU
(`trtllm::silu_and_mul` with `swiglu_limit`) clamps the gate before SiLU, and
in bf16 that lands at most one ulp from the right answer. So the cell gate is
bit-level (each element within one ulp, and almost every element exact), and
the identity test shows that gate, and fp32, refusing the pre-SiLU clamp.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

# Registers torch.ops.trtllm.silu_and_mul, the rejected pre-SiLU clamp.
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.activation.clamped_swiglu import (  # noqa: E501
    LIMIT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.activation.clamped_swiglu import (  # noqa: E501
    clamped_swiglu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "clamped_swiglu requires a CUDA device"

#: The fp32 evaluation (expf-based SiLU, one product) carries a few fp32 ulps
#: of relative error, about 2^-22. It rounds to a different bf16 value than the
#: exact result only when the exact result lies that close to a bf16 rounding
#: midpoint: a probability of about 2 * 2^-22 / 2^-8 = 1.2e-4 per element,
#: and the miss is then exactly one bf16 ulp. The gate allows 8x that rate.
MAX_ULPS = 1
MAX_OFF_FRACTION = 1e-3

#: Bulk spread of the synthetic activand: N(0, 3^2) puts about 2% of each half
#: beyond +-7 on its own; the planted band below adds the boundary.
BULK_STD = 3.0
BAND = (6.9, 9.0)
PLANT_EVERY = 8


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _activand(rows: int, inter: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """`[rows, 2 * inter]`, gate half first, with one element in `PLANT_EVERY`
    of each half in the clamp boundary band: gate in [6.9, 9] (across
    silu^-1(7) = 7.0064), up in +-[6.9, 9] (across +-7).

    The two halves are planted at different columns, so a planted gate meets a
    bulk up and the clamped SiLU is multiplied by varied values: planted at the
    same column, both clamps saturate together and every product is 49, which
    the pre-SiLU clamp's 6.99362 * 7 also rounds to in bf16.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = BULK_STD * torch.randn(rows, 2 * inter, generator=g, device="cuda")
    band = torch.empty(rows, 2 * inter, device="cuda").uniform_(*BAND, generator=g)
    sign = torch.randint(0, 2, (rows, inter), generator=g, device="cuda") * 2.0 - 1.0
    band[:, inter:] *= sign
    planted = torch.zeros(rows, 2 * inter, dtype=torch.bool, device="cuda")
    planted[:, :inter][:, ::PLANT_EVERY] = True
    planted[:, inter:][:, PLANT_EVERY // 2 :: PLANT_EVERY] = True
    return torch.where(planted, band, x).to(dtype)


def _ordered(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns as integers ordered like the values they encode."""
    bits = t.contiguous().view(torch.int16).int()
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _ulp_distance(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    assert out.dtype == ref.dtype == torch.bfloat16
    return (_ordered(out) - _ordered(ref)).abs()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    """Each element within `MAX_ULPS` bf16 ulp of the fp64 reference, and at most
    `MAX_OFF_FRACTION` of them off at all (derivation at `MAX_ULPS`).
    """
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert not torch.isnan(out).any()
    dist = _ulp_distance(out, ref)
    worst = dist.max().item()
    off = (dist > 0).float().mean().item()
    assert worst <= MAX_ULPS, f"{worst} bf16 ulps from the reference"
    assert off <= MAX_OFF_FRACTION, f"{off:.3e} of elements off by one ulp"
    return worst, off


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    gate_up = _activand(spec["rows"], spec["inter"], spec["dtype"], _seed(cell.why))
    before = gate_up.clone()
    out = op(gate_up, spec["limit"])
    assert out.shape == (spec["rows"], spec["inter"])
    assert out.dtype == gate_up.dtype and out.device == gate_up.device
    assert torch.equal(before, gate_up), "the entry must not mutate its input"
    worst, off = _compare(out, op.reference(gate_up, spec["limit"]))
    print(f"\n  {cell.why}: max {worst} ulp, off-by-ulp fraction {off:.3e}")


def test_cuda_graph_capture_and_replay(certifying):  # noqa: F811
    """The decode site is captured: warm up on a side stream, capture at the
    largest captured batch of the widest site, refill the input in place,
    replay, and check the replayed output against the new input.
    """
    certifying(op)
    rows, inter = 128, 13824
    gate_up = _activand(rows, inter, torch.bfloat16, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(gate_up, LIMIT)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(gate_up, LIMIT)
    fresh = _activand(rows, inter, torch.bfloat16, 2)
    gate_up.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(fresh, LIMIT))


def _pre_silu_clamp64(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """What `trtllm::silu_and_mul(swiglu_limit=limit)` computes, in fp64."""
    width = gate_up.shape[-1] // 2
    g = gate_up[..., :width].double().clamp(max=limit)
    u = gate_up[..., width:].double().clamp(-limit, limit)
    return g * torch.sigmoid(g) * u


def _variants64(gate_up: torch.Tensor, limit: float) -> dict[str, torch.Tensor]:
    width = gate_up.shape[-1] // 2
    g = gate_up[..., :width].double()
    u = gate_up[..., width:].double()
    silu_g = g * torch.sigmoid(g)
    silu_u = u * torch.sigmoid(u)
    return {
        "pre-SiLU gate clamp (trtllm silu_and_mul)": _pre_silu_clamp64(gate_up, limit),
        "halves swapped (up first)": silu_u.clamp(max=limit) * g.clamp(-limit, limit),
        "no gate clamp": silu_g * u.clamp(-limit, limit),
        "no up clamp": silu_g.clamp(max=limit) * u,
        "no clamp (plain SwiGLU)": silu_g * u,
    }


def test_identity_is_the_post_silu_clamp():
    """In fp32, where every variant is visible, and in bf16 through the actual
    engine op the contract rejects, so the bf16 gate is shown to refuse it.
    """
    gate_up32 = _activand(512, 1536, torch.float32, 21)
    out = op(gate_up32, LIMIT).double()
    ref = op.reference(gate_up32.double(), LIMIT)
    correct = (out - ref).abs().max().item()
    print(f"\n  fp32: correct-path max deviation {correct:.3e}")
    assert correct < 1e-5, correct
    for name, wrong in _variants64(gate_up32, LIMIT).items():
        dev = (out - wrong).abs().max().item()
        print(f"  {name:42s} max dev {dev:.6e} ({dev / correct:10.1f}x correct)")
        assert dev > 100.0 * correct, (name, dev, correct)

    # bf16, through the engine's own pre-SiLU clamp.
    gate_up = gate_up32.to(torch.bfloat16)
    ref_b = op.reference(gate_up, LIMIT)
    ours = _ulp_distance(op(gate_up, LIMIT), ref_b)
    engine = _ulp_distance(torch.ops.trtllm.silu_and_mul(gate_up, swiglu_limit=LIMIT), ref_b)
    ours_off, engine_off = (ours > 0).float().mean().item(), (engine > 0).float().mean().item()
    print(
        f"  bf16: entry off-by-ulp fraction {ours_off:.3e} (max {ours.max().item()} ulp); "
        f"trtllm silu_and_mul(swiglu_limit=7) {engine_off:.3e} (max {engine.max().item()} ulp)"
    )
    assert ours_off <= MAX_OFF_FRACTION and ours.max().item() <= MAX_ULPS
    assert engine_off > 10.0 * MAX_OFF_FRACTION, engine_off
    with pytest.raises(AssertionError, match="off by one ulp"):
        _compare(torch.ops.trtllm.silu_and_mul(gate_up, swiglu_limit=LIMIT), ref_b)


def test_guard_refuses_an_odd_width():
    """Width 3 splits 2 | 1 and broadcasts: unguarded, the op answers `[T, 2]`."""
    gate_up = _activand(4, 2, torch.bfloat16, 3)[:, :3].contiguous()
    out = op(gate_up, LIMIT)
    assert out.shape == (4, 2), out.shape
    with pytest.raises(AssertionError, match="not gate\\|up halves"):
        with validating(op):
            op(gate_up, LIMIT)


@pytest.mark.parametrize("limit", [-LIMIT, float("nan")], ids=["negative", "nan"])
def test_guard_refuses_a_limit_that_is_not_a_bound(limit):
    """`clamp(-L, L)` with `-L > L` is `L` everywhere, and a NaN limit poisons
    every element: unguarded, the op answers without an error.
    """
    gate_up = _activand(4, 384, torch.bfloat16, 4)
    out = op(gate_up, limit)
    assert out.shape == (4, 384)
    with pytest.raises(AssertionError, match="must be non-negative"):
        with validating(op):
            op(gate_up, limit)


def test_bf16_band_is_one_rounding_of_the_fp32_expression():
    """The band's premise: the entry is the fp32 expression rounded once, not a
    chain of bf16 roundings. Rounding SiLU to bf16 before the clamp and the
    product (what `F.silu` on the bf16 half would do) falls outside it.
    """
    gate_up = _activand(512, 3456, torch.bfloat16, 5)
    ref = op.reference(gate_up, LIMIT)
    gate, up = gate_up.chunk(2, dim=-1)
    double_rounded = F.silu(gate).clamp(max=LIMIT) * up.clamp(-LIMIT, LIMIT)
    off = (_ulp_distance(double_rounded, ref) > 0).float().mean().item()
    print(f"\n  bf16 SiLU then bf16 product: off-by-ulp fraction {off:.3e}")
    with pytest.raises(AssertionError):
        _compare(double_rounded, ref)
