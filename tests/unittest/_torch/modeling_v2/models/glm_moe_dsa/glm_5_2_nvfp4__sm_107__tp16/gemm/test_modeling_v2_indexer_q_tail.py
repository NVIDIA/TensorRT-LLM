# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_moe_dsa tp16 indexer_q_tail entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.indexer_q_tail import (  # noqa: E501
    indexer_q_tail as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 entries are certified on sm_107"
)

#: GLM-5.2's fused-A output row with the indexer's wk (128) and weights_proj (32) rows: the
#: MLA's 2624 columns, then the indexer K, then the head weights.
_FUSED_A_WIDTH = 2784
_WEIGHTS_COLUMN = 2624 + 128
#: Rows of the test's RoPE table, and GLM-5.2's rope_theta.
_POSITIONS = 8192
_THETA = 8.0e6
#: An E4M3 NaN: what every Q byte holds until the kernel stores it.
_E4M3_NAN = 0x7F


def _indexer_cos_sin(positions: int, rope_dim: int) -> torch.Tensor:
    """The indexer's RoPE table: per position the rope_dim / 2 cosines, then the sines."""
    inv_freq = _THETA ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    angle = torch.arange(positions, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat((angle.cos(), angle.sin()), dim=-1).float().cuda().contiguous()


def _build(spec: dict, seed: int) -> dict:
    """The op's arguments for one cell.

    q_a and the head weights are views of one fused-A output, as the target passes them. The
    weight is scaled so that wq_b's outputs have a standard deviation of 2. The outputs start as
    NaN, so anything the kernel does not store fails the comparison.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k = spec["m"], spec["k"]
    heads, head_dim = spec["n_heads"], spec["head_dim"]
    fused_a = torch.randn(m, _FUSED_A_WIDTH, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(heads * head_dim, k, generator=g, device="cuda") * 2.0 * k**-0.5).to(
        torch.bfloat16
    )
    positions = torch.randint(0, _POSITIONS, (m,), generator=g, device="cuda", dtype=torch.int32)
    q_fp8 = torch.full((m, heads, head_dim), _E4M3_NAN, dtype=torch.uint8, device="cuda")
    return dict(
        q_a=fused_a[:, :k],
        weight=weight,
        positions=positions,
        cos_sin=_indexer_cos_sin(_POSITIONS, spec["rope_dim"]),
        weights_in=fused_a[:, _WEIGHTS_COLUMN : _WEIGHTS_COLUMN + heads],
        q_fp8=q_fp8.view(torch.float8_e4m3fn),
        q_scale=torch.full((m, heads, 1), float("nan"), dtype=torch.float32, device="cuda"),
        weights_out=torch.full((m, heads), float("nan"), dtype=torch.float32, device="cuda"),
        eps=spec["eps"],
        n_heads=heads,
        head_dim=head_dim,
        rope_dim=spec["rope_dim"],
        interleave=spec["interleave"],
        ue8m0=spec["ue8m0"],
        weight_scale=spec["weight_scale"],
    )


def _compare(got: tuple, want: tuple) -> None:
    """Scales and weights exact where the head's scale agrees; Q within one E4M3 ulp.

    wq_b's FP32 outputs differ from the reference's by the split-K summation order of a
    2048-long dot product and the approximate rsqrt (a few times 1e-6 relative), which changes
    the BF16 rounding of about one value in a thousand by one BF16 ulp. Those flips reach the
    E4M3 Q only where they cross an E4M3 rounding midpoint (1 in 16 of them) and reach a head's
    UE8M0 scale only where the head's amax crosses a power of two (far rarer). Allowed: at most
    2 (token, head) scales differing, exact scaled weights where the scale agrees, every Q value
    within one E4M3 ulp of the coarser of the two scales, at most max(8, 0.1 %) Q bytes
    differing. A wrong position, pair, sign, scale rule or amax moves whole heads.
    """
    q, scale, weights = got
    q_ref, scale_ref, weights_ref = want
    assert torch.isfinite(scale).all() and torch.isfinite(weights).all(), "an output is unwritten"
    q_value = q.float()
    assert torch.isfinite(q_value).all(), "a Q byte is unwritten (NaN)"
    same_scale = scale == scale_ref
    assert int((~same_scale).sum()) <= 2, f"{int((~same_scale).sum())} head scales differ"
    same = same_scale.squeeze(-1)
    assert torch.equal(weights[same], weights_ref[same]), "the scaled weights differ"
    coarser = torch.maximum(scale, scale_ref)
    got_value = q_value * scale
    want_value = q_ref.float() * scale_ref
    magnitude = (torch.maximum(got_value.abs(), want_value.abs()) / coarser).clamp_min(2.0**-6)
    ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 3) * coarser
    worst = ((got_value - want_value).abs() / ulp).max().item()
    assert worst <= 1.0, f"a Q value is {worst} E4M3 ulp off"
    differing = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
    assert differing <= max(8, q.numel() // 1000), f"{differing} of {q.numel()} Q bytes differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    q_a_before, weights_in_before = args["q_a"].clone(), args["weights_in"].clone()
    got = op(**args)
    assert got[0] is args["q_fp8"] and got[1] is args["q_scale"] and got[2] is args["weights_out"]
    assert torch.equal(args["q_a"], q_a_before)
    assert torch.equal(args["weights_in"], weights_in_before)
    _compare(got, op.reference(**args))


def test_scales_are_powers_of_two() -> None:
    """UE8M0: every scale is a power of two at or above amax / 448, so the Q never saturates."""
    args = _build(op.CELLS[0].spec, seed=51)
    with validating(op):
        q, scale, _ = op(**args)
    mantissa = scale.view(torch.int32) & 0x007FFFFF
    assert int(mantissa.abs().sum()) == 0, "a scale is not a power of two"
    assert float(q.float().abs().max()) <= 448.0


def test_guard_refuses_a_position_past_the_table() -> None:
    args = _build(op.CELLS[0].spec, seed=52)
    args["positions"][0] = _POSITIONS
    with pytest.raises(AssertionError, match="every position must index a row"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_q_over_q_a() -> None:
    """The owner CTAs store while other clusters still read q_a: silently wrong."""
    args = _build(op.CELLS[0].spec, seed=53)
    m, heads, head_dim = args["q_fp8"].shape
    over = args["q_a"].view(torch.uint8)[:, : heads * head_dim].view(m, heads, head_dim)
    args["q_fp8"] = over.view(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="q_fp8 must not share memory with q_a"):
        with validating(op):
            op(**args)
