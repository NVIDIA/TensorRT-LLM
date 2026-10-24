# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_moe_dsa tp16 q_b_fp8_rope entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.gemm.q_b_fp8_rope import (  # noqa: E501
    q_b_fp8_rope as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the glm_moe_dsa tp16 entries are certified on sm_107"
)

#: GLM-5.2's fused-A output row: q_lora_rank 2048, kv_lora_rank 512, qk_rope_head_dim 64.
_FUSED_A_WIDTH = 2624
#: Rows of the test's RoPE table, and GLM-5.2's rope_theta.
_POSITIONS = 8192
_THETA = 8.0e6
#: An E4M3 NaN: what every output byte holds until the kernel stores it.
_E4M3_NAN = 0x7F


def _mla_cos_sin(positions: int, rope_dim: int) -> torch.Tensor:
    """The MLA RoPE table in the layout the attention backend builds for MLA: per position
    rope_dim (cos, sin) pairs, the rope_dim / 2 frequencies theta ** (-2i / rope_dim) twice."""
    inv_freq = _THETA ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    angle = torch.arange(positions, dtype=torch.float64)[:, None] * inv_freq[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    return torch.stack((angle.cos(), angle.sin()), dim=-1).float().cuda().contiguous()


def _build(spec: dict, seed: int) -> dict:
    """The op's arguments for one cell.

    q_a is the q_lora_rank-column row view of a fused-A output, as the target passes it. The
    weight is scaled so that the FP32 values before the conversion have a standard deviation of
    4 (times q_scale): E4M3 normals of every exponent from 2**-6 up, and subnormals below.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, k, seq_len = spec["m"], spec["k"], spec["seq_len"]
    n = spec["heads"] * spec["head_dim"]
    fused_a = torch.randn(m, _FUSED_A_WIDTH, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * 4.0 * k**-0.5).to(torch.bfloat16)
    kv_lens = torch.randint(
        seq_len,
        _POSITIONS + 1,
        (m // seq_len,),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    return dict(
        q_a=fused_a[:, :k],
        weight=weight,
        out=torch.full((m, n), _E4M3_NAN, dtype=torch.uint8, device="cuda"),
        kv_lens=kv_lens,
        cos_sin=_mla_cos_sin(_POSITIONS, spec["rope_dim"]),
        q_scale=torch.full((1,), spec.get("q_scale", 1.0), dtype=torch.float32, device="cuda"),
        eps=spec["eps"],
        head_dim=spec["head_dim"],
        rope_dim=spec["rope_dim"],
        seq_len=seq_len,
        plan=spec["plan"],
    )


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Every E4M3 value within one E4M3 ulp, and only the rare ones not equal.

    Before its single rounding, the kernel's FP32 value differs from the reference's by the
    split-K summation order of a 2048-long dot product (about 3e-6 relative), the approximate
    rsqrt of the row RMS (about 2**-22) and the contraction of the RoPE products into FMAs (one
    FP32 ulp): a few times 1e-6 relative. E4M3 spaces its values 2**-4 to 2**-3 apart
    (relative), so a value converts differently only within that distance of a rounding
    midpoint: a probability of about 1e-4 per element, about two of the 18432 values of a verify
    cell. Allowed: one E4M3 ulp per element, and at most max(4, 0.1 %) differing bytes. A wrong
    position, pair, sign or scale moves most RoPE values by many ulps, and a lost K slice moves
    every value.
    """
    got = out.view(torch.float8_e4m3fn).float()
    want = ref.view(torch.float8_e4m3fn).float()
    assert torch.isfinite(got).all(), "the op left an output byte unwritten (NaN) or wrote NaN"
    magnitude = torch.maximum(got.abs(), want.abs()).clamp_min(2.0**-6)
    ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 3)
    worst = ((got - want).abs() / ulp).max().item()
    assert worst <= 1.0, f"an E4M3 value is {worst} ulp off"
    differing = int((out != ref).sum())
    assert differing <= max(4, out.numel() // 1000), f"{differing} of {out.numel()} bytes differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    q_a_before, weight_before = args["q_a"].clone(), args["weight"].clone()
    assert op(**args) is args["out"], "the op writes in place and returns what it wrote"
    assert torch.equal(args["q_a"], q_a_before) and torch.equal(args["weight"], weight_before)
    _compare(args["out"], op.reference(**args))


def test_q_scale_is_applied_and_saturates() -> None:
    """A scale of 32 pushes the values past 448: the conversion saturates instead of producing
    NaN, and the scale multiplies before it (c1 runs at 1, where the scale is invisible)."""
    spec = dict(op.CELLS[0].spec, q_scale=32.0)
    args = _build(spec, seed=41)
    with validating(op):
        op(**args)
    got = args["out"].view(torch.float8_e4m3fn).float()
    assert (got.abs() == 448.0).any(), "no value reached the saturation bound"
    _compare(args["out"], op.reference(**args))


def test_plans_agree() -> None:
    """The plans change the launch and the FP32 summation order, not the definition."""
    spec = op.CELLS[0].spec
    results = {}
    for plan in ("regular", "beside_topk", "early_gather"):
        args = _build(dict(spec, plan=plan), seed=43)
        op(**args)
        results[plan] = args["out"]
    _compare(results["beside_topk"], results["regular"])
    _compare(results["early_gather"], results["regular"])


def test_cuda_graph_capture_and_replay() -> None:
    """The target captures the whole decode step: the op must replay on new contents of the same
    buffers. Warm-up compiles the kernel and builds its placeholders outside the capture."""
    spec = op.CELLS[0].spec
    args = _build(spec, seed=44)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(**args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    fresh = _build(spec, seed=45)
    args["q_a"].copy_(fresh["q_a"])
    args["kv_lens"].copy_(fresh["kv_lens"])
    args["out"].fill_(_E4M3_NAN)
    graph.replay()
    torch.cuda.synchronize()
    _compare(args["out"], op.reference(**args))


def test_guard_refuses_a_table_of_another_row_width() -> None:
    """A table with rope_dim / 2 pairs per position (no repeated half) is read at the wrong rows."""
    args = _build(op.CELLS[0].spec, seed=46)
    args["cos_sin"] = args["cos_sin"][:, : args["rope_dim"] // 2].contiguous()
    with pytest.raises(AssertionError, match="cos_sin must be"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_position_past_the_table() -> None:
    args = _build(op.CELLS[0].spec, seed=47)
    args["kv_lens"].fill_(_POSITIONS + 1)
    with pytest.raises(AssertionError, match="cos/sin table must hold every position"):
        with validating(op):
            op(**args)


def test_guard_refuses_kv_lens_shorter_than_the_step() -> None:
    """kv_lens below seq_len gives negative positions: reads before the table."""
    args = _build(op.CELLS[0].spec, seed=48)
    args["kv_lens"].fill_(args["seq_len"] - 1)
    with pytest.raises(AssertionError, match="kv_lens must cover the step's own tokens"):
        with validating(op):
            op(**args)


def test_guard_refuses_an_out_over_q_a() -> None:
    """The owner CTAs store while other clusters still read q_a: silently wrong."""
    args = _build(op.CELLS[0].spec, seed=49)
    m, n = args["out"].shape
    args["out"] = args["q_a"].view(torch.uint8)[:, :n]
    assert args["out"].shape == (m, n)
    with pytest.raises(AssertionError, match="out must not share memory with q_a"):
        with validating(op):
            op(**args)
