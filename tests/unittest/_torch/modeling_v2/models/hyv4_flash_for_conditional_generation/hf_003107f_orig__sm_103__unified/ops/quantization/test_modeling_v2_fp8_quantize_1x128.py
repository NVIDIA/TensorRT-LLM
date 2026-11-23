# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128 entry of the Hy4-Flash unified target.

The entry owns the contract. This file turns a cell into an input, holds every code
disagreement of a cell to a rounding tie, and adds what a cell cannot carry: CUDA-graph
capture and replay of the decode shapes, the error bound against the input itself, and the
loud rejection of a strided input.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    BLOCK,
    block_scales,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

DEV = "cuda"

#: The default build divides exactly, so its scales are the reference's. A `FAST_MATH=ON` build
#: divides approximately, each division within CUDA's documented 2 ulp, and the returned scale is
#: two divisions deep: the gate admits that build.
SCALE_ULP_GATE = 4

#: A code can only disagree with the exactly rounded one when the scale it multiplies by is off,
#: so when the exact product sits within that division's 2 ulp of an e4m3 rounding midpoint.
TIE_REL_GATE = 2.0 * 2.0**-23

assert torch.cuda.is_available(), "fp8_quantize_1x128 requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="fp8_quantize_1x128 is certified on sm_103"
)


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude ordinal of an e4m3 byte: adjacent codes differ by 1."""
    b = codes.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b & 0x7F)


def _compare(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scales within 4 fp32 ulp; codes at most one e4m3 step apart, >=99.5% bit-exact.

    The default build divides exactly, and every certified cell is 100% bit-exact against
    the reference (the printed output has the per-cell numbers). The gate is slack for a
    `--use_fast_math` build, which compiles each of the kernel's two divisions within
    CUDA's documented 2 ulp, so a code can only disagree when the exact product sits
    within that band of an e4m3 rounding midpoint.
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
        assert exact >= 0.995, f"only {100 * exact:.4f}% of codes bit-exact"


def _expand(v: torch.Tensor, n: int) -> torch.Tensor:
    """`[T, nb]` block values -> `[T, n]` per-column values."""
    return v.repeat_interleave(BLOCK, dim=1)[:, :n].contiguous()


def _input(rows: int, cols: int, seed: int, magnitude: float = 3.0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(rows, cols, device=DEV, generator=g) * magnitude).to(torch.bfloat16)


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


def _check_cell(label, x, use_ue8m0, data, scales) -> None:
    """Shapes and layout, the gate against the entry's reference, and every miss a tie."""
    nb = (x.shape[1] + BLOCK - 1) // BLOCK
    assert data.shape == x.shape and data.is_contiguous() and data.device == x.device
    assert scales.shape == (nb, x.shape[0]) and scales.is_contiguous()
    ref = op.reference(x, use_ue8m0)
    _compare((data, scales), ref)
    rel = _assert_misses_are_ties(x, data, ref[0]) if not use_ue8m0 else 0.0
    exact = (data.view(torch.uint8) == ref[0].view(torch.uint8)).float().mean().item()
    print(f"  {label}: {100 * exact:.4f}% bit-exact, worst tie {rel:.3e}")


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    x = _input(spec["rows"], spec["cols"], seed=seed)
    data, scales = op(x, spec["use_ue8m0"])
    _check_cell(cell.why[:44], x, spec["use_ue8m0"], data, scales)


# ── what a cell cannot carry ──────────────────────────────────────────────


@pytest.mark.parametrize("rows", [1, 6, 64])
def test_cuda_graph_capture_and_replay(rows):
    """A replay reproduces a fresh eager call bitwise; so does a side stream.

    Rows 1 and 64 are the decode batch's ends and 6 is the drafter's block, at q_b's
    1024-wide input: the shapes the target captures.
    """
    x = _input(rows, 1024, seed=17, magnitude=1.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_data, g_sf = op(x)
    x.copy_(_input(rows, 1024, seed=18, magnitude=1.0))
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


def test_the_quantization_round_trips_within_the_block_bound() -> None:
    """Independent of the reference: dequantizing has to return the input.

    The cells compare two implementations of the same arithmetic, which would both be wrong
    together if the arithmetic itself were wrong. This bounds the error against the input
    instead: e4m3's 3 mantissa bits round a value to within half an ulp, 2^-4 of it, and no
    element exceeds the block maximum, so the dequantized error is at most `amax * 2^-4`
    (measured 3.571e-02 of amax at this shape). The square shape (64 rows == 8192/128) also
    reads the scales K-block-major, so a transposed layout fails here rather than passing a
    shape check.
    """
    x = _input(64, 8192, seed=5)
    data, scales = op(x)
    rows, nb = x.shape[0], 8192 // BLOCK
    scale = scales.t().float().reshape(rows, nb, 1)
    deq = (data.float().view(rows, nb, BLOCK) * scale).view(rows, 8192)
    block_amax = x.float().view(rows, nb, BLOCK).abs().amax(-1, keepdim=True)
    err = (deq - x.float()).view(rows, nb, BLOCK).abs().amax(-1, keepdim=True)
    worst = float((err / block_amax.clamp_min(1e-30)).max())
    print(f"  {torch.cuda.get_device_name(0)}: worst block error {worst:.3e} of amax")
    assert torch.all(err <= block_amax * 2.0**-4), f"worst block error {worst:.3e} of amax"


def test_rejects_a_strided_input():
    """The op's own precondition, loud, which is why there is no guard."""
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device=DEV)[:, ::2]
    with pytest.raises((RuntimeError, NotImplementedError)):
        op(x)
