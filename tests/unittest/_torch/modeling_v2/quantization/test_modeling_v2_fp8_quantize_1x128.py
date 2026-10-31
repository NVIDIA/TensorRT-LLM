# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128 entry (the unified target's local variant r41).

The entry owns the contract. This file turns a cell into an input, holds
every code disagreement of a cell to a rounding tie, and adds what a cell
cannot carry: CUDA-graph capture and replay, and the loud rejection of a
strided input.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (  # noqa: E501
    SCALE_ULP_GATE,
    block_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128 as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp8_quantize_1x128 requires a CUDA device"


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
    over the certified column: scales within 2 ulp of both `amax / 448` and
    `1 / (448 / amax)` (~80% bit-exact at T=1024); codes at worst
    99.6875% bit-exact at H 2560 (99.8566% elsewhere), every miss exactly
    one code away with the exact product within 1.122e-07 relative of the
    midpoint between the two -- a rounding tie the scale's last ulp decides.
    `use_ue8m0=True` is bit-exact (the power-of-two scale makes the
    multiply exact), and passes this gate trivially.

    Wrong recipes at T=256, H=2560 (bytes agreeing / worst relative scale
    error): power-of-two 0.043 / 0.99, per-tensor 0.0002 / 2.0, per-row
    0.083 / 1.6, blocks down the token axis 0.19 / 1.0, blocks offset by 64
    0.48 / 0.93, e4m3 max as 240 0 / 0.87, signed max 0.58 / 0.58; the
    correct one 0.9993 / 1.4e-07.
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


DEV = "cuda"
# A code can only disagree when the exact product sits within the first
# division's 2 ulp of an e4m3 rounding midpoint.
TIE_REL_GATE = 2.0 * 2.0**-23


def _expand(v: torch.Tensor, n: int) -> torch.Tensor:
    """`[T, nb]` block values -> `[T, n]` per-column values."""
    return v.repeat_interleave(128, dim=1)[:, :n].contiguous()


def _input(tokens: int, hidden: int, seed: int, magnitude: float = 3.0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(tokens, hidden, device=DEV, generator=g) * magnitude).to(torch.bfloat16)


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


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    x = _input(spec["tokens"], spec["hidden"], seed=zlib.crc32(cell.why.encode()))
    data, scales = op(x, spec["use_ue8m0"])
    nb = (spec["hidden"] + 127) // 128
    assert data.shape == x.shape and data.is_contiguous() and data.device == x.device
    assert scales.shape == (nb, spec["tokens"]) and scales.is_contiguous()
    ref = op.reference(x, spec["use_ue8m0"])
    _compare((data, scales), ref)
    rel = _assert_misses_are_ties(x, data, ref[0]) if not spec["use_ue8m0"] else 0.0
    exact = (data.view(torch.uint8) == ref[0].view(torch.uint8)).float().mean().item()
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    print(
        f"  {cell.why[:44]}: {100 * exact:.4f}% bit-exact, worst tie {rel:.3e}, "
        f"{spec['tokens'] * nb} groups over a {sms * 8 * 8 * 2}-group grid sweep"
    )


# ── what a cell cannot carry ──────────────────────────────────────────────


@pytest.mark.parametrize("rows", (64, 128, 1024))
def test_cuda_graph_capture_and_replay(rows):
    """A replay reproduces a fresh eager call bitwise; so does a side stream.

    128 and 1024 are the captured drafter rows at max_batch_size 128: one row
    per request, and mtp7's 8 verify rows per request on the first draft.
    """
    x = _input(rows, 2560, seed=17, magnitude=1.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_data, g_sf = op(x)
    x.copy_(_input(rows, 2560, seed=18, magnitude=1.0))
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


def test_rejects_a_strided_input():
    """The op's own precondition, loud, which is why there is no guard."""
    x = torch.randn(8, 5120, dtype=torch.bfloat16, device=DEV)[:, :2560]
    with pytest.raises((RuntimeError, NotImplementedError)):
        op(x)
