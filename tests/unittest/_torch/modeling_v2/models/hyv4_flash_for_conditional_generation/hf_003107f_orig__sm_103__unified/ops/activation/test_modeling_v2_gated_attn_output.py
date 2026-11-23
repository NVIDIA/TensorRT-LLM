# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_attn_output entry of the Hy4-Flash unified target.

The entry owns the contract. This file turns a cell into a pair of operands, gates the
result against the entry's own reference with a band derived from the two implementations,
holds the gate and the buffer past the last row untouched, adds what a cell cannot carry
(the error against a float64 product, CUDA-graph capture and replay of the decode shapes,
and the band refusing a wrong gate read), and drives the op's loud layout refusals.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.activation.gated_attn_output import (  # noqa: E501
    gated_attn_output as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 — path above

DEV = "cuda"

#: One bf16 ulp of a row's scale, in the units `assert_within_ulp` gates on.
_ULP = 2.0**-8

#: The kernel widens both operands to fp32, computes `attention * tl.sigmoid(gate)` and
#: rounds once to bf16; the reference computes the same product with torch's fp32 sigmoid.
#: The two differ only in sigmoid's accuracy (~1e-7 relative), far below bf16's spacing, so
#: a disagreeing element is one bf16 ulp of its row's scale. The gate is slack for the
#: approximation: element-wise 2 ulp of the row scale, and the relative RMS well inside one.
_ELEMENT_ULP = 2.0
_RMS_ULP = 1.0

#: Elements past `attention_output`'s last row, inside the buffer it views: a masked lane
#: that stored anyway would land here (one kernel block is 2048 wide).
_TAIL = 2048

assert torch.cuda.is_available(), "gated_attn_output requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(out, out_buffer, gate)` for one spec: `out` the compact head of a buffer that runs
    `_TAIL` elements past its last row, `gate` a compact `[rows, width]` tensor."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, width, dtype = spec["rows"], spec["width"], spec["dtype"]
    buffer = torch.randn(rows * width + _TAIL, generator=g, device=DEV).to(dtype)
    out = buffer[: rows * width].view(rows, width)
    gate = torch.randn(rows, width, generator=g, device=DEV).to(dtype)
    return out, buffer, gate


def _measure(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """`(bit-exact fraction, worst element in row-scale ulp, relative L2)`."""
    diff = (out.float() - ref.float()).abs()
    scale = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    exact = (out.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
    worst = float((diff / scale).max()) / _ULP
    rel = (
        (out.float() - ref.float()).pow(2).mean().sqrt()
        / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
    ).item()
    return exact, worst, rel


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    out, buffer, gate = _build(cell.spec, seed=_seed(cell.why))
    original = out.clone()
    gate_before = gate.clone()
    tail_before = buffer[out.numel() :].clone()

    result = op(out, gate)

    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    ref = op.reference(original, gate)
    assert_within_ulp(result, ref, _ELEMENT_ULP, _RMS_ULP)
    assert torch.equal(gate, gate_before), "the gate was written"
    assert torch.equal(buffer[out.numel() :], tail_before), "a masked lane stored"
    exact, worst, rel = _measure(result, ref)
    print(
        f"  {cell.why[:44]}: {100 * exact:.6f}% bit-exact, worst {worst:.3e} row-scale ulp, "
        f"rel L2 {rel:.3e}"
    )


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_matches_a_float64_product() -> None:
    """Independent of the reference: the op and the reference both have to sit at bf16
    output rounding of the exact product.

    The cells compare two implementations of the same arithmetic, which would both be
    wrong together if the arithmetic itself were wrong. This bounds the error against a
    float64 product instead: bf16 rounds a value to within half an ulp, 2^-9 of it, so the
    RMS relative error is at most 2^-9 / sqrt(3) ~ 2^-9.8. The gate is 2^-8 (measured at
    this shape: the printed value).
    """
    spec = dict(rows=64, width=16384, dtype=torch.bfloat16)
    out, _, gate = _build(spec, seed=_seed("float64"))
    original = out.clone()
    op(out, gate)
    exact = original.double() * torch.sigmoid(gate.double())
    rel = ((out.double() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()).item()
    print(f"  {torch.cuda.get_device_name(0)}: relative L2 {rel:.3e} from the float64 product")
    assert rel <= 2.0**-8, f"relative L2 {rel:.3e} from the float64 product"


@pytest.mark.parametrize(
    "rows,width", [(1, 8192), (6, 8192), (64, 8192), (8, 16384), (1, 4096), (64, 4096)]
)
def test_cuda_graph_capture_and_replay(rows, width) -> None:
    """The decode and draft call sites run captured: the op replays on refilled inputs.

    Rows 1, 6 and 64 are the decode batch's ends and the drafter's block at the sparse
    width; 8 rows at the SWA width is a captured decode batch; 1 and 64 rows at the
    context engine's `attn_tp` 2 width (4096) are its captured decode ends.
    """
    spec = dict(rows=rows, width=width, dtype=torch.bfloat16)
    out, _, gate = _build(spec, seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(out, gate)  # JIT compile outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(out, gate)

    for seed in (11, 12):
        fresh, _, fresh_gate = _build(spec, seed=seed)
        out.copy_(fresh)
        gate.copy_(fresh_gate)
        expected = op.reference(out, gate)
        graph.replay()
        torch.cuda.synchronize()
        assert_within_ulp(out, expected, _ELEMENT_ULP, _RMS_ULP)


@pytest.mark.parametrize("rows,width", [(6, 8192), (64, 8192), (6, 4096), (64, 4096)])
def test_the_band_refuses_a_wrong_gate_read(rows, width) -> None:
    """The band refuses the two ways the gate could be read wrong: not applied at all, and
    each row reading its neighbour's gate. Both widths: the sparse layers' and the context
    engine's `attn_tp` 2 share of them."""
    spec = dict(rows=rows, width=width, dtype=torch.bfloat16)
    out, _, gate = _build(spec, seed=_seed(f"wrong read {rows} {width}"))
    original = out.clone()
    op(out, gate)
    for name, wrong in (
        ("gate ignored", original),
        ("neighbour row", op.reference(original, gate.roll(1, dims=0))),
    ):
        with pytest.raises(AssertionError):
            assert_within_ulp(out, wrong, _ELEMENT_ULP, _RMS_ULP)
        separation = (out.float() - wrong.float()).abs().max().item()
        print(f"  {rows} rows, {name}: max abs separation {separation:.3f}")
        assert separation > 0.5, name


def test_metadata_misuse_raises() -> None:
    """Every layout violation is refused loudly by the op's own bare assert, which is why
    the entry has no `is_valid`: a strided gate, a gate of the wrong shape, a 1-D
    attention_output."""
    g = torch.Generator(device=DEV).manual_seed(8)
    attention_output = torch.randn(4, 8192, generator=g, device=DEV).to(torch.bfloat16)
    strided_gate = torch.randn(4, 16384, generator=g, device=DEV).to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError):
        op(attention_output, strided_gate)
    with pytest.raises(AssertionError):
        op(attention_output, torch.randn(3, 8192, generator=g, device=DEV).to(torch.bfloat16))
    with pytest.raises(AssertionError):
        op(attention_output.flatten(), torch.randn(4 * 8192, generator=g, device=DEV))
    # A CPU tensor is outside the op's contract: Triton refuses the pointer, which is the
    # ValueError the entry's note records. Nothing is written on the host.
    host_g = torch.Generator().manual_seed(9)
    host = torch.randn(4, 8192, generator=host_g)
    with pytest.raises(ValueError):
        op(host, torch.randn(4, 8192, generator=host_g))
