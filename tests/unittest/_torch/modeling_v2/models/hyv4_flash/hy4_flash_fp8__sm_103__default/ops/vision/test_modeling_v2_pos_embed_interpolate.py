# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the pos_embed_interpolate entry of the Hy4-Flash unified target.

The entry owns the contract. This file turns a cell into a table, gates the result against
the entry's own reference with a band derived from the two implementations, checks the same
result against the HF tower's own `interpolate_pos_encoding` (which the entry's mirror would
follow into the same mistake if the formula were wrong), holds the table untouched, and adds
what a cell cannot carry: the band refusing each plausible wrong reading of the grid, and
the op's `is_valid` refusals after showing the wrong answer they prevent.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.vision.pos_embed_interpolate import (  # noqa: E501
    pos_embed_interpolate as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401 — path above

DEV = "cuda"

#: One bf16 ulp, the unit `assert_within_ulp` gates in.
_ULP = 2.0**-8

#: The kernel blends the four corners in fp32 and rounds once to bf16; the reference does
#: the same blend in fp64 and rounds once. A boundary flip moves an element by one bf16 ulp
#: of its own value, and a bf16 ulp runs up to 2^-7 of that value inside its binade, so the
#: row-scale unit used here reads up to 2.0 for a single flip. The gate is 3.0; the printed
#: worst over the cells is 1.896 and the relative RMS 0.041 of one ulp.
_ELEMENT_ULP = 3.0
_RMS_ULP = 1.0

assert torch.cuda.is_available(), "pos_embed_interpolate requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _spec(t: int, h: int, w: int) -> dict:
    """One grid at the checkpoint's table geometry, taken from the entry's own cells."""
    spec = dict(op.CELLS[0].spec)
    spec.update(t=t, h=h, w=w)
    return spec


def _table(spec, seed):
    """The checkpoint's table shape and dtype: `[position_edge**2 + 1, hidden]`, row 0 the
    cls token.

    Independent normal rows rather than a smooth field: the op rounds once from an fp32
    blend, so its error is measured against the row's scale and does not grow with the
    corners' disagreement the way a dtype-accumulating kernel's does. Uncorrelated rows put
    cancellation in the blend, which is the harder table.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = spec["position_edge"] ** 2 + 1
    return torch.randn(rows, spec["hidden"], generator=g, device=DEV).to(spec["dtype"])


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


def _hf_path(table: torch.Tensor, t: int, h: int, w: int, edge: int, dtype) -> torch.Tensor:
    """`HYV4VLVisionPatchEmbed.interpolate_pos_encoding`, the HF tower's math verbatim.

    Rows 1..edge**2 of the table, read as a square grid, `.float()`, bilinear with an
    explicit `scale_factor` of `(size + 0.1) / edge` and `align_corners=False`, back to
    raster order, `.to(dtype)`, then `.repeat(t, 1)` by the caller. torch's own identity
    when the output size equals the input size comes along with the call.
    """
    pos = table[1:].reshape(1, edge, edge, -1).permute(0, 3, 1, 2).float()
    pos = torch.nn.functional.interpolate(
        pos,
        scale_factor=((h + 0.1) / edge, (w + 0.1) / edge),
        mode="bilinear",
        align_corners=False,
    )
    return pos.permute(0, 2, 3, 1).reshape(h * w, -1).to(dtype).repeat(t, 1)


def _src(size: int, scale: float, dev) -> torch.Tensor:
    return ((torch.arange(size, dtype=torch.float64, device=dev) + 0.5) * scale - 0.5).clamp_min(
        0.0
    )


def _blend(grid: torch.Tensor, src_h: torch.Tensor, src_w: torch.Tensor) -> torch.Tensor:
    """The four-corner bilinear blend in fp64 over `[edge, edge, hidden]`, `[h, w, hidden]`.

    Only the wrong-variant test builds grids and source positions other than the op's; the
    certified cells go through the entry's own reference.
    """
    hf = src_h.floor().to(torch.int64)
    wf = src_w.floor().to(torch.int64)
    hc = (hf + 1).clamp_max(grid.shape[0] - 1)
    wc = (wf + 1).clamp_max(grid.shape[1] - 1)
    dh = (src_h - hf.double())[:, None, None]
    dw = (src_w - wf.double())[None, :, None]
    top = (1.0 - dw) * grid[hf][:, wf] + dw * grid[hf][:, wc]
    bot = (1.0 - dw) * grid[hc][:, wf] + dw * grid[hc][:, wc]
    return (1.0 - dh) * top + dh * bot


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    s = cell.spec
    table = _table(s, seed=_seed(cell.why))
    before = table.clone()

    result = op(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])

    rows = s["t"] * s["h"] * s["w"]
    assert result.shape == (rows, s["hidden"]), result.shape
    assert result.dtype == s["dtype"]
    ref = op.reference(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    exact, worst, rel = _measure(result, ref)
    print(
        f"  {cell.why[:44]}: {100 * exact:.6f}% bit-exact, worst {worst:.3e} row-scale ulp, "
        f"rel L2 {rel:.3e}"
    )
    assert_within_ulp(result, ref, _ELEMENT_ULP, _RMS_ULP)
    assert torch.equal(table, before), "the table was written"
    if s["t"] > 1:
        frame = result[: s["h"] * s["w"]]
        assert torch.equal(result, frame.repeat(s["t"], 1)), "the frame does not repeat"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_matches_the_hf_reference(cell) -> None:
    """Independent of the entry's reference: the HF tower's own path, on the same table.

    A mirror that followed the op into a wrong grid would pass the cell and fail here.
    """
    s = cell.spec
    table = _table(s, seed=_seed(cell.why) + 1)
    result = op(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    hf = _hf_path(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    exact, worst, rel = _measure(result, hf)
    print(
        f"  {cell.why[:44]}: {100 * exact:.6f}% bit-exact against HF, worst {worst:.3e} "
        f"row-scale ulp, rel L2 {rel:.3e}"
    )
    assert_within_ulp(result, hf, _ELEMENT_ULP, _RMS_ULP)


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_the_band_refuses_a_wrong_reading() -> None:
    """Each plausible wrong reading of the grid, against the op's output.

    The `+ 0.1` in the scale factor is the HF tower's own and the difference from SigLIP's
    `size=` interpolation; corner alignment, the cls row, the row order and the scope of the
    identity at the table's own size are the other ways a bilinear resample of this table
    goes wrong. The grid is `96x128` so the last one is expressible: one side at the table's
    size, one not.
    """
    s = _spec(1, 96, 128)
    edge, dtype = s["position_edge"], s["dtype"]
    table = _table(s, seed=_seed("wrong readings"))
    out = op(table, s["t"], s["h"], s["w"], edge, dtype)
    dev = table.device
    grid = table.double()[1:].reshape(edge, edge, -1)
    src_h = _src(s["h"], 1.0 / ((s["h"] + 0.1) / edge), dev)
    src_w = _src(s["w"], 1.0 / ((s["w"] + 0.1) / edge), dev)

    wrongs = {
        # SigLIP's `size=` interpolation: the scale factor without the + 0.1.
        "size= without the + 0.1": _blend(
            grid, _src(s["h"], edge / s["h"], dev), _src(s["w"], edge / s["w"], dev)
        ),
        "align_corners=True": _blend(
            grid,
            torch.arange(s["h"], dtype=torch.float64, device=dev) * ((edge - 1) / (s["h"] - 1)),
            torch.arange(s["w"], dtype=torch.float64, device=dev) * ((edge - 1) / (s["w"] - 1)),
        ),
        # The grid read from row 0, the cls token included.
        "row 0 not skipped": _blend(
            table.double()[: edge * edge].reshape(edge, edge, -1), src_h, src_w
        ),
        # The qwen4_exp entry's row order: 2x2 merge blocks row-major.
        "spatial-merge order": (
            _blend(grid, src_h, src_w)
            .reshape(s["h"] // 2, 2, s["w"] // 2, 2, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, table.shape[1])
        ),
        # torch's identity at the table's own size taken per axis instead of whole-op: the
        # 128 side would copy the table instead of following the + 0.1 grid.
        "identity on the matching side only": _blend(grid, src_h, _src(s["w"], 1.0, dev)),
    }
    for name, wrong in wrongs.items():
        wrong = wrong.reshape(-1, table.shape[1]).to(dtype)
        _, worst, _ = _measure(out, wrong)
        separation = (out.float() - wrong.float()).abs().max().item()
        print(f"  {name}: {worst:.3e} row-scale ulp, max abs separation {separation:.3f}")
        with pytest.raises(AssertionError):
            assert_within_ulp(out, wrong, _ELEMENT_ULP, _RMS_ULP)
        assert worst > 8.0, name

    # The qwen4_exp entry's dtype-accumulating arithmetic. A different arithmetic, not a
    # different grid: it sits inside this band (the printed value), so it is a measurement
    # and not a refusal -- both it and the op are within a few ulp of the exact value.
    hf = src_h.floor().to(torch.int64)
    wf = src_w.floor().to(torch.int64)
    hc = (hf + 1).clamp_max(edge - 1)
    wc = (wf + 1).clamp_max(edge - 1)
    dh = (src_h - hf.double())[:, None, None].to(dtype).double()
    dw = (src_w - wf.double())[None, :, None].to(dtype).double()
    top = (1.0 - dw) * grid[hf][:, wf] + dw * grid[hf][:, wc]
    bot = (1.0 - dw) * grid[hc][:, wf] + dw * grid[hc][:, wc]
    bf16_weights = ((1.0 - dh) * top + dh * bot).reshape(-1, table.shape[1]).to(dtype)
    _, worst, _ = _measure(out, bf16_weights)
    print(
        f"  bf16 corner weights (qwen4_exp arithmetic): {worst:.3e} row-scale ulp, inside the band"
    )


def test_metadata_misuse_raises() -> None:
    """Each `is_valid` refusal, after showing the wrong answer it prevents.

    The op's `__call__` goes straight to the kernel (the guard is a test-time check, see
    `_validating.py`), so the unguarded call is the demonstration.
    """
    s = _spec(1, 32, 32)
    edge, hidden, dtype = s["position_edge"], s["hidden"], s["dtype"]
    args = (s["t"], s["h"], s["w"], edge, dtype)
    g = torch.Generator(device=DEV).manual_seed(8)
    table = torch.randn(edge * edge + 1, hidden, generator=g, device=DEV).to(dtype)

    # A column slice of a wider buffer: shape right, row stride 2 * hidden, last-dim
    # stride 1 -- the stride the kernel assumes is hidden.
    base = torch.randn(edge * edge + 1, 2 * hidden, generator=g, device=DEV).to(dtype)
    sliced = base[:, :hidden]
    assert sliced.stride() == (2 * hidden, 1) and not sliced.is_contiguous()
    wrong = op(sliced, *args)
    ref = op.reference(sliced, *args)
    separation = (wrong.float() - ref.float()).abs().max().item()
    print(f"  strided table: max abs separation {separation:.3f} from the same values")
    assert wrong.shape == ref.shape and separation > 0.5
    with validating(op):
        with pytest.raises(AssertionError):
            op(sliced, *args)

    # A grid side that does not describe the table: 64 on a 128-edge table, read out of
    # the table's first 4096 rows.
    wrong = op(table, s["t"], s["h"], s["w"], edge // 2, dtype)
    ref = op.reference(table, *args)
    separation = (wrong.float() - ref.float()).abs().max().item()
    print(f"  edge 64 on the 128-edge table: max abs separation {separation:.3f}")
    assert wrong.shape == ref.shape and separation > 0.5
    with validating(op):
        with pytest.raises(AssertionError):
            op(table, s["t"], s["h"], s["w"], edge // 2, dtype)

    # A 3-D table: `hidden` becomes shape[1], so the output's width is that dimension.
    # Shape [2, 9, hidden] with edge 8 keeps every read inside the allocation; a table
    # passed as [1, edge**2 + 1, hidden] instead makes shape[1] the row count and reads
    # far past the tensor.
    three_d = torch.randn(2, 9, hidden, generator=g, device=DEV).to(dtype)
    wrong = op(three_d, 1, 4, 4, edge // 16, dtype)
    print(f"  3-D table: output {tuple(wrong.shape)}, not ({4 * 4}, {hidden})")
    assert wrong.shape != (4 * 4, hidden)
    with validating(op):
        with pytest.raises(AssertionError):
            op(three_d, 1, 4, 4, edge // 16, dtype)

    # An empty grid side: no rows, no error, where the HF path's `F.interpolate` raises on
    # a zero output size.
    wrong = op(table, 1, 0, 32, edge, dtype)
    assert wrong.shape == (0, hidden)
    with pytest.raises(RuntimeError):
        _hf_path(table, 1, 0, 32, edge, dtype)
    with validating(op):
        with pytest.raises(AssertionError):
            op(table, 1, 0, 32, edge, dtype)
