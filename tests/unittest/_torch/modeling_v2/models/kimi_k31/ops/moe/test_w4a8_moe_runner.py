# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 pinned-tactic W4A8 routed-expert entry: every cell bitwise against the
catalog runner entry with its autotuned tactic, the pin table's tactics bitwise at the targets' row counts, and the
routed experts' runner entry (``k31_runner_moe``: the table's tactic where it has one, else the catalog entry)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    k31_runner_moe,
    k31_w4a8_moe_runner,
    pinned_tactic,
)
from tensorrt_llm._utils import is_sm_100f

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

op = k31_w4a8_moe_runner

if not torch.cuda.is_available() or not is_sm_100f():
    pytest.skip("the TRTLLM-Gen MXFP4 x MXFP8 runner needs an SM10x GPU", allow_module_level=True)

# The rows the runner sees per rank at the targets that call it (each cell runs the table's tactic for its rows): the
# TEP target at MoE TP4 x EP8 (TEP32: 8 rows) and MoE TP4 x EP4 (TEP16: 6 / 14 / 24 / 72 rows), the DEP16 target at
# EP16 (96 / 140 / 168 / 224 rows per rank x 16 ranks), with its 56 experts per rank and under per-layer dynamic
# EPLB with 1024 slots (56 resident experts + 8 helper slots: 64 slots per rank).
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__tep": (
        *(
            Cell(why=f"TEP16 rows {r}", spec=dict(rows=r, local=224, inter=768, tactic="table"))
            for r in (6, 14, 24, 72)
        ),
        Cell(why="TEP32 rows 8", spec=dict(rows=8, local=112, inter=768, tactic="table")),
        *(
            Cell(
                why=f"TEP32 rows {r}: table edge",
                spec=dict(rows=r, local=112, inter=768, tactic="table"),
            )
            for r in (16, 48)
        ),
        Cell(
            why="TEP16 rows 192: table edge",
            spec=dict(rows=192, local=224, inter=768, tactic="table"),
        ),
    ),
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__dep": (
        *(
            Cell(
                why=f"DEP16 small batch {r} rows/rank x 16",
                spec=dict(rows=16 * r, local=56, inter=3072, tactic="table"),
            )
            for r in (10, 24)
        ),
        *(
            Cell(
                why=f"DEP16 {r} rows/rank x 16",
                spec=dict(rows=16 * r, local=56, inter=3072, tactic="table"),
            )
            for r in (96, 140, 168, 224)
        ),
        *(
            Cell(
                why=f"DEP16 per-layer dynamic EPLB {r} rows/rank x 16 over 64 slots",
                spec=dict(rows=16 * r, local=64, inter=3072, tactic="table"),
            )
            for r in (10, 24, 96, 224)
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    """Random prepared operands of one rank (any bytes are a valid MXFP4 / UE8M0 operand; the scale
    exponents stay in 2^-9..2^-4 so the BF16 outputs are finite) and random distinct top-16 ids."""
    rows, local, inter = spec["rows"], spec["local"], spec["inter"]
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, device=dev, dtype=torch.uint8)

    def sf(*shape):
        return torch.randint(118, 124, shape, generator=g, device=dev, dtype=torch.uint8)

    x = (torch.randn(rows, HIDDEN, generator=g, device=dev) * 2).to(torch.bfloat16)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
    ids = torch.argsort(torch.rand(rows, NUM_EXPERTS, generator=g, device=dev), dim=1)[:, :TOP_K]
    w = torch.rand(rows, TOP_K, generator=g, device=dev) + 0.5
    w = (w / w.sum(1, keepdim=True)).to(torch.bfloat16)
    tactic = spec["tactic"]
    if tactic == "table":
        tactic = pinned_tactic(local, inter, rows)
        assert tactic is not None, f"no pin for {spec}"
    return (
        xq,
        xsf.view(rows, -1),
        ids.to(torch.int32).contiguous(),
        w.contiguous(),
        nib(local, 2 * inter, HIDDEN // 2),
        sf(local, 2 * inter, HIDDEN // 32),
        torch.full((local,), 4.0, dtype=torch.float32, device=dev),
        torch.full((local,), 25.0, dtype=torch.float32, device=dev),
        nib(local, HIDDEN, inter // 2),
        sf(local, HIDDEN, inter // 32),
        inter,
        inter,
        0,
        local,
        tactic,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    ref = op.reference(*args)
    assert out.shape == (cell.spec["rows"], HIDDEN) and out.dtype is torch.bfloat16
    assert torch.isfinite(ref.float()).all()
    # A tactic changes the token tiling and the CTA schedule, not the per-expert K order: exact.
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), "pinned tactic is not bitwise"


def test_output_buffer() -> None:
    args = _build(dict(rows=1536, local=56, inter=3072, tactic="table"), seed=3)
    out = torch.empty(1536, HIDDEN, dtype=torch.bfloat16, device="cuda")
    got = op(*args, output=out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out, op.reference(*args))


def _runner_moe(spec, seed):
    a = _build(spec, seed)
    xq, xsf, ids, w, w13, w13s, al, be, w2, w2s, inter, vinter, off, local, _ = a
    return a, k31_runner_moe(
        xq, xsf, ids, w, w13, w13s, al, be, w2, w2s, inter, vinter, off, local, 8192, True
    )


def test_runner_entry_pins_the_table_and_falls_back_to_the_engine_op(monkeypatch) -> None:
    """The table's tactic where it lists one (bitwise to the engine op), else the engine op itself."""
    spec = dict(rows=1536, local=56, inter=3072, tactic="table")
    a, out = _runner_moe(spec, seed=11)
    assert torch.equal(out, op.reference(*a))
    tactics = []
    real_call = type(op).__call__
    monkeypatch.setattr(
        type(op),
        "__call__",
        lambda self, *x, **k: tactics.append(x[14]) or real_call(self, *x, **k),
    )
    _runner_moe(spec, seed=11)
    _runner_moe(dict(rows=40, local=56, inter=3072, tactic=(32, 18)), seed=12)
    assert tactics == [pinned_tactic(56, 3072, 1536), None]
