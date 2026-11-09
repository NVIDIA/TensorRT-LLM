# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for flashinfer_rmsnorm, using the shared operator comparison."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = ["../../../../norm", "../../../.."]
import test_modeling_v2_flashinfer_rmsnorm as shared  # noqa: E402
from _validating import certifying  # noqa: E402,F401

_BEAM_DENSE_BLOCK = tuple(
    Cell(
        why=f"beam dense {name}, {tokens} tokens: {rows} rows of {width}",
        spec=dict(shape=(rows, width), dtype=torch.bfloat16, eps=1e-6),
    )
    for tokens in (1, 17, 129)
    for name, rows, width in (
        ("sandwich norm", tokens, 5120),
        ("q_norm", tokens * 48, 128),
        ("k_norm", tokens * 8, 128),
    )
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/full_model_nvfp4__sm_103__tp4": tuple(
        Cell(
            why=f"beam TP4 {name}, {tokens} tokens: {rows} rows of {width}",
            spec=dict(shape=(rows, width), dtype=torch.bfloat16, eps=1e-6),
        )
        for tokens in (1, 17, 129)
        for name, rows, width in (
            ("sandwich/final norm", tokens, 5120),
            ("q_norm", tokens * 12, 128),
            ("k_norm", tokens * 2, 128),
        )
    ),
    "beam/dense_block": _BEAM_DENSE_BLOCK,
}

# A shared shape used by multiple components needs only one kernel comparison.
CELLS = tuple(
    {
        repr(sorted(cell.spec.items())): cell for cells in TARGET_CELLS.values() for cell in cells
    }.values()
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_certified_cells(cell: Cell, certifying) -> None:  # noqa: F811
    assert shared.op is op
    shared.test_certified_cells(cell, certifying)
