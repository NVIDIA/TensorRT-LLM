# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for fp4_quantize, using the shared operator comparison."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = ["../../../../quantization", "../../../.."]
import test_modeling_v2_fp4_quantize as shared  # noqa: E402

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/full_model_nvfp4__sm_103__tp4": tuple(
        Cell(
            why=f"beam TP4 expert activation, {rows} rows at width {width}",
            spec=dict(shape=(rows, width), dtype=torch.bfloat16, vec=16, swizzled=True),
        )
        for width in (5120, 640)
        for rows in (1, 17, 129)
    ),
}

# A shared shape used by multiple components needs only one kernel comparison.
CELLS = tuple(
    {
        repr(sorted(cell.spec.items())): cell for cells in TARGET_CELLS.values() for cell in cells
    }.values()
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_certified_cells(cell: Cell) -> None:
    shared.test_certified_cells(cell)
