# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for flashinfer_silu_and_mul, using the shared operator comparison."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = ["../../../../activation", "../../../.."]
import test_modeling_v2_flashinfer_silu_and_mul as shared  # noqa: E402

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/dense_block": tuple(
        (
            Cell(
                why=f"beam dense block, {tokens} tokens: 2 x 20480",
                spec=dict(shape=(tokens, 40960), dtype=torch.bfloat16),
            )
            for tokens in (1, 17, 129)
        )
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
    assert shared.op is op
    shared.test_certified_cells(cell)
