# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for nvfp4_gemm, using the shared operator comparison."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = ["../../../../gemm", "../../../.."]
import test_modeling_v2_nvfp4_gemm as shared  # noqa: E402

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/full_model_nvfp4__sm_103__tp4": tuple(
        Cell(
            why=f"beam TP4 expert {name} {k}->{n}, {m} rows",
            spec=dict(m=m, k=k, n=n, bias=False, backend="cutlass"),
        )
        for name, k, n in (("gate/up", 5120, 640), ("down", 640, 5120))
        for m in (1, 17, 129)
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
