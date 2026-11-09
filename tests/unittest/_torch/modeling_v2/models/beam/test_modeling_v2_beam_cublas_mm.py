# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for cublas_mm, using the shared operator comparison."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm as op

__extra_import_path__ = ["../../gemm", "../.."]
import test_modeling_v2_cublas_mm as shared  # noqa: E402

_BEAM_DENSE_BLOCK = tuple(
    Cell(
        why=f"beam dense {name} {k}->{n}, {m} tokens",
        spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
    )
    for name, k, n in (
        ("q/gate", 5120, 6144),
        ("k/v", 5120, 1024),
        ("o", 6144, 5120),
        ("gate/up", 5120, 20480),
        ("down", 20480, 5120),
    )
    for m in (1, 17, 129)
)

_BEAM_HEAD = tuple(
    Cell(
        why=f"beam head 5120->151936, {m} tokens",
        spec=dict(m=m, k=5120, n=151936, dtype=torch.bfloat16, bias=False, out_dtype=None),
    )
    for m in (1, 17, 129)
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/full_model_nvfp4__sm_103__tp4": tuple(
        Cell(
            why=f"beam TP4 {name} {k}->{n}, {m} tokens",
            spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
        )
        for name, k, n in (
            ("q/gate", 5120, 1536),
            ("k/v", 5120, 256),
            ("o", 1536, 5120),
            ("dense gate/up", 5120, 5120),
            ("dense down", 5120, 5120),
            ("head", 5120, 37984),
        )
        for m in (1, 17, 129)
    ),
    "beam/dense_block": _BEAM_DENSE_BLOCK,
    "beam/six_layer_model": _BEAM_HEAD,
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
