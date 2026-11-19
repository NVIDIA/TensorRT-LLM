# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 window-rows entry (CuTe DSL).

A sample of the certified cells is gated element for element against the entry's reference (the drafter's
torch index lists): the same int32 rows, ``-1`` padding and lengths. Page tables hold random
distinct page ids, so a wrong page column, offset or window edge moves entries.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (  # noqa: E501
    dflash2_window_rows as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, 107, reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x")


def _ctx_lengths(kind: str, batch: int, g: torch.Generator) -> torch.Tensor:
    if kind == "short":
        return torch.full((batch,), 3)
    if kind == "long":
        return torch.full((batch,), 5000)
    if kind == "edge":
        return torch.tensor([1018, 1019, 1020, 1023, 1024, 0, 4096])[:batch]
    lengths = torch.randint(0, 20000, (batch,), generator=g)
    lengths[0] = 0
    return lengths


def _build(spec, seed):
    g = torch.Generator().manual_seed(seed)
    batch, width, page_size = spec["batch"], spec["width"], spec["page_size"]
    ctx_len = _ctx_lengths(spec["ctx"], batch, g)
    ring = spec["ring"]
    if ring:
        # A ring slot holds ring_rows rows whatever the context length.
        pages = ring // page_size + spec["spare_pages"]
    else:
        pages = max(
            1, (int(ctx_len.max()) + width + page_size - 1) // page_size + spec["spare_pages"]
        )
    tables = torch.randperm(1 << 20, generator=g)[: batch * pages].view(batch, pages)
    if spec["table"] == "strided":
        wide = torch.full((2 * batch, pages + 5), -7, dtype=torch.int64)
        wide[::2, :pages] = tables
        tables = wide.cuda()[::2, :pages]
    else:
        tables = tables.cuda()
    dtype = spec["dtype"]
    return (
        ctx_len.to(dtype).cuda(),
        tables.to(dtype) if tables.dtype != dtype else tables,
        width,
        page_size,
        spec["window"],
        spec["index_width"],
        ring,
    )


CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    for got, want in zip(op(*args), op.reference(*args)):
        assert got.dtype == want.dtype and torch.equal(got, want)
