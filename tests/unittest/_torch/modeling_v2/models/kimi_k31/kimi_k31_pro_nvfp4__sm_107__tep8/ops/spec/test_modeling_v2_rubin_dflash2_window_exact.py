# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 FP32 window attention entry (``rubin_dflash2_window_exact``), on CPU.

A sample of the cells each builds a paged draft history and the window rows of each request's
block, as ``dflash2_window_rows`` lays them out. The entry is the drafter's FP32 decode: it equals its
float64 ``reference`` within the output dtype's rounding.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.ops.spec.rubin_dflash2_window_exact import (  # noqa: E501
    rubin_dflash2_window_exact as op,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.cpu_only

#: A short window keeps the CPU cells small; the arithmetic does not depend on its length.
_WINDOW = 96


def _inputs(spec: dict, seed: int):
    gen = torch.Generator().manual_seed(seed)
    batch, block, heads = spec["batch"], spec["block_size"], spec["heads"]
    page, width = spec["page_size"], spec["latent"] + spec["rope"]
    ctx_len = torch.randint(0, 3 * _WINDOW, (batch,), generator=gen, dtype=torch.int32)
    pages_per_request = -(-(int(ctx_len.max()) + block) // page)
    num_pages = batch * pages_per_request + 1
    page_tables = (
        torch.randperm(num_pages - 1, generator=gen)[: batch * pages_per_request]
        .view(batch, pages_per_request)
        .to(torch.int32)
    )
    rows = _window_rows(ctx_len, page_tables, block, page)
    pool, kv_scale = _pool(torch.randn(num_pages * page, width, generator=gen), spec["pool"])
    query = torch.randn(batch * block, heads, width, generator=gen).to(torch.bfloat16)
    return query, pool, rows, spec["scale"], spec["latent"], kv_scale


def _window_rows(ctx_len, page_tables, block: int, page: int) -> torch.Tensor:
    """Each block query's window as global pool rows (``page * page_size + offset``), ascending
    positions, then -1: the query at position ``p`` attends ``[max(0, p - W + 1), p]``."""
    out = torch.full((ctx_len.shape[0] * block, _WINDOW), -1, dtype=torch.int32)
    for b, length in enumerate(ctx_len.tolist()):
        for j in range(block):
            p = length + j
            for col, pos in enumerate(range(max(0, p - _WINDOW + 1), p + 1)):
                out[b * block + j, col] = int(page_tables[b, pos // page]) * page + pos % page
    return out


def _pool(values: torch.Tensor, kind: str) -> tuple[torch.Tensor, float]:
    """The pool rows as the cache stores them, and their dequant scale."""
    if kind == "fp8":
        return (values * 4).to(torch.float8_e4m3fn), 0.25
    return values.to(torch.bfloat16), 1.0


CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_cell_is_the_fp32_decode_and_matches_the_reference(cell):
    args = _inputs(dict(cell.spec), seed=len(cell.why))
    op.is_valid(*args)
    # Both round to BF16 once; FP32 accumulation moves a value by far less than a BF16 unit.
    torch.testing.assert_close(
        op(*args).float(), op.reference(*args).float(), rtol=2**-7, atol=2**-12
    )
