# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window decode with Triton-staged operands (``rubin_dflash2_window_fp32``).

The entry claims ``sparse_mla_rows_eager``'s output bit for bit, so every gate here is equality,
never a band. Each cell builds a paged history whose window pages are random pages of the pool
and whose other page-table columns point at page 0 (finite values, never read in a live column).

GPU (SM107): the DEP16 generation cell against ``reference`` bit for bit.
"""

from __future__ import annotations

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.spec.rubin_dflash2_window_fp32 import (  # noqa: E501
    rubin_dflash2_window_fp32 as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (  # noqa: E501
    dflash2_window_rows,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

gpu = on_sm(107)
# The FP8 pool's largest cell of the sample: DEP16 generation, 12 requests at draft 7, 96 heads.
_CELLS = sample_cells(op.CELLS, size_key="heads", group_keys=("pool",))[2:3]


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode())


def build(spec: dict, seed: int, device):
    """``(query, pool_rows, rows)``: window pages drawn at random from the pool, the rest page 0."""
    g = torch.Generator().manual_seed(seed)
    lengths = list(spec["lengths"])
    batch, blk, page = len(lengths), spec["block_size"], spec["page_size"]
    dim = spec["latent"] + spec["rope"]
    span = (spec["window"] + blk) // page + 2  # pages a window and its block can touch
    columns = (max(lengths) + blk) // page + 1
    num_pages = 1 + batch * span
    table = torch.zeros(batch, columns, dtype=torch.int64)
    perm = torch.randperm(num_pages - 1, generator=g) + 1
    for b, length in enumerate(lengths):
        lo = max(0, length - spec["window"]) // page
        hi = min(columns, (length + blk - 1) // page + 1)
        table[b, lo:hi] = perm[b * span : b * span + (hi - lo)]
    pool = 0.5 * torch.randn(num_pages * page, dim, generator=g)
    pool = pool.to(torch.float8_e4m3fn if spec["pool"] == "fp8" else torch.bfloat16)
    query = (0.5 * torch.randn(batch * blk, spec["heads"], dim, generator=g)).to(torch.bfloat16)
    ctx = torch.tensor(lengths, dtype=torch.int64)
    rows, _ = dflash2_window_rows(ctx, table, blk, page, spec["window"], spec["window"])
    return query.to(device), pool.to(device), rows.to(device)


def _args(spec, query, pool, rows):
    return (query, pool, rows, spec["scale"], spec["latent"], spec["kv_scale"])


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)


@gpu
@pytest.mark.parametrize("cell", _CELLS, ids=["dep16"])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    args = _args(spec, *build(spec, _seed(cell), "cuda"))
    assert torch.equal(_bits(op(*args)), _bits(op.reference(*args))), "the staged decode differs"
