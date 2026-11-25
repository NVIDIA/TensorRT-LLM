# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for engram_lookup_packed (bitwise against the byte-copy reference) and engram_packed_projection (relative RMS
1e-2 against the float64 product of the exact operands)."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.kernels.blackwell.engram.lookup import packed_row_bytes
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E501
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_packed import (  # noqa: E501
    engram_lookup_packed,
    engram_packed_projection,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 kernels",
)
BLOCK = 32


def _rel(a, b):
    return (
        (a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()
    ).item()


def _table(rows, dim, gen, mxfp4=False):
    if mxfp4:  # E2M1 pairs, dim / 2 bytes a row
        weight = torch.randint(
            0, 256, (rows, dim // 2), generator=gen, device=DEV, dtype=torch.uint8
        )
    else:
        weight = (
            (torch.randn(rows, dim, generator=gen, device=DEV) * 4)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    scale = torch.randint(
        118, 134, (rows, dim // BLOCK), generator=gen, device=DEV, dtype=torch.uint8
    )
    return weight, scale


@pytest.mark.parametrize(
    "cell", engram_lookup_packed.CELLS, ids=[c.why[:44] for c in engram_lookup_packed.CELLS]
)
def test_lookup_cells(cell, certifying):  # noqa: F811
    certifying(engram_lookup_packed)
    s = cell.spec
    gen = gen_for(cell)
    weight, scale = _table(s["rows"], s["dim"], gen, s.get("mxfp4", False))
    bucket = s["rows"] // s["heads"]
    offsets = (torch.arange(s["heads"], device=DEV) * bucket)[
        s["head_start"] : s["head_start"] + s["local_heads"]
    ]
    offsets = offsets.contiguous()
    ids = torch.randint(
        0, bucket, (s["tokens"], s["heads"]), generator=gen, device=DEV, dtype=torch.int64
    )
    kw = dict(
        head_start=s["head_start"],
        local_heads=s["local_heads"],
        block_size=BLOCK,
        num_ctas=64,
        mxfp4=s.get("mxfp4", False),
    )
    got = engram_lookup_packed(weight, scale, ids, offsets, **kw)
    ref = engram_lookup_packed.reference(weight, scale, ids, offsets, **kw)
    k = s["local_heads"] * s["dim"]
    assert got.dtype == torch.uint8 and got.shape == ref.shape
    assert torch.equal(
        got[:, : k + k // BLOCK], ref[:, : k + k // BLOCK]
    )  # the row padding is not written


@pytest.mark.parametrize(
    "cell", engram_packed_projection.CELLS, ids=[c.why[:44] for c in engram_packed_projection.CELLS]
)
def test_projection_cells(cell, certifying):  # noqa: F811
    certifying(engram_packed_projection)
    s = cell.spec
    gen = gen_for(cell)
    k = s["heads"] * s["dim"]
    local_k = k // s["sources"]
    rb = packed_row_bytes(s["heads"] // s["sources"], s["dim"], BLOCK)
    total = 2 * s["m"] if s.get("slice") else s["m"]
    packed = torch.zeros(s["sources"], total, rb, dtype=torch.uint8, device=DEV)
    vals = (torch.randn(s["sources"], total, local_k, generator=gen, device=DEV) * 4).to(
        torch.float8_e4m3fn
    )
    packed[:, :, :local_k] = vals.view(torch.uint8)
    packed[:, :, local_k : local_k + local_k // BLOCK] = torch.randint(
        120,
        130,
        (s["sources"], total, local_k // BLOCK),
        generator=gen,
        device=DEV,
        dtype=torch.uint8,
    )
    rows = packed[:, s["m"] : 2 * s["m"]] if s.get("slice") else packed
    weight = (torch.randn(s["n"], k, generator=gen, device=DEV) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(
        118, 126, (s["n"], k // BLOCK), generator=gen, device=DEV, dtype=torch.uint8
    )
    alpha = torch.ones(1, dtype=torch.float32, device=DEV)
    got = engram_packed_projection(rows, weight, scale, mxfp8_swizzled_scales(scale), alpha)
    ref = engram_packed_projection.reference(rows, weight, scale, None, alpha)
    assert got.shape == (s["m"], s["n"]) and got.dtype == torch.bfloat16
    assert _rel(got, ref) < 1e-2, _rel(got, ref)
