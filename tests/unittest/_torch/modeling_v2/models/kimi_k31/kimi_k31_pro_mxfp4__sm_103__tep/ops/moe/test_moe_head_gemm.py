# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_moe_head_gemm entry: router logits and routed latent in one launch, bit for bit the
two-GEMM head (dsv3_router_gemm_op's cuBLAS router, k31_fused_a_gemm's latent down projection) at every row it
serves."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_head_gemm import (  # noqa: E501
    LATENT_ROWS,
    MAX_ROWS,
    MIN_ROWS,
    ROUTER_ROWS,
    head_config,
    k31_moe_head_gemm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="certified on sm_103",
)

HIDDEN = 7168


def _build(rows: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    w = (
        torch.randn(ROUTER_ROWS + LATENT_ROWS, HIDDEN, generator=gen, device="cuda") * HIDDEN**-0.5
    ).to(torch.bfloat16)
    return x, w


def _assert_bitwise(x: torch.Tensor, w: torch.Tensor) -> None:
    logits, latent = k31_moe_head_gemm(x, w)
    ref_logits, ref_latent = k31_moe_head_gemm.reference(x, w)
    rows = x.shape[0]
    assert logits.shape == (rows, ROUTER_ROWS) and logits.dtype == torch.float32
    assert latent.shape == (rows, LATENT_ROWS) and latent.dtype == torch.bfloat16
    assert torch.equal(logits.view(torch.int32), ref_logits.view(torch.int32)), (
        f"router logits at {rows} rows"
    )
    assert torch.equal(latent.view(torch.int16), ref_latent.view(torch.int16)), (
        f"latent at {rows} rows"
    )


@pytest.mark.parametrize(
    "cell", k31_moe_head_gemm.CELLS, ids=[c.why[:48] for c in k31_moe_head_gemm.CELLS]
)
def test_certified_cells(cell) -> None:
    x, w = _build(cell.spec["rows"], zlib.crc32(cell.why.encode()) % 2**31)
    k31_moe_head_gemm.is_valid(x, w)
    _assert_bitwise(x, w)


def test_every_served_row_is_bitwise() -> None:
    for rows in range(MIN_ROWS, MAX_ROWS + 1):
        x, w = _build(rows, rows)
        _assert_bitwise(x, w)


def test_rows_outside_the_range_are_not_served() -> None:
    assert head_config(MIN_ROWS - 1) is None and head_config(MAX_ROWS + 1) is None
    x, w = _build(MIN_ROWS - 1, 1)
    with pytest.raises(ValueError, match="serves"):
        k31_moe_head_gemm(x, w)
