# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_moe_head_gemm entry: router logits and routed latent in one launch, bit for bit the
two-GEMM head (dsv3_router_gemm_op's cuBLAS router, k31_fused_a_gemm's latent down projection) on a sample of its
cells (``sample_cells``)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_head_gemm import (  # noqa: E501
    LATENT_ROWS,
    ROUTER_ROWS,
    k31_moe_head_gemm,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import on_sm, sample_cells

pytestmark = on_sm(103)

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


CELLS = sample_cells(k31_moe_head_gemm.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell) -> None:
    x, w = _build(cell.spec["rows"], zlib.crc32(cell.why.encode()) % 2**31)
    k31_moe_head_gemm.is_valid(x, w)
    _assert_bitwise(x, w)
