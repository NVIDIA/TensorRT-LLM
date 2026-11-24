# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's bf16_linear entry: every cell against the FP64 product, within the
band of one FP32-accumulated BF16 GEMM."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_linear import (  # noqa: E501
    bf16_linear as op,
)

__extra_import_path__ = ["../../../..", "."]
from _bf16_band import assert_bf16_gemm_close  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, n, k = spec["rows"], spec["n"], spec["k"]
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    bank = spec["bank"]
    if bank is None:
        weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    else:
        bank_rows, offset = bank
        full = torch.randn(bank_rows, k, generator=g, device="cuda") * k**-0.5
        weight = full.to(torch.bfloat16)[offset : offset + n]
    return x, weight


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    x, weight = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(x, weight)
    torch.cuda.synchronize()
    assert_bf16_gemm_close(out, op.reference(x, weight), x, weight)
