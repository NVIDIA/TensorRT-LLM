# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's bf16_low_m_gemm entry: every cell (each plan tactic kind) against
the FP64 product, the product in the caller's ``out``, and the guard's refusal of an aliased ``out``."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_low_m_gemm import (  # noqa: E501
    bf16_low_m_gemm as op,
)

__extra_import_path__ = ["../../../..", "."]
from _bf16_band import assert_bf16_gemm_close  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the low-M CuTe DSL kernels need an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, n, k = spec["rows"], spec["n"], spec["k"]
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    return x, weight, out


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no_pdl"])
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, pdl) -> None:
    x, weight, out = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    tactic = cell.spec["tactic"]
    with validating(op):
        result = op(x, weight, out, tactic, pdl)
    torch.cuda.synchronize()
    assert result is out, "the op writes the caller's out and returns it"
    assert_bf16_gemm_close(out, op.reference(x, weight, out, tactic, pdl), x, weight)


def test_is_valid_refuses_an_out_aliasing_an_operand() -> None:
    x, weight, _ = _build(CELLS[0].spec, seed=1)
    bank = torch.empty(x.shape[0], x.shape[1] + weight.shape[0], dtype=x.dtype, device="cuda")
    aliased = bank[:, : weight.shape[0]]
    with pytest.raises(ValueError, match="share memory with x"):
        op.is_valid(bank[:, weight.shape[0] :], weight, aliased, CELLS[0].spec["tactic"], True)
    op.is_valid(x, weight, torch.empty_like(aliased), CELLS[0].spec["tactic"], True)
