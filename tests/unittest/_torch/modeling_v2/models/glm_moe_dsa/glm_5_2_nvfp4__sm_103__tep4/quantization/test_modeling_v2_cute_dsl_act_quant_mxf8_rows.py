# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_act_quant_mxf8_rows entry.

The operand starts as sentinel bytes, so a byte written outside the rows (or the padding
scale rows) the entry owns, or one it misses, fails the byte-for-byte comparison."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_act_quant_mxf8_rows as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_act_quant_mxf8_rows


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled) -> None:
    row0, rows, k = cell.spec["row0"], cell.spec["rows"], cell.spec["k"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # Per-block magnitudes from 2^-10 to 2^5: the scales vary block by block.
    x = torch.randn(rows, k, generator=g, device="cuda")
    exp = torch.randint(-10, 6, (rows, k // 32), generator=g, device="cuda").float()
    x = (x * torch.exp2(exp).repeat_interleave(32, dim=1)).to(torch.bfloat16)
    t = row0 + rows + 3
    q = torch.full((t, k), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.full((entry.sf_bytes(t, k),), 0x33, dtype=torch.uint8, device="cuda")
    ref_q, ref_sf = op.reference(x, q, sf, row0, compiled)
    with validating(op):
        got_q, got_sf = op(x, q, sf, row0, compiled)
    torch.cuda.synchronize()
    assert got_q is q and got_sf is sf
    assert torch.equal(q, ref_q), f"{int((q != ref_q).sum())} operand bytes differ"
    assert torch.equal(sf, ref_sf), f"{int((sf != ref_sf).sum())} scale bytes differ"


def test_is_valid_refuses_rows_past_the_operand(compiled) -> None:
    x = torch.zeros(8, 4096, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(100, 4096, dtype=torch.uint8, device="cuda")
    sf = torch.empty(entry.sf_bytes(100, 4096), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="must fit"):
        with validating(op):
            op(x, q, sf, 95, compiled)
