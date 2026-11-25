# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_hidden_add entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_hidden_add as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_hidden_add
HIDDEN = entry.HIDDEN


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """Bit-exact: one fp32 add of two bf16 values and one rounding, which is what
    torch.add on bf16 does."""
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows = cell.spec["rows"]

    def operand(pitch: int) -> torch.Tensor:
        wide = torch.randn(rows, pitch, generator=g, device="cuda").to(torch.bfloat16)
        return wide[:, pitch - HIDDEN :]

    a, b = operand(cell.spec["pitch_a"]), operand(cell.spec["pitch_b"])
    assert entry.takes(a, b)
    out = op(a, b, compiled)
    assert out.is_contiguous() and out.shape == (rows, HIDDEN)
    ref = op.reference(a, b, compiled)
    assert ref.float().abs().sum() > 0, "a zero reference proves nothing"
    assert torch.equal(out, ref)
