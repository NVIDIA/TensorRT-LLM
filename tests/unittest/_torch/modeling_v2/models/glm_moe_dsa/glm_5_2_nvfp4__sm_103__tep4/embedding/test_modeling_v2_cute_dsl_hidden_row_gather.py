# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_hidden_row_gather entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.embedding import (  # noqa: E402, E501
    cute_dsl_hidden_row_gather as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_hidden_row_gather
HIDDEN = entry.HIDDEN


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """Bit-exact: a copy."""
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    n, rows = cell.spec["n"], cell.spec["rows"]
    hidden = torch.randn(n, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    idx = torch.randint(0, n, (rows,), generator=g, device="cuda").to(cell.spec["dtype"])
    assert entry.takes(hidden, idx)
    with validating(op):
        out = op(hidden, idx, compiled)
    assert out.is_contiguous() and out.shape == (rows, HIDDEN)
    assert torch.equal(out, op.reference(hidden, idx, compiled))


def test_is_valid_refuses_an_index_past_the_rows(compiled) -> None:
    """The kernel reads the row with no bound."""
    hidden = torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
    idx = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="must index"):
        with validating(op):
            op(hidden, idx, compiled)
