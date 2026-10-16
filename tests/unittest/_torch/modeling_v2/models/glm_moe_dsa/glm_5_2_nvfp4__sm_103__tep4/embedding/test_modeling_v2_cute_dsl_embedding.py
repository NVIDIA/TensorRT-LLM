# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_embedding entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.embedding import (  # noqa: E402, E501
    cute_dsl_embedding as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_embedding
VOCAB, HIDDEN = 154880, 6144


@pytest.fixture(scope="module")
def table():
    """The full-size table (1.9 GB) and its runner, built once for every cell."""
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    return weight, entry.build(weight)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, table, certifying) -> None:  # noqa: F811 — fixture by name
    """Bit-exact: the op copies rows (or writes zeros) and does no arithmetic."""
    certifying(op)
    weight, runner = table
    rows = cell.spec["rows"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    ids = torch.randint(0, VOCAB, (rows,), generator=g, device="cuda", dtype=torch.int32)
    if cell.spec.get("invalid"):
        # Every third id out of range, on both sides, and the two boundary ids.
        ids[0::3] = -1 - ids[0::3] % 7
        ids[1::6] = VOCAB + ids[1::6] % 7
        ids[2] = VOCAB - 1
        ids[5] = 0
    out = op(ids, weight, runner)
    assert out.shape == (rows, HIDDEN) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert torch.equal(out, op.reference(ids, weight, runner))


def test_guard_refuses_a_runner_built_on_another_table(table) -> None:
    """The runner gathers from its own table and ignores the one passed."""
    weight, runner = table
    other = weight.clone()
    ids = torch.zeros(4, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="different weight"):
        with validating(op):
            op(ids, other, runner)
