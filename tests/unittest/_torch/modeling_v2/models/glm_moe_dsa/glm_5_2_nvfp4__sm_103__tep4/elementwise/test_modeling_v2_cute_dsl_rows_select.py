# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_rows_select entry: rows move bitwise, rows not
named stay as they were."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_rows_select as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_rows_select


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


def _rows_of(spec, g) -> torch.Tensor:
    n, src = spec["sel"], spec["src_rows"]
    if spec["scatter"]:
        return torch.randperm(src, generator=g, device="cuda")[:n].to(torch.int32)
    return torch.randint(0, src, (n,), generator=g, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    dt = torch.float8_e4m3fn if spec["dtype"] == "float8" else torch.bfloat16
    width = spec["width"] // torch.empty(0, dtype=dt).element_size()
    rows = _rows_of(spec, g)
    big = torch.randint(0, 256, (spec["src_rows"], spec["width"]), generator=g, device="cuda",
                        dtype=torch.uint8).view(dt).view(spec["src_rows"], width)  # fmt: skip
    small = torch.randint(0, 256, (spec["sel"], spec["width"]), generator=g, device="cuda",
                          dtype=torch.uint8).view(dt).view(spec["sel"], width)  # fmt: skip
    x, y = (small, big.clone()) if spec["scatter"] else (big, small.clone())
    ref = op.reference(x, rows, y, compiled, scatter=spec["scatter"])
    with validating(op):
        assert op(x, rows, y, compiled, scatter=spec["scatter"]) is y
    torch.cuda.synchronize()
    assert torch.equal(y.view(torch.uint8), ref.view(torch.uint8)), "rows differ"


def test_is_valid_refuses_an_index_past_the_source(compiled) -> None:
    x = torch.zeros(10, 64, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda")
    rows = torch.tensor([1, 10], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="outside"):
        with validating(op):
            op(x, rows, y, compiled)
