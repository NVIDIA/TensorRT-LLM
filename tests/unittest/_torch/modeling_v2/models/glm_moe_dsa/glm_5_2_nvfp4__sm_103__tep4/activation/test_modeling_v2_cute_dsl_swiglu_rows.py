# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_swiglu_rows entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_swiglu_rows as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.cute_dsl_swiglu_rows


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """2 ulp per element against the row scale, 0.5 ulp relative RMS. Each output is
    one fp32 product rounded once; the kernel's exp is the fast one, a few fp32 ulps
    from the fp64 reference, which moves a bf16 result by at most one ulp where the
    fp32 value sits on a rounding midpoint. Swapping gate and up, or a wrong width
    split, moves every element."""
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = cell.spec["rows"], cell.spec["width"]
    src = (torch.randn(rows, 2 * width, generator=g, device="cuda") * 4).to(torch.bfloat16)
    assert entry.takes(src)
    out = op(src, compiled)
    assert out.is_contiguous() and out.shape == (rows, width)
    ref = op.reference(src, compiled)
    assert ref.float().abs().sum() > 0, "a zero reference proves nothing"
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
