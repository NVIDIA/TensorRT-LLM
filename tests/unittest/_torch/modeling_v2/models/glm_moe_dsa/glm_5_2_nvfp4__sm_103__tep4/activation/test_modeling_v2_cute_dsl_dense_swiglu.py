# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_dense_swiglu entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_dense_swiglu as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.cute_dsl_dense_swiglu


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    Elementwise, one bf16 rounding on each side. The kernel's sigmoid uses an
    approximate exp2 and reciprocal (a few fp32 ulps); the reference is fp64.
    So the results differ only where the fp32 value sits that close to a bf16
    rounding midpoint, by one bf16 ulp of that element: up to 2 units of
    `ULP[bf16]` against the row scale when it is the row's largest. Those
    flips are rare enough that their RMS stays far below half an ulp. Swapped
    gate/up halves or a wrong activation miss by O(1).
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # The gate/up projection's output scale: a few units, so the sigmoid sees
    # both tails.
    source = (3.0 * torch.randn(cell.spec["rows"], 6144, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    before = source.clone()
    out = op(source, compiled)
    assert out.shape == (cell.spec["rows"], 3072) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    assert torch.equal(source, before), "the op must not write its input"
    _compare(out, op.reference(source, compiled))
