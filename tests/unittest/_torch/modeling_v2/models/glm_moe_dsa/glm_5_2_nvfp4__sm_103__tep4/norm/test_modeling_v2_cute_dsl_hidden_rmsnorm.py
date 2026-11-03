# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_hidden_rmsnorm entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_hidden_rmsnorm as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

op = entry.cute_dsl_hidden_rmsnorm
HIDDEN = entry.HIDDEN


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """2 ulp per element against the row scale, 0.5 ulp relative RMS: the kernel's
    fp32 path differs from the fp64 reference by its sum-of-squares order and an
    approximate rsqrt, a few fp32 ulps, so a bf16 result moves only where the fp32
    value sits that close to a rounding midpoint (one bf16 ulp of that element, up
    to 2 against the row scale). A wrong eps, weight or row statistic moves every
    element by far more."""
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, pitch = cell.spec["rows"], cell.spec["pitch"]
    wide = (torch.randn(rows, pitch, generator=g, device="cuda") * 3).to(torch.bfloat16)
    x = wide[:, pitch - HIDDEN :]
    weight = (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    assert entry.takes(x, weight)
    before = wide.clone()
    out = op(x, weight, 1e-5, compiled)
    assert out.is_contiguous() and out.shape == (rows, HIDDEN)
    ref = op.reference(x, weight, 1e-5, compiled)
    assert ref.float().abs().sum() > 0, "a zero reference proves nothing"
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
    assert torch.equal(wide, before), "the input was written"


def test_takes_refuses_what_the_kernel_cannot_read() -> None:
    w = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(4, HIDDEN + 1, dtype=torch.bfloat16, device="cuda")[:, 1:]
    assert not entry.takes(x, w), "a misaligned row must go to the stock norm"
    assert not entry.takes(torch.zeros(4, HIDDEN, device="cuda"), w), "fp32 rows"
    assert not entry.takes(torch.zeros(4, 4096, dtype=torch.bfloat16, device="cuda"), w)
