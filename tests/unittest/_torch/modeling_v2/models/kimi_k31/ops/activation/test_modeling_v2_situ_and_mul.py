# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 situ_and_mul entry: a sample of its certified cells against its reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.situ_and_mul import situ_and_mul

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)

# Both sides compute in fp32 and round once to bf16; libdevice tanh differs from torch.tanh in the
# last fp32 bits, which moves the rounded bf16 result by at most one ulp.
RTOL = 2 * torch.finfo(torch.bfloat16).eps


CELLS = sample_cells(situ_and_mul.CELLS)[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(situ_and_mul)
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = spec["shape"]
    # The scale of a gate/up projection output: well inside the soft caps, with a tail reaching them.
    x = (torch.randn(rows, width, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    if spec.get("saturated"):
        half = width // 2
        for row, (gate, up) in enumerate(((1e4, 1e4), (1e4, -1e4), (-1e4, 1e4), (0.0, 1e4))):
            x[row, :half], x[row, half:] = gate, up
    out = situ_and_mul(x, spec["beta"], spec["linear_beta"])
    ref = situ_and_mul.reference(x, spec["beta"], spec["linear_beta"])
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=1e-5)
