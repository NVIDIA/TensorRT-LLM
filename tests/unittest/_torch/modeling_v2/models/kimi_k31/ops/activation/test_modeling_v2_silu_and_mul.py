# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 silu_and_mul entry: close to the reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.silu_and_mul import (  # noqa: E501
    silu_and_mul as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "silu_and_mul requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = cell.spec["rows"], cell.spec["width"]
    x = (torch.randn(rows, 2 * width, generator=g, device="cuda") * 3).to(torch.bfloat16)
    with validating(op):
        out = op(x)
    torch.testing.assert_close(out, op.reference(x), rtol=1.6e-2, atol=1e-2)
