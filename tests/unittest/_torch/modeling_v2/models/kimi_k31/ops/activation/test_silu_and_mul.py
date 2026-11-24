# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 silu_and_mul entry: bitwise the engine SwiGLU call, close to the reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.silu_and_mul import (  # noqa: E501
    silu_and_mul as op,
)
from tensorrt_llm._torch.modules.swiglu import swiglu

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "silu_and_mul requires a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = cell.spec["rows"], cell.spec["width"]
    x = (torch.randn(rows, 2 * width, generator=g, device="cuda") * 3).to(torch.bfloat16)
    with validating(op):
        out = op(x)
    # The call the engine's BF16 GatedMLP made: the same kernel, bit for bit.
    assert torch.equal(out.view(torch.int16), swiglu(x).view(torch.int16))
    torch.testing.assert_close(out, op.reference(x), rtol=1.6e-2, atol=1e-2)
