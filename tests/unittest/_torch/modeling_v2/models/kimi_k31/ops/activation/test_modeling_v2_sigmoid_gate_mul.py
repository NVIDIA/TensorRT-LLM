# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 sigmoid_gate_mul entry: a sample of its cells bitwise against the reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.sigmoid_gate_mul import (  # noqa: E501
    k31_sigmoid_gate_mul as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "sigmoid_gate_mul requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, cols, stride = spec["rows"], spec["cols"], spec["gate_row_stride"]
    x = torch.randn(rows, cols, generator=g, device="cuda").to(torch.bfloat16)
    wide = (torch.randn(rows, stride or cols, generator=g, device="cuda") * 4).to(torch.bfloat16)
    gate = wide[:, (stride or cols) - cols :]
    return x, gate


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    ref = op.reference(*args)
    # Bitwise: the kernel's claim is the two PyTorch kernels' exact bits.
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
