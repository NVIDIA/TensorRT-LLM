# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 q_a RMSNorm with the launch-time PDL release: bitwise the FlashInfer RMSNorm."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rmsnorm_release import (
    k31_rmsnorm_release as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the RMSNorm requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(rows, cols, row_stride, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wide = torch.randn(rows, row_stride, generator=g, device="cuda").to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(cols, generator=g, device="cuda")).to(torch.bfloat16)
    return wide[:, :cols], w


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    s = cell.spec
    x, w = _build(s["rows"], s["cols"], s["row_stride"], zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(x, w, 1e-6)
    ref = op.reference(x, w, 1e-6)
    assert out.is_contiguous() and out.shape == ref.shape
    assert torch.equal(out.view(torch.int16), ref.contiguous().view(torch.int16))
