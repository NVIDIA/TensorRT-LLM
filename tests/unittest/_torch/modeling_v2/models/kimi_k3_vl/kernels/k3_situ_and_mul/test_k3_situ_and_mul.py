# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's trtllm::k3_situ_and_mul is a copy of the built-in model's trtllm::situ_and_mul: same bits.

Both ops are registered in one process (the built-in one by importing tensorrt_llm._torch.modules.situ), and every
cell of the situ_and_mul OpWrapper, with PDL on and off, must give identical outputs.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.modules.situ  # noqa: F401  (registers trtllm::situ_and_mul, the comparison)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.situ_and_mul import (
    situ_and_mul,
)

assert torch.cuda.is_available(), "k3_situ_and_mul requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)


def _input(cell) -> torch.Tensor:
    """The situ_and_mul op test's input for ``cell``."""
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows, width = spec["shape"]
    x = torch.randn(
        rows, spec.get("row_width", width), generator=gen, dtype=spec["dtype"], device="cuda"
    ) * spec.get("scale", 4.0)
    x = x[:, :width]
    if spec.get("saturated"):
        x[0, :256], x[0, 256:] = 1e4, 1e4
        x[1, :256], x[1, 256:] = 1e4, -1e4
        x[2, :256], x[2, 256:] = -1e4, 1e4
        x[3, :256], x[3, 256:] = 0.0, 1e4
    return x


@pytest.mark.parametrize("pdl", ("1", "0"))
@pytest.mark.parametrize("cell", situ_and_mul.CELLS, ids=[c.why[:40] for c in situ_and_mul.CELLS])
def test_bitwise_equal_to_builtin_op(cell, pdl, monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", pdl)
    x = _input(cell)
    beta, linear_beta = cell.spec["beta"], cell.spec["linear_beta"]
    family = torch.ops.trtllm.k3_situ_and_mul(x, beta, linear_beta)
    builtin = torch.ops.trtllm.situ_and_mul(x, beta, linear_beta)
    assert family.shape == builtin.shape and family.dtype == builtin.dtype
    assert torch.equal(family.view(torch.uint8), builtin.view(torch.uint8)), cell.why
