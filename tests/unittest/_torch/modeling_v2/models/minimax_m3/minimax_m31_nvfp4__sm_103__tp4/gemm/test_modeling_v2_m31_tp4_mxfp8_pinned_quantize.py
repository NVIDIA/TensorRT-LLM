# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_pinned_quantize entry of the MiniMax-M3.1 TP4 target (QPIN): every tactic is bitwise the
stock MXFP8 quantize."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.gemm.m3_mxfp8_pinned_quantize import (  # noqa: E501
    m3_mxfp8_pinned_quantize as op,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner as QR

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_pinned_quantize requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@pytest.mark.parametrize("tactic", [QR.CUTE_DSL, QR.TRTLLM], ids=["cute_dsl", "trtllm"])
def test_certified_cells_bitwise(cell, tactic) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(cell.spec["m"], cell.spec["k"], generator=g, device="cuda") * 3.0).to(
        torch.bfloat16
    )
    with validating(op):
        q, sf = op(x, tactic)
    rq, rsf = op.reference(x, tactic)
    assert torch.equal(q.view(torch.uint8).view(-1), rq.view(torch.uint8).view(-1))
    assert torch.equal(sf.view(-1), rsf.view(-1))
