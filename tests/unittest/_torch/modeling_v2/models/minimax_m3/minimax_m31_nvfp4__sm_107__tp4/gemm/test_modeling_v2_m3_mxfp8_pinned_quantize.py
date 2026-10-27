# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_pinned_quantize entry (MX-TAC): the pinned decode-graph MXFP8 activation quantize is the
stock quantize bit for bit, at every tactic stock's AutoTuner can pick."""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.gemm.m3_mxfp8_pinned_quantize import (  # noqa: E501
    m3_mxfp8_pinned_quantize as op,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8QuantizeRunner

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_pinned_quantize requires a CUDA device"

TACTIC = modeling.MXFP8_DECODE_QUANT_TACTIC


def test_the_pinned_tactic_is_cute_dsl() -> None:
    assert TACTIC == MXFP8QuantizeRunner.CUTE_DSL


def _check(x: torch.Tensor) -> None:
    m, k = x.shape
    with validating(op):
        xq, xsf = op(x, TACTIC)
    tq, tsf = op.reference(x, TACTIC)
    assert torch.equal(xq.view(torch.uint8).view(m, k), tq.view(torch.uint8).view(m, k))
    assert torch.equal(xsf.view(torch.uint8).flatten(), tsf.view(torch.uint8).flatten())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_certified_cells(cell, scale) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(spec["m"], spec["k"], generator=g, device="cuda") * scale).to(torch.bfloat16)
    x[0, ::32] *= 1e3  # one outlier per 32-element block on the first row
    _check(x)
