# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_situ_and_mul_mxfp8, SiTU handing the shared experts' down GEMM its MXFP8 operand.

GPU (sm_107): the dep16 context chunk's cell bit for bit against the two-kernel path it replaces
(the family's situ_and_mul, then the catalog mxfp8_quantize of its BF16 rows): the E4M3 codes and
the whole scale buffer, padding rows included.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.activation.rubin_situ_and_mul_mxfp8 import (  # noqa: E501
    rubin_situ_and_mul_mxfp8 as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.situ_and_mul import situ_and_mul

__extra_import_path__ = ["../../.."]
from _k31_utils import on_sm, sample_cells

on_sm107 = on_sm(107)
# The largest cell of the sample: a dep16 context chunk of 16384 rows.
_CELLS = sample_cells(op.CELLS, size_key="shape")[-1:]


def _inputs(spec, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, n = spec["shape"]
    x = torch.randn(rows, n, generator=gen, device=device) * 3
    if spec.get("saturated"):
        x = torch.sign(x) * 1.0e4
        x[:, :: n // 8] = 0
    x[0, : n // 2] = 0  # a zero gate row: all-zero groups
    return x.to(torch.bfloat16), spec["beta"], spec["linear_beta"]


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_cells_bitwise_against_the_two_kernel_path(cell):
    from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize

    x, beta, linear_beta = _inputs(cell.spec, "cuda", seed=cell.spec["shape"][0])
    want_codes, want_scales = mxfp8_quantize(situ_and_mul(x, beta, linear_beta), True, 32)
    codes, scales = op(x, beta, linear_beta)
    assert torch.equal(codes.view(torch.uint8), want_codes.view(torch.uint8)), "codes differ"
    assert torch.equal(scales, want_scales), "scale bytes differ"
