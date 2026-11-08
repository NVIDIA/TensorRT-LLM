# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local VR200 coverage of the TEP/DEP/TP K128 FP8 projections at four and eight ranks."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.gemm.cute_dsl_fp8_block_scale_linear import (  # noqa: E501
    cute_dsl_fp8_block_scale_linear as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.gemm.cute_dsl_fp8_block_scale_linear import (  # noqa: E501
    prepare_weight_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.gemm.fp8_prequantized_swap_ab_gemm import (  # noqa: E501
    fp8_prequantized_swap_ab_gemm as packed_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.quantization.fp8_quantize_1x128_packed_ue8m0 import (  # noqa: E501
    fp8_quantize_1x128_packed_ue8m0 as quantize,
)
from tensorrt_llm.quantization.utils.fp8_utils import get_col_major_tma_aligned_packed_tensor

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why for cell in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    m, n, k = (cell.spec[name] for name in ("m", "n", "k"))
    generator = torch.Generator(device="cuda").manual_seed(cell.seed_base())
    input = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    exponent = torch.randint(-6, 4, (n // 128, k // 128), device="cuda", generator=generator)
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    packed_scale = get_col_major_tma_aligned_packed_tensor(scale.repeat_interleave(128, dim=0))
    prepared = prepare_weight_scale(packed_scale, k)
    actual = op(input, weight, prepared)
    reference = op.reference(input, weight, prepared)
    codes, input_scale = quantize.reference(input)
    packed_reference = packed_op.reference(
        codes, input_scale, weight, packed_scale, torch.bfloat16, True
    )
    torch.testing.assert_close(reference, packed_reference, rtol=0, atol=0)
    # FP8 products are exact in FP32; accumulation order can straddle a BF16
    # rounding boundary. Use the packed GEMM's four-element/two-RMS ULP band.
    assert_within_ulp(actual, reference, element_ulp=4, rms_ulp=2)
