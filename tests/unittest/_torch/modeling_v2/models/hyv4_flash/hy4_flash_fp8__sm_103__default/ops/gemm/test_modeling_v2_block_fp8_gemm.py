# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the block_fp8_gemm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.block_fp8_gemm import (  # noqa: E501
    BLOCK,
    SPARSE_Q_B,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.block_fp8_gemm import (  # noqa: E501
    block_fp8_gemm as op,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

DEV = "cuda"

assert torch.cuda.is_available(), "block_fp8_gemm requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="block_fp8_gemm is certified on sm_103"
)

# Both sides read identical e4m3 codes and fp32 scales; they differ only in fp32 summation order and where the
# scales are applied, then each rounds once to bf16.
ELEMENT_ULP = 4.0
RMS_ULP = 2.0


def _blocks(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK


def _build(spec: dict, seed: int) -> tuple[torch.Tensor, ...]:
    """Block-FP8 operands at O(1) output scale, every value exact in its storage type."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    m, n, k = spec["m"], spec["n"], spec["k"]
    x = (2.0 * torch.randn(m, k, generator=g, device=DEV)).to(torch.float8_e4m3fn)
    x_scale = 0.5 + torch.rand(_blocks(k), m, generator=g, device=DEV)
    w = torch.randn(n, k, generator=g, device=DEV).to(torch.float8_e4m3fn)
    w_scale = (0.5 + torch.rand(_blocks(n), _blocks(k), generator=g, device=DEV)) / k**0.5
    return x, w, x_scale, w_scale


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, w, x_scale, w_scale = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, w, x_scale, w_scale)
    assert out.is_contiguous()
    assert_within_ulp(
        out, op.reference(x, w, x_scale, w_scale), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP
    )


def test_block_fp8_linear_from_bf16_rows() -> None:
    """The model feeds `fp8_quantize_1x128`'s scales straight in: its layout must be the one the GEMM reads."""
    g = torch.Generator(device=DEV).manual_seed(23)
    rows = torch.randn(64, SPARSE_Q_B["k"], generator=g, device=DEV).to(torch.bfloat16)
    x, x_scale = fp8_quantize_1x128(rows)
    _, w, _, w_scale = _build(dict(m=1, **SPARSE_Q_B), seed=29)
    with validating(op):
        out = op(x, w, x_scale, w_scale)
    assert_within_ulp(
        out, op.reference(x, w, x_scale, w_scale), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP
    )
