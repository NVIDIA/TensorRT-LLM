# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of ``rubin_mxfp8_linear``'s reference.

The entry's reference is checked against a float64 product built from the checkpoint's own form:
E4M3 values and row-major K32 scales, never relaid.
"""

import pytest
import torch

__extra_import_path__ = [".", "../.."]
from _k31_utils import cpu_or_sm
from _rubin_gemm_util import mxfp8_weight_scales  # noqa: E402 — needs the path declared above

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm import rubin_mxfp8_gemm as G
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm import rubin_mxfp8_linear as L

pytestmark = [pytest.mark.cpu_only, cpu_or_sm(107)]

op = L.rubin_mxfp8_linear


def _checkpoint_weight(rows: int, k: int, gen: torch.Generator):
    """E4M3 values and row-major K32 scales of a ``[rows, k]`` weight, as the export stores them.
    Column magnitudes span four decades, so the 32-value scales differ from block to block."""
    w = torch.randn(rows, k, generator=gen) * torch.logspace(-2, 2, k)
    return G.quantize_mxfp8_reference(w.to(torch.bfloat16))


def _activation(rows: int, k: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(rows, k, generator=gen) * 4.0
    x[:, ::7] *= 30.0
    return x.to(torch.bfloat16)


def _modelopt_float64(x: torch.Tensor, values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The ModelOpt W8A8 product in float64, from the checkpoint form (``scales`` row-major).

    Each 32 input values take the smallest power of two at or above ``amax / 448`` (found with
    frexp, not with the entry's bit arithmetic) and round to E4M3. Weight value ``(r, j)`` is
    ``values[r, j] * 2 ** (scales[r, j // 32] - 127)``.
    """
    m, k = x.shape
    n = values.shape[0]
    blocks = x.float().view(m, k // 32, 32)
    ratio = blocks.abs().amax(-1) / 448.0
    mantissa, exponent = torch.frexp(ratio)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    scale = torch.ldexp(torch.ones_like(ratio, dtype=torch.float64), exponent)
    scale = torch.where(ratio > 0, scale, torch.full_like(scale, 2.0**-127))
    xq = (blocks.double() / scale.unsqueeze(-1)).float().to(torch.float8_e4m3fn)
    xd = xq.double() * scale.unsqueeze(-1)
    wd = values.double().view(n, k // 32, 32) * torch.exp2(scales.double() - 127.0).unsqueeze(-1)
    return xd.view(m, k) @ wd.view(n, k).t()


def _within_one_rounding(out: torch.Tensor, exact: torch.Tensor) -> None:
    """One BF16 rounding of an FP32 sum of exact terms: within 2^-8 of the row's largest value."""
    assert out.dtype == torch.bfloat16 and out.shape == exact.shape
    row_max = exact.abs().amax(dim=1, keepdim=True)
    assert torch.all((out.double() - exact).abs() <= row_max * 2.0**-8)


@pytest.mark.parametrize("m, n, k", [(8, 128, 7168)])
def test_reference_is_the_modelopt_recipe(m, n, k) -> None:
    gen = torch.Generator().manual_seed(m * n + k)
    x = _activation(m, k, gen)
    values, scales = _checkpoint_weight(n, k, gen)
    out = op.reference(x, values, mxfp8_weight_scales(scales))
    _within_one_rounding(out, _modelopt_float64(x, values, scales))
