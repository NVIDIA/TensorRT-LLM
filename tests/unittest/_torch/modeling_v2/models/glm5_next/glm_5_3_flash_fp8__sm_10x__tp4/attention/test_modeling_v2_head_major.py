# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 head-major FP8 cast entries."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.head_major import (  # noqa: E501
    quantize_1x128,
    saturate_fp8,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)


def _input(spec: dict, seed: int, scale: float) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(spec["h"], spec["r"], spec["d"], generator=g, device="cuda") * scale
    return x.to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("cell", saturate_fp8.CELLS, ids=[c.why[:44] for c in saturate_fp8.CELLS])
def test_saturate_matches_bitwise(cell, certifying):  # noqa: F811
    """Values past +-448 must saturate, not become NaN; the rest round as torch's cast does."""
    certifying(saturate_fp8)
    x = _input(cell.spec, zlib.crc32(cell.why.encode()) % 2**31, scale=300.0)
    saturate_fp8.is_valid(x)
    assert torch.equal(_bits(saturate_fp8(x)), _bits(saturate_fp8.reference(x)))


@pytest.mark.parametrize(
    "cell", quantize_1x128.CELLS, ids=[c.why[:44] for c in quantize_1x128.CELLS]
)
def test_quantize_matches_bitwise(cell, certifying):  # noqa: F811
    """Same codes and scales as ``fp8_quantize_1x128`` on the transposed copy."""
    certifying(quantize_1x128)
    x = _input(cell.spec, zlib.crc32(cell.why.encode()) % 2**31, scale=1.0)
    x[0, :3] = 0  # all-zero blocks take the 1e-10 amax floor
    quantize_1x128.is_valid(x)
    (q, s), (q_ref, s_ref) = quantize_1x128(x), quantize_1x128.reference(x)
    assert torch.equal(_bits(q), _bits(q_ref))
    assert torch.equal(s, s_ref)
