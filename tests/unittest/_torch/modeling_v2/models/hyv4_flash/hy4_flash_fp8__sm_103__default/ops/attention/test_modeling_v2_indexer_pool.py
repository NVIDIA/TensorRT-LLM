# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the hyv4_flash compressed-indexer pool entry `indexer_pool`."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.indexer_pool import (  # noqa: E501
    indexer_pool,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "indexer_pool requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the hyv4_flash target's entries are certified on sm_103"
)

DEVICE = "cuda"
HD = 128
KPOOL = 4
EPS = 1e-5
#: Checkpoint scales of the four pooling tensors; token keys are unit-RMS like the normed keys.
VALUE_STD, GATE_STD, APE_STD, NORM_MEAN, NORM_STD = 0.022, 0.049, 0.196, 1.055, 0.122


def _args(spec: dict, seed: int) -> tuple:
    """`(token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight)`."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    batch, tokens = spec["batch"], spec["tokens"]
    token_keys = torch.randn(batch, tokens, HD, generator=g, device=DEVICE).to(torch.bfloat16)
    value_weight = torch.randn(HD, HD, generator=g, device=DEVICE) * VALUE_STD
    gate_weight = torch.randn(HD, HD, generator=g, device=DEVICE) * GATE_STD
    intra_group_ape = torch.randn(KPOOL, HD, generator=g, device=DEVICE) * APE_STD
    key_norm_weight = NORM_MEAN + torch.randn(HD, generator=g, device=DEVICE) * NORM_STD
    return token_keys, value_weight, gate_weight, intra_group_ape, key_norm_weight


@pytest.mark.parametrize("cell", indexer_pool.CELLS, ids=[c.why[:44] for c in indexer_pool.CELLS])
def test_indexer_pool_matches_reference(cell, certifying):  # noqa: F811
    """Both sides are fp32 inside and round to BF16 once, so at most 1 in 10000 keys may differ
    (a BF16 rounding-boundary flip from `exp2`/`rsqrt`), each within one BF16 step of the row scale."""
    certifying(indexer_pool)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = indexer_pool(*args, EPS, kpool=KPOOL)
    ref = indexer_pool.reference(*args, EPS, kpool=KPOOL)
    assert out.shape == (cell.spec["batch"], cell.spec["tokens"] // KPOOL, HD)
    assert out.dtype is torch.bfloat16 and ref.dtype is torch.bfloat16
    count = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
    assert count <= max(16, out.numel() // 10_000), f"{count} of {out.numel()} group keys differ"
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25, ulp=ULP[torch.bfloat16])
