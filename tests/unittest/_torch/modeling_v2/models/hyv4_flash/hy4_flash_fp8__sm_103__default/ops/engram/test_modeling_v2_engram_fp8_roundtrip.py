# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_fp8_roundtrip entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.engram.engram_fp8_roundtrip import (  # noqa: E501
    engram_fp8_roundtrip as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "engram_fp8_roundtrip requires a CUDA device"

#: ``engram_n_embed_per_ngram`` (768) / ``engram_n_head_per_ngram`` (8).
WIDTH = 96
#: One e4m3 step at the row maximum, in bf16 ulps of the row maximum: 2^-3 / 2^-8.
ONE_E4M3_STEP = 32.0
# A bf16 quotient may flip an element to the neighbouring e4m3 value; the element band alone cannot
# tell the round trip from its absence, so RMS and the bit-identical fraction are gated too.
ELEMENT_ULP = 1.25 * ONE_E4M3_STEP
RMS_ULP = 5.0
EXACT = 0.95

CELLS = op.CELLS


def _build(rows: int, seed: int) -> torch.Tensor:
    """Per-row magnitudes over ~11 binades; row 0 all zero, row 1 at amax ~1e-9."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, WIDTH), generator=g, device="cuda")
    magnitude = torch.exp2(torch.empty((rows, 1), device="cuda").uniform_(-8.0, 3.0, generator=g))
    x = (x * magnitude).to(torch.bfloat16)
    if rows >= 2:
        x[0] = 0.0
        x[1] *= 2.0**-30
    return x


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 - pytest resolves the fixture by this name
    certifying(op)
    rows = _build(cell.spec["rows"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(rows)
    ref = op.reference(rows)
    assert out.dtype == torch.bfloat16 and out.shape == rows.shape
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
    exact = (out == ref).float().mean().item()
    assert exact >= EXACT, f"{exact:.4f} of elements bit-identical to the reference"
