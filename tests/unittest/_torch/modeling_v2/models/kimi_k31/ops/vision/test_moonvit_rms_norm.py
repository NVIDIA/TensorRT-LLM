# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_rms_norm entry: each certified cell against its reference within the band
stated below, and bit for bit against the engine MoonViT3d code it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_rms_norm import (
    moonvit_rms_norm,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 (sm_107 driven separately)",
)

#: bf16 results against an fp32 / fp64 reference rounded once: the entries round intermediate bf16 values, each half
#: an ulp; four ulps of the row's scale covers them.
_BF16_BAND = dict(rtol=4 * 2.0**-8, atol=4 * 2.0**-8)


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _ids(op):
    return [c.why[:48] for c in op.CELLS]


@pytest.mark.parametrize("cell", moonvit_rms_norm.CELLS, ids=_ids(moonvit_rms_norm))
def test_rms_norm_cells(cell, certifying) -> None:  # noqa: F811
    certifying(moonvit_rms_norm)
    gen = _gen(cell)
    rows, width, eps = cell.spec["rows"], cell.spec["width"], cell.spec["eps"]
    x = torch.randn(rows, width, generator=gen, device="cuda").bfloat16()
    norm = torch.nn.RMSNorm(width, eps=eps).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(width, generator=gen, device="cuda"))

    out = moonvit_rms_norm(x, norm.weight, eps)
    with torch.no_grad():
        assert torch.equal(out, norm(x))
    torch.testing.assert_close(out, moonvit_rms_norm.reference(x, norm.weight, eps), **_BF16_BAND)
