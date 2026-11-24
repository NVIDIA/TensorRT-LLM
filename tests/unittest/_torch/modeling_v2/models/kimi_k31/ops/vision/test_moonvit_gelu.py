# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_gelu entry: each certified cell against its reference within the band
stated below, and bit for bit against the engine MoonViT3d code it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_gelu import (
    moonvit_gelu,
)
from tensorrt_llm._torch.models import modeling_kimi_k25 as engine_k25

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


@pytest.mark.parametrize("cell", moonvit_gelu.CELLS, ids=_ids(moonvit_gelu))
def test_gelu_cells(cell, certifying) -> None:  # noqa: F811
    certifying(moonvit_gelu)
    gen = _gen(cell)
    approximate = cell.spec["approximate"]
    x = (
        3.0 * torch.randn(cell.spec["rows"], cell.spec["width"], generator=gen, device="cuda")
    ).bfloat16()

    out = moonvit_gelu(x, approximate)
    engine = engine_k25._gelu_tanh(x) if approximate == "tanh" else torch.nn.GELU()(x)
    assert torch.equal(out, engine)
    torch.testing.assert_close(out, moonvit_gelu.reference(x, approximate), **_BF16_BAND)
