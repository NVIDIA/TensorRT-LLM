# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_argmax entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.sampling import (  # noqa: E402, E501
    cute_dsl_argmax as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_argmax
VOCAB = entry.VOCAB
#: Columns in different segments of the 8-CTA cluster (each scans VOCAB / 8).
_SEG = VOCAB // 8


def _logits(spec, seed) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, case = spec["rows"], spec["case"]
    x = torch.randn(rows, VOCAB, generator=g, device="cuda") * 4
    if case == "ties":
        # The same maximum in several segments, and twice in one segment.
        for r in range(rows):
            for c in (5 * _SEG + 17, 2 * _SEG + 3 + r, 7 * _SEG + 1, 2 * _SEG + 900):
                x[r, c] = 100.0
    elif case == "signed_zero":
        x.fill_(-1.0)
        x[0, 50_000], x[0, 9] = 0.0, -0.0
        x[1, 77], x[1, 120_000] = -0.0, 0.0
        x[2, 3 * _SEG] = -0.0
    elif case == "nan":
        x[0, 140_000] = float("nan")
        x[1, 30], x[1, 31] = float("nan"), float("nan")
        x[2, 5] = float("inf")
        x[2, 6] = float("nan")
    elif case == "inf":
        x[0, 1234] = float("inf")
        x[1].fill_(float("-inf"))
        x[2, 4 * _SEG], x[2, 6 * _SEG] = float("inf"), float("inf")
        x[3, :] = float("-inf")
        x[3, VOCAB - 1] = -1e30
    return x


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """Exact: an argmax has one right answer per row, ties included."""
    certifying(op)
    logits = _logits(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    assert entry.takes(logits)
    tokens = torch.full((logits.shape[0],), -7, dtype=torch.int32, device="cuda")
    got = op(logits, tokens, compiled)
    assert got is tokens
    assert torch.equal(got, op.reference(logits, tokens, compiled))


def test_takes_refuses_a_strided_view() -> None:
    wide = torch.zeros(2, VOCAB + 4, device="cuda")
    assert not entry.takes(wide[:, :VOCAB])
