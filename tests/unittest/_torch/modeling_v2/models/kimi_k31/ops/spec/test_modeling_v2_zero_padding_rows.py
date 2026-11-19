# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_zero_padding_rows entry: a sample of its cells bitwise against its reference (the engine's
``SpecWorkerBase._zero_padding_rows``, in place)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.zero_padding_rows import (  # noqa: E501
    k31_zero_padding_rows as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(103, reason="the K3.1 padding-rows entry is certified on sm_103")

DUMMY = 9


def _padding(kind: str, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    if kind == "last":
        mask[-1] = True
    elif kind == "all":
        mask[:] = True
    elif kind == "ends":
        mask[0] = mask[-1] = True
    elif kind == "some":
        mask[1::5] = True
    return mask


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, rpr, vocab = spec["n"], spec["rows_per_request"], spec["vocab"]
    logits = 8.0 * torch.randn(n * rpr, vocab, generator=g, device="cuda")
    mask = _padding(spec["padding"], n).cuda()
    if spec["special"]:
        logits[:, ::7] = float("nan")
        logits[:, 3::11] = float("inf")
        logits[:, 5::13] = -0.0
    real = torch.randperm(DUMMY + 4 * n, generator=g, device="cuda")
    real = real[real != DUMMY][:n]
    slot_ids = torch.where(mask, torch.full_like(real, DUMMY), real).to(spec["ids"])
    return logits, slot_ids, DUMMY, rpr


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


CELLS = sample_cells(op.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    logits, slot_ids, dummy, rpr = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    expected = op.reference(logits.clone(), slot_ids, dummy, rpr)
    assert torch.equal(_bits(op(logits, slot_ids, dummy, rpr)), _bits(expected))
