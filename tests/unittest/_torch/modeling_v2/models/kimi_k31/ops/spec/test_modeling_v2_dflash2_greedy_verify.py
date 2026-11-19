# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 strict-verification entry (CuTe DSL).

A sample of the certified cells is gated bitwise against the entry's reference (the family's torch
``dflash2_greedy_verify``).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_greedy_verify import (  # noqa: E501
    dflash2_greedy_verify as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, 107, reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x")

_DRAWS = 4


def _build(spec, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n, g, vocab = spec["batch"], spec["drafts"], spec["vocab"]
    logits = 4.0 * torch.randn(n, g + 1, vocab, device="cuda", generator=gen)
    if spec["ties"]:
        # A repeated maximum in every row: the lower index must win.
        top = logits.argmax(-1, keepdim=True)
        twin = (
            top + 1 + torch.randint(0, vocab - 1, top.shape, device="cuda", generator=gen)
        ) % vocab
        logits.scatter_(-1, twin, logits.gather(-1, top))
    logits = logits.to(spec["dtype"])
    for b in range(spec["nan_rows"]):
        logits[b, b % (g + 1), (7919 * (b + 1)) % vocab] = float("nan")
        if b == 0:
            logits[b, 0, 11] = float("nan")  # two NaNs in one row: the first wins
    best = logits.argmax(-1)
    agree = torch.rand(n, g, device="cuda", generator=gen) < spec["agree"]
    other = torch.randint(0, vocab, (n, g), device="cuda", generator=gen)
    drafts = torch.where(agree, best[:, :g], other).to(spec["draft_dtype"])
    return logits, drafts


CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    base = zlib.crc32(cell.why.encode()) % 2**31
    for draw in range(_DRAWS):
        args = _build(cell.spec, seed=base + draw)
        for got, want in zip(op(*args), op.reference(*args)):
            assert got.dtype == want.dtype and torch.equal(got, want), f"draw {draw}"
