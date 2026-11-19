# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_proposal_load entry: a sample of its cells bitwise against the store's
load."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_proposal_load import (  # noqa: E501
    k31_dflash2_proposal_load as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(
    103, 107, reason="the K3.1 DFlash2 proposal entry is certified on sm_103 and sm_107"
)

K_MAX = 7


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots, rows = spec["n"], spec["slots"], spec["rows"]
    store_cand = torch.randint(0, 163840, (rows, K_MAX, 32), generator=g, device="cuda")
    store_q = torch.rand(rows, K_MAX, 32, generator=g, device="cuda")
    store_tok = torch.randint(0, 163840, (rows, K_MAX), generator=g, device="cuda")
    store_tok[-1] = -1  # an invalidated row
    slot_ids = torch.randperm(rows, generator=g, device="cuda")[:n]
    drafts = store_tok[slot_ids, :slots].clone()
    if spec["walked"] != "all":
        changed = torch.rand(n, generator=g, device="cuda") < 0.5
        if spec["walked"] == "none":
            changed[:] = True
        pos = torch.randint(0, slots, (n,), generator=g, device="cuda")
        bumped = drafts.clone()
        bumped[torch.arange(n, device="cuda"), pos] += 1
        drafts = torch.where(changed[:, None], bumped, drafts)
    return [store_cand, store_q, store_tok, slot_ids.to(spec["ids"]), drafts.to(spec["ids"])]


CELLS = sample_cells(op.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    for got, want in zip(op(*args), op.reference(*args)):
        assert torch.equal(got, want)
