# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_prefill entry (CUDA radix prefill top-k).

Each row's window holds distinct logits and everything outside it holds +inf, so a kernel that
reads outside its window selects a wrong column. Rows are compared as sets (the kernel's slot
order is atomic).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (
    indexer_topk_prefill as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key=None)[:2]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    ks, ke, num_keys = inputs.chunk_windows(spec["requests"], spec["q_rows"])
    logits = inputs.distinct_logits(ks.shape[0], num_keys, gen, spec["logits_pad"])
    inputs.poison_outside(logits, ks, ke)
    indices = torch.full((ks.shape[0], spec["index_topk"]), -7, dtype=torch.int32, device="cuda")
    return dict(
        logits=logits, row_starts=ks, row_ends=ke, indices=indices, index_topk=spec["index_topk"]
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    ref = torch.full_like(args["indices"], -9)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    out = args["indices"]
    inputs.assert_same_sets(out, ref)
    # Short windows are written as 0 .. len - 1 in order, then -1.
    lengths = (args["row_ends"] - args["row_starts"]).long()
    short = lengths <= args["index_topk"]
    assert torch.equal(out[short], ref[short])
