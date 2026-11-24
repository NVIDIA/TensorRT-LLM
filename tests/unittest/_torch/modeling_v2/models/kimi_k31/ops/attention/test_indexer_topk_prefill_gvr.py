# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 indexer_topk_prefill_gvr entry: each row's set equals the reference's
(ties to the smaller column) and, on distinct scores, the CUDA radix's; a repeated call writes the same
output, order included, also on fresh prompts of a few thousand tokens with tied scores."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill import (  # noqa: E501
    indexer_topk_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill_gvr import (  # noqa: E501
    indexer_topk_prefill_gvr as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _build(spec, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ks, ke, start = [], [], 0
    for cached, new in spec["requests"]:
        for i in range(new):
            ks.append(start)
            ke.append(start + cached + i + 1)
        start += cached + new
    rows, n = len(ks), start
    stride = n + spec["logits_pad"]
    base = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    if spec["data"] == "ties":
        base[:, :n] = torch.randint(0, 64, (rows, n), generator=gen, device="cuda").float()
    else:
        # Distinct values per row: a random permutation of a fixed grid (no tie anywhere).
        base[:, :n] = (
            torch.rand(rows, n, generator=gen, device="cuda").argsort(dim=1).float() - n / 2
        ) * 0.25
    return dict(
        logits=base[:, :n],
        row_starts=torch.tensor(ks, dtype=torch.int32, device="cuda"),
        row_ends=torch.tensor(ke, dtype=torch.int32, device="cuda"),
        indices=torch.full((rows, spec["index_topk"]), -5, dtype=torch.int32, device="cuda"),
        index_topk=spec["index_topk"],
    )


def _sorted(t):
    return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(_sorted(out), _sorted(ref))
    if cell.spec["data"] != "ties":
        radix = torch.empty_like(out)
        indexer_topk_prefill(
            args["logits"], args["row_starts"], args["row_ends"], radix, args["index_topk"]
        )
        assert torch.equal(_sorted(out), _sorted(radix))


def test_guard_refuses_a_window_past_the_logits():
    args = _build(op.CELLS[5].spec, seed=1)
    ends = args["row_ends"].clone()
    ends[-1] = args["logits"].shape[1] + 1
    with pytest.raises(AssertionError, match="past the logits"):
        with validating(op):
            op(**dict(args, row_ends=ends))


@pytest.mark.parametrize("n", [2100, 3000, 6000, 12000])
@pytest.mark.parametrize("data", ["ties", "distinct"])
def test_fresh_prompt_repeats_bit_for_bit(n, data):
    """A fresh n-token prompt (row i's window is keys 0 .. i): the selected set is the stable-sort
    reference's (ties at the k-th score to the smaller column), and 100 repeated calls write the same
    output, order included (the radix the narrow tiles used to take picks among ties in atomic order)."""
    args = _build(dict(requests=((0, n),), index_topk=2048, data=data, logits_pad=0), seed=n)
    op(**args)
    out = args["indices"].clone()
    ref = torch.empty_like(out)
    op.reference(**dict(args, indices=ref))
    torch.cuda.synchronize()
    assert torch.equal(_sorted(out), _sorted(ref))
    for _ in range(100):
        args["indices"].fill_(-5)
        op(**args)
        assert torch.equal(args["indices"], out)
