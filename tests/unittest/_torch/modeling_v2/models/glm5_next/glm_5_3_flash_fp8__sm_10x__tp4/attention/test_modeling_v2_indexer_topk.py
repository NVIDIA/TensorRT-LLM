# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry indexer_topk."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.indexer_topk import (  # noqa: E501
    indexer_topk,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
KPOOL, SELECT_K = 4, 512


def _args(spec: dict, seed: int):
    """Scores and candidate counts the way the target hands them over.

    Context cells: rows ``lo .. hi - 1`` of a prompt see ``row + 1`` positions; the scores are
    a view of DeepGEMM's row-padded logits. Generation cells: request ``b``'s KV length after
    the step is ``kv - b % 4`` and its rows see ``kv_b - T + 1 .. kv_b`` positions, over rows
    a full block table wide. Every column a row must not read (at
    or past its count, and the row padding) holds +inf, so a read past the count wins the
    selection. ``levels`` rounds the scores to quarter steps: the k-th score ties many times.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if "rows" in spec:
        lo, hi = spec["rows"]
        visible = torch.arange(lo + 1, hi + 1, device=DEVICE)
    else:
        kv = spec["kv"] - torch.arange(spec["requests"], device=DEVICE) % 4
        steps = torch.arange(spec["tokens"], device=DEVICE)
        visible = (kv[:, None] - spec["tokens"] + 1 + steps).reshape(-1)
    num_cand = (visible // KPOOL).to(torch.int32)
    rows, width = visible.shape[0], spec["width"]
    buf = torch.randn(rows, width + spec.get("row_pad", 0), generator=g, device=DEVICE)
    if "levels" in spec:
        buf = torch.round(buf * spec["levels"]) / spec["levels"]
    col = torch.arange(buf.shape[1], device=DEVICE)
    buf = torch.where(col < num_cand[:, None].long(), buf, float("inf"))
    scores = buf[:, :width]
    selected = torch.full((rows, SELECT_K), 12345, dtype=torch.int32, device=DEVICE)
    return scores, num_cand, selected


def _check(out, ref, scores, num_cand):
    """Rows with at most ``SELECT_K`` candidates: exactly the reference (``0 .. n - 1``, then
    -1). Longer rows: ``SELECT_K`` distinct candidates whose scores, sorted, equal the
    reference's bit for bit (true of every valid top-k, whatever the tie rule), and the same
    pools as the reference wherever the k-th score is not tied."""
    short = num_cand.long() <= SELECT_K
    assert torch.equal(out[short], ref[short])
    long_rows = (~short).nonzero().flatten()
    if long_rows.numel() == 0:
        return
    got = out[long_rows].long().sort(dim=1).values
    want = ref[long_rows].long()
    count = num_cand[long_rows, None].long()
    assert bool(((got >= 0) & (got < count)).all()), "a pool outside the row's candidates"
    assert bool((got.diff(dim=1) > 0).all()), "a pool selected twice"
    got_scores = torch.gather(scores[long_rows], 1, got).sort(dim=1).values
    want_scores = torch.gather(scores[long_rows], 1, want).sort(dim=1).values
    assert torch.equal(got_scores, want_scores)
    # Without a tie at the k-th score the selected set is unique.
    col = torch.arange(scores.shape[1], device=DEVICE)
    candidates = torch.where(col < count, scores[long_rows], float("-inf"))
    next_best = candidates.topk(SELECT_K + 1, dim=1).values[:, -1]
    untied = want_scores[:, 0] > next_best
    assert torch.equal(got[untied], want[untied])


@pytest.mark.parametrize("cell", indexer_topk.CELLS, ids=[c.why[:44] for c in indexer_topk.CELLS])
def test_indexer_topk_matches_reference(cell, certifying):  # noqa: F811
    """See ``_check``: the kernel's choice among tied pools and its order are unspecified, so
    the comparison is of the selected scores and, where nothing ties, of the pool sets."""
    certifying(indexer_topk)
    scores, num_cand, selected = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = indexer_topk(scores, num_cand, selected, select_k=SELECT_K)
    assert out is selected
    ref = indexer_topk.reference(scores, num_cand, selected, select_k=SELECT_K)
    _check(out, ref, scores, num_cand)


def test_guard_refuses_counts_past_the_row():
    """A candidate count past the row reads the next row's scores as this row's."""
    scores, num_cand, selected = _args(dict(requests=1, tokens=4, kv=4096, width=1000), seed=0)
    with pytest.raises(AssertionError, match="candidate counts"):
        with validating(indexer_topk):
            indexer_topk(scores, num_cand, selected, select_k=SELECT_K)
