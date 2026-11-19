# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large CSA2 decode batches select from the scorer's block maxima with the masked TopK's results."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as csa2_indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2 import metadata as csa2_metadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Params

from ._utils import _paged_decode_batch, _replay_paged_decode


def _tie_only(old, new, scores):
    """Rows of two increasing selections differ only by entries scoring the new k-th score."""
    for row in (old != new).any(1).nonzero().flatten().tolist():
        a = set(old[row][old[row] >= 0].tolist())
        b = set(new[row][new[row] >= 0].tolist())
        kth = scores[row, sorted(b)].min()
        assert bool((scores[row, sorted(a ^ b)] == kth).all()), f"row {row} differs beyond ties"


def _block_scores(logits, valid, visible):
    """Masked 8-position block maxima with the newest visible block pinned, as published."""
    width = logits.shape[1]
    columns = torch.arange(width, device=logits.device)
    ok = (columns[None, :] < visible[:, None]) & valid[:, columns // 64]
    masked = F.pad(logits.masked_fill(~ok, -torch.inf), (0, -width % 8), value=-torch.inf)
    blocks = masked.view(logits.shape[0], -1, 8).amax(-1)
    newest = ((visible - 1) // 8).clamp_min(0)
    return blocks.scatter(1, newest[:, None].long(), torch.inf)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="The block selection runs on SM100/SM103",
)
@pytest.mark.parametrize("next_n", [6])
@torch.inference_mode()
def test_block_select_matches_topk_decode(monkeypatch, next_n):
    torch.manual_seed(4401 + next_n)
    batch = _paged_decode_batch(next_n, (4100, 1000))
    metadata, state, visible = batch.metadata, batch.state, batch.visible
    rows, width = batch.rows, batch.width
    options = CSA2Params(layout=batch.layout, enable_heuristic_topk=True)
    source, consumer = (CSA2Indexer(batch.layout, i, 32, 128, options) for i in (0, 1))
    # A FULL layer without a publication, block-selecting at this batch size too.
    plain_layout = CSA2Layout((1,), (0,), (0,), None, index_topk=512)
    plain = CSA2Indexer(
        plain_layout, 0, 32, 128, CSA2Params(layout=plain_layout, enable_heuristic_topk=True)
    )
    monkeypatch.setattr(csa2_metadata, "BLOCK_SELECT_FULL_MIN_ROWS", 1)
    scored = {}
    paged = source._paged_mqa_logits

    def keep_logits(*args, **kwargs):
        logits = paged(*args, **kwargs)
        scored["logits"] = logits.clone()
        return logits

    monkeypatch.setattr(source, "_paged_mqa_logits", keep_logits)
    launched = []
    kernel = csa2_indexer.block_select_
    monkeypatch.setattr(
        csa2_indexer, "block_select_", lambda *a, **kw: launched.append(1) or kernel(*a, **kw)
    )

    def forward():
        metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
        selected = source(state, 0, rows)
        return selected, consumer(state, 0, rows), plain(state, 0, rows)

    def outputs_of_metadata():
        return (
            metadata.csa2_candidates[0],
            metadata.csa2_candidate_blocks[0],
            metadata.csa2_candidate_counts[0],
        )

    def outputs(result):
        return [t.clone() for t in (*result, *outputs_of_metadata(), scored["logits"])]

    min_rows = csa2_metadata.BLOCK_SELECT_SOURCE_MIN_ROWS

    def run(block_select):
        # A row bound above the batch keeps the plain scorer and the masked TopK.
        monkeypatch.setattr(
            csa2_metadata, "BLOCK_SELECT_SOURCE_MIN_ROWS", min_rows if block_select else 1 << 30
        )
        for _ in range(2):
            result = forward()
        torch.cuda.synchronize()
        return outputs(result)

    def check(old, new):
        selected, chosen, full, published, blocks, counts, logits = new
        valid = metadata.csa2_indexer_valid_pages[:, : -(-width // 64)]
        # Both paths score every visible, backed position with DeepGEMM's bits.
        columns = torch.arange(width, device="cuda")
        seen = (columns[None, :] < visible[:, None]) & valid[:, columns // 64]
        assert torch.equal(
            logits[:, :width][seen].view(torch.int32), old[6][:, :width][seen].view(torch.int32)
        )
        _tie_only(old[0], selected, logits)
        _tie_only(old[2], full, logits)
        # Publications agree up to ties of the last selected block's score; consumers of rows
        # with equal publications select the same positions.
        same = (old[3] == published).all(1)
        if not bool(same.all()):
            scores = _block_scores(logits, valid, visible)
            _tie_only(old[3][~same][:, ::8] // 8, published[~same][:, ::8] // 8, scores[~same])
        assert torch.equal(old[1][same], chosen[same])
        assert torch.equal(old[4][same], blocks[same]) and torch.equal(old[5][same], counts[same])

    old = run(False)
    assert not launched
    new = run(True)
    assert len(launched) == 4
    check(old, new)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Eager forwards publish into new tensors: keep the graph's own.
        captured = (*forward(), *outputs_of_metadata(), scored["logits"])

    def check_replay():
        replayed = [t.clone() for t in captured]
        check(run(False), replayed)

    _replay_paged_decode(batch, graph, check_replay)
