# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A candidate source publishes what it published with its buffers filled at allocation and its block
maxima unpadded, eagerly and in replayed graphs."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as csa2_indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2 import metadata as csa2_metadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params

from ._utils import _DECODE_PAGES, _paged_decode_batch, _PagedIndexOwner, _replay_paged_decode


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="The candidate source fills on publish and aligns its block maxima on SM100/SM103",
)
@torch.inference_mode()
def test_candidate_fill_on_publish_and_aligned_blocks_match_plain(monkeypatch):
    torch.manual_seed(4701)
    # 3,067 candidate blocks per row: not a whole number of the block Top-K's 8-column vectors.
    width = _DECODE_PAGES * _PagedIndexOwner.tokens_per_block - 40
    batch = _paged_decode_batch(4, (4100, 1000), width=width)
    metadata, state, rows = batch.metadata, batch.state, batch.rows
    options = CSA2Params(layout=batch.layout, enable_heuristic_topk=True)
    source, consumer = (CSA2Indexer(batch.layout, i, 32, 128, options) for i in (0, 1))
    # The source publishes from its masked logits' block maxima, not from a block selection.
    monkeypatch.setattr(csa2_metadata, "BLOCK_SELECT_SOURCE_MIN_ROWS", 1 << 30)
    aligned = []
    block_max = csa2_indexer.candidate_block_max_

    def probe(*args, **kwargs):
        aligned.append(args[8].shape[1] % 8 == 0)
        return block_max(*args, **kwargs)

    monkeypatch.setattr(csa2_indexer, "candidate_block_max_", probe)

    def forward():
        metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
        selected, chosen = source(state, 0, rows), consumer(state, 0, rows)
        tables = (
            metadata.csa2_candidates,
            metadata.csa2_candidate_blocks,
            metadata.csa2_candidate_counts,
        )
        return (selected, chosen, *(table[0] for table in tables))

    def run(optimized):
        monkeypatch.setattr(csa2_indexer, "_candidate_fill_on_publish", lambda: optimized)
        monkeypatch.setattr(csa2_indexer, "_aligned_candidate_blocks", lambda: optimized)
        for _ in range(2):
            outputs = forward()
        torch.cuda.synchronize()
        return [t.clone() for t in outputs]

    def check(plain, new):
        # Selections of both layers, candidates, sparse blocks and counts, bit for bit.
        for i, (expected, actual) in enumerate(zip(plain, new)):
            assert torch.equal(actual, expected), i

    plain = run(False)
    assert aligned and not any(aligned)
    aligned.clear()
    check(plain, run(True))
    assert aligned and all(aligned)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Eager forwards publish into new tensors: keep the graph's own.
        captured = forward()

    def check_replay():
        replayed = [t.clone() for t in captured]
        check(run(False), replayed)

    _replay_paged_decode(batch, graph, check_replay)
