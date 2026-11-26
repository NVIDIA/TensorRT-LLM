# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate consumers' large decode batches select with the block Top-K what the radix Top-K selects."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as csa2_indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params
from tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits import sparse_logits_topk_dsl

from ._utils import _paged_decode_batch, _replay_paged_decode


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="The block Top-K runs on SM100/SM103",
)
@torch.inference_mode()
def test_consumer_block_topk_matches_radix_topk(monkeypatch):
    torch.manual_seed(4605)
    # Full, partial-block and short (at most 512 columns) rows.
    batch = _paged_decode_batch(4, (4100, 1000, 300))
    metadata, state, rows = batch.metadata, batch.state, batch.rows
    options = CSA2Params(layout=batch.layout, enable_heuristic_topk=True)
    source, consumer = (CSA2Indexer(batch.layout, i, 32, 128, options) for i in (0, 1))
    launched = []
    kernel = sparse_logits_topk_dsl.sparse_logits_topk_
    monkeypatch.setattr(
        sparse_logits_topk_dsl,
        "sparse_logits_topk_",
        lambda *a, **kw: launched.append(1) or kernel(*a, **kw),
    )

    def forward():
        metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
        source(state, 0, rows)
        return consumer(state, 0, rows)

    min_rows = csa2_indexer._CONSUMER_BLOCK_TOPK_MIN_ROWS

    def run(block_topk):
        # A row bound above the batch keeps the radix TopK.
        monkeypatch.setattr(
            csa2_indexer, "_CONSUMER_BLOCK_TOPK_MIN_ROWS", min_rows if block_topk else 1 << 30
        )
        for _ in range(2):
            selected = forward()
        torch.cuda.synchronize()
        return selected.clone()

    old = run(False)
    assert not launched
    new = run(True)
    assert len(launched) == 2
    # The same positions in the same order, ties at the k-th score included.
    assert torch.equal(new, old)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()

    def check_replay():
        replayed = captured.clone()
        assert torch.equal(replayed, run(False))

    _replay_paged_decode(batch, graph, check_replay)
