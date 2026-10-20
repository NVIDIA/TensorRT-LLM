# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash prefill: the prepare-time host snapshot of the draft page counts."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_SLOTS, MAX_BLOCKS = 9, 16


def _worker(divisor: int = 2) -> DFlashWorker:
    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = torch.zeros(NUM_SLOTS, MAX_BLOCKS, dtype=torch.int32, device="cuda")
    worker._ctx_block_counts = torch.zeros(NUM_SLOTS, dtype=torch.long, device="cuda")
    worker._ctx_pool_idx = 1
    worker._ctx_block_divisor = divisor
    worker._ctx_kv_manager = object()
    worker._ctx_offsets_src = None
    worker._ctx_counts_snapshot = None
    return worker


def _fill(offsets: torch.Tensor, counts: list[int], divisor: int) -> None:
    """Rows of pool 1 with counts[i] encoded pages, the rest placeholders (-1)."""
    offsets.fill_(-1)
    for row, count in enumerate(counts):
        pages = torch.arange(count, dtype=torch.int32, device="cuda") * divisor + 7 * row
        offsets[1, row, :, :count] = pages


def test_snapshot_matches_refreshed_counts() -> None:
    worker = _worker()
    offsets = torch.empty(2, NUM_SLOTS, 2, MAX_BLOCKS, dtype=torch.int32, device="cuda")
    meta = SimpleNamespace(draft_kv_cache_block_offsets=offsets)
    # No refresh seen yet: nothing to snapshot.
    worker._snapshot_ctx_block_counts(3)
    assert worker._ctx_counts_snapshot is None
    _fill(offsets, [1, 2, 3, 4], 2)
    assert worker._refresh_ctx_block_tables(meta, 4)
    for step, counts in enumerate(([5, 0, 16], [16, 3, 1, 9, 2], [2])):
        # The next iteration's attn_metadata.prepare() refills the same tensor.
        _fill(offsets, counts, 2)
        n = len(counts)
        worker._snapshot_ctx_block_counts(n)
        # The forward runs, then refreshes the table from the same tensor.
        torch.cuda._sleep(1_000_000)
        assert worker._refresh_ctx_block_tables(meta, n)
        snap = worker._take_ctx_block_counts(meta, n)
        assert snap is not None and not snap.is_cuda, step
        assert snap.tolist() == counts == worker._ctx_block_counts[:n].tolist()
        # Consumed once.
        assert worker._take_ctx_block_counts(meta, n) is None


@pytest.mark.parametrize("change", ["num_contexts", "tensor", "manager", "pool"])
def test_snapshot_is_rejected_when_the_binding_changed(change: str) -> None:
    worker = _worker()
    offsets = torch.empty(2, NUM_SLOTS, 2, MAX_BLOCKS, dtype=torch.int32, device="cuda")
    meta = SimpleNamespace(draft_kv_cache_block_offsets=offsets)
    _fill(offsets, [3, 1], 2)
    worker._refresh_ctx_block_tables(meta, 2)
    worker._snapshot_ctx_block_counts(2)
    n = 2
    if change == "num_contexts":
        n = 1
    elif change == "tensor":
        meta = SimpleNamespace(draft_kv_cache_block_offsets=offsets.clone())
    elif change == "manager":
        worker._ctx_kv_manager = object()
    else:
        worker._ctx_pool_idx = 0
    assert worker._take_ctx_block_counts(meta, n) is None


def test_no_snapshot_under_graph_capture() -> None:
    worker = _worker()
    offsets = torch.empty(2, NUM_SLOTS, 2, MAX_BLOCKS, dtype=torch.int32, device="cuda")
    meta = SimpleNamespace(draft_kv_cache_block_offsets=offsets)
    _fill(offsets, [3, 1], 2)
    worker._refresh_ctx_block_tables(meta, 2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._snapshot_ctx_block_counts(2)
    assert worker._ctx_counts_snapshot is None
