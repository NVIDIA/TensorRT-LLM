# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark slot bookkeeping when requests join or leave a decode batch must not wait for the GPU."""

import collections
import contextlib

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata, DSv4DSparkWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_ROWS = 6  # five request slots and the scratch row
WINDOW = 4


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _worker():
    worker = object.__new__(DSv4DSparkWorker)
    worker._win_inited = True
    worker._scratch_slot = NUM_ROWS - 1
    worker._graph_dummy_id_floor = 1 << 40
    worker._ctx_len = torch.full((NUM_ROWS,), 7, dtype=torch.long, device="cuda")
    worker._valid_len = torch.full((NUM_ROWS,), 3, dtype=torch.long, device="cuda")
    worker._position_initialized = torch.ones(NUM_ROWS, dtype=torch.bool, device="cuda")
    worker._kv_windows = torch.ones(NUM_ROWS, 2, WINDOW, 8, dtype=torch.bfloat16, device="cuda")
    worker._batch_to_slot = torch.zeros(NUM_ROWS - 1, dtype=torch.long, device="cuda")
    worker._confidence_logits = None
    worker._free_slots = collections.deque(range(NUM_ROWS - 1))
    worker._req_to_slot = {}
    return worker


def _assert_reset(worker, row):
    assert worker._ctx_len[row].item() == 0
    assert worker._valid_len[row].item() == 0
    assert not worker._position_initialized[row].item()
    assert worker._kv_windows[row].abs().sum().item() == 0


def test_assign_slot_is_stream_ordered():
    worker = _worker()
    with _no_host_sync():
        first = worker._assign_slot(42, reset=False)
        again = worker._assign_slot(42, reset=True)
    assert (first, again) == (0, 1)
    for row in (0, 1):
        _assert_reset(worker, row)
    assert worker._ctx_len[2].item() == 7 and worker._position_initialized[2].item()


def test_join_and_leave_are_stream_ordered():
    worker = _worker()
    metadata = object.__new__(DSparkSpecMetadata)
    metadata.batch_indices_cuda = torch.zeros(NUM_ROWS, dtype=torch.int, device="cuda")
    metadata._dspark_worker = worker
    with _no_host_sync():
        # Two requests join, then one leaves and a third joins, the way prepare()
        # sees consecutive decode batches.
        for request_ids in ([100, 101], [101, 102]):
            metadata.request_ids = request_ids
            metadata.num_generations = len(request_ids)
            metadata.prepare()
    assert worker._req_to_slot == {101: 1, 102: 2}
    for row in (0, 1, 2):
        _assert_reset(worker, row)
    # The scratch row restarts its position every iteration; its window is never read.
    assert worker._ctx_len[worker._scratch_slot].item() == 0
    assert worker._valid_len[worker._scratch_slot].item() == 0
    assert not worker._position_initialized[worker._scratch_slot].item()
    torch.testing.assert_close(
        worker._batch_to_slot[:2], torch.tensor([1, 2], device="cuda"), rtol=0, atol=0
    )
