# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host-side values on the drafter's mixed-step path, so the step does not wait for the GPU.

* DFlash's context store sizes each context row from the draft block table's per-row count. With
  KVCacheManagerV2's shared-page-table copy that count is the row width, which the host knows.
  The GPU test checks the claim against the real copy kernel with missing pages in the table.
* DSpark (DSv4) seeds its rolling windows from the chunk's first position, read from the host
  position ids.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, host_counted_block_width
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

POOLS, ROWS, WIDTH, PAGE = 2, 6, 16, 64


def _manager(per_layer: bool = False, width: int = WIDTH, pin: bool = False) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.host_kv_cache_block_offsets = torch.zeros(
        POOLS, ROWS, 2, width, dtype=torch.int32, pin_memory=pin
    )
    manager._use_per_layer_page_tables = per_layer
    return manager


def _metadata(manager, width: int = WIDTH, device: str = "cpu") -> SimpleNamespace:
    return SimpleNamespace(
        draft_kv_cache_manager=manager,
        draft_kv_cache_block_offsets=torch.zeros(
            POOLS, ROWS, 2, width, dtype=torch.int32, device=device
        ),
    )


@pytest.mark.cpu_only
def test_the_width_is_known_for_the_shared_page_table_copy():
    assert host_counted_block_width(_metadata(_manager())) == WIDTH


@pytest.mark.cpu_only
def test_other_writers_leave_the_count_to_the_device():
    class OwnCopy(KVCacheManagerV2):
        def copy_batch_block_offsets(self, *args, **kwargs):
            raise NotImplementedError

    own = object.__new__(OwnCopy)
    own.host_kv_cache_block_offsets = torch.zeros(POOLS, ROWS, 2, WIDTH, dtype=torch.int32)
    own._use_per_layer_page_tables = False

    assert host_counted_block_width(_metadata(_manager(per_layer=True))) is None
    assert host_counted_block_width(_metadata(own)) is None
    assert host_counted_block_width(_metadata(object())) is None
    assert host_counted_block_width(_metadata(None)) is None
    assert host_counted_block_width(_metadata(_manager(width=WIDTH + 4))) is None
    assert host_counted_block_width(SimpleNamespace(draft_kv_cache_manager=_manager())) is None


def _worker(counts: torch.Tensor) -> DFlashWorker:
    worker = object.__new__(DFlashWorker)
    worker._ctx_block_counts = counts
    worker._ctx_page_size = PAGE
    return worker


@pytest.mark.cpu_only
def test_ctx_alloc_uses_the_host_width_or_falls_back_to_the_counts():
    counts = torch.tensor([3, 5, 7, 0, 0, 0])
    worker = _worker(counts)
    assert worker._ctx_alloc_tokens(_metadata(_manager()), 3) == [WIDTH * PAGE] * 3
    fallback = _metadata(_manager(per_layer=True))
    assert worker._ctx_alloc_tokens(fallback, 3) == [3 * PAGE, 5 * PAGE, 7 * PAGE]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seed", range(4))
def test_the_copy_kernel_leaves_every_row_counted_at_full_width(seed):
    """The refresh's count (entries >= 0) over rows the V2 copy wrote equals the row width.

    Rows mix real pages and BAD_PAGE_INDEX (front-dropped window blocks, unallocated tail);
    the destination starts at -1 so a column the copy skipped would show up as uncounted.
    """
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    rng = random.Random(seed)
    manager = _manager(pin=True)
    host = manager.host_kv_cache_block_offsets
    for pool in range(POOLS):
        for row in range(ROWS):
            for col in range(WIDTH):
                host[pool, row, 0, col] = (
                    BAD_PAGE_INDEX if rng.random() < 0.4 else rng.randrange(0, 1000)
                )
    meta = _metadata(manager, device="cuda")
    meta.draft_kv_cache_block_offsets.fill_(-1)
    rows = rng.sample(range(ROWS), 4)
    copy_idx = torch.tensor(rows, dtype=torch.int32, pin_memory=True)
    scales = torch.tensor([2, 3], dtype=torch.int32, pin_memory=True)
    offsets = torch.tensor([1, 5], dtype=torch.int32, pin_memory=True)
    copy_batch_block_offsets_to_device(
        host,
        meta.draft_kv_cache_block_offsets,
        copy_idx,
        scales,
        offsets,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    width = host_counted_block_width(meta)
    for pool in range(POOLS):
        device_counts = (meta.draft_kv_cache_block_offsets[pool, : len(rows), 0] >= 0).sum(dim=1)
        assert device_counts.tolist() == [width] * len(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ctx_alloc_on_the_host_path_does_not_synchronize():
    counts = torch.full((ROWS,), WIDTH, dtype=torch.long, device="cuda")
    worker = _worker(counts)
    meta = _metadata(_manager(), device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        alloc = worker._ctx_alloc_tokens(meta, 4)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert alloc == (counts[:4] * PAGE).tolist()


class _FakeDraftModel:
    def __init__(self):
        self.writes = []

    def write_context_windows(self, hidden, positions, window):
        self.writes.append((hidden.shape[0], positions))


def _dspark_worker(num_slots: int = 4, win: int = 8):
    from collections import deque

    from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker

    worker = object.__new__(DSv4DSparkWorker)
    worker._win = win
    worker._req_to_slot = {}
    worker._free_slots = deque(range(num_slots))
    worker._ctx_len = torch.zeros(num_slots + 1, dtype=torch.long, device="cuda")
    worker._valid_len = torch.zeros(num_slots + 1, dtype=torch.long, device="cuda")
    worker._position_initialized = torch.zeros(num_slots + 1, dtype=torch.bool, device="cuda")
    worker._kv_windows = torch.zeros(num_slots + 1, 1, win, 4, device="cuda")
    worker._confidence_logits = None
    return worker


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("padded", [False, True])
def test_dspark_window_seeding_reads_host_positions(padded, monkeypatch):
    """Same state as reading the device ids, without reading a device value back."""
    seq_lens = [5, 3]
    host_positions = torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int32)
    device_positions = host_positions.to("cuda")
    if padded:
        device_positions = torch.cat(
            [device_positions, torch.zeros(4, dtype=torch.int32, device="cuda")]
        )
    num_tokens = device_positions.numel()

    def run(with_host: bool):
        worker = _dspark_worker()
        # Request 11 continues a chunk (position 7), request 10 starts at 0.
        worker._assign_slot(11, reset=True)
        spec_metadata = SimpleNamespace(
            get_hidden_states=lambda n: torch.ones(num_tokens, 4, device="cuda"),
            context_capture_lens=None,
            request_ids=[10, 11],
            host_position_ids=host_positions if with_host else None,
        )
        attn_metadata = SimpleNamespace(num_contexts=2, _seq_lens=torch.tensor(seq_lens))
        model = _FakeDraftModel()
        with monkeypatch.context() as patch:
            if with_host:
                # The slot bookkeeping writes host scalars into device tensors, which
                # never waits on the GPU; reading a device value back would.
                def no_readback(tensor):
                    raise AssertionError("read a device value back")

                patch.setattr(torch.Tensor, "item", no_readback)
                patch.setattr(torch.Tensor, "tolist", no_readback)
            worker._seed_context_windows(
                model, spec_metadata, attn_metadata, device_positions, num_tokens
            )
        return worker, model

    host_worker, _ = run(True)
    device_worker, _ = run(False)
    assert host_worker._req_to_slot == device_worker._req_to_slot
    for name in ("_ctx_len", "_valid_len", "_position_initialized"):
        assert torch.equal(getattr(host_worker, name), getattr(device_worker, name)), name
