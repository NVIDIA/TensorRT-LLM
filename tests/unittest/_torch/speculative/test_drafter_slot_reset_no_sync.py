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
"""Drafter slot bookkeeping when requests join or leave a decode batch must not wait for the GPU.

The overlap executor prepares the next batch while the previous decode step is still queued on
the stream. A host sync there (e.g. assigning a Python scalar through an index of a CUDA tensor,
which stages the scalar and copies it with a blocking copy) leaves the GPU idle until the next
launch.
"""

import collections
import contextlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FLOOR = 1 << 40  # request ids from here on are CUDA-graph padding dummies


@contextlib.contextmanager
def _no_host_sync():
    """Fail on any host-device synchronization inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _worker(num_slots=8):
    lengths = list(range(10, 10 + num_slots + 1))
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _has_unified_draft_cache=lambda: False,
        _req_to_slot={},
        _req_ctx_pos={},
        _free_slots=collections.deque(range(num_slots)),
        _graph_dummy_id_floor=FLOOR,
        _dummy_slot=num_slots,
        _ctx_len=torch.tensor(lengths, dtype=torch.long, device="cuda"),
        _ctx_len_host=list(lengths),
        _batch_to_slot_buffer=DeviceCopyBuffer(
            torch.zeros(num_slots, dtype=torch.long, device="cuda"), pin_memory=True
        ),
    )
    worker._write_ctx_len = partial(DFlashWorker._write_ctx_len, worker)
    worker._assign_slot = partial(DFlashWorker._assign_slot, worker)
    return worker


def _prepare(worker, request_ids):
    metadata = SimpleNamespace(
        request_ids=request_ids,
        num_generations=len(request_ids),
        _dflash_worker=worker,
        _batch_indices_buffer=DeviceCopyBuffer(
            torch.zeros(16, dtype=torch.int, device="cuda"), pin_memory=True
        ),
    )
    DFlashSpecMetadata.prepare(metadata)


def test_dflash_join_and_leave_are_stream_ordered():
    worker = _worker()
    initial = worker._ctx_len.clone()
    with _no_host_sync():
        # Two requests join (each gets a fresh slot), then one of them leaves and a third
        # joins, the way prepare() sees consecutive decode batches.
        _prepare(worker, [100, 101])
        _prepare(worker, [101, 102])
    torch.cuda.synchronize()
    assert worker._req_to_slot == {101: 1, 102: 2}
    assert list(worker._free_slots) == [3, 4, 5, 6, 7, 0]
    expected = initial.clone()
    expected[[0, 1, 2]] = 0
    assert torch.equal(worker._ctx_len, expected)
    assert worker._batch_to_slot_buffer.tensor[:2].tolist() == [1, 2]


def test_dflash_reset_slot_is_stream_ordered():
    worker = _worker()
    worker._req_to_slot = {7: 4}
    worker._free_slots.remove(4)
    worker._ctx_len[4] = 33
    worker._ctx_len_host[4] = 33
    with _no_host_sync():
        slot = worker._assign_slot(7, reset=True)
    torch.cuda.synchronize()
    assert slot == 0
    assert worker._ctx_len[4].item() == 0 and worker._ctx_len[0].item() == 0
    assert worker._ctx_len_host[4] == 0 and worker._ctx_len_host[0] == 0
