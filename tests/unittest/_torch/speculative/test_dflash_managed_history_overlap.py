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
"""The managed draft history never rewinds a resident request in overlap order.

Under the overlap scheduler the host prepares step N+1 (``prepare_managed_draft_cache`` ->
``_prepare_managed_history``) before step N's draft history reaches the KV manager
(``snapshot_managed_draft_history`` -> ``publish``). A resident request's published history is
then one step late. The worker keeps the context length and position offset the device advanced
for such a request, and binds the published history only for a request that becomes resident:
a new request, or one whose cache changed. Both orders end in the same state.

The snapshot only records what to read. Its gathers are queued when the executor issues it, after
the iteration's sampler, or at the latest before the worker's next write to the state they read,
so they always precede the next preparation's writes on the execution stream.

Every DFlash worker inherits this host preparation from the engine worker, so it is checked here
once, on CPU, with a KV manager stand-in.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.speculative.dflash as dflash
from tensorrt_llm._torch.pyexecutor.kv_cache.standalone_draft_cache import StandaloneDraftHistory
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_MAX_BATCH = 4


class _Manager:
    """The KV manager's draft-history surface: a cache object per request and the history each
    request last published."""

    has_standalone_draft_cache = True
    draft_cp_rank = 0

    def __init__(self):
        self.kv_cache_map = {}
        self.published = {}
        self._draft_dummy_request_ids = set()

    def admit(self, request_id: int, length: int) -> None:
        """A request becomes resident with a history of ``length`` tokens (its prompt)."""
        self.kv_cache_map[request_id] = SimpleNamespace(is_active=True)
        self.published[request_id] = StandaloneDraftHistory(length, length)

    def get_draft_history(self, request_id: int):
        return self.published.get(request_id)

    def set_draft_history(self, request_id: int, length: int, position: int) -> None:
        self.published[request_id] = StandaloneDraftHistory(length, position)


class _SlotTable:
    def __init__(self):
        self.tensor = None

    def update(self, mapping: torch.Tensor) -> None:
        self.tensor = mapping.clone()


def _worker(manager: _Manager, monkeypatch) -> DFlashWorker:
    """An engine DFlash worker with the unified draft cache's slot state, on CPU tensors."""
    monkeypatch.setattr(dflash, "prefer_pinned", lambda: False)
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=3), Mapping())
    num_slots = _MAX_BATCH + 1
    worker._ctx_kv_manager = manager
    worker._dummy_slot = _MAX_BATCH
    worker._graph_dummy_id_floor = 1 << 62
    worker._max_ctx = 1 << 20
    worker._ctx_len = torch.zeros(num_slots, dtype=torch.long)
    worker._ctx_position_offset = torch.zeros_like(worker._ctx_len)
    worker._ctx_len_host = [0] * num_slots
    worker._batch_to_slot_buffer = _SlotTable()
    worker._managed_snapshot_slots = torch.empty(_MAX_BATCH, dtype=torch.long)
    worker._free_slots = deque(range(_MAX_BATCH))
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._managed_cache_bindings = {}
    worker._managed_request_ids = ()
    return worker


def _device_step(worker: DFlashWorker, request_ids, accepted) -> None:
    """The device side of a generation step: each row's context grows by its accepted tokens."""
    slots = torch.tensor([worker._req_to_slot[r] for r in request_ids], dtype=torch.long)
    worker._advance_ctx_len(slots, torch.tensor(accepted, dtype=torch.long))


def _state(worker: DFlashWorker, request_ids) -> dict:
    out = {}
    for r in request_ids:
        slot = worker._req_to_slot[r]
        out[r] = (
            int(worker._ctx_len[slot]),
            int(worker._ctx_position_offset[slot]),
            worker._req_ctx_pos.get(r),
        )
    return out


def _two_steps(order: str, monkeypatch, forget_bindings: bool = False):
    """Step one on requests 1 and 2; request 3 joins; step two's host preparation, with step one's
    history published before it ("sequential") or after it ("overlap")."""
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    manager.admit(2, 20)
    assert worker._prepare_managed_history([1, 2]) == [0, 1]
    _device_step(worker, [1, 2], [3, 2])
    update = worker.snapshot_managed_draft_history()
    if order == "sequential":
        update.publish()
    manager.admit(3, 30)
    if forget_bindings:
        worker._managed_cache_bindings.clear()
    restored = worker._prepare_managed_history([1, 2, 3])
    if order == "overlap":
        update.publish()
    return worker, manager, restored


@pytest.mark.parametrize("order", ["sequential", "overlap"])
def test_resident_requests_keep_the_device_advanced_history(order, monkeypatch):
    worker, manager, restored = _two_steps(order, monkeypatch)
    # Only request 3 (row 2) is bound from the manager; 1 and 2 keep what the device advanced.
    assert restored == [2]
    assert _state(worker, [1, 2, 3]) == {1: (13, 0, 10), 2: (22, 0, 20), 3: (30, 0, 30)}
    assert worker._ctx_len_host[worker._req_to_slot[3]] == 30
    # Step one's readback reaches the manager in either order.
    assert manager.published[1] == StandaloneDraftHistory(13, 13)
    assert manager.published[2] == StandaloneDraftHistory(22, 22)
    assert worker._managed_request_ids == (1, 2, 3)


def test_overlap_order_equals_sequential_order(monkeypatch):
    sequential, _, _ = _two_steps("sequential", monkeypatch)
    overlap, _, _ = _two_steps("overlap", monkeypatch)
    assert _state(overlap, [1, 2, 3]) == _state(sequential, [1, 2, 3])
    assert torch.equal(overlap._ctx_len, sequential._ctx_len)
    assert torch.equal(overlap._ctx_position_offset, sequential._ctx_position_offset)
    assert torch.equal(
        overlap._batch_to_slot_buffer.tensor, sequential._batch_to_slot_buffer.tensor
    )


def test_a_rebind_from_the_late_history_would_rewind(monkeypatch):
    """The trap the rule avoids, shown on purpose: a worker that rebound every resident row would
    take step one's lengths from the one-step-late history in overlap order."""
    worker, _, restored = _two_steps("overlap", monkeypatch, forget_bindings=True)
    assert restored == [0, 1, 2]
    assert _state(worker, [1, 2])[1][0] == 10 and _state(worker, [1, 2])[2][0] == 20


def test_a_request_whose_cache_changed_binds_the_published_history(monkeypatch):
    """A new cache under the same request id (the old one released) is a new history: it is
    bound from the manager, not kept from the old slot."""
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    worker._prepare_managed_history([1])
    _device_step(worker, [1], [4])
    manager.admit(1, 50)  # released and admitted again: a new cache object
    assert worker._prepare_managed_history([1]) == [0]
    assert _state(worker, [1]) == {1: (50, 0, 50)}


def test_a_request_that_leaves_frees_its_slot(monkeypatch):
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    manager.admit(2, 20)
    worker._prepare_managed_history([1, 2])
    slot = worker._req_to_slot[2]
    worker._prepare_managed_history([1])
    assert 2 not in worker._req_to_slot and slot in worker._free_slots
    assert int(worker._ctx_len[slot]) == 0 and int(worker._ctx_position_offset[slot]) == 0


# The snapshot is deferred: the executor issues its gathers once the iteration's sampling is
# queued, and the worker issues a pending one before its next write to the state they read. The
# gathers therefore precede the next preparation's writes in either order.


def _two_steps_issued(issue_order: str, monkeypatch):
    """``_two_steps`` in overlap order, the snapshot issued by the executor before the next
    preparation ("executor") or only by the worker's next write ("worker")."""
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    manager.admit(2, 20)
    worker._prepare_managed_history([1, 2])
    _device_step(worker, [1, 2], [3, 2])
    update = worker.snapshot_managed_draft_history()
    assert not update.issued
    if issue_order == "executor":
        update = update.issue()
    manager.admit(3, 30)
    worker._prepare_managed_history([1, 2, 3])
    update.publish()
    return worker, manager, update


@pytest.mark.parametrize("issue_order", ["executor", "worker"])
def test_the_gathers_precede_the_next_preparation(issue_order, monkeypatch):
    worker, manager, _ = _two_steps_issued(issue_order, monkeypatch)
    sequential, sequential_manager, _ = _two_steps("sequential", monkeypatch)
    assert _state(worker, [1, 2, 3]) == _state(sequential, [1, 2, 3])
    assert manager.published == sequential_manager.published
    assert manager.published[1] == StandaloneDraftHistory(13, 13)
    assert manager.published[2] == StandaloneDraftHistory(22, 22)


def test_the_next_write_issues_a_pending_snapshot(monkeypatch):
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    worker._prepare_managed_history([1])
    _device_step(worker, [1], [5])
    update = worker.snapshot_managed_draft_history()
    worker._write_ctx_len({worker._req_to_slot[1]: 0})
    assert update.issued and worker._pending_history is None
    assert update.issue().values_host.tolist() == [[15, 15]]


def test_deferred_values_equal_the_eager_gathers(monkeypatch):
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    for request_id, length in ((1, 10), (2, 20), (3, 7)):
        manager.admit(request_id, length)
    worker._prepare_managed_history([3, 1, 2])
    _device_step(worker, [3, 1, 2], [1, 4, 2])
    worker._ctx_position_offset[worker._req_to_slot[2]] = 5
    slots = worker._managed_snapshot_slots[:3]
    lengths = worker._ctx_len[slots]
    eager = torch.stack((lengths, lengths + worker._ctx_position_offset[slots]), dim=1)
    update = worker.snapshot_managed_draft_history().issue()
    assert update.request_ids == (3, 1, 2)
    assert torch.equal(update.values_host, eager)
    assert update.issue() is update
    assert worker.snapshot_managed_draft_history().issue() is not update


def test_a_read_after_the_next_preparation_would_rewind(monkeypatch):
    """The order the guard keeps, broken on purpose: the gathers queued only after the next
    preparation rebinds requests 1 and 2 read the preparation's lengths, not the forward's."""
    manager = _Manager()
    worker = _worker(manager, monkeypatch)
    manager.admit(1, 10)
    manager.admit(2, 20)
    worker._prepare_managed_history([1, 2])
    _device_step(worker, [1, 2], [3, 2])
    update = worker.snapshot_managed_draft_history()
    worker._pending_history = None  # bypass the guard
    worker._managed_cache_bindings.clear()
    worker._prepare_managed_history([1, 2])
    update.publish()
    assert manager.published[1] == StandaloneDraftHistory(10, 10)
    assert manager.published[2] == StandaloneDraftHistory(20, 20)
