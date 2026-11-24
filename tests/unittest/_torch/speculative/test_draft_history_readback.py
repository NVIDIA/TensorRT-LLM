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
"""The DFlash worker's deferred draft-history readback on a GPU.

Every step replays a CUDA graph that advances the context lengths and position offsets of the
step's slots, then takes the reference readback at once (the gathers and the copy on the
execution stream, as ``DraftHistoryUpdate.capture`` queues them), records the worker's snapshot,
queues a stand-in for the sampler, issues the snapshot (gathers on the execution stream, copy on
the readback side stream), and at once queues the next step's writes of the same tables and
slots. The host runs ahead: the execution stream is held by a sleep before each replay and the
side stream by another before some copies. Updates are published one step late, as the executor
does. The host values and the published histories must be bitwise equal to the reference, with
requests arriving, leaving and being replaced between the snapshot and its publication.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import standalone_draft_cache as sdc
from tensorrt_llm._torch.pyexecutor.kv_cache.standalone_draft_cache import DraftHistoryUpdate
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.spec_tail_kernels import draft_history_values

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
NUM_SLOTS = 9
MAX_BATCH = 8
SLEEP = 2_000_000  # cycles: about 1 ms, longer than the host's work for a step


class _Manager:
    has_standalone_draft_cache = True

    def __init__(self):
        self.kv_cache_map = {}
        self.history = {}

    def set_draft_history(self, request_id, length, position):
        self.history[request_id] = (length, position)


def _worker():
    worker = object.__new__(DFlashWorker)
    worker._ctx_len = torch.randint(0, 1 << 20, (NUM_SLOTS,), dtype=torch.long, device=DEV)
    worker._ctx_position_offset = torch.randint(-64, 64, (NUM_SLOTS,), dtype=torch.long, device=DEV)
    worker._managed_snapshot_slots = torch.zeros(MAX_BATCH, dtype=torch.long, device=DEV)
    worker._managed_request_ids = ()
    worker._pending_history = None
    return worker


def _reference(worker, manager):
    slots = worker._managed_snapshot_slots[: len(worker._managed_request_ids)]
    lengths = worker._ctx_len[slots]
    positions = lengths + worker._ctx_position_offset[slots]
    return DraftHistoryUpdate.capture(
        manager, worker._managed_request_ids, torch.stack((lengths, positions), dim=1)
    )


def _batch(step):
    """Request ids of a step: none, then arrivals, departures and a slot reused by another id."""
    plan = [(), (100,), (100, 101, 102), (101,), (101, 103), (), (104, 103, 105, 106), (104,)]
    return plan[step % len(plan)]


@pytest.mark.parametrize("n", [0, 1, 3, 8])
def test_gather_kernel_equals_the_eager_ops(n):
    gen = torch.Generator().manual_seed(n)
    ctx_len = torch.randint(0, 1 << 40, (NUM_SLOTS,), generator=gen, dtype=torch.long).to(DEV)
    offset = torch.randint(-(1 << 20), 1 << 20, (NUM_SLOTS,), generator=gen).to(DEV)
    slots = torch.randperm(NUM_SLOTS, generator=gen)[:n].to(DEV)
    lengths = ctx_len[slots]
    want = torch.stack((lengths, lengths + offset[slots]), dim=1)
    got = draft_history_values(ctx_len, offset, slots)
    assert got.shape == (n, 2) and got.dtype == torch.long
    assert torch.equal(got, want)
    on_cpu = draft_history_values(ctx_len.cpu(), offset.cpu(), slots.cpu())
    assert torch.equal(on_cpu, want.cpu())


@pytest.mark.parametrize("hold_side", [False, True])
def test_deferred_readback_matches_the_reference(hold_side):
    rng = torch.Generator().manual_seed(7 + hold_side)
    worker = _worker()
    engine, reference = _Manager(), _Manager()

    # The forward: advance the lengths of the step's slots by an accepted count, in a graph.
    acc = torch.zeros(MAX_BATCH, dtype=torch.long, device=DEV)
    slots_static = torch.zeros(MAX_BATCH, dtype=torch.long, device=DEV)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        worker._ctx_len.index_add_(0, slots_static, acc)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._ctx_len.index_add_(0, slots_static, acc)
        worker._ctx_position_offset.index_add_(0, slots_static, acc // 3)
    torch.cuda.synchronize()
    sampler_store = torch.zeros(1 << 16, device=DEV)

    pending = []
    checked = 0
    for step in range(24):
        rids = _batch(step)
        if step:
            # Departures and a replaced cache instance between the snapshot and its publication.
            for m in (engine, reference):
                m.kv_cache_map.pop(102, None)
                if step == 9:
                    m.kv_cache_map[103] = SimpleNamespace(is_active=True)
        for m in (engine, reference):
            for rid in rids:
                m.kv_cache_map.setdefault(rid, SimpleNamespace(is_active=True))
        # This step's preparation: the slots of its requests (execution stream).
        worker._managed_request_ids = tuple(rids)
        if rids:
            slots = torch.tensor([(r - 100) % (NUM_SLOTS - 1) for r in rids], dtype=torch.long)
            worker._managed_snapshot_slots[: len(rids)].copy_(slots.pin_memory(), non_blocking=True)
        acc.copy_(torch.randint(1, 9, (MAX_BATCH,), generator=rng).pin_memory(), non_blocking=True)
        slots_static.copy_(worker._managed_snapshot_slots, non_blocking=True)
        torch.cuda._sleep(SLEEP)
        graph.replay()

        want = _reference(worker, reference) if rids else None
        worker._ctx_kv_manager = engine
        deferred = worker.snapshot_managed_draft_history()
        sampler_store.add_(1.0)  # the sampler, queued between the snapshot and its issue
        if hold_side and step % 3 == 0:
            with torch.cuda.stream(sdc._readback_stream(torch.device(DEV))):
                torch.cuda._sleep(4 * SLEEP)
        if not rids:
            assert want is None and deferred is None
        else:
            got = deferred.issue()
            assert got.request_ids == want.request_ids
            pending.append((want, got))
        # The next step's preparation writes the same tables and slots at once (execution
        # stream, waiting for no copy).
        idx = torch.randint(0, NUM_SLOTS, (3,), generator=rng).pin_memory()
        vals = torch.randint(0, 1 << 20, (3,), generator=rng).pin_memory()
        worker._ctx_len.index_copy_(
            0, idx.to(DEV, non_blocking=True), vals.to(DEV, non_blocking=True)
        )
        worker._ctx_position_offset.add_(1)
        # The executor publishes a step's update after the next step is queued.
        while len(pending) > 1:
            a, b = pending.pop(0)
            a.publish()
            b.publish()
            assert torch.equal(a.values_host, b.values_host)
            assert engine.history == reference.history
            checked += 1
    for a, b in pending:
        a.publish()
        b.publish()
        assert torch.equal(a.values_host, b.values_host)
        checked += 1
    assert engine.history == reference.history
    assert checked >= 15


def test_no_unified_cache_takes_no_snapshot():
    worker = _worker()
    worker._ctx_kv_manager = SimpleNamespace(has_standalone_draft_cache=False)
    worker._managed_request_ids = (100,)
    assert worker.snapshot_managed_draft_history() is None


def test_issue_under_graph_capture_raises():
    """Neither the executor's issue nor the worker's guard may record the readback into a graph."""
    worker = _worker()
    manager = _Manager()
    manager.kv_cache_map[100] = SimpleNamespace(is_active=True)
    worker._ctx_kv_manager = manager
    worker._managed_request_ids = (100,)
    deferred = worker.snapshot_managed_draft_history()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with pytest.raises(RuntimeError, match="graph capture"):
            with torch.cuda.graph(graph, stream=stream):
                worker._issue_pending_history()
    torch.cuda.current_stream().wait_stream(stream)
    assert not deferred.issued and worker._pending_history is deferred
    worker._issue_pending_history()
    assert deferred.issued and worker._pending_history is None
    assert deferred.issue().values_host.shape == (1, 2)
