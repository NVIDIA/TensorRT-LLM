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
"""Admission back-pressure of a pipeline-parallel DFlash context server in the real PP loop.

The real ``PyExecutor._executor_loop_pp`` with the stage agreement on the
threaded loop harness of ``test_pp_ring_depth.py``, with the admission ledger
of a pipeline-parallel DFlash context server (``pp_admission.py``). Stand-in: a block pool
per stage in which a running prompt holds all of its blocks from its first
chunk to its last; the last stage's pool fits only about one prompt, the first
rank's all of them.

Checked: the followers' reports carry their pool snapshots on the existing
report message (every message matched); the first rank holds new prompts
back until the last stage has room, so no follower ever fails to allocate a
chunk it adopts; every stage runs the same chunks; every prompt finishes. On
any other server nothing changes: no snapshot rides on the reports and nothing
is held.
"""

import sys
import types

import pytest
import test_pp_ring_depth as ring

from tensorrt_llm._torch.disaggregation import pp_dflash_context
from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.pp_admission import (
    ADMISSION_REPORT_KEY,
    Constraint,
    PPAdmissionLedger,
    StageSnapshot,
    blocks_for,
)
from tensorrt_llm._torch.pyexecutor.pp_ctx_diag import PPCtxDiag
from tensorrt_llm._torch.pyexecutor.pp_stage_agree import PPStageAgreement
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only

TPB = 64
PROMPTS = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1, 64 + 7, 4 * 64, 64]


class _BlockPool:
    """A stage's KV pool in blocks with one full-attention life cycle: a running
    prompt holds ``blocks_for(prompt)`` blocks from its first chunk until its last
    chunk was handled on this stage; a reservation (reused depth 0) holds none."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.kv_cache_map = {}
        self.cost = {}
        self.peak = 0
        self.snapshots = 0

    def probe_context_reuse(self, req, include_claimed=False):
        return 0

    def used(self) -> int:
        return sum(self.cost.values())

    def reserve(self, req, cap):
        self.release(req)
        self.kv_cache_map[req.py_request_id] = "reserved"
        return 0

    def release(self, req):
        if self.kv_cache_map.get(req.py_request_id) == "reserved":
            del self.kv_cache_map[req.py_request_id]

    def run(self, req) -> bool:
        need = blocks_for(req.prompt_len, TPB)
        if self.used() + need > self.capacity:
            return False
        self.kv_cache_map[req.py_request_id] = "running"
        self.cost[req.py_request_id] = need
        self.peak = max(self.peak, self.used())
        return True

    def finish(self, request_id: int) -> None:
        self.kv_cache_map.pop(request_id, None)
        self.cost.pop(request_id, None)

    def pp_admission_snapshot(self) -> StageSnapshot:
        self.snapshots += 1
        return StageSnapshot(
            TPB, (Constraint(self.capacity - self.used(), None, self.capacity, (None,)),)
        )


class _BlockExecutor(ring._AgreeingExecutor):
    """First-rank FCFS pass over a ``_BlockPool``: a first chunk is admitted only
    if its prompt fits the first rank's own pool, as the V2 scheduler would."""

    def _schedule(self):
        pool = self.kv_cache_manager
        self._refused = False
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self._schedulable_active_requests():
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            if req.context_current_position == 0 and not pool.run(req):
                req.py_pp_page_refused = True
                self._refused = True
                continue
            self._first_scheduled.setdefault(req.request_id, self.iter_counter)
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)

    def _pp_stage_agree_exchange(self):
        super()._pp_stage_agree_exchange()
        if self.dist.rank == 0 and self._pp_admission is not None:
            self._held_log.append(set(self._pp_admission_held))

    def _handle_executed_batch(self, executed_batch):
        super()._handle_executed_batch(executed_batch)
        if executed_batch is not None:
            for req in executed_batch.scheduled_requests.context_requests_last_chunk:
                self.kv_cache_manager.finish(req.request_id)


def _run(
    pp_size,
    ring_depth,
    capacities,
    admission,
    monkeypatch,
    eager_mpi=False,
    reuse_enabled=True,
    ctx_diag=False,
):
    num_micro_batches = ring.ring_slots(pp_size, ring_depth)
    lag = ring.probe_lag(ring.relay_lag(pp_size, num_micro_batches, False))
    reports = []
    adoptions = []
    held_log = []
    original = ring._make_executor

    def make(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        exe.__class__ = _BlockExecutor
        exe._first_scheduled = {}
        exe._held_log = held_log
        exe._refused = False
        for req in exe.active_requests:
            req.py_draft_tokens = None
        exe._pp_stage_agree = PPStageAgreement(rank, pp_size, lag, reuse_enabled=reuse_enabled)
        exe._pp_admission = PPAdmissionLedger(pp_size, TPB) if admission else None
        if ctx_diag:
            # Only the admission holds are counted here; the per-iteration chunk
            # counting has its own test.
            exe._pp_ctx_diag = PPCtxDiag(rank, pp_size)
            exe._pp_ctx_diag_step = lambda scheduled_batch: None
        pool = _BlockPool(capacities[rank])
        exe.kv_cache_manager = pool
        exe.enable_joint_kv_cache_reuse = False
        exe.scheduler.context_pages_refused_last_pass = lambda: exe._refused
        exe.scheduler.pages_return_pending = lambda active: False
        exe.scheduler.release_context_admission = pool.release
        exe.scheduler.reserve_first_context_chunk = pool.reserve
        exe.scheduler.reservation_leaves_chunk_room = lambda req: True

        def adopt_first(req, position, size):
            if reuse_enabled:
                assert pool.kv_cache_map.get(req.request_id) == "reserved", "no reservation"
            if not pool.run(req):
                return False
            adoptions.append((rank, req.request_id))
            return ring._adopt_first(req, position, size)

        exe.scheduler.adopt_first_context_chunk = adopt_first
        send = exe.dist.isend_object

        def isend_object(obj, dest, tag):
            if tag == PPCommTag.STAGE_REUSE_REPORT:
                reports.append((rank, dict(obj)))
            return send(obj, dest, tag)

        exe.dist.isend_object = isend_object
        return exe

    monkeypatch.setattr(ring, "_make_executor", make)
    executors, log, _, n = ring._run_pipeline(
        pp_size,
        ring_depth,
        PROMPTS,
        64,
        True,
        2,
        lambda mb: 0.002,
        monkeypatch,
        eager_mpi=eager_mpi,
    )
    return executors, log, reports, adoptions, held_log, n


@pytest.mark.parametrize("pp_size, ring_depth", [(2, None), (4, None), (4, 8)])
@pytest.mark.parametrize("eager_mpi", [False, True])
@pytest.mark.parametrize("reuse_enabled", [True, False], ids=["reuse", "no_reuse"])
def test_first_rank_holds_new_prompts_until_the_last_stage_has_room(
    pp_size, ring_depth, eager_mpi, reuse_enabled, monkeypatch
):
    """With block reuse off the admission ledger alone drives the report
    exchange, its matching and the drain."""
    largest = max(blocks_for(p, TPB) for p in PROMPTS)
    capacities = [10 * largest] * (pp_size - 1) + [largest + 1]
    executors, log, reports, adoptions, held_log, n = _run(
        pp_size, ring_depth, capacities, True, monkeypatch, eager_mpi, reuse_enabled
    )
    leader = executors[0]
    ledger = leader._pp_admission
    first = log.chunks[0]
    for rank in range(pp_size):
        assert log.chunks[rank] == first, rank
        assert executors[rank].active_requests == []
        assert executors[rank]._pp_stage_agree.iteration == leader._pp_stage_agree.iteration
        assert not executors[rank]._pp_stage_agree.pending_sends
        assert executors[rank].kv_cache_manager.peak <= capacities[rank]
    for request_id, prompt_len in enumerate(PROMPTS):
        chunks = [(pos, size) for rid, pos, size in first if rid == request_id]
        assert chunks[0][0] == 0 and sum(size for _, size in chunks) == prompt_len
    # Every follower adopted every prompt's first chunk, none refused.
    assert sorted(adoptions) == sorted(
        (rank, rid) for rank in range(1, pp_size) for rid in range(len(PROMPTS))
    )
    # Every follower report carried its snapshot; the first rank held prompts back.
    assert reports and all(ADMISSION_REPORT_KEY in report for _, report in reports)
    assert ledger.num_held_iterations > 0 and any(held_log)
    assert ledger.num_started == 0
    assert leader._pp_stage_agree.num_undone == 0
    assert n == (ring_depth or pp_size)
    print(ledger.summary())


@pytest.mark.parametrize("pp_size", [2, 4])
def test_the_context_counters_count_every_iteration_the_first_rank_held(pp_size, monkeypatch):
    """With the per-stage counters attached, each iteration in which the ledger
    held first chunks back (at startup or for lack of room) is counted once."""
    largest = max(blocks_for(p, TPB) for p in PROMPTS)
    capacities = [10 * largest] * (pp_size - 1) + [largest + 1]
    executors, _, _, _, held_log, _ = _run(
        pp_size, None, capacities, True, monkeypatch, ctx_diag=True
    )
    ledger = executors[0]._pp_admission
    diag = executors[0]._pp_ctx_diag
    assert ledger.num_held_iterations > 0
    assert diag.admission_hold_iters == ledger.num_held_iterations + ledger.num_startup_holds
    assert diag.admission_held_chunk_iters == sum(len(held) for held in held_log)
    assert 0 < diag.admission_held_requests <= len(PROMPTS)
    assert diag.admission_held_now == 0
    for follower in executors[1:]:
        assert follower._pp_ctx_diag.admission_hold_iters == 0


@pytest.mark.parametrize("pp_size", [2, 4])
def test_other_server_sends_no_snapshot_and_holds_nothing(pp_size, monkeypatch):
    """With block reuse on the reports still flow, without the snapshot."""
    largest = max(blocks_for(p, TPB) for p in PROMPTS)
    capacities = [10 * largest] * pp_size
    executors, log, reports, adoptions, held_log, _ = _run(
        pp_size, None, capacities, False, monkeypatch
    )
    assert reports and all(ADMISSION_REPORT_KEY not in report for _, report in reports)
    assert not held_log
    assert all(e._pp_admission is None and not e._pp_admission_held for e in executors)
    assert all(e.kv_cache_manager.snapshots == 0 for e in executors)
    for rank in range(pp_size):
        assert log.chunks[rank] == log.chunks[0]


@pytest.mark.parametrize("pp_size", [2, 4])
def test_other_server_without_reuse_exchanges_no_reports(pp_size, monkeypatch):
    """Without block reuse or the admission ledger no report is exchanged at all."""
    largest = max(blocks_for(p, TPB) for p in PROMPTS)
    capacities = [10 * largest] * pp_size
    executors, log, reports, _, held_log, _ = _run(
        pp_size, None, capacities, False, monkeypatch, reuse_enabled=False
    )
    assert not reports and not held_log
    assert all(not e._pp_stage_agree.pending_sends for e in executors)
    assert all(e.kv_cache_manager.snapshots == 0 for e in executors)
    for rank in range(pp_size):
        assert log.chunks[rank] == log.chunks[0]


# --------------------------------------------------------------------------- construction


def _bare_executor(pp_size, agreement, manager, max_num_tokens=8192):
    exe = object.__new__(PyExecutor)
    exe.dist = types.SimpleNamespace(pp_size=pp_size, pp_rank=0, rank=0)
    exe._pp_stage_agree = agreement
    exe.kv_cache_manager = manager
    exe.enable_joint_kv_cache_reuse = False
    exe.scheduler = types.SimpleNamespace(max_num_tokens=max_num_tokens)
    exe.max_num_tokens = None
    return exe


def test_construction_follows_the_server(monkeypatch):
    agreement = PPStageAgreement(0, 4, 3, reuse_enabled=True)
    pool = _BlockPool(10)
    # Not a pipeline-parallel DFlash context server: no ledger.
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", False)
    assert _bare_executor(4, agreement, pool)._create_pp_admission() is None
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", True)
    assert _bare_executor(1, None, pool)._create_pp_admission() is None
    # Without the stage agreement there are no reports to carry the snapshots.
    assert _bare_executor(4, None, pool)._create_pp_admission() is None
    ledger = _bare_executor(4, agreement, pool)._create_pp_admission()
    assert isinstance(ledger, PPAdmissionLedger) and ledger.max_chunk_tokens == 8192
    # KV cache managers that cannot report their pools: no back-pressure, logged.
    infos = []
    monkeypatch.setattr(py_executor_module.logger, "info", infos.append)
    assert _bare_executor(4, agreement, object())._create_pp_admission() is None
    assert any("report their" in line for line in infos)


def test_class_defaults_keep_the_pass_unchanged():
    exe = object.__new__(PyExecutor)
    exe.active_requests = [object(), object()]
    assert exe._pp_admission is None
    assert exe._pp_admission_held == frozenset() and exe._pp_admission_blocked is False
    assert exe._schedulable_active_requests() is exe.active_requests
    assert exe._pp_admission_failure_note() == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
