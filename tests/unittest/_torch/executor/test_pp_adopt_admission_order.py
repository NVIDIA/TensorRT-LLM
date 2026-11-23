# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PP followers adopt the first rank's chunks in its allocation order.

With the PP stage agreement (``pp_stage_agree.py``) the first rank's
scheduler pass allocates context chunks in its admission order, and the batch
lists them in another one (non-last chunks before last chunks). A KV cache
resume is refused above a pool utilization (``max_util_for_resume``), not when
the pages run out, so a follower that grows a started prompt before it resumes
a reserved first chunk can be refused the resume the first rank was granted.
Under a host tier the follower then stops (``PP rank N: cannot allocate KV
cache for the N-token context chunk ... that the first PP rank scheduled``).

Built on the stand-in ranks of ``test_pp_stage_agreement.py`` (real scheduler,
real reuse claim, real pickled schedule message, real adoption), with a pool
that has a host tier's page states (a suspended reservation's pages are
evictable, so they count as free until the reservation is resumed) and the
core's resume gate.
"""

import pickle
import re
import types

import pytest
import test_pp_stage_agreement as base
from test_pp_stage_agreement import _request_states  # noqa: F401  (autouse fixture)
from test_pp_stage_agreement_adversarial import _set_micro_batches

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests, SerializableSchedulerOutput
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2 as scheduler_v2_module

pytestmark = pytest.mark.cpu_only

TPB = base.TOKENS_PER_BLOCK
CHUNK = base.MAX_NUM_TOKENS
STALL = scheduler_v2_module.KVCacheV2Scheduler._DEADLOCK_STALL_ITERS

# Request 9 starts a 10-chunk prompt with nothing to reuse. Request 8 arrives
# two iterations later: a 24576-token committed prefix (held by its stage
# reservation) and a 100-token remainder. Shortest-prefill-first admits 8 first
# (100 tokens to go) and gives 9 the rest of the token budget, so the first
# rank resumes 8's reservation before it grows 9; the batch lists 9 (a non-last
# chunk) before 8 (a last chunk).
STARTED, RESERVED = 9, 8
PREFIX_BLOCKS = 768
PROMPTS = {STARTED: 10 * CHUNK, RESERVED: PREFIX_BLOCKS * TPB + 100}
BLOCKS = {STARTED: 0, RESERVED: PREFIX_BLOCKS}
ARRIVALS = {2: {RESERVED: PROMPTS[RESERVED]}}
CAPACITY = 20000


class _HostTierPool(base._CapacityPool):
    """``_CapacityPool`` with a host tier's page states and the resume gate.

    A suspended cache's held pages can move down to the host tier, so they are
    evictable: only the pages of active caches and the chunks' allocations are
    locked, and only locked pages count against the capacity. A resume locks
    the held pages and is refused when the locked fraction of the pool is above
    ``gate`` (``KvCache::resume``'s ``max_util_for_resume`` check, made before
    any page moves), or when the held pages do not fit.
    """

    def __init__(self, cores, capacity_pages, gate=1.0):
        super().__init__(cores, capacity_pages)
        self.gate = gate
        self.refused_resumes = 0
        self.resume_log = []

    def used_pages(self):
        held = sum(len(cache.holds) for cache in self.kv_cache_map.values() if cache.is_active)
        return held + sum(self.allocated.values())

    def available_pages_per_level(self):
        return [[self.capacity_pages - self.used_pages()], [0]]

    def _resume_and_restore(self, req_id, kv_cache, cuda_stream=None):
        if kv_cache.is_active:
            return True
        locked = self.used_pages()
        self.resume_log.append((req_id, locked))
        if locked / self.capacity_pages > self.gate:
            self.refused_resumes += 1
            return False
        if locked + len(kv_cache.holds) > self.capacity_pages:
            return False
        kv_cache.resume(None)
        return True

    def admission_state(self, req):
        return f"stand-in pool: {self.used_pages()} of {self.capacity_pages} pages locked"


def _pools(num_stages, gate=1.0):
    return [
        _HostTierPool(
            {rid: base._full_attention_core(blocks) for rid, blocks in BLOCKS.items()},
            CAPACITY,
            gate,
        )
        for _ in range(num_stages)
    ]


def _enable_order(ranks):
    """What ``PyExecutor.__init__`` sets with the stage agreement."""
    for executor in ranks:
        executor._pp_adopt_in_admission_order = True
        executor.scheduler.record_context_admission_order = True


def _iteration(ranks):
    """``test_pp_stage_agreement._pp_iteration``, with the first rank's
    allocation order on the wire as ``_pp_schedule_and_propagate`` sends it."""
    leader, followers = ranks[0], ranks[1:]
    leader._pp_stage_agree_exchange()
    output = leader.scheduler.schedule_request(leader._schedulable_active_requests(), set())
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests(list(output.context_requests))
    leader._pp_stage_agree_check_admitted(scheduled)
    wire = pickle.loads(
        pickle.dumps(
            SerializableSchedulerOutput.from_scheduler_result(
                scheduled,
                [],
                1,
                record_first_chunks=True,
                reuse_proposals=leader._pp_stage_agree.take_proposals(),
                context_admission_order=PyExecutor._pp_leader_admission_order(leader),
            )
        )
    )
    for executor in followers:
        executor._pp_stage_agree_exchange()
        batch, _, _ = wire.to_scheduler_result(executor.active_requests)
        if executor._pp_follower_early_release:
            executor._pp_stage_agree_early_release(batch)
        local = executor.scheduler.schedule_request(
            executor._pp_stage_agree_local_requests(batch), set()
        )
        PyExecutor._pp_adopt_leader_schedule(executor, batch, local)
        executor._pp_stage_agree_answer(batch)
    return wire


def _ranks(num_stages, ordered, monkeypatch, gate=1.0):
    pools = _pools(num_stages, gate)
    ranks = base._ranks_with_requests(pools, {STARTED: PROMPTS[STARTED]}, True, monkeypatch)
    _set_micro_batches(ranks, num_stages)
    if ordered:
        _enable_order(ranks)
    else:
        for executor in ranks:
            executor._pp_adopt_in_admission_order = False
    monkeypatch.setattr(base, "_pp_iteration", _iteration)
    return ranks


def _leader_resume_point(num_stages, monkeypatch):
    """On pools without a gate: the pages locked on the first rank when it
    resumed the reservation, and the pages the started prompt's chunk of that
    batch adds (what a follower adopting in batch order locks before it)."""
    ranks = _ranks(num_stages, True, monkeypatch)
    leader_pool = ranks[0].kv_cache_manager
    grown = {}
    original = leader_pool.resize_context

    def resize_context(req, num_tokens):
        before = leader_pool.allocated.get(req.py_request_id, 0)
        ok = original(req, num_tokens)
        if (
            ok
            and req.py_request_id == STARTED
            and any(rid == RESERVED for rid, _ in leader_pool.resume_log)
        ):
            grown.setdefault("pages", leader_pool.allocated[STARTED] - before)
        return ok

    leader_pool.resize_context = resize_context
    base._drive(ranks, STALL + 64, arrivals=ARRIVALS)
    locked = [locked for rid, locked in leader_pool.resume_log if rid == RESERVED]
    assert len(locked) == 1 and grown["pages"] > 0
    return locked[0], grown["pages"]


def _gate_between(num_stages, monkeypatch):
    """A gate the first rank's resume passes and a follower's batch-order resume does not."""
    locked, grown = _leader_resume_point(num_stages, monkeypatch)
    return (locked + grown / 2) / CAPACITY


@pytest.mark.parametrize("num_stages", [2, 4])
def test_batch_order_adoption_is_refused_the_resume_the_first_rank_was_granted(
    num_stages, monkeypatch
):
    """Control (batch order): the failure the admission order fixes. Every stage has the same pool;
    the follower grows the started prompt first, crosses the gate, and cannot
    adopt the reserved first chunk."""
    gate = _gate_between(num_stages, monkeypatch)
    ranks = _ranks(num_stages, False, monkeypatch, gate)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"cannot allocate KV cache for the 100-token context chunk of request {RESERVED} "
            "that the first PP rank scheduled"
        ),
    ):
        base._drive(ranks, STALL + 64, arrivals=ARRIVALS)
    assert ranks[0].kv_cache_manager.refused_resumes == 0
    assert ranks[1].kv_cache_manager.refused_resumes == 1


@pytest.mark.parametrize("num_stages", [2, 4])
def test_admission_order_adoption_resumes_where_the_first_rank_did(num_stages, monkeypatch):
    """Admission order: the follower resumes the reservation before it grows the started
    prompt, as the first rank did, and both prompts run on every stage with no
    refused resume (``_drive``: every first chunk from the stage's own
    reservation at the admitted depth)."""
    gate = _gate_between(num_stages, monkeypatch)
    ranks = _ranks(num_stages, True, monkeypatch, gate)
    first = base._drive(ranks, STALL + 64, arrivals=ARRIVALS)
    assert first == {STARTED: 0, RESERVED: PREFIX_BLOCKS * TPB}
    assert all(r.kv_cache_manager.refused_resumes == 0 for r in ranks)
    resumes = [[locked for rid, locked in r.kv_cache_manager.resume_log] for r in ranks]
    assert all(log == resumes[0] for log in resumes[1:]), resumes


def _run_to_admission(ranks):
    """Iterations until the reserved first chunk is admitted; returns every wire."""
    wires = []
    for iteration in range(16):
        for rid, prompt_len in ARRIVALS.get(iteration, {}).items():
            for executor in ranks:
                executor.active_requests.append(base._Request(rid, prompt_len))
        wire = _iteration(ranks)
        wires.append(wire)
        base._advance_launched_chunks(ranks, wire)
        if RESERVED in (wire.context_chunks or {}):
            return wires
    raise AssertionError("the reserved first chunk was never admitted")


def test_the_wire_carries_the_allocation_order_and_the_batch_another(monkeypatch):
    """The admission batch: the first rank allocated the reserved first chunk
    first; the batch (and the wire's chunk map) lists the started prompt first."""
    ranks = _ranks(2, True, monkeypatch)
    wires = _run_to_admission(ranks)
    wire = wires[-1]
    assert list(wire.context_chunks) == [STARTED, RESERVED]
    assert wire.context_admission_order == [RESERVED, STARTED]
    batch, _, _ = wire.to_scheduler_result(ranks[1].active_requests)
    assert [r.request_id for r in batch.context_requests] == [STARTED, RESERVED]
    assert [r.request_id for r in PyExecutor._pp_context_adoption_order(ranks[1], batch)] == [
        RESERVED,
        STARTED,
    ]
    # Earlier batches (the started prompt alone) carried a one-element order.
    assert all(w.context_admission_order == list(w.context_chunks) for w in wires[:-1])


def test_batch_order_sends_no_order_and_adopts_in_batch_order(monkeypatch):
    ranks = _ranks(2, False, monkeypatch)
    assert ranks[0].scheduler.record_context_admission_order is False
    wires = _run_to_admission(ranks)
    assert all(w.context_admission_order is None for w in wires)
    batch, _, _ = wires[-1].to_scheduler_result(ranks[1].active_requests)
    assert batch.leader_context_admission_order is None
    assert [r.request_id for r in PyExecutor._pp_context_adoption_order(ranks[1], batch)] == [
        STARTED,
        RESERVED,
    ]
    assert ranks[0].scheduler.last_context_admission_order is None


def test_order_leaves_out_requests_the_batch_no_longer_holds():
    reqs = [base._Request(rid, 4 * CHUNK) for rid in (1, 2, 3)]
    for req in reqs:
        req.context_chunk_size = CHUNK
    scheduled = ScheduledRequests()
    scheduled.reset_context_requests([reqs[0], reqs[2]])
    wire = pickle.loads(
        pickle.dumps(
            SerializableSchedulerOutput.from_scheduler_result(
                scheduled, [], 2, record_context_chunks=True, context_admission_order=[3, 2, 1]
            )
        )
    )
    assert wire.context_admission_order == [3, 1]
    batch, _, _ = wire.to_scheduler_result(reqs)
    assert batch.leader_context_admission_order == [3, 1]
    follower = types.SimpleNamespace()
    assert [r.request_id for r in PyExecutor._pp_context_adoption_order(follower, batch)] == [
        3,
        1,
    ]
    plain = SerializableSchedulerOutput.from_scheduler_result(scheduled, [], 2)
    assert plain.context_admission_order is None


def test_executor_class_default_adopts_in_batch_order():
    """An executor built without ``__init__`` (no stage agreement) keeps the batch order."""
    assert getattr(PyExecutor, "_pp_adopt_in_admission_order", False) is False


def test_leader_order_is_taken_once():
    leader = types.SimpleNamespace(
        _pp_adopt_in_admission_order=True,
        scheduler=types.SimpleNamespace(last_context_admission_order=[5, 4]),
    )
    assert PyExecutor._pp_leader_admission_order(leader) == [5, 4]
    assert PyExecutor._pp_leader_admission_order(leader) is None
    leader._pp_adopt_in_admission_order = False
    leader.scheduler.last_context_admission_order = [5, 4]
    assert PyExecutor._pp_leader_admission_order(leader) is None


def test_failure_detail_names_the_order_and_the_pool_state(monkeypatch):
    """In admission order, a follower that still cannot adopt (its pool is smaller) says
    which order it adopted in and what its pool held."""
    gate = _gate_between(2, monkeypatch)
    ranks = _ranks(2, True, monkeypatch, gate)
    ranks[1].kv_cache_manager.capacity_pages = CAPACITY // 2
    with pytest.raises(RuntimeError) as error:
        base._drive(ranks, STALL + 64, arrivals=ARRIVALS)
    message = str(error.value)
    assert "that the first PP rank scheduled" in message
    assert "[adoption order: first rank's allocation order [" in message
    assert "stand-in pool:" in message


def test_failure_detail_is_empty_in_batch_order():
    follower = types.SimpleNamespace(_pp_adopt_in_admission_order=False)
    assert PyExecutor._pp_adopt_failure_detail(follower, ScheduledRequests(), None) == ""


class _FakeCache:
    is_active = False
    num_committed_tokens = 339078
    capacity = 339078


def test_admission_state_reports_the_resume_gate(monkeypatch):
    """``KVCacheManagerV2.admission_state``: per-group GPU utilization against
    ``max_util_for_resume``, the free + evictable slots, the request's cache."""
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = types.SimpleNamespace(
        init_config=types.SimpleNamespace(max_util_for_resume=0.95)
    )
    manager.kv_cache_map = {11: _FakeCache()}
    manager.available_pages_per_level = lambda: [[17000, 9000], [70000, 650000]]
    utilization = [0.9512, 0.31]
    monkeypatch.setattr(
        kv_v2_module._introspection, "storage_utilization", lambda impl, level: utilization
    )
    req = base._Request(11, 341517)
    line = manager.admission_state(req)
    assert "GPU utilization per pool group [0.9512, 0.31]" in line
    assert "max_util_for_resume 0.950 (a resume is refused now)" in line
    assert "[[17000, 9000], [70000, 650000]]" in line
    assert "request 11: cache suspended, 339078 committed tokens" in line
    utilization[0] = 0.9
    assert "(a resume is allowed now)" in manager.admission_state(req)
    assert "request 12: no cache" in manager.admission_state(base._Request(12, 10))
