# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the PP followers' own allocations and page-pressure declines.

Built on the stand-in ranks of ``test_pp_stage_agreement.py`` (real scheduler,
real reuse claim, real pickled schedule message, real adoption; stand-in KV
cores with page capacity).
"""

import math
import random
import types

import pytest
import test_pp_ring_depth as ring
import test_pp_stage_agree_lagged_loop as lagged
import test_pp_stage_agreement as base
from test_pp_stage_agreement import _request_states  # noqa: F401  (autouse fixture)
from test_pp_stage_agreement_adversarial import _all_swa_core, _set_micro_batches

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2 as scheduler_v2_module

pytestmark = pytest.mark.cpu_only

TPB = base.TOKENS_PER_BLOCK
CHUNK = base.MAX_NUM_TOKENS
STALL = scheduler_v2_module.KVCacheV2Scheduler._DEADLOCK_STALL_ITERS


class _HetPool(base._CapacityPool):
    """``_CapacityPool`` whose running allocations cost ``factor`` pages per block
    (a stage with more layers per life cycle, or more life cycles), not 2."""

    def __init__(self, cores, capacity_pages, factor):
        super().__init__(cores, capacity_pages)
        self.factor = factor
        self.target = {}

    def used_pages(self):
        held = sum(len(core.held) for core in self.cores.values())
        return held + sum(self.allocated.values())

    def available_pages_per_level(self):
        return [[self.capacity_pages - self.used_pages()]]

    def free_resources(self, req, pin_on_release=False):
        super().free_resources(req, pin_on_release)
        self.allocated.pop(req.py_request_id, None)
        self.target.pop(req.py_request_id, None)

    def has_room_for_context_tokens(self, num_tokens):
        return self.capacity_pages - self.used_pages() >= self.factor * (
            math.ceil(num_tokens / TPB) + 1
        )

    def _pages_between(self, start, target):
        return self.factor * max(0, math.ceil(target / TPB) - math.ceil(start / TPB))

    def resize_context(self, req, num_tokens):
        rid = req.py_request_id
        start = self.kv_cache_map[rid].num_committed_tokens
        target = max(self.target.get(rid, 0), req.context_current_position + num_tokens)
        pages = self._pages_between(start, target)
        if self.used_pages() - self.allocated.get(rid, 0) + pages > self.capacity_pages:
            return False
        previous = self.target.get(rid, 0)
        req.py_ctx_pre_resize_cap = previous if target > previous else None
        self.target[rid] = target
        self.allocated[rid] = pages
        self.resizes.append((rid, req.context_current_position, num_tokens))
        return True

    def revert_allocate_context(self, req):
        previous, req.py_ctx_pre_resize_cap = req.py_ctx_pre_resize_cap, None
        if previous is None:
            return True
        rid = req.py_request_id
        start = self.kv_cache_map[rid].num_committed_tokens
        self.target[rid] = previous
        self.allocated[rid] = self._pages_between(start, previous)
        self.kv_cache_map[rid].is_active = False
        return True


def _core(kind: str, num_blocks: int) -> base._SwaCore:
    if kind == "full" or num_blocks * TPB <= base.SWA_WINDOW:
        return base._full_attention_core(num_blocks)
    if kind == "swa":
        return _all_swa_core(num_blocks)
    return base._hybrid_window_core(num_blocks, [num_blocks * TPB])


def _het_pools(kinds, factors, capacities, blocks):
    return [
        _HetPool({rid: _core(kind, b) for rid, b in blocks.items()}, capacity, factor)
        for kind, factor, capacity in zip(kinds, factors, capacities)
    ]


def _real_revert(ranks):
    for executor in ranks:
        executor._revert_ctx_alloc = types.MethodType(PyExecutor._revert_ctx_alloc, executor)


def _stock(pools, initial, arrivals, max_iterations, monkeypatch):
    """Adoption alone on ``pools``: (iterations or None, error or None)."""
    ranks = base._ranks_with_requests(pools, initial, False, monkeypatch)
    _real_revert(ranks)
    last = max(arrivals or {0: None})
    try:
        for iteration in range(max_iterations):
            for rid, prompt_len in arrivals.get(iteration, {}).items():
                for executor in ranks:
                    executor.active_requests.append(base._Request(rid, prompt_len))
            wire = base._adopt_only_pp_iteration(ranks)
            if wire is not None:
                base._advance_launched_chunks(ranks, wire)
            base._finish_done(ranks)
            if not any(e.active_requests for e in ranks) and iteration >= last:
                return iteration + 1, None
    except RuntimeError as error:
        return None, str(error)[:200]
    return None, "stuck"


KINDS = ("full", "swa", "hybrid")


def _het_case(pp_size: int, micro: int, seed: int):
    rng = random.Random(7919 * pp_size + 101 * micro + seed)
    num_requests = rng.randint(4, 14)
    blocks, prompts = {}, {}
    for rid in range(100, 100 + num_requests):
        blocks[rid] = rng.choice([0, 128, 256, 512, 768])
        prompts[rid] = blocks[rid] * TPB + rng.randint(1, 3 * CHUNK)
    kinds = [rng.choice(KINDS) for _ in range(pp_size)]
    factors = [rng.choice([2, 2, 3, 4]) for _ in range(pp_size)]
    longest = max(prompts.values())
    capacities = [
        f * math.ceil(longest / TPB) + 2 * 768 * rng.choice([0, 0, 1]) + rng.choice([0, 256, 1024])
        for f in factors
    ]
    rids = list(prompts)
    initial = {rid: prompts[rid] for rid in rids[: num_requests // 2]}
    arrivals = {}
    for rid in rids[num_requests // 2 :]:
        arrivals.setdefault(rng.randint(1, 30), {})[rid] = prompts[rid]
    return kinds, factors, capacities, blocks, prompts, initial, arrivals


# Stock runs these, the stage agreement crashes on a follower:
# (4, 8, 5) a lone reservation next to a started prompt. (3, 3, 8), a full-attention
# follower keeping 7 reservations each checked against its own first chunk only,
# runs now (strict XPASS in CI) and is no longer listed.
_HET_FOLLOWER_CRASHES = {(4, 8, 5)}


def _het_params():
    params = []
    for pp_size, micro in ((2, 2), (3, 3), (4, 4), (4, 8)):
        for seed in range(10):
            marks = []
            if (pp_size, micro, seed) in _HET_FOLLOWER_CRASHES:
                marks = [
                    pytest.mark.xfail(strict=True, raises=RuntimeError, reason="follower room")
                ]
            params.append(
                pytest.param(pp_size, micro, seed, marks=marks, id=f"pp{pp_size}-mb{micro}-s{seed}")
            )
    return params


@pytest.mark.parametrize("pp_size, micro, seed", _het_params())
def test_heterogeneous_tight_pools_run_what_stock_runs(pp_size, micro, seed, monkeypatch):
    """Whatever stock runs on heterogeneous tight pools, the agreement runs too,
    with _drive's per-adoption checks (no follower holds a reservation the
    first rank gave back; every first chunk from the stage's own reservation at
    the admitted depth), nothing left held and no admission undone."""
    kinds, factors, capacities, blocks, prompts, initial, arrivals = _het_case(pp_size, micro, seed)
    stock, error = _stock(
        _het_pools(kinds, factors, capacities, blocks), initial, arrivals, 3000, monkeypatch
    )
    if stock is None:
        pytest.skip(f"stock does not run this workload either: {error}")
    pools = _het_pools(kinds, factors, capacities, blocks)
    ranks = base._ranks_with_requests(pools, initial, True, monkeypatch)
    _real_revert(ranks)
    _set_micro_batches(ranks, micro)
    first = base._drive(ranks, STALL + 4 * stock + 200, arrivals=arrivals)
    agreement = ranks[0]._pp_stage_agree
    assert set(first) == set(prompts)
    assert all(not pool.kv_cache_map and not pool.allocated for pool in pools)
    assert all(not core.held for pool in pools for core in pool.cores.values())
    assert agreement.num_undone == 0
    declined = sum(r._pp_stage_agree.num_declined for r in ranks[1:])
    print(f"{kinds} f={factors}: stock {stock} it; {declined} declined; {agreement.summary()}")


# --------------------------------------------------------------------------- real PP loop


class _HetPressurePool(ring._PressurePool):
    """``_PressurePool`` in units with a per-stage cost: a reservation pins
    ``pin`` units, a running request (first chunk admitted, prompt not finished
    on this stage) ``run`` units."""

    def __init__(self, capacity: int, pin: int, run: int):
        super().__init__(capacity)
        self.pin, self.run = pin, run

    def used(self) -> int:
        return sum(self.pin if s == "reserved" else self.run for s in self.kv_cache_map.values())

    def room_for(self, req) -> bool:
        # The request's reservation is already counted; its admission adds run - pin.
        return self.used() + self.run - self.pin <= self.capacity


class _HetPressureExecutor(ring._PressureExecutor):
    """First-rank FCFS pass over a ``_HetPressurePool``."""

    def _schedule(self):
        pool = self.kv_cache_manager
        self._refused = False
        batch = base.ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self._schedulable_active_requests():
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            if req.context_current_position == 0:
                state = pool.kv_cache_map.get(req.request_id)
                extra = pool.run - (pool.pin if state == "reserved" else 0)
                if pool.used() + extra > pool.capacity:
                    req.py_pp_page_refused = True
                    self._refused = True
                    continue
                pool.kv_cache_map[req.request_id] = "running"
            self._first_scheduled.setdefault(req.request_id, self.iter_counter)
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)


# Per stage (capacity, pin, run). "l": pins 1 unit per reserved prefix, fits
# one running request and one held prefix. "h": pins as much as a running
# request (a full-attention stage holds the whole prefix), fits one held prefix
# next to one running request, never two. With an "h" first rank, which runs
# up to two requests at once, "H" is an equal follower and "L" a cheap one
# that fits two running requests and a held prefix (a follower smaller than
# the first rank for running work fails in stock too).
_STAGE_COSTS = {"l": (3, 1, 2), "h": (4, 2, 2), "H": (4, 2, 2), "L": (6, 1, 2)}


@pytest.mark.parametrize(
    "layout, ring_depth",
    [
        ("lh", None),
        ("lhh", 5),
        ("lhlh", None),
        ("lhhh", 8),
        ("llhl", 4),
        ("lhlh", 8),
        ("hHL", None),
        ("hLHL", 8),
    ],
)
@pytest.mark.parametrize("eager_mpi", [False, True])
def test_real_loop_followers_decline_what_they_cannot_hold(
    layout, ring_depth, eager_mpi, monkeypatch
):
    """The real ``_executor_loop_pp`` (threads, ring depth, eager and rendezvous
    sends): followers that pin a running request's worth per reserved prefix
    ("h") decline every reservation beyond the first while a first rank with
    cheap pins ("l") reserves up to its bound; a first rank with heavy pins
    relieves its own pressure too. No deadlock, no follower adoption failure,
    every prompt runs, every follower adopts each first chunk from a
    reservation it holds and never holds one the first rank gave back, the
    declines reach the first rank and no report is left unmatched."""
    pp_size = len(layout)
    prompts = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1, 64 + 7, 4 * 64, 64]
    num_micro_batches = ring.ring_slots(pp_size, ring_depth)
    lag = ring.probe_lag(ring.relay_lag(pp_size, num_micro_batches, False))
    leader_log = {}
    adoptions = []
    original = ring._make_executor

    def make(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        exe.__class__ = _HetPressureExecutor
        exe._first_scheduled = {}
        exe._pressure_log = leader_log
        exe._refused = False
        exe._pp_stage_agree = ring.PPStageAgreement(rank, pp_size, lag, reuse_enabled=True)
        pool = _HetPressurePool(*_STAGE_COSTS[layout[rank]])
        exe.kv_cache_manager = pool
        exe.enable_joint_kv_cache_reuse = False
        exe.scheduler.context_pages_refused_last_pass = lambda: exe._refused
        exe.scheduler.pages_return_pending = lambda active: False
        exe.scheduler.release_context_admission = pool.release
        exe.scheduler.reserve_first_context_chunk = pool.reserve
        exe.scheduler.reservation_leaves_chunk_room = pool.room_for

        def adopt_first(req, position, size):
            iteration = exe._pp_stage_agree.iteration
            held = {rid for rid, s in pool.kv_cache_map.items() if s == "reserved"}
            assert pool.kv_cache_map.get(req.request_id) == "reserved", "no reservation"
            assert held <= leader_log[iteration], (held, leader_log[iteration])
            if not pool.room_for(req):
                return False
            adoptions.append((rank, req.request_id))
            pool.kv_cache_map[req.request_id] = "running"
            return ring._adopt_first(req, position, size)

        exe.scheduler.adopt_first_context_chunk = adopt_first
        return exe

    monkeypatch.setattr(ring, "_make_executor", make)
    executors, log, _, n = ring._run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        64,
        True,
        2,
        lambda mb: 0.002,
        monkeypatch,
        eager_mpi=eager_mpi,
    )
    leader = executors[0]
    agreement = leader._pp_stage_agree
    first = log.chunks[0]
    for rank in range(pp_size):
        assert log.chunks[rank] == first, rank
        assert executors[rank].active_requests == []
        assert not executors[rank].kv_cache_manager.kv_cache_map
        assert executors[rank]._pp_stage_agree.iteration == agreement.iteration
        assert not executors[rank]._pp_stage_agree.pending_sends
    for request_id, prompt_len in enumerate(prompts):
        chunks = [(pos, size) for rid, pos, size in first if rid == request_id]
        assert chunks[0][0] == 0 and sum(size for _, size in chunks) == prompt_len
    assert sorted(adoptions) == sorted(
        (rank, rid) for rank in range(1, pp_size) for rid in range(len(prompts))
    )
    assert agreement.num_admitted == len(prompts) and agreement.num_undone == 0
    declined = sum(executors[r]._pp_stage_agree.num_declined for r in range(1, pp_size))
    assert agreement.num_declines_received <= declined
    if layout[0] == "l":
        assert declined > 0 and agreement.num_declines_received > 0
    assert n == (ring_depth or pp_size)
    print(f"{layout} ring {n}: {agreement.summary()} sent declines {declined}")


@pytest.mark.parametrize("pp_size, ring_depth", [(2, None), (3, 5), (4, None), (4, 8)])
def test_lagged_consensus_loop_with_declining_followers(pp_size, ring_depth, monkeypatch):
    """The PP stage agreement with the lagged send agreement in one
    real loop (the lagged-loop harness), with every odd follower declining any
    reservation beyond the one it holds: the declines ride in the existing
    reports (no PP collective, every message matched), every prompt is admitted
    at the common depth from each stage's reservation, the sends and
    terminations agree on every stage, and no follower pass sees a context
    request (it gets generation work only)."""
    passes = {"context_requests": 0, "calls": 0}
    original = lagged._make

    def make(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        pool = exe._pool
        if rank % 2:
            exe.scheduler.reservation_leaves_chunk_room = lambda req: not (
                set(pool.reserved) - {req.py_request_id}
            )
        schedule = exe.scheduler.schedule_request

        def schedule_request(active, inflight):
            passes["calls"] += 1
            passes["context_requests"] += sum(1 for r in active if r.is_context_init_state)
            return schedule(active, inflight)

        exe.scheduler.schedule_request = schedule_request
        return exe

    monkeypatch.setattr(lagged, "_make", make)
    prompts = lagged.PROMPTS
    executors, world, log, n, seen = lagged._run(
        monkeypatch, pp_size=pp_size, ring_depth=ring_depth, prompts=prompts
    )
    leader = executors[0]
    agreement = leader._pp_stage_agree
    assert all(not calls for calls in world.collectives.values()), world.collectives
    unmatched = {k: q.qsize() for k, q in world._queues.items() if not q.empty()}
    assert not unmatched, unmatched
    for e in executors:
        assert e._pp_stage_agree.iteration == e._pp_disagg_lag.iteration == leader.iter_counter
        assert not e._pp_stage_agree.pending_sends
        assert e._pool.num_reclaimed == 0 and e._pool.reserved == {}
        for rid, p in enumerate(prompts):
            assert e._pool.admitted_at[rid] == lagged._expected_depth(pp_size, rid, p)
    for r in range(1, pp_size):
        assert log.released[r] == log.released[0]
        assert log.terminated[r] == log.terminated[0]
    assert agreement.num_admitted == len(prompts) and agreement.num_undone == 0
    declined = sum(e._pp_stage_agree.num_declined for e in executors[1:])
    assert declined > 0 and 0 < agreement.num_declines_received <= declined
    assert passes["calls"] > 0 and passes["context_requests"] == 0
    print(f"pp{pp_size} ring {n}: {agreement.summary()} sent declines {declined}")


# --------------------------------------------------------------------------- follower room, deterministic


def _swa_full_pools(blocks, follower_capacity):
    """PP2: an all-sliding-window first rank (a reserved prefix pins a window) and
    a full-attention follower (it pins the whole prefix), two life cycles each."""
    leader = base._CapacityPool({rid: _core("swa", b) for rid, b in blocks.items()}, 4000)
    follower = base._CapacityPool(
        {rid: base._full_attention_core(b) for rid, b in blocks.items()}, follower_capacity
    )
    return [leader, follower]


# Request 7: a 24576-token prefix and one 8192-token chunk (512 pages for the
# chunk on the follower); request 8, arriving one iteration later: a 4096-token
# prefix and a 100-token chunk. The follower holds 1536 + 256 pages for the two
# reserved prefixes: 300 pages are left, enough for 8's chunk, not for 7's.
_ORDER_BLOCKS = {7: 768, 8: 128}
_ORDER_PROMPTS = {7: 768 * TPB + CHUNK, 8: 128 * TPB + 100}
_ORDER_CAPACITY = 1536 + 256 + 300
_ORDER_ARRIVALS = {1: {8: _ORDER_PROMPTS[8]}}


def _order_run(micro, monkeypatch):
    pools = _swa_full_pools(_ORDER_BLOCKS, _ORDER_CAPACITY)
    ranks = base._ranks_with_requests(pools, {7: _ORDER_PROMPTS[7]}, True, monkeypatch)
    _real_revert(ranks)
    _set_micro_batches(ranks, micro)
    return base._drive(ranks, STALL + 64, arrivals=_ORDER_ARRIVALS)


def test_room_per_request_control(monkeypatch):
    """Stock runs both prompts on these pools; so does the agreement with report
    lag 1 (8's answer reaches the first rank after 7 was admitted)."""
    pools = _swa_full_pools(_ORDER_BLOCKS, _ORDER_CAPACITY)
    stock, error = _stock(pools, {7: _ORDER_PROMPTS[7]}, _ORDER_ARRIVALS, 64, monkeypatch)
    assert stock == 2, error
    assert _order_run(2, monkeypatch) == {7: 768 * TPB, 8: 128 * TPB}


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="a follower checks room for the new reservation's own chunk only",
)
@pytest.mark.parametrize("micro", [4, 8])
def test_a_small_reservation_does_not_take_the_room_of_a_held_larger_chunk(micro, monkeypatch):
    """Report lag 3 / 7: the follower holds 7's reservation (its only one) when
    8 is proposed, and checks 8 against 8's own 100-token chunk: room, so it
    keeps it. That leaves 300 pages, and 7's 8192-token chunk needs 512: when the
    first rank (an all-SWA stage, a window per prefix, no pressure) admits 7,
    the follower cannot allocate it. No follower declined anything."""
    assert _order_run(micro, monkeypatch) == {7: 768 * TPB, 8: 128 * TPB}


# Request 9 (no prefix, three 8192-token chunks: 1536 pages on the follower)
# runs; request 7 (a 24576-token prefix: 1536 pinned pages, one chunk) arrives
# while 9 is mid-prompt. 2600 follower pages hold 9 alone (1536) and 7 alone
# (2048), never 9 next to 7's held prefix (3072).
_LONE_BLOCKS = {9: 0, 7: 768}
_LONE_PROMPTS = {9: 3 * CHUNK, 7: 768 * TPB + CHUNK}
# Arrival iteration per micro-batch count: 9 has started by then.
_LONE_ARRIVAL = {2: 2, 4: 3, 8: 3}


@pytest.mark.parametrize("arrival", sorted(set(_LONE_ARRIVAL.values())))
def test_lone_reservation_control_stock_runs_both(arrival, monkeypatch):
    pools = _swa_full_pools(_LONE_BLOCKS, 2600)
    arrivals = {arrival: {7: _LONE_PROMPTS[7]}}
    stock, error = _stock(pools, {9: _LONE_PROMPTS[9]}, arrivals, 64, monkeypatch)
    assert stock is not None, error


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="a lone reservation is never declined, also next to a growing started prompt",
)
@pytest.mark.parametrize("micro", sorted(_LONE_ARRIVAL))
def test_a_lone_reservation_next_to_a_started_prompt_on_a_heavier_follower(micro, monkeypatch):
    """The follower holds no other reservation when 7 is proposed, so it keeps
    7's 1536-page prefix next to started prompt 9; the first rank (an all-SWA
    stage, no pressure) keeps scheduling 9's chunks, and the follower cannot
    allocate one of them. Stock leaves 7's prefix evictable and runs both."""
    pools = _swa_full_pools(_LONE_BLOCKS, 2600)
    ranks = base._ranks_with_requests(pools, {9: _LONE_PROMPTS[9]}, True, monkeypatch)
    _real_revert(ranks)
    _set_micro_batches(ranks, micro)
    arrivals = {_LONE_ARRIVAL[micro]: {7: _LONE_PROMPTS[7]}}
    assert base._drive(ranks, STALL + 64, arrivals=arrivals) == {9: 0, 7: 768 * TPB}


# --------------------------------------------------------------------------- unit gaps


def test_a_declined_head_stops_counting_as_pressure_once_relieved():
    """A decline is consumed by the pass that reads it: the declined head is
    retried (still proposed) and the next pass without pressure raises the
    bound again. A decline of a request forgotten before the pass (cancelled)
    is not pressure."""
    agreement = base._unit_agreement(pp_size=2, lag=1)
    requests = [base._Request(rid, 100) for rid in (1, 2, 3)]
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    agreement.take_proposals()
    agreement.advance()
    report = {"iteration": 0, "answers": {r: (0, 0, 0) for r in (1, 2, 3)}, "declined": [1]}
    report["answers"][1] = (0, None, 0)
    agreement.accept(1, report, 0)
    agreement.step(requests, lambda req, cap: 0, lambda req: True)
    assert 1 in agreement._proposed and agreement.num_retries == 1

    assert agreement.relieve(False, False, set(), False, lambda rid: None) == [3, 2]
    assert agreement.budget == 1 and agreement.num_pressure_passes == 1
    # Head 1 is still proposed; the decline was consumed.
    assert agreement.relieve(False, False, set(), False, lambda rid: None) == []
    assert agreement.budget == 2 and agreement.num_pressure_passes == 1

    # A decline of a request cancelled before the pass is not pressure.
    agreement.take_proposals()
    agreement.advance()
    agreement.accept(1, {"iteration": 1, "answers": {1: (0, None, 0)}, "declined": [1]}, 1)
    agreement.step(requests[1:], lambda req, cap: 0, lambda req: True)
    assert 1 not in agreement._proposed
    assert agreement.relieve(False, False, set(), False, lambda rid: None) == []
    assert agreement.num_pressure_passes == 1


class _RoomManager:
    def __init__(self, room):
        self.room = room
        self.asked = []

    def has_room_for_context_tokens(self, num_tokens):
        self.asked.append(num_tokens)
        return self.room


@pytest.mark.parametrize("draft_room", [True, False])
def test_reservation_room_asks_for_the_largest_first_chunk_in_every_paired_pool(draft_room):
    """The chunk asked for is the prompt after the reserved depth, at most the
    token budget, plus the draft tokens, in the target and the paired draft pool."""
    scheduler = object.__new__(scheduler_v2_module.KVCacheV2Scheduler)
    scheduler.max_num_tokens = CHUNK
    scheduler.kv_cache_manager = _RoomManager(True)
    scheduler._joint_draft_manager = _RoomManager(draft_room)
    req = base._Request(1, 30000)
    req.py_draft_tokens = [0, 0, 0]
    req.py_pp_reserved_depth = 24576
    assert scheduler.reservation_leaves_chunk_room(req) is draft_room
    req.py_pp_reserved_depth = 0
    scheduler.reservation_leaves_chunk_room(req)
    assert scheduler.kv_cache_manager.asked == [30000 - 24576 + 3, CHUNK + 3]
    assert scheduler._joint_draft_manager.asked == [30000 - 24576 + 3, CHUNK + 3]


@pytest.mark.parametrize(
    "source", ["none", "transfer_state", "connector_load", "preemption", "async"]
)
def test_pages_return_pending_names_each_source(source):
    """Each source of pages that come back without a scheduling decision keeps a
    pass from counting as stalled; with none of them it counts."""
    scheduler = object.__new__(scheduler_v2_module.KVCacheV2Scheduler)
    holding = next(iter(scheduler_v2_module.KVCacheV2Scheduler._TRANSFER_HOLDING_STATE_VALUES))
    req = types.SimpleNamespace(state_value=holding if source == "transfer_state" else -1)
    connector = types.SimpleNamespace(has_pending_load=lambda r: source == "connector_load")
    scheduler.kv_cache_manager = types.SimpleNamespace(
        kv_connector_manager=connector,
        has_pending_preemption=lambda: source == "preemption",
    )
    scheduler._async_transfer_manager = types.SimpleNamespace(
        has_any_inflight_requests=lambda: source == "async"
    )
    assert scheduler.pages_return_pending([req]) is (source != "none")


@pytest.mark.xfail(strict=True, reason="a decline-driven give-back is logged as 'preemption'")
def test_a_give_back_caused_by_a_decline_is_logged_as_one(monkeypatch):
    """The first rank's give-back log names its cause; a pass that scheduled work
    and preempted nothing but read a follower decline is not a preemption."""
    from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module

    ranks = base._make_ranks(
        [base._StagePool(0), base._StagePool(0)], base.PROMPT_LEN, True, monkeypatch
    )
    leader = ranks[0]
    leader.active_requests += [base._Request(rid, 1000) for rid in (20, 21)]
    leader._pp_stage_agree_exchange()
    agreement = leader._pp_stage_agree
    assert {20, 21} <= set(agreement._proposed)
    agreement._declined.add(21)
    lines = []
    monkeypatch.setattr(py_executor_module.logger, "info", lambda msg, *a, **k: lines.append(msg))
    batch = base.ScheduledRequests()
    batch.generation_requests = [base._Request(99, 1000)]
    leader._pp_stage_agree_relieve(batch)
    assert agreement.num_pressure_passes == 1
    (line,) = [m for m in lines if "KV page pressure on the first rank" in m]
    assert "preemption" not in line, line
