# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks of the PP stage agreement's page-pressure give-back.

Built on the stand-in ranks of ``test_pp_stage_agreement.py`` (real scheduler,
real reuse claim, real pickled schedule message, stand-in KV cores):

* randomized workloads with tiny last tiers, many concurrent first chunks,
  arrivals over time, PP2 / PP3 / PP4 and report lags up to the ring-8 one,
  with a stock oracle (the leader-batch adoption alone): whatever
  stock runs to the end, the agreement runs to the end too, every first chunk
  from its own reservation, with exactly one report per follower and iteration;
* the same with every follower evicting all its unheld pages every iteration
  (eviction during the report lag) and with the given-back requests re-agreed;
* a follower whose stage pins more pages per reserved prefix than the first
  rank's (MuseSpark Open PP stages differ in their sliding-window / full-attention mix):
  the first rank sees no pressure itself; the follower declines the reservation
  that would leave it no room for a chunk, and both prompts run as in stock;
* a follower whose own pass would size a started prompt's chunk larger than
  the first rank did (equal stages): the follower's pass schedules no context
  request, so its allocation is the first rank's;
* a started prompt that shares the pool with one large waiting first chunk.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import pytest
import test_pp_stage_agreement as base
from test_pp_stage_agreement import _request_states  # noqa: F401  (autouse fixture)

from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2 as scheduler_v2_module

TPB = base.TOKENS_PER_BLOCK
CHUNK = base.MAX_NUM_TOKENS
STALL = scheduler_v2_module.KVCacheV2Scheduler._DEADLOCK_STALL_ITERS


def _pages(tokens: int) -> int:
    return 2 * math.ceil(tokens / TPB)


def _set_micro_batches(ranks, num_micro_batches: int) -> None:
    for executor in ranks:
        executor.num_micro_batches = num_micro_batches
        executor._pp_stage_agree = executor._create_pp_stage_agreement()


def _count_reports(ranks) -> Dict[str, List[int]]:
    """Wrap every rank's point-to-point calls and count them."""
    counts = {"sent": [0] * len(ranks), "received": [0] * len(ranks)}
    for executor in ranks:
        rank = executor.dist.pp_rank
        isend, recv = executor.dist.isend_object, executor.dist.recv_object

        def counted_isend(obj, dest, tag, rank=rank, isend=isend):
            counts["sent"][rank] += 1
            return isend(obj, dest, tag)

        def counted_recv(src, tag, recv=recv):
            counts["received"][src] += 1
            return recv(src, tag)

        executor.dist.isend_object = counted_isend
        executor.dist.recv_object = counted_recv
    return counts


def _stock_run(pools, prompts, arrivals, max_iterations, monkeypatch) -> Optional[int]:
    """Adoption alone (stock schedule): iterations until every prompt ran, or None
    when the V2 deadlock check fires or prompts are left."""
    ranks = base._ranks_with_requests(pools, prompts, False, monkeypatch)
    last_arrival = max(arrivals or {0: None})
    try:
        for iteration in range(max_iterations):
            for rid, prompt_len in arrivals.get(iteration, {}).items():
                for executor in ranks:
                    executor.active_requests.append(base._Request(rid, prompt_len))
            wire = base._adopt_only_pp_iteration(ranks)
            if wire is not None:
                base._advance_launched_chunks(ranks, wire)
            base._finish_done(ranks)
            if not any(e.active_requests for e in ranks) and iteration >= last_arrival:
                return iteration + 1
    except RuntimeError:
        return None
    return None


def _workload(rng: random.Random, num_requests: int):
    """Random prompts: an own committed prefix (0..24576 tokens) and up to 3 chunks after it."""
    blocks, prompts = {}, {}
    for rid in range(100, 100 + num_requests):
        blocks[rid] = rng.choice([0, 64, 256, 512, 768])
        prompts[rid] = blocks[rid] * TPB + rng.randint(1, 3 * CHUNK)
    return blocks, prompts


def _pools(num_stages, blocks, capacity):
    return [
        base._CapacityPool(
            {rid: base._full_attention_core(b) for rid, b in blocks.items()}, capacity
        )
        for _ in range(num_stages)
    ]


# A seed on which the agreement's schedule leaves two started prompts that do
# not fit together (no reservation held; the V2 scheduler does not preempt on a
# context server). Stock reaches the same state on other seeds.
_STARTED_PROMPTS_OVERCOMMIT = {(2, 2, 2, False)}


def _fuzz_params():
    params = []
    for pp_size, micro in ((2, 2), (3, 3), (4, 4), (4, 8)):
        for seed in range(6):
            for evict in (False, True):
                key = (pp_size, micro, seed, evict)
                marks = []
                if key in _STARTED_PROMPTS_OVERCOMMIT:
                    marks = [
                        pytest.mark.xfail(
                            strict=True,
                            raises=RuntimeError,
                            reason="started prompts overcommit the pool",
                        )
                    ]
                params.append(
                    pytest.param(
                        *key,
                        marks=marks,
                        id=f"pp{pp_size}-mb{micro}-s{seed}-{'evict' if evict else 'keep'}",
                    )
                )
    return params


@pytest.mark.parametrize("pp_size, micro, seed, evict", _fuzz_params())
def test_random_tight_pools_run_what_stock_runs(pp_size, micro, seed, evict, monkeypatch):
    """Equal stages, a last tier barely above the longest prompt, 6..20 requests,
    half arriving later. Whatever stock runs, the agreement runs, with the
    invariants ``_drive`` checks every iteration, one report per follower and
    iteration (received exactly ``lag`` later), nothing left held, and no
    admission undone. Followers that evict every unheld page every iteration
    (eviction during the report lag) only lower the agreed depths."""
    rng = random.Random(1000 * pp_size + 10 * micro + seed)
    num_requests = rng.randint(6, 20)
    blocks, prompts = _workload(rng, num_requests)
    capacity = max(_pages(n) for n in prompts.values()) + rng.choice([0, 256, 1024])
    rids = list(prompts)
    initial = {rid: prompts[rid] for rid in rids[: num_requests // 2]}
    arrivals = {}
    for rid in rids[num_requests // 2 :]:
        arrivals.setdefault(rng.randint(1, 30), {})[rid] = prompts[rid]

    stock = _stock_run(_pools(pp_size, blocks, capacity), initial, arrivals, 3000, monkeypatch)
    if stock is None:
        pytest.skip("stock does not run this workload either")

    pools = _pools(pp_size, blocks, capacity)
    ranks = base._ranks_with_requests(pools, initial, True, monkeypatch)
    _set_micro_batches(ranks, micro)
    agreement = ranks[0]._pp_stage_agree
    counts = _count_reports(ranks)
    iterations = [0]

    def before(ranks):
        done = iterations[0]
        for stage in range(1, pp_size):
            assert counts["sent"][stage] == done
            assert counts["received"][stage] == max(0, done - agreement.lag)
        iterations[0] += 1
        if evict:
            for executor in ranks[1:]:
                for core in executor.kv_cache_manager.cores.values():
                    core.evict_all_unheld()

    first = base._drive(ranks, STALL + 4 * stock + 200, arrivals=arrivals, before=before)

    assert set(first) == set(prompts)
    assert all(not pool.kv_cache_map and not pool.allocated for pool in pools)
    assert all(not core.held for pool in pools for core in pool.cores.values())
    assert agreement.num_undone == 0
    for rid, position in first.items():
        assert position <= blocks[rid] * TPB
    print(
        f"pp{pp_size} micro{micro} seed{seed} evict={evict}: {num_requests} requests, "
        f"stock {stock} it, agreement {iterations[0]} it; {agreement.summary()}"
    )


# --------------------------------------------------------------------------- tight follower


def _all_swa_core(num_blocks: int) -> base._SwaCore:
    """An all-sliding-window stage: a committed prompt keeps only its trailing
    window, so a reserved prefix pins a window's pages, not the whole prefix."""
    swa = [0] * num_blocks
    end = num_blocks * TPB
    for i in range(int((end + 1 - base.SWA_WINDOW) / TPB), num_blocks):
        swa[i] = TPB
    return base._SwaCore(num_blocks, [(base.SWA_WINDOW, 0), (base.SWA_WINDOW, 0)], [swa, list(swa)])


# Two one-chunk prompts on a 24576-token own prefix: 24576 + 8192 tokens. A
# full-attention stage pins 1536 pages per reserved prefix and needs 512 for the
# chunk; two pins and a chunk (3584) do not fit 2600 pages, one pin and a chunk
# (2048) do. An all-SWA stage pins about 130 per prefix.
_TIGHT_PROMPTS = {7: 24576 + CHUNK, 8: 24576 + CHUNK}


def _tight_pools(layout: str) -> list:
    cores = {
        "swa": lambda: {rid: _all_swa_core(768) for rid in _TIGHT_PROMPTS},
        "full": lambda: {rid: base._full_attention_core(768) for rid in _TIGHT_PROMPTS},
    }
    return [base._CapacityPool(cores[kind](), 2600) for kind in layout.split("-")]


LAYOUTS = ["swa-full", "swa-swa-full-swa", "full-swa-full-full"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_tight_follower_control_stock_runs_both_prompts(layout, monkeypatch):
    """Control: adoption alone runs both prompts on the same pools; on every
    stage the other prompt's prefix stays evictable."""
    assert _stock_run(_tight_pools(layout), dict(_TIGHT_PROMPTS), {}, 64, monkeypatch)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_follower_that_pins_more_than_the_first_rank_still_adopts(layout, monkeypatch):
    """Rank 0 an all-SWA stage, a follower full attention, equal page capacity.
    Rank 0's own pool shows no pressure; a full-attention follower that already
    holds one prefix declines the second (two pins and a chunk do not fit), the
    first rank gives it back, and both prompts run, each first chunk adopted
    from the follower's own reservation (``_drive``'s checks)."""
    pools = _tight_pools(layout)
    ranks = base._ranks_with_requests(pools, dict(_TIGHT_PROMPTS), True, monkeypatch)
    first = base._drive(ranks, STALL + 64)
    assert first == {7: 24576, 8: 24576}
    agreement = ranks[0]._pp_stage_agree
    declined = sum(executor._pp_stage_agree.num_declined for executor in ranks[1:])
    assert agreement.num_declines_received == declined
    if layout.split("-")[0] == "swa":
        # Rank 0's own pool never refused pages: only the declines kept the
        # full-attention follower from holding both prefixes.
        assert declined > 0 and agreement.num_pressure_passes > 0
    print(f"{layout}: {declined} decline(s); {agreement.summary()}")


@pytest.mark.parametrize("max_logged", [1, 16])
def test_every_stage_logs_the_pages_a_reservation_pins(max_logged, monkeypatch):
    """The first ``MAX_LOGGED_PINS`` reservations of every rank log the pages they
    pinned: about a window per prefix on the all-SWA first rank, the whole
    24576-token prefix (1536 pages) on the full-attention follower."""
    from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module

    monkeypatch.setattr(py_executor_module, "MAX_LOGGED_PINS", max_logged)
    lines = []
    info = py_executor_module.logger.info
    monkeypatch.setattr(
        py_executor_module.logger,
        "info",
        lambda msg, *a, **k: (lines.append(msg), info(msg, *a, **k)),
    )
    pools = _tight_pools("swa-full")
    ranks = base._ranks_with_requests(pools, dict(_TIGHT_PROMPTS), True, monkeypatch)
    base._drive(ranks, STALL + 64)
    pins = [line for line in lines if line.startswith("PP stage agreement: reservation pins ")]
    per_stage = {
        stage: [int(line.split()[5]) for line in pins if f"on stage {stage} " in line]
        for stage in (0, 1)
    }
    print("\n".join(pins))
    assert all(0 < len(counts) <= max_logged for counts in per_stage.values())
    assert all(0 < n < 200 for n in per_stage[0])
    assert 1536 in per_stage[1]
    assert all(r._pp_stage_agree.num_pin_logs == len(per_stage[r.dist.pp_rank]) for r in ranks)


def test_one_reservation_avoids_the_heavier_follower_crash(monkeypatch):
    """A cap of one reservation pins one prefix per stage, which a stage that
    fits one prompt can always hold."""
    monkeypatch.setattr(base.py_executor_module, "DEFAULT_MAX_RESERVED", 1)
    pools = _tight_pools("swa-full")
    ranks = base._ranks_with_requests(pools, dict(_TIGHT_PROMPTS), True, monkeypatch)
    assert ranks[0]._pp_stage_agree.max_reserved == 1
    assert base._drive(ranks, STALL + 64) == {7: 24576, 8: 24576}


# --------------------------------------------------------------------------- follower chunk sizing

# Request 9 is started (3 chunks, no prefix); request 7 (16384-token own prefix,
# one 1000-token chunk) arrives at iteration 1. Rank 0 admits 7 next to 9's
# second chunk and gives 9 the rest of the token budget (7192). A follower pass
# that leaves 7 out but schedules 9 sizes 9's chunk at 8192, and adopting the
# smaller chunk keeps the larger allocation (resize grows to a maximum):
# 2050..2111 pages fit rank 0's batch and not that follower's.
_SIZING_BLOCKS = {9: 0, 7: 512}
_SIZING_PROMPTS = {9: 3 * CHUNK, 7: 16384 + 1000}


@pytest.mark.parametrize("capacity", [2080, 2112])
def test_a_follower_keeps_its_own_larger_chunk_allocation_control(capacity, monkeypatch):
    pools = _pools(2, _SIZING_BLOCKS, capacity)
    arrivals = {1: {7: _SIZING_PROMPTS[7]}}
    assert _stock_run(pools, {9: _SIZING_PROMPTS[9]}, arrivals, 64, monkeypatch) == 4


@pytest.mark.parametrize("capacity", [2080, 2112])
def test_a_follower_keeps_its_own_larger_chunk_allocation(capacity, monkeypatch):
    """Equal stages: under the stage agreement the follower's own pass used to
    size request 9's chunk with the whole budget and keep that allocation, and
    ran out of pages where rank 0 did not (2080). Its pass now schedules no
    context request: every stage allocates exactly rank 0's chunks."""
    pools = _pools(2, _SIZING_BLOCKS, capacity)
    ranks = base._ranks_with_requests(pools, {9: _SIZING_PROMPTS[9]}, True, monkeypatch)
    first = base._drive(ranks, 64, arrivals={1: {7: _SIZING_PROMPTS[7]}})
    assert first == {9: 0, 7: 16384}


# --------------------------------------------------------------------------- started prompt


def _started_vs_waiting(stage_agree: bool, micro: int, monkeypatch) -> Tuple[int, int]:
    """Request 9 (no prefix, 12 chunks) starts first; request 7 (24576-token
    prefix, one chunk) arrives one iteration later. The pool fits 9's whole
    cache, but not next to 7's held prefix after 9's first few chunks; stock
    leaves 7's prefix evictable, so 9 runs one chunk per pass and 7 after it.
    Returns (iterations until 9 finished, iterations until both finished)."""
    blocks = {9: 0, 7: 768}
    prompts = {9: 12 * CHUNK, 7: 24576 + CHUNK}
    capacity = _pages(12 * CHUNK) + 256
    pools = _pools(2, blocks, capacity)
    ranks = base._ranks_with_requests(pools, {9: prompts[9]}, stage_agree, monkeypatch)
    if stage_agree:
        _set_micro_batches(ranks, micro)
    done_9 = None
    for iteration in range(STALL + 400):
        if iteration == 1:
            for executor in ranks:
                executor.active_requests.append(base._Request(7, prompts[7]))
        if stage_agree:
            wire = base._pp_iteration(ranks)
        else:
            wire = base._adopt_only_pp_iteration(ranks)
        if wire is not None:
            base._advance_launched_chunks(ranks, wire)
        base._finish_done(ranks)
        if done_9 is None and all(r.py_request_id != 9 for r in ranks[0].active_requests):
            done_9 = iteration + 1
        if not any(e.active_requests for e in ranks):
            return done_9, iteration + 1
    raise AssertionError("prompts left")


@pytest.mark.parametrize("micro", [2, 8])
def test_a_started_prompt_next_to_one_large_waiting_first_chunk(micro, monkeypatch):
    """A 12-chunk started prompt and one waiting first chunk whose held prefix
    blocks it: each give-back lets about one chunk run before the waiting one is
    proposed again and pins its prefix for another agreement round. Not a
    deadlock; a slowdown that grows with the report lag."""
    stock_9, stock_all = _started_vs_waiting(False, micro, monkeypatch)
    agreed_9, agreed_all = _started_vs_waiting(True, micro, monkeypatch)
    print(
        f"micro={micro}: stock 9 done {stock_9}, all {stock_all}; "
        f"agreement 9 done {agreed_9}, all {agreed_all}"
    )
    # The agreement costs a few extra iterations here, more at the deeper report lag.
    assert agreed_all <= 2 * stock_all


# --------------------------------------------------------------------------- mutation gaps


def test_held_first_chunks_do_not_count_before_a_sweep(monkeypatch):
    """Before any sweep found nothing admissible, a pass in which every waiting
    first chunk is held is idle for the V2 deadlock check: a follower that
    needs many proposal rounds (here: no free index slot) does not trip it."""
    monkeypatch.setattr(scheduler_v2_module.KVCacheV2Scheduler, "_DEADLOCK_STALL_ITERS", 20)
    pools = [
        base._CapacityPool({base.REQUEST_ID: base._full_attention_core(768)}, 10**9),
        base._CapacityPool({base.REQUEST_ID: base._full_attention_core(768)}, 10**9, index_slots=0),
    ]
    ranks = base._ranks_with_requests(pools, {base.REQUEST_ID: base.PROMPT_LEN}, True, monkeypatch)
    for _ in range(80):
        assert not base._pp_iteration(ranks).context_chunks
    assert not ranks[0]._pp_stage_agree.exhausted
    assert ranks[0].scheduler.stage_held_context_candidates == 0
    pools[1].index_slots = 64
    wire, _ = base._run_until_admitted(ranks)
    assert wire.context_chunks[base.REQUEST_ID] == (24576, CHUNK)


def test_a_generation_step_is_progress(monkeypatch):
    """A pass that scheduled only generation steps is not a stalled pass, even
    with a context admission refused pages: no reservation is given back."""
    ranks = base._make_ranks(
        [base._StagePool(0), base._StagePool(0)], base.PROMPT_LEN, True, monkeypatch
    )
    leader = ranks[0]
    leader.active_requests += [base._Request(rid, 1000) for rid in (20, 21, 22)]
    leader._pp_stage_agree_exchange()
    agreement = leader._pp_stage_agree
    leader.scheduler._ctx_page_refused = True
    batch = base.ScheduledRequests()
    batch.generation_requests = [base._Request(99, 1000)]

    leader._pp_stage_agree_relieve(batch)

    assert agreement.num_pressure_passes == 0 and agreement.num_given_back == 0


def test_an_arrival_after_a_sweep_stops_counting_the_held_chunks(monkeypatch):
    """After a sweep found nothing admissible the held first chunks count for the
    deadlock check; a new request changes the waiting set, so the count stops
    while the sweep starts over and the newcomer is admitted with no raise."""
    monkeypatch.setattr(scheduler_v2_module.KVCacheV2Scheduler, "_DEADLOCK_STALL_ITERS", 14)
    big = {7: base.PROMPT_LEN, 8: base.PROMPT_LEN}
    cores = {rid: base._full_attention_core(768) for rid in big}
    cores[9] = base._full_attention_core(0)
    pools = [base._CapacityPool(dict(cores), 1700) for _ in range(2)]
    ranks = base._ranks_with_requests(pools, big, True, monkeypatch)
    _set_micro_batches(ranks, 8)
    agreement = ranks[0]._pp_stage_agree
    for _ in range(400):
        base._pp_iteration(ranks)
        if agreement.exhausted:
            break
    assert agreement.exhausted
    for executor in ranks:
        executor.active_requests.append(base._Request(9, 1000))
    for waited in range(400):
        wire = base._pp_iteration(ranks)
        if 9 in wire.context_chunks:
            break
    assert 9 in wire.context_chunks
    # The arrival waits behind a new sweep over the two that never fit (oldest
    # first, one reservation at a time): about two agreement rounds.
    print(f"lag {agreement.lag}: arrival admitted after {waited + 1} iterations")


def test_a_request_refused_alone_and_then_cancelled_leaves_the_rotation(monkeypatch):
    """Request 7 never fits, even alone, and goes to the back of the rotation;
    it is then cancelled on every stage. The rotation forgets it, and the
    requests that fit run to the end."""
    prompts = {7: base.PROMPT_LEN, 8: 2048 + 4000}
    pools = base._rotation_pools(2)
    ranks = base._ranks_with_requests(pools, prompts, True, monkeypatch)
    agreement = ranks[0]._pp_stage_agree
    for _ in range(64):
        base._pp_iteration(ranks)
        if agreement.num_refused_alone:
            break
    assert 7 in agreement._tried_alone
    for executor in ranks:
        req = next(r for r in executor.active_requests if r.py_request_id == 7)
        executor._pp_stage_agree_release(req)
        executor.active_requests.remove(req)
    for pool in pools:
        pool.cores[10] = base._full_attention_core(64)
    for executor in ranks:
        executor.active_requests.append(base._Request(10, 2048 + 4000))
    for _ in range(200):
        wire = base._pp_iteration(ranks)
        base._advance_launched_chunks(ranks, wire)
        base._finish_done(ranks)
        if not any(executor.active_requests for executor in ranks):
            break
    assert not any(executor.active_requests for executor in ranks)
    assert 7 not in agreement._tried_alone


# --------------------------------------------------------------------------- follower early release


def _recorded_reports(ranks):
    reports = {executor.dist.pp_rank: [] for executor in ranks[1:]}
    for executor in ranks[1:]:
        isend = executor.dist.isend_object

        def recorded(obj, dest, tag, rank=executor.dist.pp_rank, isend=isend):
            reports[rank].append(obj)
            return isend(obj, dest, tag)

        executor.dist.isend_object = recorded
    return reports


@pytest.mark.parametrize(
    "pp_size, micro, seed, evict",
    [
        pytest.param(p, m, s, e, id=f"pp{p}-mb{m}-s{s}-{'evict' if e else 'keep'}")
        for p, m in ((3, 3), (4, 8))
        for s in range(3)
        for e in (False, True)
    ],
)
def test_random_tight_pools_early_release_reports_what_the_answer_alone_reports(
    pp_size, micro, seed, evict, monkeypatch
):
    """The tight-pool workloads above, run with and without
    the follower early release: drops (give-backs under pressure) and
    lowered depths (followers evicting during the lag) applied before the
    adoption leave every follower report, every first-chunk position and the
    iteration count unchanged (these pools have no resume gate)."""
    rng = random.Random(1000 * pp_size + 10 * micro + seed)
    num_requests = rng.randint(6, 20)
    blocks, prompts = _workload(rng, num_requests)
    capacity = max(_pages(n) for n in prompts.values()) + rng.choice([0, 256, 1024])
    rids = list(prompts)
    initial = {rid: prompts[rid] for rid in rids[: num_requests // 2]}
    arrivals = {}
    for rid in rids[num_requests // 2 :]:
        arrivals.setdefault(rng.randint(1, 30), {})[rid] = prompts[rid]

    def before(ranks):
        if evict:
            for executor in ranks[1:]:
                for core in executor.kv_cache_manager.cores.values():
                    core.evict_all_unheld()

    results = []
    for early in (False, True):
        pools = _pools(pp_size, blocks, capacity)
        ranks = base._ranks_with_requests(pools, initial, True, monkeypatch)
        _set_micro_batches(ranks, micro)
        for executor in ranks:
            executor._pp_follower_early_release = early
        reports = _recorded_reports(ranks)
        try:
            first = base._drive(ranks, STALL + 3000, arrivals=arrivals, before=before)
        except RuntimeError as error:
            first = f"raised: {error}"
        results.append((first, reports))
        applied = sum(
            e._pp_stage_agree.num_early_releases + e._pp_stage_agree.num_early_lowerings
            for e in ranks[1:]
        )
        assert sum(e._pp_stage_agree.num_early_skipped_in_batch for e in ranks[1:]) == 0
        if not early:
            assert applied == 0
    assert results[0] == results[1]
    print(f"pp{pp_size} mb{micro} seed{seed} evict={evict}: {applied} applied early")


def test_early_release_drop_and_lowering_in_one_message_then_a_stale_message():
    """A drop and a lowering of different requests in one message are both
    applied before the adoption; a ``pre_release`` whose answer never ran (the
    request left in between) does not leak into the next message's answer."""
    from tensorrt_llm._torch.pyexecutor.pp_stage_agree import PPStageAgreement, set_reserved_depth

    follower = PPStageAgreement(2, 3, 1, True)
    requests = {rid: base._Request(rid, 4 * CHUNK) for rid in (1, 2, 3)}
    calls = []

    def reserve(req, cap):
        calls.append(("reserve", req.py_request_id, cap))
        set_reserved_depth(req, cap)
        return cap

    def release(req):
        calls.append(("release", req.py_request_id))
        set_reserved_depth(req, None)

    for rid, depth in ((1, 640), (2, 640), (3, 640)):
        set_reserved_depth(requests[rid], depth)
    applied = follower.pre_release({2: 320, 1: None}, requests, set(), reserve, release)
    assert applied == {1: (640, None), 2: (640, 320)}
    assert calls == [("release", 1), ("reserve", 2, 320)]

    # The answer of that message never ran; the next message proposes 3 lower and
    # nothing for 1 and 2. The stale entries must not change its answer.
    calls.clear()
    applied = follower.pre_release({3: 128}, requests, set(), reserve, release)
    assert applied == {3: (640, 128)} and follower._pre_applied == {3: 128}
    follower.answer_proposals({3: 128, 1: 64}, requests, reserve, release, lambda req: 0)
    # 1 was dropped by the stale message: its new proposal is claimed normally.
    assert calls == [("reserve", 3, 128), ("reserve", 1, 64)]
    assert follower.build_report()["answers"] == {3: (128, 128, 0), 1: (64, 64, 0)}


def test_early_release_of_a_request_cancelled_before_the_answer():
    """A request lowered before the adoption and cancelled before the answer is
    answered as gone, as without the agreement; its reservation is not touched again."""
    from tensorrt_llm._torch.pyexecutor.pp_stage_agree import PPStageAgreement, set_reserved_depth

    follower = PPStageAgreement(1, 2, 1, True)
    req = base._Request(4, 4 * CHUNK)
    set_reserved_depth(req, 640)
    calls = []

    def reserve(r, cap):
        calls.append(("reserve", cap))
        set_reserved_depth(r, cap)
        return cap

    follower.pre_release({4: 256}, {4: req}, set(), reserve, lambda r: calls.append("release"))
    req.state = base._State.GENERATION_COMPLETE  # cancelled: no longer a waiting context
    follower.answer_proposals(
        {4: 256}, {4: req}, reserve, lambda r: calls.append("release"), lambda r: 0
    )
    assert calls == [("reserve", 256)]
    assert follower.build_report()["answers"] == {4: (256, None, None)}
