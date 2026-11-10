# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GEN preemption policy, startup prerequisites, and resident-state regressions."""

import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import compute_max_num_sequences
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import gen_preemption
from tensorrt_llm._torch.pyexecutor.scheduler.gen_admission_queue import (
    GenAdmissionConfig,
    GenAdmissionWaitingQueue,
)
from tensorrt_llm._torch.pyexecutor.scheduler.gen_preemption import (
    GenPreemption,
    _Req,
    create_gen_preemption,
)
from tensorrt_llm.llmapi.llm_args import GenPreemptionConfig

pytestmark = pytest.mark.cpu_only
GEN = LlmRequestState.GENERATION_IN_PROGRESS
ON = GenPreemptionConfig(tpot_budget=0.25)
SLO = GenPreemptionConfig(target_output_tps=8.0)
AUTO = GenAdmissionConfig(ttft_percentile=90.0)


def request(rid, **overrides):
    req = SimpleNamespace(
        py_request_id=rid,
        state=GEN,
        py_decoding_iter=100,
        prompt_len=1000,
        py_max_new_tokens=4096,
        is_dummy_request=False,
        generated=100,
        py_batch_idx=rid,
    )
    req.__dict__.update(overrides)
    req.get_num_tokens = lambda beam: req.prompt_len + req.generated
    return req


def policy(settings=ON, **knobs):
    p = GenPreemption(settings, 128)
    p.mean_decode_s, p.mean_tpot_s = 20.0, 0.1
    for name, value in {"min_run_frac": 0.0, "min_remaining_tokens": 0, **knobs}.items():
        setattr(p, name, value)
    return p


def running(p, n=2, start_id=0):
    reqs = [request(i) for i in range(start_id, start_id + n)]
    p.observe(reqs, 90.0)
    for req in reqs:
        req.generated += 100
    p.observe(reqs, 100.0)
    return reqs


def queue(ages=(), x=10.0, guard=100.0, percentile=90.0):
    q = GenAdmissionWaitingQueue(GenAdmissionConfig(ttft_percentile=percentile))
    if q.adaptive is not None:
        q.adaptive.X, q.adaptive.G, q.adaptive.s2f_s = x, guard, 0.5
    items = [RequestQueueItem(i + 1000, Mock()) for i in range(len(ages))]
    for item, (gen_age, orig_age) in zip(items, ages):
        item.admit_arrival, item.admit_orig_arrival = 100.0 - gen_age, 100.0 - orig_age
    q.add_requests(items)
    q.set_now(100.0)
    return q


def complete(p, tokens=1000, paused=0.0, target_ready=True, sacrificed=False):
    p._now = 20.0 + paused
    p._on_left(
        _Req(0, 0.0, first_token_t=0.0, gen=tokens, paused_total=paused, sacrificed=sacrificed),
        target_ready,
    )


@pytest.mark.parametrize("settings,extra", [(GenPreemptionConfig(), 0), (ON, 48), (SLO, 48)])
def test_pool_headroom(settings, extra):
    assert gen_preemption.resident_headroom(128, settings) == extra
    mapping = Mock(pp_size=1)
    mapping.has_pp.return_value = False
    mamba = object.__new__(MambaHybridCacheManagerV2)
    mamba.max_batch_size, mamba.mapping = 128, mapping
    mamba.gen_preemption_headroom = extra
    assert (
        compute_max_num_sequences(mapping, 128, False, gen_preemption_headroom=extra) == 128 + extra
    )
    assert (
        compute_max_num_sequences(
            mapping, 128, False, enable_overlap_headroom=True, gen_preemption_headroom=extra
        )
        == 256 + extra
    )
    other = object.__new__(MambaHybridCacheManagerV2)
    other.max_batch_size, other.mapping = 128, mapping
    other.gen_preemption_headroom = 0
    assert other._max_resident_sequences() == 128
    assert mamba._max_resident_sequences() == 128 + extra


def test_headroom_follows_each_executor_configuration(monkeypatch):
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SchedulerConfig

    monkeypatch.setenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", "90")
    enabled = SimpleNamespace(
        scheduler_config=SchedulerConfig(gen_preemption=ON),
        kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
        max_beam_width=1,
        sleep_config=None,
    )
    disabled = SimpleNamespace(scheduler_config=SchedulerConfig())
    for args, expected in ((enabled, 48), (disabled, 0), (enabled, 48)):
        assert gen_preemption.headroom_for_args(128, args) == expected
    monkeypatch.setenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", "100")
    assert gen_preemption.headroom_for_args(128, enabled) == 0
    assert enabled.scheduler_config.gen_preemption.enabled


@pytest.mark.parametrize(
    "changes,eligible",
    [
        ({}, True),
        ({"py_decoding_iter": 0}, False),
        ({"is_dummy_request": True}, False),
        ({"state": LlmRequestState.GENERATION_TO_COMPLETE}, False),
        ({"py_max_new_tokens": 232}, False),
        ({"py_max_new_tokens": 233}, True),
    ],
)
def test_victim_eligibility(changes, eligible):
    p = policy(min_remaining_tokens=32)
    req = running(p, 1)[0]
    req.__dict__.update(changes)
    assert (p._victim([req], 1) is req) == eligible


def test_victim_order_limits_and_cleanup():
    p = policy(min_run_frac=0.07, hysteresis_iters=10, max_paused=1)
    reqs = [request(0), request(1)]
    p.observe(reqs, 0.0)
    p.observe(reqs, 1.0)
    assert p._victim(reqs, 1) is None
    reqs.append(request(2))
    p.observe(reqs, 2.0)
    assert p._victim(reqs, 3) is reqs[1]  # oldest resident, newest tie
    assert p._victim(reqs, 0) is None
    for count in (1, 2):
        p._pause(reqs[1])
        assert p._victim(reqs, 1) is None  # paused cap
        assert p._resume(1) == 1
        assert not p._eligible(reqs[1])  # resume hysteresis
        reqs[1].py_decoding_iter += 10
        assert p._eligible(reqs[1]) == (count == 1)
    p._pause(reqs[0])
    p.observe(reqs[1:], 3.0)
    assert not p.paused_ids and 0 not in p._reqs


@pytest.mark.parametrize("settings", [ON, SLO])
def test_admission_pause_and_resume(settings):
    p = policy(settings)
    reqs = running(p, 4)
    at_risk = queue([(1.0, 9.2)])
    assert p.step(reqs, 100.0, 5, 4, at_risk) == 4  # a free slot
    assert p.step(reqs, 100.0, 4, 4, queue([(1.0, 1.0)])) == 4
    assert p.step(reqs, 100.0, 4, 4, at_risk) == 3 and p.paused_ids == {3}
    assert p.step(reqs[1:], 100.1, 4, 3, queue()) == 3 and not p.paused_ids
    reqs[3].py_decoding_iter += p.hysteresis_iters
    p.step(reqs, 101.0, 4, 4, at_risk)
    assert p.step(reqs, 107.5, 4, 5, queue()) == 4 and p.paused_ids
    assert p.step(reqs, 108.5, 4, 5, queue()) == 5 and not p.paused_ids
    assert p.stats["forced"] == 1


@pytest.mark.parametrize(
    "ages,guard,slack,expected",
    [
        ([(1, 9.2), (0.5, 5)], 100, 5, 2),
        ([(1, 9.2)], 100, 0.2, 0),
        ([(1, 20), (1, 15)], 100, 5, 2),
        ([(0.1, 9.2), (6, 6)], 5, 5, 2),
        ([(1, 20), (0.1, 0)], 100, 5, 0),
    ],
)
def test_admission_risk(ages, guard, slack, expected):
    assert queue(ages, guard=guard).num_at_risk(slack) == expected


@pytest.mark.parametrize("percentile", [90.0, 100.0])
def test_warmup(percentile):
    q = queue([(1, 9)], x=None, percentile=percentile)
    assert q.num_at_risk(100.0) == 0
    p = policy(mean_decode_s=None)
    reqs = running(p)
    assert p.step(reqs, 100.0, 2, 2, queue([(1, 9.2)])) == 2
    assert p.step(reqs, 100.0, 2, 2, queue([(1, 20)])) == 1


def test_controller_accounting():
    p = GenPreemption(ON, 128)
    assert (p.max_paused, p.gain, p.slack_s, p.max_pause_s, p.min_run_s) == (48, 1, 0, 10, 2)
    complete(p)
    assert (p.mean_decode_s, p.mean_tpot_s, p.max_pause_s) == (20.0, 0.02, 7.0)
    theta = p.theta
    complete(p, paused=5.0)
    assert p.theta == pytest.approx(theta - 0.05 / (0.2 * 3000))
    assert p.mean_tpot_s == pytest.approx(0.02)  # pause excluded from decode time
    theta = p.theta
    complete(p, target_ready=False)
    assert p.theta == theta
    complete(p, tokens=2000)
    assert p.mean_tpot_s == pytest.approx(0.02 + (0.01 - 0.02) / 500)


@pytest.mark.parametrize("gain,paused", [(2.0, 0.0), (0.25, 10.0)])
def test_controller_clamps(gain, paused):
    p = policy()
    p.theta = math.log(gain)
    complete(p, tokens=200, paused=paused)
    assert p.gain == pytest.approx(gain)


def slo_requests(specs):
    p = policy(SLO)
    reqs = [request(i, generated=1, py_max_new_tokens=n) for i, (n, _) in enumerate(specs)]
    p.observe(reqs, 0.0)
    for req, (_, rate) in zip(reqs, specs):
        req.generated += int(10 * rate)
    p.observe(reqs, 10.0)
    return p, reqs


def test_output_rate_victims_and_credit():
    p, reqs = slo_requests([(201, 10), (401, 10), (201, 8.1), (201, 5)])
    assert p.laxity(reqs[0]) == pytest.approx(5.0)
    assert p.laxity(reqs[3]) == pytest.approx(-15.0)
    unknown = policy(SLO, mean_tpot_s=None)
    req = request(9, generated=1, py_max_new_tokens=201)
    unknown.observe([req], 0.0)
    assert unknown.laxity(req) is None
    unknown.mean_tpot_s = 0.1
    assert unknown.laxity(req) == pytest.approx(5.0)
    for rid in (1, 0, 3):
        victim = next(r for r in reqs if r.py_request_id == rid)
        assert p._victim(reqs, 4) is victim
        reqs.remove(victim)
    assert p._victim(reqs, 1) is None  # remaining request has too little laxity
    p.credit = 1.5
    assert p._victim(reqs, 1) is reqs[0]
    assert p._reqs[2].sacrificed and p.credit == pytest.approx(0.5)
    assert p._victim(reqs, 1) is reqs[0] and p.credit == pytest.approx(0.5)
    p._reqs[2].pauses = 3
    assert p._victim(reqs, 1) is None  # below-target pause limit
    p, reqs = slo_requests([(401, 10)])
    p._reqs[0].pauses, p.credit = 8, 10
    assert p._victim(reqs, 1) is None and p.credit == 10


def test_budget_resume_fifo():
    p = policy()
    reqs = running(p, 3)
    for req in reversed(reqs):
        p._pause(req)
    for rid in (2, 1, 0):
        before = p.paused_ids
        assert p._resume(1) == 1
        assert before - p.paused_ids == {rid}


def test_output_rate_resume_order_and_laxity():
    p, reqs = slo_requests([(401, 10), (201, 10), (161, 10), (201, 5)])
    for req in reqs:
        p._pause(req)
    assert [p._resume_at(p._reqs[i]) for i in range(4)] == pytest.approx([17, 14.8, 13.8, 17])
    p._now = 13.79
    assert p._resume(0) == 0
    p._now = 13.81
    assert p._resume(1) == 1 and p.paused_ids == {0, 1, 3}
    assert p.laxity(reqs[2]) == pytest.approx(0.19)
    p._now = 14.0
    assert p._resume(1) == 1 and p.paused_ids == {0, 3}
    p._now = 17.0
    assert p._resume(0) == 2 and p.stats["forced"] == 2
    p, reqs = slo_requests([(201, 5), (2001, 10)])
    p._reqs[0].sacrificed = True
    p._pause(reqs[0])
    p.observe(reqs, 12.0)
    p._pause(reqs[1])
    assert p._resume(1) == 1 and p.paused_ids == {0}  # sacrifice resumes last
    p.observe(reqs, 17.0)
    assert p._resume(0) == 1 and p.stats["forced"] == 1


@pytest.mark.parametrize(
    "tokens,paused,sacrificed,credit",
    [(1000, 0, False, 0.2), (100, 0, False, -0.8), (200, 6, False, -0.8), (200, 6, True, 0.2)],
)
def test_output_rate_completion(tokens, paused, sacrificed, credit):
    p = policy(SLO)
    complete(p, tokens=tokens, paused=paused, sacrificed=sacrificed)
    assert p.credit == pytest.approx(credit)
    complete(p, target_ready=False)
    assert p.credit == pytest.approx(credit) and p.theta == 0
    p.credit = 600
    complete(p)
    assert p.credit == 600
    p.credit = -600
    complete(p, tokens=1)
    assert p.credit == -600


@pytest.mark.parametrize("settings", [ON, SLO])
def test_decisions_are_rank_identical(settings):
    results = []
    for _ in range(3):
        p = policy(settings)
        reqs = running(p, 4)
        for i, req in enumerate(reqs):
            req.generated += 3 * i
            req.py_max_new_tokens += 40 * i
        clock = Mock(side_effect=AssertionError("local clock"))
        with patch("time.monotonic", clock), patch("time.time", clock):
            p.step(reqs, 100.0, 4, 4, queue([(1, 9.2)]))
            p.step(reqs[1:], 103.0, 4, 3, queue())
            complete(p, tokens=50)
        results.append((p.credit, p.theta, p.paused_ids, repr(p._reqs)))
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize(
    "failure", [None, "off", "scheduler", "admission", "slots", "draft", "recurrent"]
)
def test_startup(failure):
    slots = Mock()
    slots.get_max_resource_count.return_value = 176
    ex = SimpleNamespace(
        scheduler=Mock(spec=["set_gen_paused_ids"]),
        _gen_admission=AUTO,
        enable_attention_dp=False,
        dist=SimpleNamespace(has_pp=False, rank=0),
        kv_cache_manager=Mock(spec=[], max_admissible_sequences=352),
        draft_kv_cache_manager=None,
        max_num_active_requests=128,
        resource_manager=SimpleNamespace(
            resource_managers={ResourceManagerType.SEQ_SLOT_MANAGER: slots}
        ),
    )
    if failure == "scheduler":
        ex.scheduler = object()
    elif failure == "admission":
        ex._gen_admission = GenAdmissionConfig(ttft_percentile=100)
    elif failure == "slots":
        slots.get_max_resource_count.return_value = 175
    elif failure == "draft":
        ex.draft_kv_cache_manager = Mock(spec=[], max_admissible_sequences=175)
    elif failure == "recurrent":
        ex.kv_cache_manager._max_resident_sequences = lambda: 175
    with patch.object(gen_preemption, "logger") as log:
        result = create_gen_preemption(ex, GenPreemptionConfig() if failure == "off" else ON)
    assert (result is not None) == (failure is None)
    assert log.warning.called == (failure not in (None, "off"))


def drafter(kind, num_slots=4):
    """A DFlash or DSpark drafter worker and its metadata with ``num_slots``
    request slots plus a scratch slot (CPU stand-ins for device buffers)."""
    import collections
    from functools import partial

    import torch

    from tensorrt_llm._device_copy import DeviceCopyBuffer
    from tensorrt_llm._torch.speculative import dflash, dspark

    rows = num_slots + 1
    if kind == "dflash":
        worker = object.__new__(dflash.DFlashWorker)
        worker._ctx_buf_inited, worker._req_ctx_pos, worker._dummy_slot = True, {}, num_slots
        worker._ctx_len_host = [0] * rows
        worker._batch_to_slot_buffer = DeviceCopyBuffer(torch.full((num_slots,), num_slots))

        worker._ctx_base_host = [0] * rows
        worker._ctx_base_pages = torch.zeros(rows, dtype=torch.long)
        worker._ctx_block_tables = None
        worker._write_ctx_len = partial(dflash.DFlashWorker._write_ctx_len, worker)
        worker._assign_slot = partial(dflash.DFlashWorker._assign_slot, worker)
        worker._seed_generation_slot = partial(dflash.DFlashWorker._seed_generation_slot, worker)
        meta = object.__new__(dflash.DFlashSpecMetadata)
        meta._batch_indices_buffer = DeviceCopyBuffer(torch.zeros(num_slots, dtype=torch.int))
        meta.batch_indices_cuda = meta._batch_indices_buffer.tensor
        meta._dflash_worker = worker
        meta.host_position_ids = torch.zeros(num_slots, dtype=torch.long)
        meta.seq_lens = [1] * num_slots
    else:
        worker = object.__new__(dspark.DSv4DSparkWorker)
        worker._win_inited, worker._scratch_slot, worker._confidence_logits = True, num_slots, None
        worker._valid_len = torch.zeros(rows, dtype=torch.long)
        worker._position_initialized = torch.zeros(rows, dtype=torch.bool)
        worker._kv_windows = torch.zeros(rows, 1, 2, 2)
        worker._batch_to_slot = torch.full((num_slots,), num_slots, dtype=torch.long)
        meta = object.__new__(dspark.DSparkSpecMetadata)
        meta.batch_indices_cuda = torch.zeros(num_slots, dtype=torch.int)
        meta._dspark_worker = worker
    worker._req_to_slot, worker._free_slots = {}, collections.deque(range(num_slots))
    worker._graph_dummy_id_floor = 1 << 40
    worker._ctx_len = torch.zeros(rows, dtype=torch.long)

    def prepare(ids, paused=()):
        meta.request_ids, meta.num_generations = list(ids), len(ids)
        meta.paused_in_place_ids = frozenset(paused)
        with patch(f"tensorrt_llm._torch.speculative.{kind}.prefer_pinned", return_value=False):
            meta.prepare()

    return worker, prepare


def executor_for_preemption(p, reqs, scheduler):
    return SimpleNamespace(
        _gen_preemption=p,
        scheduler=scheduler,
        active_requests=reqs,
        inflight_req_ids=set(),
        kv_cache_manager=Mock(spec=[]),
        kv_cache_transceiver=None,
        is_encoder_decoder=False,
        enable_attention_dp=False,
        enable_batch_waiting=False,
        _cap_context_by_total_kv_len=lambda requests: requests,
        _maybe_record_hang_diagnostic_phase=Mock(),
    )


@pytest.mark.parametrize("kind", ["dflash", "dspark"])
@pytest.mark.parametrize("settings", [ON, SLO])
def test_forced_resume_accounting_and_drafter_lifecycle(kind, settings):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SchedulerOutput

    p = policy(settings)
    # Request ID 0 is reserved for the attention-DP dummy and gets no drafter slot.
    reqs = running(p, start_id=1)
    p._pause(reqs[0])
    reqs[0].py_batch_idx = None
    assert p.step(reqs, 108.0, 1, 2, queue()) == 2  # forced resume blocks admission
    scheduler = Mock()
    scheduler.schedule_request.side_effect = lambda requests, _: SchedulerOutput(
        [],
        [],
        sorted(requests, key=lambda req: req.py_request_id not in p.resumed_ids)[:1],
        [],
        [],
        1,
    )
    ex = executor_for_preemption(p, reqs, scheduler)
    worker, prepare = drafter(kind)
    prepare([1, 2])
    slots = dict(worker._req_to_slot)
    assert set(slots) == {1, 2}
    worker._ctx_len[slots[2]] = 77
    for now, expected, paused in [(108.0, [1], {2}), (110.0, [1], {2}), (116.0, [2], {1})]:
        if now == 116.0:
            assert p.step(reqs, now, 1, 2, queue()) == 2
            assert p.stats["forced"] == 2 and p._reqs[2].paused_total == 8.0
        else:
            p.observe(reqs, now)
        scheduled, _, _ = PyExecutor._schedule(ex)
        assert [req.py_request_id for req in scheduled.generation_requests] == expected
        assert scheduled.paused_in_place_ids == p.paused_ids == paused
        assert next(req for req in reqs if req.py_request_id in paused).py_batch_idx is None
        prepare(expected, paused)
        for batch_idx, rid in enumerate(expected):
            next(req for req in reqs if req.py_request_id == rid).py_batch_idx = batch_idx
        assert worker._req_to_slot == slots and int(worker._ctx_len[slots[2]]) == 77
        if now < 116.0:
            assert p._reqs[2].paused_t == 108.0 and p._reqs[2].pauses == 1
    assert p._reqs[2].paused_t is None
    assert p._reqs[2].resumed_at_iter == reqs[1].py_decoding_iter
    reqs[1].generated += 20
    p.observe(reqs, 117.0)
    p.observe(reqs[:1], 118.0, target_ready=True)
    assert p.mean_decode_s == pytest.approx(20.0)
    assert p.stats["scored"] == 1 and p.stats["infl"] > 0
    prepare([2, 3, 4], paused={1})
    assert len(set(worker._req_to_slot.values())) == 4
    prepare([1, 2, 3])
    assert worker._req_to_slot[2] == slots[2] and int(worker._ctx_len[slots[2]]) == 77
    assert 4 not in worker._req_to_slot and len(worker._free_slots) == 1


@pytest.mark.parametrize("settings", [ON, SLO])
@pytest.mark.parametrize("force_resume", [False, True])
@pytest.mark.parametrize("replay_only", [False, True])
def test_unrelated_unscheduled_decodes_keep_pause_budget(settings, force_resume, replay_only):
    """Replay-only batches must not turn skipped decodes into preemption victims."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SchedulerOutput

    p = policy(settings)
    reqs = running(p, start_id=1)
    if force_resume:
        p._pause(reqs[0])
        assert p.step(reqs, 108.0, 1, 2, queue()) == 2
        assert p.forced_resumed_ids == {1}
    scheduler = Mock()
    reqs[1].py_batch_idx = None
    generation = [] if replay_only else reqs[:1]
    scheduler.schedule_request.return_value = SchedulerOutput(
        [], [], generation, [], [], len(generation)
    )
    ex = executor_for_preemption(p, reqs, scheduler)
    before = {rid: r.pauses for rid, r in p._reqs.items()}
    for now in (108.0, 110.0):
        p.observe(reqs, now)
        scheduled, _, _ = PyExecutor._schedule(ex)
        assert not scheduled.paused_in_place_ids
        assert not p.paused_ids
    assert {rid: r.pauses for rid, r in p._reqs.items()} == before
    assert p._victim(reqs, 1) is not None
