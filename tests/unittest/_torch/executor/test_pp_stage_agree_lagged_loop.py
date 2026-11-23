# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PP stage agreement and the lagged send agreement in one real PP loop.

The real ``PyExecutor._executor_loop_pp`` with the real stage agreement
(exchange, answer, admitted-depth check, adoption, drain) and the real lagged
send agreement (``PPDisaggLaggedConsensus``, the ``KvCacheTransceiverV2``
lagged code, coordinator and ``LaggedDisaggPPTerminationHandler``), on the
loop harness of ``test_pp_disagg_lagged_consensus.py``. Stand-in: a V2 pool
per stage with per-stage reusable prefixes and sliding-window-like pruning
(odd stages can claim only 0 or their whole prefix).

Checked: no PP or world collective with both agreements; every message matched; the
same iteration count on every rank and in both agreements; every first chunk
admitted at the largest depth every stage can claim exactly, from each stage's
reservation; commits at the handled chunk's end; identical send outcomes and
terminations on every stage. Generation-first requests, which the lagged step
promotes, are agreed before the first rank can admit them.
"""

import sys
import threading
import time
import types
from typing import Dict, List
from unittest.mock import MagicMock

import pytest
import test_pp_disagg_lagged_consensus as H
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.pp_stage_agree import (
    pp_stage_agree_layout,
    reuse_cap,
    set_reserved_depth,
)
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

BLOCK = 64


def _valid_endpoints(stage: int, rid: int, prompt_len: int) -> List[int]:
    """Stage-local claimable depths: full attention on even stages (every block up
    to the stage's prefix), sliding-window-like on odd stages (0 or the full
    prefix only; interior endpoints pruned)."""
    blocks = (rid * 3 + stage * 2) % 5  # 0..4 blocks reusable
    if rid % 4 == 3:
        blocks = 0  # some prompts reuse nothing
    top = blocks * BLOCK
    assert top < prompt_len
    if stage % 2 == 0:
        return list(range(0, top + 1, BLOCK))
    return sorted({0, top})


def _expected_depth(pp_size, rid, prompt_len):
    common = set(_valid_endpoints(0, rid, prompt_len))
    for s in range(1, pp_size):
        common &= set(_valid_endpoints(s, rid, prompt_len))
    return max(common)


class _Req(H._Request):
    def __init__(self, request_id, prompt_len):
        super().__init__(request_id, prompt_len)
        self.prepopulated = 0

    @property
    def is_first_context_chunk(self) -> bool:
        return self.context_current_position == self.prepopulated


class _Pool:
    """Stand-in V2 pool of one stage: reservations hold a depth; the claim
    prunes to the stage's valid endpoints."""

    # Its update_context_resources takes context_ends, as KVCacheManagerV2's does.
    _supports_pp_context_ends = True

    def __init__(self, stage, prompts, fail_first_reserve=()):
        self.stage = stage
        self.prompts = prompts
        self.kv_cache_map: Dict[int, int] = {}
        self.reserved: Dict[int, int] = {}
        self.admitted_at: Dict[int, int] = {}
        self.commits: Dict[int, List[int]] = {}
        self.launch_ends: Dict[int, List[int]] = {}
        self.num_reserve = 0
        self.num_probe = 0
        self.num_reclaimed = 0
        self.num_update_calls_without_ends = 0
        self.num_release_admission = 0
        self._fail_first = set(fail_first_reserve)
        self._lock = threading.Lock()

    def claim(self, rid, cap):
        valid = _valid_endpoints(self.stage, rid, self.prompts[rid])
        return max(v for v in valid if cap is None or v <= cap)

    # KVCacheManagerV2 surface the loop and the agreement read
    def probe_context_reuse(self, req, include_claimed=False):
        assert include_claimed
        self.num_probe += 1
        return self.claim(req.py_request_id, None)

    def update_context_resources(self, scheduled_batch, context_ends=None):
        if context_ends is None:
            self.num_update_calls_without_ends += 1
            return
        for req in scheduled_batch.context_requests:
            end = context_ends[req.py_request_id]
            assert end <= req.context_current_position, (end, req.context_current_position)
            self.commits.setdefault(req.py_request_id, []).append(end)

    # KVCacheV2Scheduler surface
    def reserve(self, req, cap):
        rid = req.py_request_id
        self.release(req)
        self.num_reserve += 1
        if rid in self._fail_first:
            self._fail_first.discard(rid)
            return None
        depth = self.claim(rid, cap)
        self.reserved[rid] = depth
        self.kv_cache_map[rid] = depth
        set_reserved_depth(req, depth)
        return depth

    def release(self, req):
        rid = req.py_request_id
        self.reserved.pop(rid, None)
        self.kv_cache_map.pop(rid, None)

    def release_admission(self, req):
        """KVCacheV2Scheduler.release_context_admission: drop the caches and
        rewind the cursor to the prompt start."""
        self.release(req)
        self.admitted_at.pop(req.py_request_id, None)
        req.context_current_position = 0
        req.prepopulated = 0
        self.num_release_admission += 1

    def admit_leader(self, req):
        """First rank's scheduler admission: resume the reservation (capped)."""
        rid = req.py_request_id
        cap = reuse_cap(req)
        depth = self.reserved.pop(rid, None)
        if depth is None or (cap is not None and depth != cap):
            depth = self.claim(rid, cap)
        self.kv_cache_map[rid] = depth
        self.admitted_at[rid] = depth
        return depth

    def adopt_first(self, req, position, chunk_size):
        rid = req.py_request_id
        depth = self.reserved.pop(rid, None)
        if depth != position:
            self.num_reclaimed += 1
            depth = self.claim(rid, position)
            if depth != position:
                raise RuntimeError(f"stage {self.stage}: cannot resume {rid} at {position}")
        self.kv_cache_map[rid] = depth
        self.admitted_at[rid] = depth
        set_reserved_depth(req, None)
        req.context_current_position = position
        req.prepopulated = position
        req.context_chunk_size = chunk_size
        return True


class _Combined(H._Executor):
    def _schedule(self):
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self._schedulable_active_requests():
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            if req.is_first_context_chunk and req.request_id not in self._pool.admitted_at:
                depth = self._pool.admit_leader(req)
                req.context_current_position = depth
                req.prepopulated = depth
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)

    def _forward_step_inter_pp(self, scheduled_batch, gpu_forward_start=None, gpu_forward_end=None):
        for req in scheduled_batch.context_requests:
            self._pool.launch_ends.setdefault(req.py_request_id, []).append(
                req.context_current_position + req.context_chunk_size
            )
        return super()._forward_step_inter_pp(scheduled_batch, gpu_forward_start, gpu_forward_end)

    def _forward_step(self, scheduled_batch, *args, **kwargs):
        for req in scheduled_batch.context_requests:
            self._pool.launch_ends.setdefault(req.py_request_id, []).append(
                req.context_current_position + req.context_chunk_size
            )
        return super()._forward_step(scheduled_batch, *args, **kwargs)

    # V2-only loop hooks with nothing to do here
    def _can_pause_for_rebalance(self):
        return False

    def _terminate_recompute_paused_requests(self, scheduled_batch):
        pass

    def _pause_recompute_paused_requests(self, scheduled_batch):
        pass

    def _maybe_finish_pp_rebalance(self):
        pass

    def _progress_recompute_pause_termination_if_idle(self, executed_batch_num):
        pass

    def _do_terminate_request(self, request):
        super()._do_terminate_request(request)
        self._pool.kv_cache_map.pop(request.py_request_id, None)


def _make(world, rank, n, prompts, budget, gpu, log, cfg):
    exe = H._make_executor(world, rank, n, list(prompts), budget, gpu, log, cfg)
    exe.__class__ = _Combined
    exe.active_requests = [_Req(i, p) for i, p in enumerate(prompts)]
    pool = _Pool(rank, list(prompts), cfg.get("fail_first_reserve", {}).get(rank, ()))
    exe._pool = pool
    exe.kv_cache_manager = pool
    exe.draft_kv_cache_manager = None
    exe.enable_joint_kv_cache_reuse = False
    exe._is_kv_manager_v2 = True
    exe.kv_connector_manager = None
    # Real knob wiring (overrides the harness's hard-coded adoption).
    # A disaggregated PP server, as PyExecutor.__init__ wires it: the agreement,
    # the leader-batch adoption it builds on and, with block reuse, the early
    # release (turned off for the arm that checks the release after adoption).
    exe._pp_disaggregated = True
    exe._pp_stage_agree = exe._create_pp_stage_agreement()
    exe._pp_adopt_leader_chunks = exe._pp_stage_agree is not None
    exe._pp_follower_early_release = exe._pp_stage_agree.reuse_enabled and cfg.get(
        "early_release", True
    )
    local_pass = types.SimpleNamespace(
        context_requests=[], generation_requests=[], fitting_disagg_gen_init_requests=[]
    )
    seen = cfg["seen"]

    def _schedule_request(active, inflight):
        # The follower pass gets no waiting first chunk (stage agreement).
        with seen["lock"]:
            seen["follower_pass_first_chunks"] += sum(
                1
                for r in active
                if r.is_first_context_chunk
                and not r.context_done
                and r.context_current_position == 0
                and r.request_id not in pool.admitted_at
            )
        return local_pass

    def _adopt(req, size):
        req.context_chunk_size = size
        return True

    exe.scheduler = H._Noop(
        schedule_request=_schedule_request,
        adopt_context_chunk=_adopt,
        adopt_first_context_chunk=pool.adopt_first,
        reserve_first_context_chunk=pool.reserve,
        release_context_admission=pool.release_admission,
        adopt_generation_step=lambda req: True,
        reservation_leaves_chunk_room=lambda req: True,
    )
    original_adopt = exe._pp_adopt_leader_schedule

    def _record_adopt(batch, local):
        with seen["lock"]:
            if batch.leader_context_chunks is not None:
                seen["wire_chunks"] += 1
            if batch.leader_first_context_chunks:
                seen["wire_first"] += len(batch.leader_first_context_chunks)
            if batch.leader_reuse_proposals:
                seen["wire_proposals"] += len(batch.leader_reuse_proposals)
        return original_adopt(batch, local)

    exe._pp_adopt_leader_schedule = _record_adopt
    return exe


def _run(
    monkeypatch,
    *,
    pp_size=4,
    ring_depth=None,
    prompts,
    budget=128,
    cost_s=0.002,
    fabric=None,
    fail_first_reserve=None,
    early_release=False,
):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(py_executor_module, "CUASSERT", lambda result: result)
    monkeypatch.setattr(py_executor_module, "cudart", MagicMock())
    n = H.ring_slots(pp_size, ring_depth)
    world = H._World(pp_size)
    log = H._Log(pp_size)
    links = [H._Link() for _ in range(pp_size - 1)]
    gpus = [
        H._Gpu(
            s, links[s - 1] if s > 0 else None, links[s] if s < pp_size - 1 else None, cost_s, log
        )
        for s in range(pp_size)
    ]
    seen = {
        "lock": threading.Lock(),
        "wire_chunks": 0,
        "wire_first": 0,
        "wire_proposals": 0,
        "follower_pass_first_chunks": 0,
    }
    cfg = {
        "fabric": fabric or H._Fabric(delay_s=0.003),
        "timeout_ms": 600_000,
        "block_reuse": True,
        "require_terminated": True,
        "max_iterations": 4000,
        "transceiver": True,
        "seen": seen,
        "fail_first_reserve": fail_first_reserve or {},
        "lagged": True,
        "early_release": early_release,
    }
    executors = [_make(world, r, n, prompts, budget, gpus[r], log, cfg) for r in range(pp_size)]
    errors = []

    def _rank(exe):
        try:
            exe._executor_loop_pp()
        except BaseException as exc:  # noqa: BLE001
            errors.append((exe.dist.rank, exc))

    threads = [threading.Thread(target=_rank, args=(e,), daemon=True) for e in executors]
    for t in threads:
        t.start()
    for t in threads:
        t.join(H.TIMEOUT_S * 2)
    for g in gpus:
        g.stop()
    assert not any(t.is_alive() for t in threads), "the PP loop deadlocked"
    assert not errors, errors
    assert not [g.error for g in gpus if g.error is not None]
    return executors, world, log, n, seen


PROMPTS = tuple(6 * BLOCK + 10 for _ in range(12))


@pytest.mark.parametrize(
    "pp_size,ring_depth,early_release",
    [(4, None, False), (4, 8, False), (2, None, False), (3, 5, False), (4, 8, True), (3, 5, True)],
)
def test_stage_agree_and_lagged_consensus_in_one_real_loop(
    monkeypatch, pp_size, ring_depth, early_release
):
    prompts = PROMPTS
    executors, world, log, n, seen = _run(
        monkeypatch,
        pp_size=pp_size,
        ring_depth=ring_depth,
        prompts=prompts,
        fail_first_reserve={pp_size - 1: (2,)},
        early_release=early_release,
    )
    leader = executors[0]
    iterations = leader.iter_counter
    # The agreement brings the adoption; its layout keeps all-SWA stage quotas.
    assert all(e._pp_adopt_leader_chunks for e in executors)
    assert pp_stage_agree_layout(
        pp_size=pp_size, tp_size=1, cp_size=1, enable_attention_dp=False, disaggregated=True
    )
    # No PP / world collective at all with both agreements.
    assert all(not calls for calls in world.collectives.values()), world.collectives
    # Every message on every tag was received (stage reports, lagged reports and
    # decisions, schedules, relays).
    unmatched = {k: q.qsize() for k, q in world._queues.items() if not q.empty()}
    assert not unmatched, unmatched
    tags = {k[2] for k in world._queues}
    assert int(PPCommTag.STAGE_REUSE_REPORT) in tags
    assert H.PP_DISAGG_REPORT_TAG in tags and H.PP_DISAGG_DECISION_TAG in tags
    # Same iteration count on both agreements on every rank.
    for e in executors:
        assert e.iter_counter == iterations
        assert e._pp_stage_agree.iteration == iterations
        assert e._pp_disagg_lag.iteration == iterations
        assert not e._pp_stage_agree.pending_sends
        assert not e._pp_disagg_lag._report_sends and not e._pp_disagg_lag._decision_sends
    # Every first chunk admitted at the largest depth every stage can claim
    # exactly, from each stage's reservation; nothing undone, nothing re-claimed.
    agreement = leader._pp_stage_agree
    assert agreement.num_admitted == len(prompts)
    assert agreement.num_undone == 0
    assert agreement.num_retries >= 1  # the failed follower reservation was asked again
    for e in executors:
        pool = e._pool
        assert pool.num_reclaimed == 0
        assert pool.reserved == {}
        assert pool.num_update_calls_without_ends == 0
        for rid, p in enumerate(prompts):
            assert pool.admitted_at[rid] == _expected_depth(pp_size, rid, p), (e.dist.rank, rid)
            # Commits stop at each handled chunk's end, in launch order, up to the prompt end.
            assert pool.commits[rid] == pool.launch_ends[rid], (e.dist.rank, rid)
            assert pool.commits[rid][-1] == p
        assert pool.kv_cache_map == {}
    for e in executors[1:]:
        assert e._pool.commits == leader._pool.commits
    assert seen["follower_pass_first_chunks"] == 0
    assert seen["wire_chunks"] == (pp_size - 1) * iterations
    assert seen["wire_first"] == (pp_size - 1) * len(prompts)
    assert seen["wire_proposals"] > 0
    # Every send completed and freed once, in the same iteration on every stage.
    for r in range(pp_size):
        assert set(log.released[r]) == set(range(len(prompts)))
        assert all(
            s == LlmRequestState.DISAGG_CONTEXT_COMPLETE for _, s in log.released[r].values()
        )
        assert sorted(rid for _, rid in log.terminated[r]) == list(range(len(prompts)))
    for r in range(1, pp_size):
        assert log.released[r] == log.released[0]
        assert log.terminated[r] == log.terminated[0]
    assert leader._pp_disagg_lag.num_completed == len(prompts)
    assert leader._pp_disagg_lag.num_terminated == len(prompts)
    lowered = sum(e._pp_stage_agree.num_early_lowerings for e in executors[1:])
    if early_release:
        # Odd stages claim 0 or their whole prefix: the first rank lowers, and
        # the followers that hold deeper lower before adopting.
        assert lowered > 0
        assert all(e._pp_stage_agree.num_early_skipped_in_batch == 0 for e in executors[1:])
    else:
        assert lowered == 0
    print(
        f"\n[agree+lagged pp={pp_size} ring={n} lag={agreement.lag}] iterations={iterations} "
        f"handled={log.handled[0]} max_busy={log.max_busy} max_lead={log.max_lead} "
        f"period={H._period(log) * 1e3:.2f} ms; {agreement.summary()}"
    )


def test_agree_plus_lagged_with_a_failed_send_on_one_stage(monkeypatch):
    prompts = PROMPTS[:8]
    fabric = H._Fabric(0.003, {(2, 5): "fail"})
    executors, world, log, n, seen = _run(
        monkeypatch, pp_size=4, ring_depth=8, prompts=prompts, fabric=fabric
    )
    for r in range(4):
        outcomes = {rid: s for rid, (_, s) in log.released[r].items()}
        assert outcomes[5] == LlmRequestState.DISAGG_TRANS_ERROR
        assert executors[r].kv_cache_transceiver._send_sessions == {}
    for r in range(1, 4):
        assert log.released[r] == log.released[0]
        assert log.terminated[r] == log.terminated[0]
    assert all(not calls for calls in world.collectives.values())
    assert executors[0]._pp_stage_agree.num_admitted == len(prompts)


# --------------------------------------------------------------------------- generation-first

_WAIT = LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER


def test_generation_first_requests_are_agreed_before_the_first_rank_admits_them(monkeypatch):
    """Generation-first requests wait (DISAGG_CONTEXT_WAIT_SCHEDULER) until every
    stage has its peer's info; with the lagged consensus the lagged step promotes
    them. The loop runs that step before the stage-agreement exchange, so a
    promoted first chunk is held until every stage confirmed its depth: the
    first rank never admits it unagreed, nothing is undone and no "agreed again"
    warning is logged. (With the exchange first, the first rank admitted a just
    promoted chunk with no confirmed depth and took it back out of the batch.)"""
    gen_first = {1, 4, 7, 9}
    # Per stage, when the peer's info for the generation-first requests arrives.
    arrival_s = {0: 0.0, 1: 0.03, 2: 0.15, 3: 0.08}
    t0 = time.monotonic()

    def _has_peer_info(self, rid):
        return rid not in gen_first or time.monotonic() - t0 >= arrival_s[self._rank]

    monkeypatch.setattr(H._TransferWorker, "has_all_peer_req_infos_for_send", _has_peer_info)
    monkeypatch.setattr(
        _Req,
        "is_context_init_state",
        property(lambda self: not self.context_done and self.state != _WAIT),
    )

    def _fetch(self):
        if self.iter_counter > self._max_iterations:
            raise RuntimeError("did not settle")
        if getattr(self, "_gen_first_fed", False):
            return []
        self._gen_first_fed = True
        new = [r for r in self.active_requests if r.py_request_id in gen_first]
        for r in new:
            r.py_disaggregated_params = types.SimpleNamespace(
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST, disagg_request_id=None
            )
        return new

    monkeypatch.setattr(_Combined, "_fetch_and_activate_new_requests", _fetch)
    schedule = _Combined._schedule

    def _schedule(self):
        # The first rank's scheduler never picks a request still waiting.
        held = self.active_requests
        self.active_requests = [r for r in held if r.state != _WAIT]
        try:
            return schedule(self)
        finally:
            self.active_requests = held

    monkeypatch.setattr(_Combined, "_schedule", _schedule)
    undone_warnings = []
    warning = py_executor_module.logger.warning

    def _warn(msg, *args, **kwargs):
        if "is agreed again" in str(msg):
            undone_warnings.append(msg)
        return warning(msg, *args, **kwargs)

    monkeypatch.setattr(py_executor_module.logger, "warning", _warn)
    prompts = PROMPTS[:10]
    executors, world, log, n, seen = _run(monkeypatch, pp_size=4, ring_depth=8, prompts=prompts)
    agreement = executors[0]._pp_stage_agree
    for e in executors:
        assert e._pool.num_reclaimed == 0
        for rid, p in enumerate(prompts):
            assert e._pool.admitted_at[rid] == _expected_depth(4, rid, p)
    for r in range(1, 4):
        assert log.released[r] == log.released[0]
        assert log.terminated[r] == log.terminated[0]
    assert all(not calls for calls in world.collectives.values())
    assert agreement.num_admitted == len(prompts)
    assert agreement.num_undone == 0, agreement.summary()
    assert not undone_warnings


@pytest.mark.parametrize("early_release", [False, True])
def test_follower_releases_before_its_checks_its_pass_and_the_adoption(monkeypatch, early_release):
    """Executor-level order on a follower, every iteration of the real loop: with
    the follower early release it runs first, then the
    retry check, the follower's own pass, the adoption and the answer; without
    it the early release never runs."""
    events: Dict[int, List[str]] = {}
    make = _make

    def traced_make(world, rank, n, prompts, budget, gpu, log, cfg):
        exe = make(world, rank, n, prompts, budget, gpu, log, cfg)
        trace = events.setdefault(rank, [])

        def wrap(owner, name, label):
            original = getattr(owner, name)

            def traced(*args, **kwargs):
                trace.append(label)
                return original(*args, **kwargs)

            setattr(owner, name, traced)

        wrap(exe, "_pp_stage_agree_early_release", "early")
        wrap(exe, "_pp_retry_until_can_schedule", "retry")
        wrap(exe.scheduler, "schedule_request", "pass")
        wrap(exe, "_pp_adopt_leader_schedule", "adopt")
        wrap(exe, "_pp_stage_agree_answer", "answer")
        return exe

    monkeypatch.setattr(sys.modules[__name__], "_make", traced_make)
    executors, *_ = _run(
        monkeypatch, pp_size=4, ring_depth=8, prompts=PROMPTS[:6], early_release=early_release
    )
    iterations = executors[0].iter_counter
    expected = (["early"] if early_release else []) + ["retry", "pass", "adopt", "answer"]
    for rank in range(1, 4):
        assert events[rank] == expected * iterations, (rank, events[rank][:12])
        assert executors[rank]._pp_follower_early_release is early_release
    assert "early" not in events[0]
