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
"""KVCacheV2Scheduler with extra receive slots (TRTLLM_DISAGG_GEN_EXTRA_SLOTS), mocked.

On a disaggregated generation server the executor may hold up to X requests beyond
max_batch_size while their KV cache arrives. The scheduler keeps the decode batch within
max_batch_size and seats a received request only in a free decode seat. No GPU required.
"""

import itertools
import random
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
    GEN_EXTRA_SLOTS_ENV,
    disagg_gen_extra_slots,
)
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

pytestmark = pytest.mark.cpu_only

INIT = LlmRequestState.DISAGG_GENERATION_INIT.value
RECEIVING = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS.value
RECEIVED = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE.value
DECODING = LlmRequestState.GENERATION_IN_PROGRESS.value
FINISHING = LlmRequestState.GENERATION_TO_COMPLETE.value
COMPLETE = LlmRequestState.GENERATION_COMPLETE.value
RECEIVE_PHASE = (INIT, RECEIVING, RECEIVED)

_GEN_STATES = {RECEIVED, DECODING, FINISHING}


def make_request(request_id: int, state: int, num_draft_tokens: int = 0) -> Mock:
    """A generation-only request of a disaggregated generation server."""
    req = Mock()
    req.py_csa2_remote_tail_mode = None
    req.request_id = request_id
    req.py_request_id = request_id
    req.get_beam_width_by_iter.return_value = 1
    req.py_draft_tokens = [0] * num_draft_tokens
    req.lora_task_id = None
    req.is_context_init_state = False
    req.is_first_context_chunk = True
    req.py_encoder_output_ready_event = None
    req.py_multimodal_data = None
    req.is_generation_only_request = True
    req.prompt_len = 100
    req.context_remaining_length = 100
    set_state(req, state)
    return req


def set_state(req: Mock, state: int) -> None:
    req.state_value = state
    req.is_generation_in_progress_state = state in _GEN_STATES
    req.is_generation_to_complete_state = state == FINISHING
    req.py_decoding_iter = 0 if state in RECEIVE_PHASE else 1


class _KVCacheMap(dict):
    def __missing__(self, key):
        entry = Mock()
        entry.is_active = True
        self[key] = entry
        return entry


def make_kv_cache_manager() -> Mock:
    mgr = Mock()
    mgr.kv_connector_manager = None
    mgr.tokens_per_block = 64
    mgr.enable_block_reuse = False
    mgr.enable_joint_kv_cache_reuse = False
    mgr.num_extra_kv_tokens = 0
    mgr.can_evict = False
    mgr._has_cp_helix = False
    mgr.is_vswa = False
    mgr.has_cache_tier_below_gpu = True
    mgr.kv_cache_map = _KVCacheMap()
    mgr.prepare_disagg_gen_init.side_effect = lambda req: True
    mgr.try_allocate_generation.side_effect = lambda req: True
    mgr.admit_mirror.side_effect = lambda req: True
    mgr.is_request_active.side_effect = lambda req_id: mgr.kv_cache_map[req_id].is_active
    mgr.has_pending_preemption.return_value = False
    mgr.probe_first_new_block_key.side_effect = lambda req: None
    return mgr


def make_scheduler(monkeypatch, mgr: Mock, max_batch_size: int, extra_slots: int, **env):
    if extra_slots:
        monkeypatch.setenv(GEN_EXTRA_SLOTS_ENV, str(extra_slots))
    else:
        monkeypatch.delenv(GEN_EXTRA_SLOTS_ENV, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2.KVCacheManagerV2",
        new=type(mgr),
    ):
        return scheduler_v2.KVCacheV2Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=None,
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            # The generation server never replays a context locally.
            enable_recompute_pause=False,
        )


def ids(requests) -> list[int]:
    return [r.request_id for r in requests]


def allocated_ids(mock_method) -> list[int]:
    return [c.args[0].py_request_id for c in mock_method.call_args_list]


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def test_extra_slots_are_off_by_default(monkeypatch):
    monkeypatch.delenv(GEN_EXTRA_SLOTS_ENV, raising=False)
    assert disagg_gen_extra_slots() == 0
    sched = make_scheduler(monkeypatch, make_kv_cache_manager(), 4, 0)
    assert sched.gen_extra_slots == 0


@pytest.mark.parametrize(("raw", "value"), [("", 0), ("0", 0), (" 3 ", 3), ("40", 40)])
def test_extra_slots_parse(monkeypatch, raw, value):
    monkeypatch.setenv(GEN_EXTRA_SLOTS_ENV, raw)
    assert disagg_gen_extra_slots() == value


@pytest.mark.parametrize("raw", ["-1", "two", "1.5"])
def test_extra_slots_reject_bad_values(monkeypatch, raw):
    monkeypatch.setenv(GEN_EXTRA_SLOTS_ENV, raw)
    with pytest.raises(ValueError, match=GEN_EXTRA_SLOTS_ENV):
        disagg_gen_extra_slots()


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra_slots", [0, 2])
def test_gen_init_behind_a_full_decode_batch(monkeypatch, extra_slots):
    """Off, a full decode batch ends the pass before a later gen-init request (today). On, the
    gen-init request still gets its KV cache so its receive starts now."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 2, extra_slots)
    reqs = [
        make_request(1, DECODING),
        make_request(2, DECODING),
        make_request(3, RECEIVING),
        make_request(4, INIT),
    ]
    out = sched.schedule_request(reqs, set())
    assert ids(out.generation_requests) == [1, 2]
    expected = [4] if extra_slots else []
    assert ids(out.fitting_disagg_gen_init_requests) == expected
    assert allocated_ids(mgr.prepare_disagg_gen_init) == expected


@pytest.mark.parametrize("prioritize_first_token", ["0", "1"])
def test_received_request_never_displaces_a_decoding_request(monkeypatch, prioritize_first_token):
    """A request whose KV landed waits for a seat even when it is older than the decoding
    requests (or sorted first), and nothing is allocated for it while it waits."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(
        monkeypatch,
        mgr,
        2,
        2,
        TLLM_DISAGG_GEN_PRIORITIZE_FIRST_TOKEN=prioritize_first_token,
    )
    reqs = [make_request(1, RECEIVED), make_request(2, DECODING), make_request(3, DECODING)]
    out = sched.schedule_request(reqs, set())
    assert sorted(ids(out.generation_requests)) == [2, 3]
    assert allocated_ids(mgr.try_allocate_generation) == [2, 3]


def test_received_requests_take_free_seats_in_list_order(monkeypatch):
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 3, 3)
    reqs = [
        make_request(1, DECODING),
        make_request(2, RECEIVED),
        make_request(3, RECEIVING),
        make_request(4, RECEIVED),
        make_request(5, RECEIVED),
        make_request(6, INIT),
    ]
    out = sched.schedule_request(reqs, set())
    assert sorted(ids(out.generation_requests)) == [1, 2, 4]
    assert ids(out.fitting_disagg_gen_init_requests) == [6]
    assert 5 not in allocated_ids(mgr.try_allocate_generation)


@pytest.mark.parametrize("finished_state", [FINISHING, COMPLETE])
def test_finished_request_keeps_its_seat_until_released(monkeypatch, finished_state):
    """A finished request still holds its sequence slot, so its seat is not free yet."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 2, 2)
    decoding, finished, received = (
        make_request(1, DECODING),
        make_request(2, finished_state),
        make_request(3, RECEIVED),
    )
    out = sched.schedule_request([decoding, finished, received], set())
    assert ids(out.generation_requests) == [1]
    # The executor releases it after the pass; the seat is then free.
    out = sched.schedule_request([decoding, received], set())
    assert sorted(ids(out.generation_requests)) == [1, 3]


def test_seats_count_holders_outside_the_batch(monkeypatch):
    """The free seats bound the received requests seated in one pass even when the batch
    budget would take more: a finishing holder is not in the batch but keeps its slot."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 3, 3)
    reqs = [
        make_request(1, FINISHING),
        make_request(2, RECEIVED),
        make_request(3, RECEIVED),
        make_request(4, RECEIVED),
    ]
    out = sched.schedule_request(reqs, set())
    assert ids(out.generation_requests) == [2, 3]
    assert allocated_ids(mgr.try_allocate_generation) == [2, 3]


def test_gen_init_admissions_per_pass_stay_within_max_batch_size(monkeypatch):
    """The executor prepares a pass's gen-init requests as one batch (KDA state rows sized like
    a forward batch), so at most max_batch_size of them start receiving per pass."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 2, 4)
    reqs = [make_request(i, INIT) for i in range(1, 7)]
    out = sched.schedule_request(reqs, set())
    assert ids(out.fitting_disagg_gen_init_requests) == [1, 2]
    assert allocated_ids(mgr.prepare_disagg_gen_init) == [1, 2]
    for req in reqs[:2]:
        set_state(req, RECEIVING)
    out = sched.schedule_request(reqs, set())
    assert ids(out.fitting_disagg_gen_init_requests) == [3, 4]


def test_decode_batch_and_seats_stay_within_max_batch_size(monkeypatch):
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 4, 4)
    reqs = [make_request(i, RECEIVED) for i in range(1, 9)]
    out = sched.schedule_request(reqs, set())
    assert ids(out.generation_requests) == [1, 2, 3, 4]
    for req in reqs[:4]:
        set_state(req, DECODING)
    out = sched.schedule_request(reqs, set())
    assert ids(out.generation_requests) == [1, 2, 3, 4]


def test_seat_wait_is_not_a_deadlock(monkeypatch):
    """A received request waiting behind a finishing seat holder makes no progress for a while;
    the deadlock detector must not count it, but must still count a blocked decoder."""
    mgr = make_kv_cache_manager()
    sched = make_scheduler(monkeypatch, mgr, 1, 1)
    finishing, received = make_request(1, FINISHING), make_request(2, RECEIVED)
    for _ in range(sched._DEADLOCK_STALL_ITERS + 5):
        out = sched.schedule_request([finishing, received], set())
        assert ids(out.generation_requests) == []
    assert sched._stalled_schedules == 0

    stalled = make_request(3, RECEIVED)
    with pytest.raises(RuntimeError, match="deadlock"):
        for _ in range(sched._DEADLOCK_STALL_ITERS + 1):
            sched._detect_deadlock([stalled], set(), [], set(), made_progress=False)


# ---------------------------------------------------------------------------
# Off, or never past max_batch_size: identical to today
# ---------------------------------------------------------------------------


def _trace(monkeypatch, states, max_batch_size, extra_slots, fail_ids):
    mgr = make_kv_cache_manager()
    mgr.prepare_disagg_gen_init.side_effect = lambda req: req.py_request_id not in fail_ids
    mgr.try_allocate_generation.side_effect = lambda req: req.py_request_id not in fail_ids
    sched = make_scheduler(monkeypatch, mgr, max_batch_size, extra_slots)
    reqs = [make_request(i + 1, s) for i, s in enumerate(states)]
    out = sched.schedule_request(reqs, set())
    calls = [
        (name, tuple(getattr(arg, "py_request_id", arg) for arg in args))
        for name, args, _ in mgr.method_calls
    ]
    return (
        ids(out.generation_requests),
        ids(out.fitting_disagg_gen_init_requests),
        ids(out.paused_requests),
        calls,
    )


def test_within_max_batch_size_the_option_changes_nothing(monkeypatch):
    """With at most max_batch_size active requests (all the executor admits without the extra
    slots) every pass is identical with the option on or off: same batch, same gen-init
    admissions, same KV manager calls in the same order."""
    rng = random.Random(0)
    states = (INIT, RECEIVING, RECEIVED, DECODING, FINISHING, COMPLETE)
    cases = 0
    for max_batch_size in (1, 2, 4, 8):
        for _ in range(60):
            n = rng.randint(0, max_batch_size)
            trace_states = [rng.choice(states) for _ in range(n)]
            fail_ids = {i + 1 for i in range(n) if rng.random() < 0.15}
            off = _trace(monkeypatch, trace_states, max_batch_size, 0, fail_ids)
            on = _trace(monkeypatch, trace_states, max_batch_size, 3, fail_ids)
            assert on == off, (max_batch_size, trace_states, fail_ids)
            cases += 1
    assert cases == 240


# ---------------------------------------------------------------------------
# A generation server over many passes
# ---------------------------------------------------------------------------


class _GenServer:
    """The executor side of a disaggregated generation server, pass by pass.

    Admission holds up to max_batch_size + extra_slots requests (FIFO), a gen-init request
    receives for ``receive_passes`` passes once the scheduler allocates its KV cache, and a
    seated request decodes ``decode_passes`` steps, then lingers one pass as finishing before the
    executor releases it, as under the overlap scheduler.
    """

    def __init__(self, monkeypatch, max_batch_size, extra_slots, receive_passes, decode_passes):
        self.mgr = make_kv_cache_manager()
        self.sched = make_scheduler(monkeypatch, self.mgr, max_batch_size, extra_slots)
        self.max_batch_size = max_batch_size
        self.capacity = max_batch_size + extra_slots
        self.receive_passes = receive_passes
        self.decode_passes = decode_passes
        self.active = []
        self.left = {}
        self.history = []
        self.seated_at = {}
        self.received_at = {}
        self.finished_at = {}

    def step(self, t, waiting):
        while waiting and len(self.active) < self.capacity:
            self.active.append(make_request(waiting.pop(0), INIT))
        out = self.sched.schedule_request(list(self.active), set())
        batch = ids(out.generation_requests)
        seated_before = {r.request_id for r in self.active if r.state_value in (DECODING,)}
        # Invariants of every pass.
        assert len(batch) <= self.max_batch_size
        assert len(out.fitting_disagg_gen_init_requests) <= self.max_batch_size
        assert seated_before <= set(batch), "a decoding request lost its seat"
        holders = [r for r in self.active if r.state_value not in RECEIVE_PHASE]
        newly_seated = [
            r for r in self.active if r.state_value == RECEIVED and r.request_id in batch
        ]
        assert len(holders) + len(newly_seated) <= self.max_batch_size, "sequence slots exceeded"
        self.history.append(batch)
        for req in out.fitting_disagg_gen_init_requests:
            set_state(req, RECEIVING)
            self.left[req.request_id] = self.receive_passes
        # Previous pass's finishing requests are released after this pass's scheduling.
        for req in [r for r in self.active if r.state_value == FINISHING]:
            self.active.remove(req)
            self.finished_at[req.request_id] = t
        for req in self.active:
            rid = req.request_id
            if req.state_value == RECEIVED and rid in batch:
                set_state(req, DECODING)
                self.seated_at[rid] = t
                self.left[rid] = self.decode_passes
            if req.state_value == DECODING and rid in batch:
                self.left[rid] -= 1
                if self.left[rid] == 0:
                    set_state(req, FINISHING)
            elif req.state_value == RECEIVING:
                self.left[rid] -= 1
                if self.left[rid] == 0:
                    set_state(req, RECEIVED)
                    self.received_at[rid] = t

    def run(self, num_requests, max_passes=10_000):
        waiting = list(range(1, num_requests + 1))
        t = 0
        while (waiting or self.active) and t < max_passes:
            self.step(t, waiting)
            t += 1
        return t


@pytest.mark.parametrize(
    ("max_batch_size", "extra_slots", "receive_passes", "decode_passes"),
    list(itertools.product([1, 2, 4], [1, 2, 4], [1, 3, 9], [5, 30])),
)
def test_gen_server_lifecycle(
    monkeypatch, max_batch_size, extra_slots, receive_passes, decode_passes
):
    """Every request completes; requests are seated in the order their KV landed; a seated
    request decodes every pass until it finishes; and the receive no longer holds a seat."""
    num_requests = 24
    off = _GenServer(monkeypatch, max_batch_size, 0, receive_passes, decode_passes)
    on = _GenServer(monkeypatch, max_batch_size, extra_slots, receive_passes, decode_passes)
    t_off = off.run(num_requests)
    t_on = on.run(num_requests)
    for server in (off, on):
        assert sorted(server.finished_at) == list(range(1, num_requests + 1))
        # Seated in the order the KV landed (FIFO among received requests, no starvation).
        by_landing = sorted(server.received_at, key=lambda rid: (server.received_at[rid], rid))
        by_seat = sorted(server.seated_at, key=lambda rid: (server.seated_at[rid], rid))
        assert by_seat == by_landing
        # A seated request runs in every pass from its seat to its last step.
        for rid, start in server.seated_at.items():
            steps = [t for t, batch in enumerate(server.history) if rid in batch]
            assert steps == list(range(start, start + decode_passes))
    # Without extra slots the receive sits inside the seat; with them it overlaps a decode.
    assert t_on < t_off
    busy_off = sum(len(b) for b in off.history) / (t_off * max_batch_size)
    busy_on = sum(len(b) for b in on.history) / (t_on * max_batch_size)
    assert busy_on > busy_off
    # Once the pipeline is full, a free seat is refilled in the next pass at most: the slot
    # use of the steady part is the decode passes over decode + linger.
    if extra_slots * decode_passes >= receive_passes * max_batch_size:
        steady = on.history[receive_passes + 2 : -(decode_passes + 2)]
        if steady:
            use = sum(len(b) for b in steady) / (len(steady) * max_batch_size)
            assert use >= decode_passes / (decode_passes + 2) - 1e-9
