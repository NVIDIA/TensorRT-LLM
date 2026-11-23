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
"""Per-stage context-path counters of a pipeline-parallel executor (``pp_ctx_diag.py``).

The counter arithmetic (binding stage and lost tokens of the stage agreement,
reservation lowering and drops, suspend / evict / resume with pages per cache
level, admission holds, gaps between the chunks of one prompt), the defaults,
the line format, and ``PPStageAgreement`` driven through a lowering round with
and without the counters: the same holds, proposals and summary either way, and
nothing counted without them.

Both modules import nothing from the package, so they are loaded from their
files when ``tensorrt_llm`` cannot be imported.
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def _load(name):
    try:
        return importlib.import_module(f"tensorrt_llm._torch.pyexecutor.{name}")
    except ImportError:
        path = Path(__file__).resolve().parents[4] / f"tensorrt_llm/_torch/pyexecutor/{name}.py"
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


D = _load("pp_ctx_diag")
A = _load("pp_stage_agree")


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _diag(pp_size=4, pp_rank=0, **kwargs):
    return D.PPCtxDiag(pp_rank, pp_size, clock=kwargs.pop("clock", _Clock()), **kwargs)


def _fields(line):
    """A counters line as ``{name: value}`` (values as written)."""
    assert line.startswith(D.LINE_TAG + " ")
    return dict(token.split("=", 1) for token in line[len(D.LINE_TAG) + 1 :].split(" "))


# --------------------------------------------------------------------------- defaults


def test_defaults():
    assert (D.DEFAULT_INTERVAL_S, D.DEFAULT_GAP_ITERS, D.DEFAULT_DEEP_TOKENS) == (30.0, 1, 131072)
    diag = _diag()
    assert diag.guard_timing is False
    assert _diag(guard_timing=True).guard_timing is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pp_rank": 4},
        {"pp_rank": -1},
        {"interval_s": 0},
        {"gap_iters": 0},
        {"deep_tokens": 0},
    ],
)
def test_bad_construction_raises(kwargs):
    with pytest.raises(ValueError):
        _diag(**kwargs)


@pytest.mark.parametrize(
    "gap,bucket", [(1, 0), (2, 1), (64, 1), (65, 2), (2000, 2), (2001, 3), (10**6, 3)]
)
def test_gap_buckets(gap, bucket):
    assert D.gap_bucket(gap) == bucket


def test_gap_bucket_rejects_zero():
    with pytest.raises(ValueError):
        D.gap_bucket(0)


# --------------------------------------------------------------------------- stage agreement


def test_round_binder_is_the_follower_with_the_smallest_answer():
    diag = _diag()
    diag.agreement_round(25144, {1: 25144, 2: 25144, 3: 128})
    diag.agreement_round(8192, {1: 8192, 2: 8192, 3: 8192})
    assert diag.agree_rounds == 2
    assert diag.agree_rounds_lowered == 1
    assert diag.agree_round_bound == [0, 0, 0, 1]
    assert diag.agree_round_lost_tokens == [0, 0, 0, 25016]
    assert diag.agree_rounds_tied == 0


def test_tied_round_counts_every_tied_stage_with_the_full_loss():
    diag = _diag()
    diag.agreement_round(4096, {1: 1024, 2: 4096, 3: 1024})
    assert diag.agree_round_bound == [0, 1, 0, 1]
    assert diag.agree_round_lost_tokens == [0, 3072, 0, 3072]
    assert diag.agree_rounds_tied == 1


def test_first_rank_reclaim_below_the_allowed_depth_binds_stage_zero():
    diag = _diag()
    diag.agreement_reclaim(8192, 8192)
    diag.agreement_reclaim(8192, None)
    assert diag.agree_round_bound == [0, 0, 0, 0]
    diag.agreement_reclaim(8192, 4096)
    assert diag.agree_round_bound == [1, 0, 0, 0]
    assert diag.agree_round_lost_tokens == [4096, 0, 0, 0]


def test_admitted_binder_is_the_stage_with_the_smallest_own_prefix():
    diag = _diag()
    diag.agreement_admitted(128, 25144, {1: 25144, 2: 25600, 3: 128})
    assert diag.agree_admitted == 1
    assert diag.agree_admitted_below_best == 1
    assert diag.agree_binder == [0, 0, 0, 1]
    assert diag.agree_binder_lost_tokens == [0, 0, 0, 25600 - 128]
    assert diag.agree_lost_vs_best_tokens == 25600 - 128
    assert diag.agree_lost_vs_leader_tokens == 25144 - 128
    # Admitted at the largest own prefix: nothing lost, no binder.
    diag.agreement_admitted(4096, 4096, {1: 4096, 2: 4096, 3: 4096})
    assert diag.agree_admitted == 2
    assert diag.agree_admitted_below_best == 1
    assert diag.agree_binder == [0, 0, 0, 1]


def test_admitted_without_follower_prefixes_is_counted():
    diag = _diag()
    diag.agreement_admitted(0, 0, {})
    assert diag.agree_admitted_without_prefixes == 1
    assert diag.agree_admitted_below_best == 0


def test_held_chunk_iterations_add_up():
    diag = _diag()
    for held in (3, 0, 2):
        diag.agreement_held(held)
    assert diag.agree_held_chunk_iters == 5


# --------------------------------------------------------------------------- reservations


def test_reservation_claims_count_new_lowered_and_failed():
    diag = _diag()
    diag.reservation_claimed(None, 25144)  # new
    diag.reservation_claimed(25144, 25144)  # same depth again: not lowered
    diag.reservation_claimed(25144, 128)  # lowered by 25016
    diag.reservation_claimed(128, None)  # failed while holding: dropped
    diag.reservation_claimed(None, None)  # failed, held nothing
    assert diag.resv_made == 1
    assert (diag.resv_lowered, diag.resv_lowered_tokens) == (1, 25016)
    assert diag.resv_failed == 2
    assert (diag.resv_dropped, diag.resv_dropped_tokens) == (1, 128)


def test_released_reservations_count_their_depth():
    diag = _diag()
    diag.reservation_released(17856)
    diag.reservation_released(None)
    assert (diag.resv_dropped, diag.resv_dropped_tokens) == (1, 17856)


def test_adoptions_without_a_reservation_at_the_depth_are_fresh_claims():
    diag = _diag(pp_rank=3)
    diag.first_chunk_adopted(8192, 8192)
    diag.first_chunk_adopted(None, 8192)
    diag.first_chunk_adopted(16384, 8192)
    diag.first_chunk_reclaimed()
    assert (diag.adopt_first, diag.adopt_fresh_claims, diag.adopt_reclaims) == (3, 2, 1)


# --------------------------------------------------------------------------- suspend / evict / resume


def test_suspend_evict_and_preempt_split_by_kind():
    diag = _diag()
    diag.context_suspended(started=True)
    diag.context_suspended(started=False)
    diag.context_suspended(started=True, evicted=True)
    diag.reservation_suspended()
    diag.reservation_suspended(page_refused=True)
    diag.context_preempted(started=True)
    assert diag.ctx_suspended == {"first": 1, "started": 1}
    assert diag.ctx_evicted == {"first": 0, "started": 1}
    assert diag.ctx_preempted == {"first": 0, "started": 1}
    assert diag.ctx_reservations_suspended == 2
    assert diag.ctx_reservations_page_refused == 1


def test_resume_counts_pages_below_the_gpu_and_host_time():
    diag = _diag()
    diag.context_resumed(True, True, [40, 300, 25], 0.002)
    diag.context_resumed(True, True, [10, 0, 5], 0.001)
    # The second manager of the same request: pages and time, not the event.
    diag.context_resumed(True, True, [0, 7], 0.0005, count_event=False)
    diag.context_resumed(False, True, [2, 3], 0.004)
    diag.context_resumed(False, False, [0, 9], 0.0001)
    diag.context_resumed(False, True, None, 0.0)
    assert diag.ctx_resumed == {"first": 2, "started": 2}
    assert diag.ctx_resume_refused == {"first": 1, "started": 0}
    assert diag.resume_onboarded_pages == {"started": {1: 307, 2: 30}, "first": {1: 3}}
    assert diag.resume_pages_unknown == 1
    assert diag.resume_host_s["started"] == pytest.approx(0.0035)
    assert diag.resume_host_s["first"] == pytest.approx(0.0041)
    assert diag.resume_host_s_max == pytest.approx(0.004)


# --------------------------------------------------------------------------- admission holds


def test_admission_holds_count_iterations_chunks_and_distinct_requests():
    diag = _diag()
    diag.admission_holds([1, 2])
    diag.admission_holds([])
    diag.admission_holds([2, 3])
    assert diag.admission_hold_iters == 2
    assert diag.admission_held_chunk_iters == 4
    assert diag.admission_held_requests == 3
    assert diag.admission_held_now == 2
    diag.admission_holds(())
    assert diag.admission_held_now == 0
    diag.forget_except([3])
    diag.admission_holds([2])
    assert diag.admission_held_requests == 4


# --------------------------------------------------------------------------- interrupted prefills


def _req(request_id, position, first=False, last=False):
    return types.SimpleNamespace(
        py_request_id=request_id,
        context_current_position=position,
        is_first_context_chunk=first,
        is_last_context_chunk=last,
    )


def _run(diag, schedule, iterations):
    """``schedule``: iteration -> requests; every other iteration runs no context chunk."""
    for iteration in range(iterations):
        diag.record_context_chunks(schedule.get(iteration, ()))


def test_gaps_between_chunks_of_one_prompt():
    diag = _diag(deep_tokens=16384)
    schedule = {
        0: [_req(1, 0, first=True), _req(2, 4096, first=True)],
        1: [_req(1, 8192)],  # gap 1
        5: [_req(1, 16384)],  # gap 4, deep
        6: [_req(2, 12288)],  # gap 6
        100: [_req(1, 24576)],  # gap 95, deep
        2200: [_req(1, 32768, last=True)],  # gap 2100, deep, last
        2201: [_req(2, 20480, last=True)],  # gap 2195, deep, last
    }
    _run(diag, schedule, 2202)
    assert diag.iteration == 2202
    assert diag.chunks == {"first": 2, "started": 6}
    assert diag.gaps_shallow == [1, 1, 0, 0]
    assert diag.gaps_deep == [0, 1, 1, 2]
    assert diag.interrupted == 5
    assert diag.interrupted_deep == 4
    assert diag.interrupted_max_gap == 2195
    # Last chunks leave no state behind.
    assert diag._last_chunk_iteration == {}


def test_gap_threshold_decides_what_is_interrupted():
    diag = _diag(gap_iters=64)
    _run(diag, {0: [_req(1, 0, first=True)], 64: [_req(1, 8192)], 129: [_req(1, 16384)]}, 130)
    assert diag.gaps_shallow == [0, 1, 1, 0]
    assert diag.interrupted == 1


def test_a_first_chunk_again_is_not_a_gap():
    """A prompt restarted from its first chunk (preempted, re-prefilled)."""
    diag = _diag()
    _run(diag, {0: [_req(1, 0, first=True)], 1: [_req(1, 8192)], 9: [_req(1, 0, first=True)]}, 10)
    assert sum(diag.gaps_shallow) == 1
    assert diag.interrupted == 0


def test_forget_except_drops_requests_that_left():
    diag = _diag()
    _run(diag, {0: [_req(1, 0, first=True), _req(2, 0, first=True)]}, 1)
    diag.forget_except([2])
    _run(diag, {0: [_req(1, 8192), _req(2, 8192)]}, 1)
    assert sum(diag.gaps_shallow) == 1


# --------------------------------------------------------------------------- lines


def test_periodic_lines_are_due_once_per_interval():
    clock = _Clock()
    diag = _diag(clock=clock, interval_s=30.0)
    assert not diag.due()
    clock.now += 29.0
    assert not diag.due()
    clock.now += 1.0
    assert diag.due()
    assert not diag.due()
    # A long stall owes one line, not one per missed interval, and the next
    # line stays on the interval grid (start + k * interval).
    clock.now += 95.0
    assert diag.due()
    assert not diag.due()
    clock.now += 5.0
    assert not diag.due()
    clock.now += 20.0
    assert diag.due()


def test_line_carries_every_count_as_key_value_tokens():
    clock = _Clock()
    diag = _diag(clock=clock, deep_tokens=16384)
    diag.agreement_round(25144, {1: 25144, 2: 25144, 3: 128})
    diag.agreement_admitted(128, 25144, {1: 25144, 2: 25144, 3: 128})
    diag.reservation_claimed(25144, 128)
    diag.context_resumed(True, True, [1, 30, 4], 0.0125)
    diag.admission_holds([7])
    _run(diag, {0: [_req(1, 0, first=True)], 3: [_req(1, 20480)]}, 4)
    clock.now += 12.0
    line = diag.line("periodic", {"waiting_first": 5, "agree_held": 2})
    assert re.fullmatch(r"\[pp_ctx_diag\]( [a-z_0-9]+=[^ =]+)+", line)
    fields = _fields(line)
    assert fields["kind"] == "periodic"
    assert fields["pp_rank"] == "0"
    assert fields["pp_size"] == "4"
    assert fields["seq"] == "1"
    assert fields["elapsed_s"] == "12.0"
    assert fields["iter"] == "4"
    assert fields["agree_round_bound"] == "s0:0,s1:0,s2:0,s3:1"
    assert fields["agree_round_lost_tokens"] == "s0:0,s1:0,s2:0,s3:25016"
    assert fields["agree_binder"] == "s0:0,s1:0,s2:0,s3:1"
    assert fields["resv_lowered_tokens"] == "25016"
    assert fields["resume_onboarded_pages_started"] == "L1:30,L2:4"
    assert fields["resume_onboarded_pages_first"] == "-"
    assert fields["resume_host_ms_started"] == "12.500"
    assert fields["admission_held_requests"] == "1"
    assert fields["gaps_deep"] == "1:0,2-64:1,65-2000:0,2001+:0"
    assert fields["interrupted_deep"] == "1"
    assert fields["q_waiting_first"] == "5"
    assert fields["q_agree_held"] == "2"
    assert _fields(diag.line("final"))["seq"] == "2"


def test_config_line():
    line = _diag(pp_rank=2, interval_s=15.0, gap_iters=64, deep_tokens=65536).config_line()
    assert _fields(line) == {
        "kind": "config",
        "pp_rank": "2",
        "pp_size": "4",
        "interval_s": "15",
        "gap_iters": "64",
        "deep_tokens": "65536",
        "gap_buckets": "1,2-64,65-2000,2001+",
    }


def test_guard_timing_off_leaves_the_lines_unchanged():
    diag = _diag()
    assert diag.guard_timing is False
    assert "guard_timing" not in diag.config_line()
    assert not any(name.startswith("guard_") for name in _fields(diag.line("periodic")))


def test_guard_calls_are_summed_per_kind_on_the_line():
    diag = _diag(guard_timing=True)
    assert _fields(diag.config_line())["guard_timing"] == "1"
    diag.guard_call("prefetch", 0.004, 1)
    diag.guard_call("claim", 0.0005, 1)
    diag.guard_call("note", 0.00025, 1)
    diag.guard_call("claim", 0.0015, 2)
    fields = _fields(diag.line("periodic"))
    assert fields["guard_calls_prefetch"] == "1" and fields["guard_host_ms_prefetch"] == "4.000"
    assert fields["guard_calls_claim"] == "2" and fields["guard_host_ms_claim"] == "2.000"
    assert fields["guard_calls_note"] == "1" and fields["guard_host_ms_note"] == "0.250"
    assert fields["guard_host_ms_max"] == "4.000"
    assert fields["guard_prefix_hashes"] == "2"
    with pytest.raises(ValueError, match="unknown guard call kind"):
        diag.guard_call("resume", 0.001, 2)


# --------------------------------------------------------------------------- the stage agreement


class _Stage:
    """A stage's reusable prefix per cap: claims return ``min(own, cap)``, pruned
    to the largest window end at or below it (a sliding-window stage)."""

    def __init__(self, ends):
        self.ends = sorted(ends)

    def claim(self, cap):
        own = self.ends[-1]
        limit = own if cap is None else min(cap, own)
        return max([0] + [end for end in self.ends if end <= limit])


def _agreement_run(diag):
    """PP4, lag 1: the first rank proposes its own 16384; stage 2 holds windows
    at 4096 and 8192 below its own 24576 and lowers the depth to 8192; at 8192 stage 3
    holds a window at 4096 only (own 20480) and lowers it to 4096, which every
    stage confirms.

    Returns what the agreement did, for comparison with and without counters.
    """
    stages = [
        _Stage([4096, 8192, 16384]),
        _Stage([4096, 8192, 16384]),
        _Stage([4096, 8192, 24576]),
        _Stage([4096, 16384, 20480]),
    ]
    agreement = A.PPStageAgreement(0, 4, lag=1, reuse_enabled=True)
    agreement.diag = diag
    req = types.SimpleNamespace(
        py_request_id=11,
        prompt_len=40000,
        is_context_init_state=True,
        is_first_context_chunk=True,
    )
    held_depth = {}

    def reserve(r, cap):
        held_depth[r.py_request_id] = stages[0].claim(cap)
        return held_depth[r.py_request_id]

    trace = []
    pending = None
    for iteration in range(8):
        if pending is not None:
            for stage in agreement.follower_stages:
                depth = pending
                claimed = stages[stage].claim(depth)
                own = stages[stage].ends[-1]
                report = {"iteration": iteration - 1, "answers": {11: (depth, claimed, own)}}
                agreement.accept(stage, report, iteration - 1)
        held = agreement.step([req], reserve, lambda r: r.py_request_id in held_depth)
        proposals = agreement.take_proposals()
        pending = (proposals or {}).get(11)
        confirmed = agreement.confirmed_depth(11)
        trace.append((sorted(held), proposals, confirmed))
        agreement.advance()
        if confirmed is not None:
            trace.append(agreement.check_admitted(11, confirmed))
            break
    trace.append(agreement.summary())
    return trace


def test_agreement_counts_the_binding_stage_of_each_round():
    diag = _diag()
    trace = _agreement_run(diag)
    assert trace[-2][0] is True
    # Round 1 at 16384: stage 2 answers 8192 (lost 8192); round 2 at 8192:
    # stage 3 answers 4096 (lost 4096); round 3 at 4096 is confirmed.
    assert diag.agree_rounds == 3
    assert diag.agree_rounds_lowered == 2
    assert diag.agree_round_bound == [0, 0, 1, 1]
    assert diag.agree_round_lost_tokens == [0, 0, 8192, 4096]
    # The first rank re-claimed 8192 and 4096 exactly: it bound nothing.
    # Own prefixes 16384 / 16384 / 24576 / 20480: admitted at 4096, the first
    # rank (and stage 1) had the smallest own prefix.
    assert diag.agree_admitted == 1
    assert diag.agree_binder == [1, 1, 0, 0]
    assert diag.agree_binder_lost_tokens == [24576 - 4096, 24576 - 4096, 0, 0]
    assert diag.agree_lost_vs_best_tokens == 24576 - 4096
    assert diag.agree_lost_vs_leader_tokens == 16384 - 4096
    assert diag.agree_held_chunk_iters == 3


def test_agreement_is_unchanged_by_the_counters_and_counts_nothing_without_them():
    with_counters = _agreement_run(_diag())
    without = _agreement_run(None)
    assert with_counters == without
    agreement = A.PPStageAgreement(0, 4, lag=1, reuse_enabled=True)
    assert agreement.diag is None
