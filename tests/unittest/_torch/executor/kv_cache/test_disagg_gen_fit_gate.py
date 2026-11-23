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
"""Tests for the opt-in disagg generation-init fit gate (TLLM_KV_DISAGG_GEN_FIT_GATE).

The gate tests run on CPU against a fake two-group page pool laid out like the served
generation pool (one sliding-window pool group, one full-attention pool group). The
manager-glue tests need the KV cache manager module (triton), and the end-to-end test needs
the C++ KV cache manager on a GPU. The scheduler-side FIFO wiring is covered in
``test_kv_cache_v2_scheduler.py`` (``TestDisaggGenFitGate``).
"""

import math
import random
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import disagg_gen_fit_gate as fit_gate_module
from tensorrt_llm._torch.pyexecutor.kv_cache.disagg_gen_fit_gate import (
    FIT_GATE_ENV,
    MAX_U_HI,
    MIN_RESERVE,
    MIN_RESERVE_ENV,
    MIN_U_HI,
    REPORT_UTIL,
    RESERVE_FLOOR,
    DisaggGenFitGate,
    parse_fit_gate,
    reserve_of,
)

TPB = 32
WINDOW = 2048
# Served generation pool geometry at pool_ratio [0.042, 0.958]: pool group 0 holds the
# sliding-window life cycle, pool group 1 the full-attention one.
SWA_PAGES = 7_316
FULL_PAGES = 15_222_000 // TPB
DRAFT_AND_EXTRA = 16


def full_need(prompt_len: int) -> int:
    return -(-(prompt_len + DRAFT_AND_EXTRA) // TPB)


def swa_need(prompt_len: int) -> int:
    # Blocks of the live window at the prompt end plus the draft tail.
    first = max(0, prompt_len - WINDOW) // TPB
    return -(-(prompt_len + DRAFT_AND_EXTRA) // TPB) - first


class FakeGenPool:
    """Pages per pool group, admitting and releasing whole generation-init requests."""

    def __init__(self, gate: Optional[DisaggGenFitGate], total=(SWA_PAGES, FULL_PAGES)) -> None:
        self.gate = gate
        self.total = list(total)
        self.used = [0] * len(total)
        self.residents = {}

    def need(self, prompt_len: int) -> List[int]:
        return [swa_need(prompt_len), full_need(prompt_len)]

    def util(self) -> List[float]:
        return [u / t for u, t in zip(self.used, self.total)]

    def prepare(self, rid: int, prompt_len: int) -> bool:
        """What prepare_disagg_gen_init does: the fit check, then the allocation."""
        need = self.need(prompt_len)
        if self.gate is not None and not self.gate.check(
            self.used, need, self.total, request_id=rid
        ):
            return False
        if any(u + n > t for u, n, t in zip(self.used, need, self.total)):
            return False
        self.used = [u + n for u, n in zip(self.used, need)]
        self.residents[rid] = need
        if self.gate is not None:
            self.gate.note_admitted(rid)
        return True

    def release(self, rid: int) -> None:
        need = self.residents.pop(rid)
        self.used = [u - n for u, n in zip(self.used, need)]


def schedule_pass(pool: FakeGenPool, waiting: List[tuple]) -> List[int]:
    """Phase 1 of KVCacheV2Scheduler._schedule_loop for generation-init requests only."""
    gate = pool.gate
    admitted = []
    if gate is not None:
        gate.begin_pass()
    for rid, prompt_len in waiting:
        if gate is not None and gate.pass_blocked:
            gate.note_fifo_blocked()
            continue
        if not pool.prepare(rid, prompt_len):
            if gate is not None:
                gate.note_refused(rid)
            continue
        admitted.append(rid)
    if gate is not None:
        gate.end_pass()
    return admitted


# --------------------------------------------------------------------------- knob


@pytest.mark.parametrize(
    "raw, expected",
    [(None, None), ("", None), ("  ", None), ("0", None), ("0.0", None), ("0.99", 0.99)],
)
def test_parse_off_and_valid(raw, expected):
    env = {} if raw is None else {FIT_GATE_ENV: raw}
    assert parse_fit_gate(env) == expected


@pytest.mark.parametrize("raw", [" 0.985 ", "0.5", "0.95", str(MAX_U_HI)])
def test_parse_accepts_range(raw):
    assert parse_fit_gate({FIT_GATE_ENV: raw}) == float(raw)


@pytest.mark.parametrize(
    "raw", ["abc", "-0.5", "0.1", "0.49", "0.995", "1", "1.5", "nan", "inf", "on"]
)
def test_parse_rejects(raw):
    with pytest.raises(ValueError, match=FIT_GATE_ENV):
        parse_fit_gate({FIT_GATE_ENV: raw})


def test_parse_reads_process_env(monkeypatch):
    monkeypatch.delenv(FIT_GATE_ENV, raising=False)
    assert parse_fit_gate() is None
    monkeypatch.setenv(FIT_GATE_ENV, "0.985")
    assert parse_fit_gate() == 0.985


@pytest.mark.parametrize("u_hi", [0.0, -0.1, 0.3, 0.995, 1.0])
def test_gate_rejects_out_of_range(u_hi):
    with pytest.raises(ValueError):
        DisaggGenFitGate(u_hi)


# --------------------------------------------------------------------------- refusal math


def test_refusal_math_at_served_geometry():
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    total = [SWA_PAGES, FULL_PAGES]
    need = [swa_need(150_000), full_need(150_000)]
    assert need[0] == 65 and need[1] == 4_688
    limit_full = math.floor(0.99 * FULL_PAGES)

    # The default 0.95 gate would refuse this pool (util about 0.98 before the request); the fit
    # gate admits as long as the request itself fits under U_hi.
    used = [3_000, limit_full - need[1]]
    decision = gate.decide(used, need, total)
    assert decision.admit and decision.pool_group == 1
    assert decision.util_after == pytest.approx(limit_full / FULL_PAGES)

    # One page more than U_hi allows in the full-attention group.
    used = [3_000, limit_full - need[1] + 1]
    decision = gate.decide(used, need, total)
    assert not decision.admit and decision.pool_group == 1
    assert decision.util_after > 0.99

    # The sliding-window group alone can refuse.
    limit_swa = math.floor(0.99 * SWA_PAGES)
    decision = gate.decide([limit_swa - need[0] + 1, 0], need, total)
    assert not decision.admit and decision.pool_group == 0

    # Below the old 0.95 resume threshold before the request, but a giant does not fit.
    giant = [swa_need(700_000), full_need(700_000)]
    used = [3_000, int(0.95 * FULL_PAGES)]
    assert used[1] / FULL_PAGES <= 0.95
    assert not gate.decide(used, giant, total).admit


def test_check_counts_and_tracks_max_util():
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    total = [100, 1_000]
    assert gate.check([10, 900], [1, 50], total, request_id=1)  # 0.95 after
    gate.note_admitted(1)
    assert gate.check([10, 900], [1, 80], total, request_id=2)  # 0.98 after
    gate.note_admitted(2)
    assert not gate.check([10, 950], [1, 41], total, request_id=3)  # 0.991 after
    assert gate.stats["fit_passed"] == 2
    assert gate.stats["fit_admitted"] == 2
    assert gate.stats["fit_refused"] == 1
    assert gate.stats["admitted_above_report_util"] == 1
    assert gate.max_util_after_admit == pytest.approx(0.98)
    assert REPORT_UTIL == 0.95


def test_admission_counters_count_only_allocated_requests():
    """A request the fit check passes but a later step refuses (index slots, the resume
    gate, the resize) is not an admission."""
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    total = [100, 1_000]
    gate.begin_pass()
    assert gate.check([10, 900], [1, 85], total, request_id=1)  # 0.985 after
    gate.note_refused(1)
    assert gate.pass_blocked
    gate.end_pass()
    assert gate.stats["fit_passed"] == 1
    assert gate.stats["refused_after_fit"] == 1
    assert gate.stats["fit_admitted"] == 0
    assert gate.stats["admitted_above_report_util"] == 0
    assert gate.max_util_after_admit == 0.0
    # An admission report for another request than the one checked counts nothing.
    gate.check([10, 900], [1, 85], total, request_id=2)
    gate.note_admitted(3)
    gate.note_admitted(2)
    assert gate.stats["fit_admitted"] == 0


def test_decide_requires_matching_group_counts():
    gate = DisaggGenFitGate(0.99)
    with pytest.raises(ValueError, match="pool group counts"):
        gate.decide([0, 0], [1], [10, 10])


def test_oversize_request_is_left_to_resume_gate():
    """A request needing more than U_hi of a group alone can never pass the fit check;
    the gate lets it through to the resume gate and the allocation (the default behaviour) so a
    FIFO-blocked pass cannot wait on it forever."""
    gate = DisaggGenFitGate(0.9, log_every_passes=0)
    decision = gate.decide([0, 0], [1, 95], [100, 100])
    assert decision.admit and decision.oversize and not decision.never_fits
    assert decision.nothing_held and decision.never_admissible
    assert gate.check([50, 50], [1, 95], [100, 100], request_id=4)
    assert gate.stats["oversize_passed"] == 1
    assert gate.stats["fit_admitted"] == 0
    gate.note_admitted(4)
    assert gate.stats["fit_admitted"] == 1 and gate.stats["oversize_admitted"] == 1
    # Oversize admissions do not move the utilization counters (they are the default path).
    assert gate.max_util_after_admit == 0.0


def test_never_fitting_request_does_not_block_the_pass(monkeypatch):
    """A request needing more pages than a pool group has can never be allocated: FIFO must
    not hold every later generation-init behind it, pass after pass."""
    warnings = []
    monkeypatch.setattr(fit_gate_module.logger, "warning", warnings.append)
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    pool = FakeGenPool(gate, total=(1_000, 1_000))
    pool.used = [0, 300]
    giant = (0, 32 * 2_000)  # 2_001 full-attention pages: more than the group has
    assert gate.decide(pool.used, pool.need(giant[1]), pool.total).never_fits
    for _ in range(50):
        admitted = schedule_pass(pool, [giant, (1, 32 * 10), (2, 32 * 10)])
        if admitted:
            assert admitted == [1, 2]
            pool.release(1)
            pool.release(2)
    assert gate.stats["never_admissible"] == 50
    assert gate.stats["fifo_blocked"] == 0
    # Logged once for the request, not once per pass.
    assert len(warnings) == 1 and "can never be admitted" in warnings[0]


def test_oversize_request_refused_on_an_empty_pool_does_not_block():
    gate = DisaggGenFitGate(0.9, log_every_passes=0)
    gate.begin_pass()
    # Alone more than U_hi of group 1, nothing held, and still refused by the allocation.
    assert gate.check([0, 0], [1, 95], [100, 100], request_id=9)
    gate.note_refused(9)
    assert not gate.pass_blocked
    gate.end_pass()
    assert gate.stats["never_admissible"] == 1


def test_oversize_request_refused_while_others_hold_pages_keeps_its_turn():
    """While residents hold pages, an oversize head waits for them to finish, and the
    requests behind it wait too."""
    gate = DisaggGenFitGate(0.9, log_every_passes=0)
    gate.begin_pass()
    assert gate.check([0, 10], [1, 95], [100, 100], request_id=9)
    gate.note_refused(9)
    assert gate.pass_blocked
    gate.note_fifo_blocked()
    gate.end_pass()
    assert gate.stats["never_admissible"] == 0
    assert gate.stats["refused_after_fit"] == 1
    assert gate.stats["fifo_blocked"] == 1


# --------------------------------------------------------------------------- reserve


def test_reserve_bound():
    assert MIN_RESERVE == pytest.approx(0.01)
    assert MAX_U_HI == pytest.approx(0.99)
    assert MIN_U_HI == pytest.approx(0.5)
    with pytest.raises(ValueError):
        parse_fit_gate({FIT_GATE_ENV: "0.991"})


def test_reserve_override_is_opt_in():
    assert reserve_of({}) == MIN_RESERVE
    assert reserve_of({MIN_RESERVE_ENV: " "}) == MIN_RESERVE
    assert reserve_of({MIN_RESERVE_ENV: "0.004"}) == pytest.approx(0.004)
    assert parse_fit_gate({FIT_GATE_ENV: "0.996", MIN_RESERVE_ENV: "0.004"}) == pytest.approx(0.996)
    # The override raises the bound only as far as the reserve it sets.
    with pytest.raises(ValueError, match="0.4%"):
        parse_fit_gate({FIT_GATE_ENV: "0.997", MIN_RESERVE_ENV: "0.004"})
    # Without the override the bound stays 1 - MIN_RESERVE.
    with pytest.raises(ValueError):
        parse_fit_gate({FIT_GATE_ENV: "0.996"})
    # The override never leaves the gate off when FIT_GATE_ENV is unset.
    assert parse_fit_gate({MIN_RESERVE_ENV: "0.004"}) is None


@pytest.mark.parametrize("raw", ["0.001", "0.011", "nan", "x", "-0.004"])
def test_reserve_override_refusals(raw):
    with pytest.raises(ValueError, match=MIN_RESERVE_ENV):
        reserve_of({MIN_RESERVE_ENV: raw})
    with pytest.raises(ValueError, match=MIN_RESERVE_ENV):
        parse_fit_gate({FIT_GATE_ENV: "0.99", MIN_RESERVE_ENV: raw})


def test_gate_reserve_argument():
    assert DisaggGenFitGate(0.996, log_every_passes=0, reserve=0.004).u_hi == 0.996
    with pytest.raises(ValueError):
        DisaggGenFitGate(0.996, log_every_passes=0)
    with pytest.raises(ValueError):
        DisaggGenFitGate(0.99, log_every_passes=0, reserve=RESERVE_FLOOR / 2)


@pytest.mark.parametrize("reserve", [MIN_RESERVE, 0.004, RESERVE_FLOOR])
def test_admissions_never_pass_u_hi_and_keep_the_reserve(reserve):
    rng = random.Random(7)
    u_hi = 1.0 - reserve
    gate = DisaggGenFitGate(u_hi, log_every_passes=0, reserve=reserve)
    pool = FakeGenPool(gate)
    waiting, rid = [], 0
    for _ in range(3_000):
        while len(waiting) < 16:
            waiting.append((rid, rng.choice([2_000, 30_000, 150_000, 400_000, 700_000])))
            rid += 1
        admitted = set(schedule_pass(pool, waiting))
        waiting = [w for w in waiting if w[0] not in admitted]
        for used, total in zip(pool.used, pool.total):
            assert used <= u_hi * total
            assert total - used >= reserve * total - 1
        # The CUDA-graph padding extension takes at most one page per resident (96) and life
        # cycle; the reserve of the full-attention group covers it several times over, also at
        # the smallest reserve the override accepts.
        assert pool.total[1] - pool.used[1] > 5 * 96
        if pool.residents and rng.random() < 0.6:
            pool.release(rng.choice(sorted(pool.residents)))
    assert gate.stats["fit_refused"] > 0
    assert gate.max_util_after_admit <= u_hi


# --------------------------------------------------------------------------- FIFO


def test_fifo_stops_within_a_pass_and_resets_next_pass():
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    pool = FakeGenPool(gate, total=(1_000, 1_000))
    pool.used = [0, 900]
    # Request 0 fits; request 1 (a long prompt) does not; request 2 would, but must not
    # overtake it.
    waiting = [(0, 32 * 40 - DRAFT_AND_EXTRA), (1, 32 * 200), (2, 32 * 10)]
    assert schedule_pass(pool, waiting) == [0]
    assert gate.stats["fit_refused"] == 1
    assert gate.stats["fifo_blocked"] == 1
    assert gate.stats["fifo_blocked_passes"] == 1
    assert not gate.pass_blocked  # end_pass leaves the gate unblocked

    # Next pass, the residents have left: the giant goes first, then the short one.
    pool.release(0)
    pool.used = [0, 500]
    assert schedule_pass(pool, [(1, 32 * 200), (2, 32 * 10)]) == [1, 2]
    assert gate.stats["passes"] == 2
    assert gate.stats["fifo_blocked_passes"] == 1


def test_any_refusal_blocks_the_rest_of_the_pass():
    """A refusal for a reason other than the fit check (index slots, resume gate, resize)
    also keeps the later generation-inits waiting."""
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    gate.begin_pass()
    assert not gate.pass_blocked
    gate.note_refused(5)
    assert gate.pass_blocked
    gate.note_fifo_blocked()
    gate.note_fifo_blocked()
    gate.end_pass()
    gate.begin_pass()
    assert not gate.pass_blocked
    gate.end_pass()
    assert gate.stats["fifo_blocked"] == 2
    assert gate.stats["fifo_blocked_passes"] == 1
    assert gate.stats["passes"] == 2


def test_fifo_serves_in_arrival_order():
    """Without FIFO a size-aware gate lets short prompts overtake refused long ones."""
    rng = random.Random(11)
    arrivals = [rng.choice([4_000, 600_000]) for _ in range(2_000)]

    def run(fifo: bool) -> List[int]:
        gate = DisaggGenFitGate(0.99, log_every_passes=0)
        pool = FakeGenPool(gate, total=(SWA_PAGES, 200_000))
        pending, served, rid = [], [], 0
        for step in range(1_500):
            while len(pending) < 24 and rid < len(arrivals):
                pending.append((rid, arrivals[rid]))
                rid += 1
            if fifo:
                admitted = schedule_pass(pool, pending)
            else:
                admitted = [r for r, p in pending if pool.prepare(r, p)]
            served += admitted
            pending = [w for w in pending if w[0] not in admitted]
            if pool.residents and step % 2:
                pool.release(min(pool.residents))
        return served

    fifo_order = run(fifo=True)
    assert len(fifo_order) > 500
    assert fifo_order == list(range(len(fifo_order)))
    skip_order = run(fifo=False)
    overtakes = sum(1 for a, b in zip(skip_order, skip_order[1:]) if b < a)
    assert overtakes > 0
    overtaken = [i for i in range(len(skip_order)) if i not in set(skip_order)]
    assert any(arrivals[i] == 600_000 for i in overtaken)


# --------------------------------------------------------------------------- counters / log


def test_periodic_log_line(monkeypatch):
    lines = []
    monkeypatch.setattr(fit_gate_module.logger, "info", lines.append)
    gate = DisaggGenFitGate(0.985, name=" rank=3", log_every_passes=2)
    free = iter([[9, 40], [8, 60], [7, 900], [8, 950]])
    gate.free_probe = lambda: next(free)
    gate.begin_pass()
    gate.check([0, 990], [0, 1], [10, 1_000], request_id=1)
    gate.end_pass()
    assert lines == []
    gate.begin_pass()
    gate.note_refused(2)
    gate.note_fifo_blocked()
    gate.end_pass()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[disagg-gen-fit-gate] rank=3 U_hi=0.9850 passes=2")
    for field in (
        "fit_passed=0",
        "fit_refused=1",
        "fit_admitted=0",
        "fit_admitted_above_0.95=0",
        "max_util_after_admit=0.0000",
        "fifo_blocked=1",
        "fifo_blocked_passes=1",
        "resident_blocked=0",
        "resident_suspends=0",
        "never_admissible=0",
        "window_min_free_pages=[8, 40]",
    ):
        assert field in line
    # The minimum restarts with every log window.
    gate.begin_pass()
    gate.end_pass()
    gate.begin_pass()
    gate.end_pass()
    assert "window_min_free_pages=[7, 900]" in lines[1]


def test_residents_without_cache_block_the_pass():
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    gate.begin_pass(residents_without_cache=2)
    assert gate.pass_blocked
    gate.note_fifo_blocked()
    gate.note_fifo_blocked()
    gate.end_pass()
    gate.begin_pass(residents_without_cache=0)
    assert not gate.pass_blocked
    gate.end_pass()
    assert gate.stats["resident_blocked"] == 2
    assert gate.stats["resident_blocked_passes"] == 1
    assert gate.stats["fifo_blocked"] == 0
    assert gate.stats["fifo_blocked_passes"] == 0


def test_resident_suspend_and_resume_counters(monkeypatch):
    warnings = []
    monkeypatch.setattr(fit_gate_module.logger, "warning", warnings.append)
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    gate.note_resident_suspended()
    gate.note_resident_suspended()
    gate.note_resident_resumed(False)
    gate.note_resident_resumed(True)
    assert gate.stats["resident_suspends"] == 2
    assert gate.stats["resident_resume_refused"] == 1
    assert gate.stats["resident_resumes"] == 1
    assert len(warnings) == 1 and "suspended" in warnings[0]


# --------------------------------------------------------------------------- manager glue


def _manager_module():
    pytest.importorskip("triton")
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2

    return kv_cache_manager_v2


def _fake_target(m, totals=(100, 1_000), pp_size=1, world_size=1, enable_attention_dp=False):
    fake = object.__new__(m.KVCacheManagerV2)
    fake.is_draft = False
    fake.tokens_per_block = TPB
    fake.mapping = SimpleNamespace(
        pp_size=pp_size, world_size=world_size, enable_attention_dp=enable_attention_dp
    )
    fake._get_storage_statistics = Mock(
        return_value=[SimpleNamespace(unavailable=0, total=t) for t in totals]
    )
    return fake


def test_manager_default_off_builds_nothing(monkeypatch):
    m = _manager_module()
    fake = _fake_target(m)
    monkeypatch.delenv(FIT_GATE_ENV, raising=False)
    assert m.KVCacheManagerV2._make_disagg_gen_fit_gate(fake, 0.95) is None
    monkeypatch.setenv(FIT_GATE_ENV, "0")
    assert m.KVCacheManagerV2._make_disagg_gen_fit_gate(fake, 0.95) is None
    # Instances built without __init__ see the class-level default.
    assert object.__new__(m.KVCacheManagerV2).disagg_gen_fit_gate is None


def test_manager_builds_gate_only_for_target(monkeypatch):
    m = _manager_module()
    monkeypatch.setenv(FIT_GATE_ENV, "0.99")
    target = _fake_target(m, totals=(100, 1_000))
    gate = m.KVCacheManagerV2._make_disagg_gen_fit_gate(target, 0.99)
    assert isinstance(gate, DisaggGenFitGate) and gate.u_hi == 0.99
    # The gate's free-page margin reads the same pool statistics.
    target._get_storage_statistics.return_value = [
        SimpleNamespace(unavailable=40, total=100),
        SimpleNamespace(unavailable=990, total=1_000),
    ]
    assert gate.free_probe() == [60, 10]
    draft = _fake_target(m)
    draft.is_draft = True
    assert m.KVCacheManagerV2._make_disagg_gen_fit_gate(draft, 0.99) is None


def test_manager_ignores_gate_with_pipeline_parallelism(monkeypatch):
    m = _manager_module()
    monkeypatch.setenv(FIT_GATE_ENV, "0.99")
    fake = _fake_target(m, pp_size=4, world_size=4)
    assert m.KVCacheManagerV2._make_disagg_gen_fit_gate(fake, 0.99) is None
    fake._get_storage_statistics.assert_not_called()


@pytest.mark.parametrize("same", [True, False])
def test_manager_agrees_pool_totals_across_ranks(monkeypatch, same):
    m = _manager_module()
    monkeypatch.setenv(FIT_GATE_ENV, "0.99")
    fake = _fake_target(m, totals=(100, 1_000), world_size=4)
    others = [[100, 1_000]] * 3 + [[100, 1_000] if same else [100, 999]]
    dist = Mock()
    dist.allgather = Mock(return_value=others)
    monkeypatch.setattr(m.Distributed, "get", Mock(return_value=dist))
    if same:
        assert m.KVCacheManagerV2._make_disagg_gen_fit_gate(fake, 0.99) is not None
    else:
        with pytest.raises(ValueError, match="same GPU pages"):
            m.KVCacheManagerV2._make_disagg_gen_fit_gate(fake, 0.99)
    dist.allgather.assert_called_once_with([100, 1_000])


def _glue_manager(m, gate, used, need, total):
    manager = object.__new__(m.KVCacheManagerV2)
    manager._disagg_gen_fit_gate = gate
    manager.kv_cache_map = {}
    manager._get_storage_statistics = Mock(
        return_value=[SimpleNamespace(unavailable=u, total=t) for u, t in zip(used, total)]
    )
    manager._disagg_gen_init_page_demand = Mock(return_value=list(need))
    # Stops prepare_disagg_gen_init right after the gate.
    manager.prepare_context_cache = Mock(return_value=None)
    return manager


@pytest.mark.parametrize("gate_on", [False, True])
def test_prepare_disagg_gen_init_refuses_before_creating_a_cache(gate_on):
    m = _manager_module()
    gate = DisaggGenFitGate(0.99, log_every_passes=0) if gate_on else None
    manager = _glue_manager(m, gate, used=[0, 990], need=[1, 5], total=[100, 1_000])
    req = SimpleNamespace(py_request_id=7, is_dummy=False)
    assert manager.prepare_disagg_gen_init(req) is False
    if gate_on:
        manager.prepare_context_cache.assert_not_called()
        assert gate.stats["fit_refused"] == 1
    else:
        # Off: the default path, no pool statistics read.
        manager.prepare_context_cache.assert_called_once_with(req)
        manager._get_storage_statistics.assert_not_called()
        manager._disagg_gen_init_page_demand.assert_not_called()


def test_prepare_disagg_gen_init_admits_through_to_the_cache():
    m = _manager_module()
    gate = DisaggGenFitGate(0.99, log_every_passes=0)
    manager = _glue_manager(m, gate, used=[0, 900], need=[1, 5], total=[100, 1_000])
    req = SimpleNamespace(py_request_id=7, is_dummy=False)
    manager.prepare_disagg_gen_init(req)
    manager.prepare_context_cache.assert_called_once_with(req)
    assert gate.stats["fit_passed"] == 1
    # prepare_context_cache refused (the stub returns None): not an admission.
    assert gate.stats["fit_admitted"] == 0


def test_gen_init_target_matches_allocation():
    m = _manager_module()
    manager = object.__new__(m.KVCacheManagerV2)
    manager._has_cp_helix = False
    manager.num_extra_kv_tokens = 3
    manager._shared_draft_block_width = 0
    req = Mock(prompt_len=1_000, total_input_len_cp=4_000, py_draft_tokens=[0] * 7)
    assert manager._disagg_gen_init_target(req) == (1_000 + 7 + 3, 1_000)
    manager._has_cp_helix = True
    assert manager._disagg_gen_init_target(req) == (4_000 + 7 + 3, 4_000)


# --------------------------------------------------------------------------- C++ end to end


def test_cpp_demand_matches_gen_init_resize():
    """The demand the gate checks is exactly what a generation-init resize allocates."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU and the C++ KV cache manager")
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        GPU_LEVEL,
        AttentionLayerConfig,
        BatchDesc,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        KVCacheDesc,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        _introspection,
    )

    tpb, window, extra = 16, 64, 4
    layers = [
        AttentionLayerConfig(
            layer_id=LayerId(i),
            buffers=[
                BufferConfig(role=DataRole("key"), size=8192),
                BufferConfig(role=DataRole("value"), size=8192),
            ],
            sliding_window_size=window if i % 2 == 0 else None,
        )
        for i in range(4)
    ]
    manager = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=tpb, cache_tiers=[GpuCacheTierConfig(quota=64 << 20)], layers=layers
        )
    )
    gate = DisaggGenFitGate(0.9, log_every_passes=0)
    stream = torch.cuda.Stream()
    caches = []

    def pool():
        stats = manager.get_storage_statistics(GPU_LEVEL)
        return [int(s.unavailable) for s in stats], [int(s.total) for s in stats]

    try:
        for prompt_len in [1, 15, 16, 17, 63, 64, 65, 200, 1_000, 3_333]:
            target = prompt_len + extra
            need = list(
                _introspection.compute_slots_for_batch(
                    manager,
                    BatchDesc(
                        [
                            KVCacheDesc(
                                capacity=target,
                                history_length=prompt_len,
                                prompt_length=prompt_len,
                            )
                        ]
                    ),
                    tpb,
                )
            )
            used, total = pool()
            admit = gate.check(used, need, total)
            kv_cache = manager.create_kv_cache(None, None, expected_prompt_length=prompt_len)
            kv_cache.enable_swa_scratch_reuse = False
            assert kv_cache.resume(stream.cuda_stream)
            assert kv_cache.resize(target, prompt_len)
            after, _ = pool()
            assert [a - u for a, u in zip(after, used)] == need, prompt_len
            if admit:
                assert max(a / t for a, t in zip(after, total)) <= gate.u_hi
            caches.append(kv_cache)
        stream.synchronize()
    finally:
        for kv_cache in caches:
            kv_cache.close()
        manager.shutdown()
