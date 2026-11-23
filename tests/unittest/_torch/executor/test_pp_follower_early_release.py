# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PP followers apply released reservations before adopting the batch.

With the PP stage agreement and block reuse (``pp_stage_agree.py``). The first rank
lowers a reservation (a follower stage claimed less) in
``PPStageAgreement.step``, before its scheduler pass: its deep prefix is no
longer held when the pass resumes another reserved first chunk. A follower that
claimed the same deep prefix lowers it only in its answer, after it adopted
that batch. Without a cache tier below the GPU a held prefix counts as used, so
the follower adopts above the KV core's resume gate (``max_util_for_resume``)
and is refused the resume the first rank was granted. With ``host_cache_size``
0 the follower then stops (``PP rank N: cannot allocate KV cache for the
N-token context chunk ... that the first PP rank scheduled``).

Built on the stand-in ranks of ``test_pp_stage_agreement.py`` (real scheduler,
real reuse claim, real pickled schedule message, real adoption), with a pool
in which held pages are not evictable (no host tier) and the core's resume gate.
"""

import re

import pytest
import test_pp_stage_agreement as base
from test_pp_stage_agreement import _request_states  # noqa: F401  (autouse fixture)
from test_pp_stage_agreement_adversarial import _set_micro_batches

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2 as scheduler_v2_module

pytestmark = pytest.mark.cpu_only

TPB = base.TOKENS_PER_BLOCK
STALL = scheduler_v2_module.KVCacheV2Scheduler._DEADLOCK_STALL_ITERS

# Request DEEP: a 400-block committed prefix on stages 0 and 1, none on the last
# stage, so the agreement lowers it to 0 one report lag after the first claim.
# Request SHARED: a 100-block prefix on every stage, confirmed in the same step.
DEEP, SHARED = 5, 6
DEEP_BLOCKS, SHARED_BLOCKS = 400, 100
PROMPTS = {DEEP: DEEP_BLOCKS * TPB + 100, SHARED: SHARED_BLOCKS * TPB + 100}
CAPACITY = 20000
# Pages other requests use (active caches): with the deep pin (2 * 400 pages)
# the pool is above GATE, without it below.
OCCUPIED = 9500
GATE = 0.5


class _HostZeroPool(base._CapacityPool):
    """``_CapacityPool`` without a host tier, plus pages other requests use, and the resume gate.

    Every held page -- a suspended reservation's too -- counts as used: with
    ``host_cache_size`` 0 a held page cannot move down a tier, so it is neither
    free nor evictable. A resume is refused when the used fraction is above
    ``gate`` (``KvCache::resume``'s ``max_util_for_resume`` check).
    """

    def __init__(self, cores, capacity_pages, occupied, gate):
        super().__init__(cores, capacity_pages)
        self.occupied = occupied
        self.gate = gate
        self.refused_resumes = 0
        self.resume_log = []

    def used_pages(self):
        return super().used_pages() + self.occupied

    def _resume_and_restore(self, req_id, kv_cache, cuda_stream=None):
        if kv_cache.is_active:
            return True
        used = self.used_pages()
        self.resume_log.append((req_id, used))
        if used / self.capacity_pages > self.gate:
            self.refused_resumes += 1
            return False
        kv_cache.resume(None)
        return True

    def admission_state(self, req):
        return f"stand-in pool: {self.used_pages()} of {self.capacity_pages} pages used"


def _pools(blocks_per_stage, gate):
    return [
        _HostZeroPool(
            {rid: base._full_attention_core(b) for rid, b in blocks.items()},
            CAPACITY,
            OCCUPIED,
            gate,
        )
        for blocks in blocks_per_stage
    ]


# Stage 0 and follower A (stage 1) deep, follower B (the last stage) shallow.
STAGES = [
    {DEEP: DEEP_BLOCKS, SHARED: SHARED_BLOCKS},
    {DEEP: DEEP_BLOCKS, SHARED: SHARED_BLOCKS},
    {DEEP: 0, SHARED: SHARED_BLOCKS},
]


def _ranks(early, monkeypatch, gate=GATE, stages=STAGES, prompts=PROMPTS, in_admission_order=False):
    """The stand-in ranks. ``in_admission_order`` also makes the followers adopt the
    first rank's chunks in its allocation order, as the executor does wherever it
    runs the stage agreement; without it they adopt in the batch order."""
    ranks = base._ranks_with_requests(_pools(stages, gate), dict(prompts), True, monkeypatch)
    _set_micro_batches(ranks, len(stages))
    for executor in ranks:
        executor._pp_follower_early_release = early
        if in_admission_order:
            executor._pp_adopt_in_admission_order = True
            executor.scheduler.record_context_admission_order = True
    return ranks


def _record_reports(ranks):
    """Every report each follower sends, in order."""
    reports = {executor.dist.pp_rank: [] for executor in ranks[1:]}
    for executor in ranks[1:]:
        isend = executor.dist.isend_object

        def recorded(obj, dest, tag, rank=executor.dist.pp_rank, isend=isend):
            reports[rank].append(obj)
            return isend(obj, dest, tag)

        executor.dist.isend_object = recorded
    return reports


def _counting_drive(ranks, max_iterations, arrivals=None):
    iterations = [0]

    def before(ranks):
        iterations[0] += 1

    try:
        return base._drive(ranks, max_iterations, arrivals=arrivals, before=before), iterations[0]
    finally:
        ranks[0]._drive_iterations = iterations[0]


def test_control_follower_adopts_above_the_gate_one_lag_after_the_first_claim(monkeypatch):
    """Without the early release: the first rank lowers DEEP and resumes SHARED in the same
    iteration, one report lag after it claimed both; follower A still holds
    DEEP's prefix, is above the gate and cannot adopt SHARED. Follower B never
    held it."""
    ranks = _ranks(False, monkeypatch)
    lag = ranks[0]._pp_stage_agree.lag
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"PP rank 1: cannot allocate KV cache for the 100-token context chunk of "
            f"request {SHARED} that the first PP rank scheduled"
        ),
    ):
        _counting_drive(ranks, STALL + 64)
    # Claimed at iteration 0; lowered and admitted at iteration ``lag``.
    assert ranks[0]._drive_iterations == lag + 1
    leader_pool, follower_a, follower_b = (r.kv_cache_manager for r in ranks)
    assert leader_pool.refused_resumes == 0
    assert leader_pool.resume_log == [(SHARED, OCCUPIED + 2 * SHARED_BLOCKS)]
    assert follower_a.refused_resumes == 1
    assert follower_a.resume_log == [(SHARED, OCCUPIED + 2 * (SHARED_BLOCKS + DEEP_BLOCKS))]
    assert follower_b.resume_log == []  # the error came from follower A first


def test_failure_detail_names_the_reservation_held_deeper(monkeypatch):
    """The follower's error lists the reservation it still holds deeper than the
    message proposes (printed with the early release, or with admission order)."""
    ranks = _ranks(False, monkeypatch)
    for executor in ranks:
        executor._pp_adopt_in_admission_order = True
    with pytest.raises(RuntimeError) as error:
        _counting_drive(ranks, STALL + 64)
    message = str(error.value)
    assert "stand-in pool: 10500 of 20000 pages used" in message
    assert (
        "[reservations held deeper than this message's proposal: "
        f"{{{DEEP}: ({DEEP_BLOCKS * TPB}, 0)}}; early-released this iteration: {{}}]"
    ) in message


def test_early_release_adopts_where_the_first_rank_resumed(monkeypatch):
    """With the early release: follower A lowers DEEP before it adopts the batch, resumes SHARED
    at the first rank's utilization, and both prompts run on every stage, each
    first chunk from the stage's own reservation (``_drive``'s checks)."""
    lines = []
    info = py_executor_module.logger.info
    monkeypatch.setattr(
        py_executor_module.logger,
        "info",
        lambda msg, *a, **k: (lines.append(str(msg)), info(msg, *a, **k)),
    )
    ranks = _ranks(True, monkeypatch)
    first, _ = _counting_drive(ranks, STALL + 64)
    assert first == {SHARED: SHARED_BLOCKS * TPB, DEEP: 0}
    assert all(r.kv_cache_manager.refused_resumes == 0 for r in ranks)
    logs = [r.kv_cache_manager.resume_log for r in ranks]
    assert logs[0][0] == (SHARED, OCCUPIED + 2 * SHARED_BLOCKS)
    assert all(log[0] == logs[0][0] for log in logs[1:]), logs
    follower_a, follower_b = (r._pp_stage_agree for r in ranks[1:])
    assert (
        follower_a.num_early_lowerings == 1 and follower_a.max_early_lowered_by == DEEP_BLOCKS * TPB
    )
    assert follower_b.num_early_lowerings == 0 and follower_b.num_early_releases == 0
    released = [line for line in lines if "before adopting the first rank's batch" in line]
    assert len(released) == 1
    assert (
        f"rank 1 released/lowered reservation(s) {{{DEEP}: '{DEEP_BLOCKS * TPB}->0'}}"
        in (released[0])
    )
    assert f"pages freed per cache level and pool group [[{2 * DEEP_BLOCKS}]]" in released[0]
    assert "GPU util per pool group now [0.485]" in released[0]


def test_early_release_sends_the_reports_the_answer_alone_sends(monkeypatch):
    """Without the gate both settings run; every follower report is the same."""
    results = []
    for early in (False, True):
        ranks = _ranks(early, monkeypatch, gate=1.0)
        reports = _record_reports(ranks)
        first, iterations = _counting_drive(ranks, STALL + 64)
        results.append((first, iterations, reports))
    assert results[0] == results[1]


# A stream of requests, every other one with a prefix only stages 0 and 1 hold.
_STREAM = {100 + i: (DEEP_BLOCKS if i % 2 else SHARED_BLOCKS) for i in range(24)}
_STREAM_PROMPTS = {rid: blocks * TPB + 100 + 37 * (rid % 5) for rid, blocks in _STREAM.items()}
_STREAM_STAGES = [
    dict(_STREAM),
    dict(_STREAM),
    {rid: (0 if blocks == DEEP_BLOCKS else blocks) for rid, blocks in _STREAM.items()},
]


def _stream_arrivals():
    arrivals = {}
    for i, rid in enumerate(_STREAM):
        arrivals.setdefault(1 + 2 * (i // 2), {})[rid] = _STREAM_PROMPTS[rid]
    return arrivals


@pytest.mark.parametrize("gate", [1.0, GATE])
def test_a_stream_of_divergent_prefixes_runs_with_identical_reports(gate, monkeypatch):
    """Pairs of requests arrive every other iteration, one of each pair lowered on
    the deep stages. The followers adopt in the first rank's allocation order, as
    the executor's do with the stage agreement; under shortest prefill first that
    order differs from the batch order. With the early release every prompt runs
    from each stage's reservation, no resume is refused, and the follower reports
    equal those without the early release run without the gate."""
    baseline = None
    for early in (False, True):
        if not early and gate != 1.0:
            continue
        ranks = _ranks(
            early,
            monkeypatch,
            gate=gate,
            stages=_STREAM_STAGES,
            prompts={},
            in_admission_order=True,
        )
        reports = _record_reports(ranks)
        first, iterations = _counting_drive(ranks, STALL + 2000, arrivals=_stream_arrivals())
        assert set(first) == set(_STREAM)
        # The first rank's own gate may refuse a resume (the request is agreed
        # again); a follower refused one would have raised in the adoption.
        print(
            f"gate {gate} early {early}: {iterations} iterations, refused resumes "
            f"{[r.kv_cache_manager.refused_resumes for r in ranks]}; "
            f"{ranks[1]._pp_stage_agree.follower_summary()}"
        )
        assert all(r.kv_cache_manager.refused_resumes == 0 for r in ranks[1:])
        assert all(not r.kv_cache_manager.kv_cache_map for r in ranks)
        if not early:
            baseline = (first, iterations, reports)
            continue
        assert ranks[1]._pp_stage_agree.num_early_lowerings > 0
        assert ranks[2]._pp_stage_agree.num_early_lowerings == 0
        if baseline is not None:
            assert (first, iterations, reports) == baseline
