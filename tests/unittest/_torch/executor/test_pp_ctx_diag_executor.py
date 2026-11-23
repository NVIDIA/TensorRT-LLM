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
"""The per-stage context-path counters (``pp_ctx_diag.py``) wired into the executor.

Built on the stand-in ranks of ``test_pp_stage_agreement.py`` (real scheduler,
real reuse claim and resume, real pickled schedule message, real adoption):
the stage agreement on a PP3 pipeline of sliding-window stages lowers the depth
twice before every stage confirms it, and the counters of every rank account
for the binding stages, the reservations each rank lowered, the adoptions and
the resumes. Off a disaggregated server nothing is built or attached.
"""

import types
from unittest.mock import MagicMock

import pytest
import test_pp_stage_agreement as base
from test_pp_stage_agreement import _request_states  # noqa: F401  (autouse fixture)

from tensorrt_llm._torch.pyexecutor.pp_ctx_diag import PPCtxDiag
from tensorrt_llm._torch.pyexecutor.scheduler import scheduler_v2

pytestmark = pytest.mark.cpu_only


def _ranks(monkeypatch, diag: bool):
    stages, common, _ = base.SWA_CASES["pp3_two_lowerings"]
    pools = base._swa_pools(stages)
    ranks = base._make_ranks(pools, base.PROMPT_LEN, stage_agree=True, monkeypatch=monkeypatch)
    for executor in ranks:
        # The counters run on a disaggregated PP server (one with a KV cache
        # transceiver); the stand-in ranks are one.
        executor._pp_disaggregated = diag
        executor._pp_ctx_diag = executor._create_pp_ctx_diag()
    return ranks, pools, common


def test_aggregated_server_builds_and_attaches_nothing(monkeypatch):
    ranks, pools, common = _ranks(monkeypatch, diag=False)
    for executor, pool in zip(ranks, pools):
        assert executor._pp_ctx_diag is None
        assert executor._pp_stage_agree.diag is None
        assert executor.scheduler.ctx_diag is None
        assert pool.ctx_diag is None
    wire, _ = base._run_until_admitted(ranks)
    assert wire.context_chunks[base.REQUEST_ID] == (common, base.MAX_NUM_TOKENS)


def test_counters_follow_the_agreement_on_every_rank(monkeypatch):
    ranks, pools, common = _ranks(monkeypatch, diag=True)
    for executor, pool in zip(ranks, pools):
        diag = executor._pp_ctx_diag
        assert diag is not None
        assert executor._pp_stage_agree.diag is diag
        assert executor.scheduler.ctx_diag is diag
        assert pool.ctx_diag is diag

    wire, _ = base._run_until_admitted(ranks)
    assert common == 4096
    assert wire.context_chunks[base.REQUEST_ID] == (common, base.MAX_NUM_TOKENS)

    leader, stage1, stage2 = (executor._pp_ctx_diag for executor in ranks)
    # Proposal 16384: stage 1 holds a window at 8192 only (lost 8192); proposal
    # 8192: stage 2 holds one at 4096 only (lost 4096); 4096 is confirmed.
    assert leader.agree_rounds == 3
    assert leader.agree_round_bound == [0, 1, 1]
    assert leader.agree_round_lost_tokens == [0, 8192, 4096]
    # Own prefixes 16384 / 24576 / 24576: the first rank had the smallest.
    assert leader.agree_admitted == 1
    assert leader.agree_binder == [1, 0, 0]
    assert leader.agree_lost_vs_best_tokens == 24576 - common
    assert leader.agree_lost_vs_leader_tokens == 16384 - common
    # Reservations each rank lowered on the way down.
    assert (leader.resv_made, leader.resv_lowered, leader.resv_lowered_tokens) == (1, 2, 12288)
    assert (stage1.resv_made, stage1.resv_lowered, stage1.resv_lowered_tokens) == (1, 1, 4096)
    assert (stage2.resv_made, stage2.resv_lowered, stage2.resv_lowered_tokens) == (1, 1, 12288)
    for diag in (leader, stage1, stage2):
        assert diag.resv_dropped == 0
        # The admission resumed the reservation: one first-chunk resume, whose
        # pages the stand-in caches cannot report.
        assert diag.ctx_resumed == {"first": 1, "started": 0}
        assert diag.resume_pages_unknown == 1
    for diag in (stage1, stage2):
        assert (diag.adopt_first, diag.adopt_fresh_claims, diag.adopt_reclaims) == (1, 0, 0)


def test_step_counts_the_chunks_and_logs_a_line_when_due(monkeypatch):
    ranks, _, common = _ranks(monkeypatch, diag=True)
    base._run_until_admitted(ranks)
    lines = []
    monkeypatch.setattr(base.py_executor_module.logger, "info", lines.append)
    for executor in ranks:
        diag = executor._pp_ctx_diag
        # Due now: the next line's time is the clock's current reading.
        diag._next_line = diag._clock()
        req = executor.active_requests[0]
        executor._pp_ctx_diag_step(types.SimpleNamespace(context_requests=[req]))
        assert diag.chunks == {"first": 1, "started": 0}
    assert len(lines) == len(ranks)
    for stage, line in enumerate(lines):
        fields = dict(token.split("=", 1) for token in line.split(" ")[1:])
        assert fields["kind"] == "periodic"
        assert fields["pp_rank"] == str(stage)
        assert fields["q_waiting_first"] == "1"
    assert lines[0].split(" ")[0] == "[pp_ctx_diag]"


@pytest.mark.parametrize("active", [True, False])
def test_kept_reservation_is_counted_whether_or_not_its_cache_is_still_active(monkeypatch, active):
    """A resize refused pages suspends a first chunk's cache before the
    scheduler keeps the reservation, so the count must not need it active."""
    monkeypatch.setattr(scheduler_v2, "rewind_context_after_cache_drop", lambda req, tpb: None)
    scheduler = object.__new__(scheduler_v2.KVCacheV2Scheduler)
    manager = MagicMock(name="kv_cache_manager")
    manager.kv_cache_map = {5: object()}
    manager.is_request_active.return_value = active
    scheduler.kv_cache_manager = manager
    scheduler.draft_kv_cache_manager = None
    scheduler.tokens_per_block = 32
    scheduler.ctx_diag = PPCtxDiag(1, 4)
    req = types.SimpleNamespace(py_request_id=5, py_batch_idx=3)

    scheduler._keep_stage_reservation(req, page_refused=True)
    scheduler._keep_stage_reservation(req, page_refused=False)

    assert scheduler.ctx_diag.ctx_reservations_suspended == 2
    assert scheduler.ctx_diag.ctx_reservations_page_refused == 1
    assert manager.suspend_request.call_count == 2
    assert req.py_pp_page_refused is False
