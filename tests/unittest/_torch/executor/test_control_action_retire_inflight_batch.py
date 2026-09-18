# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Control actions fire at a quiescent, lockstep step boundary.

``_handle_control_request`` runs on the executor-loop thread at a point every
rank reaches in the same iteration. Before it yields to the action it retires
the overlap loop's in-flight ``previous_batch`` (``_retire_inflight_batch_for_control``)
with the same attention-DP gathers on every rank, whether or not the rank has
a batch, so nothing consumes that batch later from a per-rank code path (that
deadlocked attention-DP context engines at an in-flight weight update). With
``drain=True`` the attention-DP ranks additionally vote so that no rank fires
alone.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


def _batch():
    batch = MagicMock(name="previous_batch")
    batch.scheduled_requests.all_requests.return_value = ["r1", "r2"]
    return batch


class _Rank:
    """Records what the retirement helper does on one rank."""

    _retire_inflight_batch_for_control = PyExecutor._retire_inflight_batch_for_control

    def __init__(self, *, previous_batch, pp_size=1, overlap=True,
                 early_first_token=False, drafter=None, use_spec_decode=False):
        self.dist = MagicMock(pp_size=pp_size, world_size=8)
        self.disable_overlap_scheduler = not overlap
        self.previous_batch = previous_batch
        self.speculation_gate = None
        self.enable_early_first_token_response = early_first_token
        self.drafter = drafter
        self.use_spec_decode = use_spec_decode
        self.has_previous_draft_tokens = True
        self.perf_manager = MagicMock()
        self.iter_counter = 7
        self.calls = []
        self.gathers = 0  # attention-DP collectives this rank takes part in

    def _update_requests(self, sample_state):
        self.calls.append("update_requests")

    def _send_kv_async(self, requests):
        self.calls.append("send_kv_async")

    def _emit_first_token_responses(self, scheduled_requests):
        self.calls.append("first_token")
        self.gathers += 1

    def _flush_pending_transfer_responses(self):
        self.calls.append("flush")
        self.gathers += 1  # gathers whenever attention-DP is on

    def _commit_kv_cache_stats(self, scheduled_requests):
        self.calls.append("commit_stats")

    def _wait_for_model_engine_input_copy(self):
        self.calls.append("wait_input_copy")

    def _process_previous_batch(self):
        self.calls.append("process_previous_batch")
        self.gathers += 1  # _handle_responses -> one _enqueue_responses gather

    def _enqueue_responses(self, responses):
        assert responses == []
        self.calls.append("enqueue_empty")
        self.gathers += 1


class TestRetireInflightBatch:

    def test_rank_with_batch_follows_the_loop_tail_order(self):
        rank = _Rank(previous_batch=_batch())

        rank._retire_inflight_batch_for_control()

        assert rank.calls == [
            "update_requests", "send_kv_async", "flush", "commit_stats",
            "wait_input_copy", "process_previous_batch"
        ]
        assert rank.previous_batch is None
        rank.perf_manager.compute_batch_gpu_times.assert_called_once_with(["r1", "r2"])
        assert rank.has_previous_draft_tokens is False

    def test_rank_without_batch_joins_every_gather(self):
        with_batch = _Rank(previous_batch=_batch())
        without = _Rank(previous_batch=None)

        with_batch._retire_inflight_batch_for_control()
        without._retire_inflight_batch_for_control()

        assert without.calls == ["flush", "enqueue_empty"]
        assert without.gathers == with_batch.gathers == 2

    def test_early_first_token_adds_one_symmetric_gather(self):
        with_batch = _Rank(previous_batch=_batch(), early_first_token=True)
        without = _Rank(previous_batch=None, early_first_token=True)

        with_batch._retire_inflight_batch_for_control()
        without._retire_inflight_batch_for_control()

        assert with_batch.calls[:4] == ["update_requests", "send_kv_async", "first_token", "flush"]
        assert without.calls == ["enqueue_empty", "flush", "enqueue_empty"]
        assert without.gathers == with_batch.gathers == 3

    @pytest.mark.parametrize("kwargs", [dict(pp_size=2), dict(overlap=False)],
                             ids=["pipeline_parallel", "non_overlap_loop"])
    def test_out_of_scope_loops_are_untouched(self, kwargs):
        """PP keeps its microbatches (BatchStatePP is consumed by
        _handle_executed_batch); the non-overlap loop has no in-flight batch."""
        batch = _batch()
        rank = _Rank(previous_batch=batch, **kwargs)

        rank._retire_inflight_batch_for_control()

        assert rank.calls == []
        assert rank.previous_batch is batch

    def test_spec_decode_cleanup_mirrors_the_loop(self):
        drafter = MagicMock()
        with_batch = _Rank(previous_batch=_batch(), drafter=drafter, use_spec_decode=True)
        without = _Rank(previous_batch=None, drafter=drafter, use_spec_decode=True)

        with_batch._retire_inflight_batch_for_control()
        drafter.cleanup_previous_draft_resources.assert_called_once()
        without._retire_inflight_batch_for_control()
        drafter.cleanup_previous_draft_resources.assert_called_once()  # not again
        assert with_batch.has_previous_draft_tokens is False
        assert without.has_previous_draft_tokens is False


class _RecordingEvent(threading.Event):

    def __init__(self, log, name):
        super().__init__()
        self._log = log
        self._name = name

    def set(self):
        self._log.append(self._name)
        super().set()


class _ControlExecutor:
    """Stand-in for PyExecutor around _handle_control_request."""

    _handle_control_request = PyExecutor._handle_control_request
    _control_drain_ready = PyExecutor._control_drain_ready

    def __init__(self, *, drain, attention_dp=True, active=(), transfers=0, votes=None):
        self.order = []
        self.control_requests = [MagicMock(control_requires_drain=drain, control_id="c1")]
        self.active_requests = list(active)
        self.waiting_queue = []
        self.enable_attention_dp = attention_dp
        self.dist = MagicMock(world_size=8, pp_size=1)
        if votes is not None:
            self.dist.tp_allgather.return_value = votes
        self._transfers = transfers
        self._control_drain_transfer_deadline = None
        self.control_request_barrier = _RecordingEvent(self.order, "barrier")
        self.control_action_done = threading.Event()
        self.control_action_done.set()  # the action completes immediately
        self.hang_detector = MagicMock()
        self._active_control_id = None

    def _num_inflight_kv_transfers(self):
        return self._transfers

    def _pop_sleep_wakeup_abort(self, control_id):
        return None

    def _retire_inflight_batch_for_control(self):
        self.order.append("retire")


@pytest.fixture
def cuda_sync(monkeypatch):
    log = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: log.append("cuda_sync"))
    return log


class TestHandleControlRequest:

    def test_in_flight_action_retires_the_batch_before_releasing_the_barrier(self, cuda_sync):
        ex = _ControlExecutor(drain=False, active=["busy"])

        ex._handle_control_request()

        assert cuda_sync == ["cuda_sync"]
        assert ex.order == ["retire", "barrier"]
        assert ex.control_requests == []
        ex.dist.tp_allgather.assert_not_called()  # drain=False never votes

    def test_drain_votes_every_iteration_even_when_locally_busy(self, cuda_sync):
        ex = _ControlExecutor(drain=True, active=["busy"], votes=[0, 1, 1, 1])

        ex._handle_control_request()

        # Every rank must take part in the vote or the allgather desyncs.
        ex.dist.tp_allgather.assert_called_once_with(0)
        assert ex.order == [] and cuda_sync == []
        assert len(ex.control_requests) == 1  # still pending

    def test_drain_waits_for_the_last_rank(self, cuda_sync):
        ex = _ControlExecutor(drain=True, votes=[1, 1, 0, 1])

        ex._handle_control_request()

        ex.dist.tp_allgather.assert_called_once_with(1)
        assert ex.order == [] and len(ex.control_requests) == 1

    def test_drain_fires_when_unanimous(self, cuda_sync):
        ex = _ControlExecutor(drain=True, votes=[1, 1, 1, 1])

        ex._handle_control_request()

        assert ex.order == ["retire", "barrier"]
        assert ex.control_requests == []

    def test_drain_without_attention_dp_does_not_vote(self, cuda_sync):
        ex = _ControlExecutor(drain=True, attention_dp=False)

        ex._handle_control_request()

        ex.dist.tp_allgather.assert_not_called()
        assert ex.order == ["retire", "barrier"]

    def test_drain_transfer_deadline_is_rank_local(self, monkeypatch):
        ex = _ControlExecutor(drain=True, attention_dp=False, transfers=1)
        now = [1000.0]
        monkeypatch.setattr(time, "time", lambda: now[0])

        assert ex._control_drain_ready(1) is False
        assert ex._control_drain_transfer_deadline > 1000.0  # armed on first sight
        now[0] = ex._control_drain_transfer_deadline + 1.0
        assert ex._control_drain_ready(1) is True  # fires anyway, with a warning
        assert ex._control_drain_ready(0) is True
        ex.active_requests = ["busy"]
        assert ex._control_drain_ready(0) is False
        assert ex._control_drain_transfer_deadline is None


if __name__ == "__main__":
    pytest.main([__file__])
