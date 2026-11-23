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

import contextlib
import threading
import time
from types import SimpleNamespace
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

    def __init__(
        self,
        *,
        previous_batch,
        pp_size=1,
        overlap=True,
        early_first_token=False,
        drafter=None,
        use_spec_decode=False,
    ):
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
            "update_requests",
            "send_kv_async",
            "flush",
            "commit_stats",
            "wait_input_copy",
            "process_previous_batch",
        ]
        assert rank.previous_batch is None
        rank.perf_manager.compute_batch_gpu_times.assert_called_once_with(["r1", "r2"])
        assert rank.has_previous_draft_tokens is False


class _RecordingEvent(threading.Event):
    def __init__(self, log, name):
        super().__init__()
        self._log = log
        self._name = name

    def set(self):
        self._log.append(self._name)
        super().set()


class _NoHangDetector:
    def pause(self):
        return contextlib.nullcontext()


class _PendingLoadsConnector:
    def has_pending_loads(self):
        return True


class _ControlExecutor:
    """Stand-in for PyExecutor around _handle_control_request."""

    _handle_control_request = PyExecutor._handle_control_request
    _control_drain_ready = PyExecutor._control_drain_ready
    _has_pending_connector_transfers = PyExecutor._has_pending_connector_transfers
    _has_pending_context_retirement = PyExecutor._has_pending_context_retirement
    _dist_size = staticmethod(PyExecutor._dist_size)

    def __init__(self, *, drain, active=(), transfers=0, connector=None):
        self.order = []
        self.control_requests = [SimpleNamespace(control_requires_drain=drain, control_id="c1")]
        self.active_requests = list(active)
        self.waiting_queue = []
        self.enable_attention_dp = False
        self.kv_connector_manager = connector
        self.dist = SimpleNamespace(world_size=1, pp_size=1, tp_size=1)
        self._transfers = transfers
        self._control_drain_transfer_deadline = None
        self.control_request_barrier = _RecordingEvent(self.order, "barrier")
        self.control_action_done = threading.Event()
        self.control_action_done.set()  # the action completes immediately
        self.hang_detector = _NoHangDetector()
        self._active_control_id = None
        self.dwdp_manager = None
        self.disagg = SimpleNamespace(has_retiring_gen_receives=False)

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

    def test_drain_waits_for_pending_connector_loads(self, cuda_sync):
        ex = _ControlExecutor(drain=True, connector=_PendingLoadsConnector())

        ex._handle_control_request()

        # A KV connector's pending loads hold the drain, with no deadline.
        assert ex.order == [] and cuda_sync == []
        assert len(ex.control_requests) == 1
        assert ex._control_drain_transfer_deadline is None

    def test_drain_transfer_deadline_arms_once_then_fires(self):
        ex = _ControlExecutor(drain=True, transfers=1)

        assert ex._control_drain_ready(1) is False
        armed = ex._control_drain_transfer_deadline
        assert armed is not None and armed > time.monotonic()
        assert ex._control_drain_ready(1) is False
        assert ex._control_drain_transfer_deadline == armed  # not re-armed while waiting
        ex._control_drain_transfer_deadline = time.monotonic() - 1.0  # the deadline passed
        assert ex._control_drain_ready(1) is True  # fires anyway, with a warning
        assert ex._control_drain_ready(0) is True
        ex.active_requests = ["busy"]
        assert ex._control_drain_ready(0) is False
        assert ex._control_drain_transfer_deadline is None


if __name__ == "__main__":
    pytest.main([__file__])
