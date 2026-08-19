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
"""Session lifecycle under pipelined KV transfer.

Pipelining makes transfer activity outlive the compute-phase request state, and
the guards that keep that safe are otherwise only exercised indirectly. Each
test here fails if its guard is removed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import TaskStatus, TxSession
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggregatedParams

pytestmark = pytest.mark.cpu_only

_RID = 4242


def _req(rid=_RID):
    return SimpleNamespace(
        request_id=rid,
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
        state=LlmRequestState.CONTEXT_INIT,
    )


def _transceiver(send=None, recv=None):
    tc = MagicMock()
    tc._send_sessions = send if send is not None else {}
    tc._recv_sessions = recv if recv is not None else {}
    return tc


def _tx_session(task_statuses, exception=None, terminal=None, aux=None):
    """A TxSession carrying the given per-slice task states."""
    s = TxSession.__new__(TxSession)
    s._terminal_status = terminal
    s._exception = exception
    s.kv_tasks = [SimpleNamespace(status=st) for st in task_statuses]
    s.aux_task = aux
    s.receiver_ready = True
    return s


class TestHasInflightTransfer:
    """Session membership, not req.state, is the ownership record."""

    def test_true_while_a_send_session_exists(self):
        req = _req()
        tc = _transceiver(send={_RID: object()})
        assert KvCacheTransceiverV2.has_inflight_transfer(tc, req) is True

    def test_true_while_a_recv_session_exists(self):
        req = _req()
        tc = _transceiver(recv={_RID: object()})
        assert KvCacheTransceiverV2.has_inflight_transfer(tc, req) is True

    def test_false_once_no_session_holds_the_request(self):
        assert KvCacheTransceiverV2.has_inflight_transfer(_transceiver(), _req()) is False

    def test_independent_of_the_compute_phase_state(self):
        """A chunk can be in flight while the request is still prefilling."""
        req = _req()
        req.state = LlmRequestState.CONTEXT_INIT  # not a TRANS_IN_PROGRESS state
        tc = _transceiver(send={_RID: object()})
        assert KvCacheTransceiverV2.has_inflight_transfer(tc, req) is True


class TestCollectDoneRegistrationGuard:
    """A pipelined session predates its request registration by many chunks."""

    def test_skips_a_session_whose_request_is_not_registered_yet(self):
        """Acting on it would close a session later chunks still need."""
        session = MagicMock()
        session.is_completed.return_value = True
        session.has_failed.return_value = False
        completed, failed = KvCacheTransceiverV2._collect_done(MagicMock(), {_RID: session}, {})
        assert completed == [] and failed == []

    def test_skips_an_unregistered_session_that_has_failed(self):
        session = MagicMock()
        session.is_completed.return_value = False
        session.has_failed.return_value = True
        completed, failed = KvCacheTransceiverV2._collect_done(MagicMock(), {_RID: session}, {})
        assert completed == [] and failed == []

    def test_collects_once_the_request_is_registered(self):
        session = MagicMock()
        session.is_completed.return_value = True
        session.has_failed.return_value = False
        completed, failed = KvCacheTransceiverV2._collect_done(
            MagicMock(), {_RID: session}, {_RID: _req()}
        )
        assert completed == [_RID] and failed == []


class TestTxSessionStatus:
    """A failed slice must surface before the last slice is ever built."""

    def test_error_on_a_failed_slice_while_others_are_pending(self):
        s = _tx_session([TaskStatus.ERROR, TaskStatus.INIT])
        assert s.status == SessionStatus.ERROR

    def test_error_on_a_session_exception(self):
        s = _tx_session([TaskStatus.INIT], exception=RuntimeError("boom"))
        assert s.status == SessionStatus.ERROR

    def test_terminal_status_still_wins(self):
        s = _tx_session([TaskStatus.ERROR], terminal=SessionStatus.CANCELLED)
        assert s.status == SessionStatus.CANCELLED

    def test_all_slices_transferred_reports_kv_transferred(self):
        s = _tx_session([TaskStatus.TRANSFERRED, TaskStatus.TRANSFERRED])
        assert s.status == SessionStatus.KV_TRANSFERRED

    def test_aux_pending_keeps_it_short_of_fully_transferred(self):
        """This is what stops a pipelined session completing mid-prefill."""
        aux = SimpleNamespace(status=TaskStatus.INIT)
        s = _tx_session([TaskStatus.TRANSFERRED], aux=aux)
        assert s.status == SessionStatus.KV_TRANSFERRED
        aux.status = TaskStatus.TRANSFERRED
        assert s.status == SessionStatus.FULLY_TRANSFERRED


class TestIsRequestInTransmission:
    """The executor must consult the transceiver, not just req.state."""

    @staticmethod
    def _executor(transceiver):
        ex = object.__new__(PyExecutor)
        ex.kv_cache_transceiver = transceiver
        return ex

    @pytest.mark.parametrize(
        "state",
        [
            LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        ],
    )
    def test_true_for_the_compute_phase_states(self, state):
        req = _req()
        req.state = state
        tc = MagicMock()
        tc.has_inflight_transfer.return_value = False
        assert PyExecutor._is_request_in_transmission(self._executor(tc), req) is True

    def test_true_for_a_chunk_in_flight_during_prefill(self):
        """Without this the KV pages could be freed under an active NIC read."""
        req = _req()
        req.state = LlmRequestState.CONTEXT_INIT
        tc = MagicMock()
        tc.has_inflight_transfer.return_value = True
        assert PyExecutor._is_request_in_transmission(self._executor(tc), req) is True

    def test_false_when_nothing_is_in_flight(self):
        req = _req()
        req.state = LlmRequestState.CONTEXT_INIT
        tc = MagicMock()
        tc.has_inflight_transfer.return_value = False
        assert PyExecutor._is_request_in_transmission(self._executor(tc), req) is False

    def test_false_without_a_transceiver(self):
        req = _req()
        req.state = LlmRequestState.CONTEXT_INIT
        assert PyExecutor._is_request_in_transmission(self._executor(None), req) is False


class TestPipelinedTransferResolution:
    """On wherever supported; the env var is the kill switch."""

    @staticmethod
    def _resolve(pp_size=1, bounce_mb=0, window=None, disabled=False, monkeypatch=None):
        tc = MagicMock()
        tc._mapping.pp_size = pp_size
        tc._page_table.layer_groups = [SimpleNamespace(sliding_window_size=window)]
        cfg = SimpleNamespace(kv_cache_bounce_size_mb=bounce_mb)
        if disabled:
            monkeypatch.setenv("TRTLLM_DISABLE_PIPELINED_KV_TRANSFER", "1")
        return KvCacheTransceiverV2._resolve_pipelined_transfer(tc, cfg)

    def test_on_when_nothing_blocks_it(self):
        assert self._resolve() is True

    def test_env_kill_switch_wins(self, monkeypatch):
        assert self._resolve(disabled=True, monkeypatch=monkeypatch) is False

    @pytest.mark.parametrize("kwargs", [{"pp_size": 2}, {"bounce_mb": 64}, {"window": 512}])
    def test_off_when_unsupported(self, kwargs):
        """Never raises: an unsupported rank stays off and says so."""
        assert self._resolve(**kwargs) is False
