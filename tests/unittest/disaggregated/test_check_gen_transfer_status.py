"""Unit tests for KvCacheTransceiverV2.check_gen_transfer_status.

Tests the timeout and retry behavior for gen-side KV transfer:
- RxSession.wait_complete is called with the configured timeout
- Completed transfers are cleaned up and marked DISAGG_GENERATION_TRANS_COMPLETE
- Timed-out transfers (WaitResult.TIMEOUT) are kept for retry
- Failed transfers are cleaned up and marked DISAGG_TRANS_ERROR
"""

from unittest.mock import MagicMock

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle


def _make_mock_session(status=SessionStatus.KV_TRANSFERRED, wait_result=WaitResult.COMPLETED):
    """Create a mock RxSession with configurable state and wait_complete result."""
    session = MagicMock()
    session.status = status
    session.is_completed.return_value = status in (
        SessionStatus.KV_TRANSFERRED,
        SessionStatus.FULLY_TRANSFERRED,
    )
    session.has_failed.return_value = status == SessionStatus.ERROR
    session.wait_complete.return_value = wait_result
    return session


def _make_mock_request(need_aux=False):
    """Create a mock LlmRequest with configurable disagg params."""
    req = MagicMock()
    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    if need_aux:
        req.py_disaggregated_params.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
    else:
        req.py_disaggregated_params = None
    return req


def _make_transceiver(recv_sessions, recv_reqs):
    """Create a mock transceiver with the attributes check_gen_transfer_status needs."""
    txr = MagicMock(spec=KvCacheTransceiverV2)
    txr._recv_sessions = dict(recv_sessions)
    txr._recv_reqs = dict(recv_reqs)
    txr._need_aux_transfer = KvCacheTransceiverV2._need_aux_transfer
    txr._collect_done = lambda self_sessions, self_reqs: KvCacheTransceiverV2._collect_done(
        txr, self_sessions, self_reqs
    )
    txr._build_to_process = lambda sessions, consensus, wait_num, block_all: (
        KvCacheTransceiverV2._build_to_process(txr, sessions, consensus, wait_num, block_all)
    )
    txr._close_failed_sessions = lambda sessions, reqs, failed: (
        KvCacheTransceiverV2._close_failed_sessions(txr, sessions, reqs, failed)
    )
    txr._gen_consensus = lambda ids: ids  # single-rank: no sync needed
    return txr


class TestCheckGenTransferStatusCompleted:
    """Tests for successful transfer completion."""

    def test_completed_transfer_sets_state_and_cleans_up(self):
        session = _make_mock_session(wait_result=WaitResult.COMPLETED)
        req = _make_mock_request()
        rid = "req-1"

        txr = _make_transceiver(
            recv_sessions={rid: session},
            recv_reqs={rid: req},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert rid not in txr._recv_sessions
        assert rid not in txr._recv_reqs
        session.close.assert_called_once()

    def test_wait_complete_called_with_blocking_flag(self):
        session = _make_mock_session(wait_result=WaitResult.COMPLETED)
        req = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"rid": session},
            recv_reqs={"rid": req},
        )

        # block_all=True when at_least_request_num is None
        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)
        session.wait_complete.assert_called_once_with(blocking=True)


class TestCheckGenTransferStatusTimeout:
    """Tests for timeout behavior (WaitResult.TIMEOUT)."""

    def test_timed_out_transfer_kept_for_retry(self):
        session = _make_mock_session(wait_result=WaitResult.TIMEOUT)
        req = _make_mock_request()
        rid = "req-1"
        original_state = req.state

        txr = _make_transceiver(
            recv_sessions={rid: session},
            recv_reqs={rid: req},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        # Request must remain in dicts for retry on next iteration
        assert rid in txr._recv_sessions
        assert rid in txr._recv_reqs
        # State must NOT be changed
        assert req.state == original_state
        # Session must NOT be closed
        session.close.assert_not_called()

    def test_timed_out_transfer_not_marked_as_error(self):
        session = _make_mock_session(wait_result=WaitResult.TIMEOUT)
        req = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"rid": session},
            recv_reqs={"rid": req},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        assert req.state != LlmRequestState.DISAGG_TRANS_ERROR


class TestCheckGenTransferStatusFailed:
    """Tests for terminal failure (WaitResult.FAILED)."""

    def test_failed_transfer_sets_error_and_cleans_up(self):
        session = _make_mock_session(status=SessionStatus.ERROR, wait_result=WaitResult.FAILED)
        req = _make_mock_request()
        rid = "req-1"

        txr = _make_transceiver(
            recv_sessions={rid: session},
            recv_reqs={rid: req},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
        assert rid not in txr._recv_sessions
        assert rid not in txr._recv_reqs
        session.close.assert_called_once()


class TestCheckGenTransferStatusMixed:
    """Tests with multiple requests in different states."""

    def test_mixed_completed_timed_out_and_failed(self):
        session_ok = _make_mock_session(wait_result=WaitResult.COMPLETED)
        session_timeout = _make_mock_session(wait_result=WaitResult.TIMEOUT)
        session_fail = _make_mock_session(status=SessionStatus.ERROR, wait_result=WaitResult.FAILED)
        req_ok = _make_mock_request()
        req_timeout = _make_mock_request()
        req_fail = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"ok": session_ok, "timeout": session_timeout, "fail": session_fail},
            recv_reqs={"ok": req_ok, "timeout": req_timeout, "fail": req_fail},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        # Completed: cleaned up
        assert "ok" not in txr._recv_sessions
        assert req_ok.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE

        # Timed-out: kept for retry
        assert "timeout" in txr._recv_sessions
        assert req_timeout.state != LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert req_timeout.state != LlmRequestState.DISAGG_TRANS_ERROR

        # Failed: cleaned up with error
        assert "fail" not in txr._recv_sessions
        assert req_fail.state == LlmRequestState.DISAGG_TRANS_ERROR

    def test_check_gen_transfer_complete_reflects_remaining(self):
        """check_gen_transfer_complete should return False when timed-out requests remain."""
        session_timeout = _make_mock_session(wait_result=WaitResult.TIMEOUT)
        req = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"rid": session_timeout},
            recv_reqs={"rid": req},
        )

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        assert not KvCacheTransceiverV2.check_gen_transfer_complete(txr)

    def test_session_in_init_state_not_selected(self):
        """Sessions that haven't reached KV_TRANSFERRED should not be polled via wait_complete."""
        session_init = _make_mock_session(status=SessionStatus.INIT, wait_result=WaitResult.TIMEOUT)
        session_init.is_completed.return_value = False
        session_init.has_failed.return_value = False
        req = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"rid": session_init},
            recv_reqs={"rid": req},
        )

        # at_least_request_num=0 means don't force-process any
        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=0)

        session_init.wait_complete.assert_not_called()

    def test_retry_on_second_call_completes(self):
        """Simulate a timed-out transfer succeeding on the next call."""
        session = _make_mock_session(wait_result=WaitResult.TIMEOUT)
        req = _make_mock_request()

        txr = _make_transceiver(
            recv_sessions={"rid": session},
            recv_reqs={"rid": req},
        )

        # First call: times out
        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)
        assert "rid" in txr._recv_sessions

        # Second call: succeeds
        session.wait_complete.return_value = WaitResult.COMPLETED
        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)
        assert "rid" not in txr._recv_sessions
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE


class TestCheckGenTransferStatusBlockAll:
    """Tests for the at_least_request_num parameter."""

    def test_block_all_processes_all_requests(self):
        sessions = {
            f"rid-{i}": _make_mock_session(wait_result=WaitResult.COMPLETED) for i in range(3)
        }
        reqs = {f"rid-{i}": _make_mock_request() for i in range(3)}

        txr = _make_transceiver(recv_sessions=sessions, recv_reqs=reqs)

        KvCacheTransceiverV2.check_gen_transfer_status(txr, at_least_request_num=None)

        assert len(txr._recv_sessions) == 0
