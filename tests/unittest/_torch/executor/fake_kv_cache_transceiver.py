# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    CtxTransferStatus,
    GenTransferStatus,
    KvCacheTransceiver,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.bindings import LlmRequestState

_SEND_OUTCOMES = ("complete", "error")
_RECV_OUTCOMES = ("complete", "error", "cancel")


class FakeKvCacheTransceiver(KvCacheTransceiver):
    """In-process CPU fake of the ``KvCacheTransceiver`` contract.

    Transfers never make progress on their own: tests enqueue outcomes via
    ``finish_send`` / ``finish_recv`` / ``cancel_recv_remotely``, and the next
    ``check_*_transfer_status`` call reports them with the same typed results
    and request-state postconditions the contract requires from real
    implementations. This makes transfer lifecycle timing fully deterministic
    for executor-level tests.

    Blocking mode (``at_least_request_num=None``) is runtime-specific in the
    real implementations and therefore not portable: the C++ runtime blocks
    until everything completes (potentially unboundedly) or rejects None when
    in-flight cancellation is enabled, while V2's bounded wait can return
    with sessions still pending. The fake REJECTS None outright — the same
    stance as the strictest real runtime — so any coordinator code that
    reaches for None fails at the exact call instead of passing here and
    breaking on C++. Portable draining is finite polls in an explicit loop;
    the fake models that: each call reports whatever outcomes are scripted
    and returns with the rest still pending.

    The fake is deliberately stricter than the real implementations about
    contract misuse: double send/receive and use after shutdown raise
    ``AssertionError`` instead of being tolerated, so orchestration bugs
    surface at the exact call that commits them.

    Requests are tracked by ``py_request_id``; test requests only need
    ``py_request_id`` and a writable ``state`` attribute.
    """

    def __init__(self, kv_transfer_timeout_ms: Optional[int] = None) -> None:
        self.kv_transfer_timeout_ms = kv_transfer_timeout_ms
        self._pending_sends: Dict[int, LlmRequest] = {}
        self._pending_recvs: Dict[int, LlmRequest] = {}
        # rid -> outcome, consumed by the next check_*_transfer_status call.
        self._send_outcomes: Dict[int, str] = {}
        self._recv_outcomes: Dict[int, str] = {}
        # Chronological record of contract calls, for call-order assertions.
        self.call_log: List[str] = []
        self._is_shut_down = False

    # -- test scripting API (not part of the contract) ----------------------

    def finish_send(self, req: LlmRequest, outcome: str = "complete") -> None:
        """Script the outcome the next context status check reports for ``req``."""
        assert outcome in _SEND_OUTCOMES, f"invalid send outcome {outcome!r}"
        assert req.py_request_id in self._pending_sends, (
            f"finish_send for request {req.py_request_id} which has no pending send"
        )
        self._send_outcomes[req.py_request_id] = outcome

    def finish_recv(self, req: LlmRequest, outcome: str = "complete") -> None:
        """Script the outcome the next generation status check reports for ``req``."""
        assert outcome in _RECV_OUTCOMES, f"invalid recv outcome {outcome!r}"
        assert req.py_request_id in self._pending_recvs, (
            f"finish_recv for request {req.py_request_id} which has no pending receive"
        )
        self._recv_outcomes[req.py_request_id] = outcome

    def cancel_recv_remotely(self, req: LlmRequest) -> None:
        """Script a remote-initiated cancellation (e.g. context-side timeout)."""
        self.finish_recv(req, outcome="cancel")

    def script_sync_recv(self, req: LlmRequest, outcome: str) -> None:
        """Script the outcome of an upcoming ``request_and_receive_sync`` for ``req``.

        Unscripted synchronous receives succeed by default; use this to make
        one fail so the blocking-receive error postcondition can be tested.
        """
        assert outcome in _SEND_OUTCOMES, f"invalid sync receive outcome {outcome!r}"
        self._recv_outcomes[req.py_request_id] = outcome

    # -- KvCacheTransceiver contract -----------------------------------------

    def respond_and_send_async(self, req: LlmRequest) -> None:
        self._assert_alive("respond_and_send_async")
        rid = req.py_request_id
        self.call_log.append(f"respond_and_send_async:{rid}")
        assert rid not in self._pending_sends, f"double send for request {rid}"
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        self._pending_sends[rid] = req

    def request_and_receive_sync(self, req: LlmRequest) -> None:
        self._assert_alive("request_and_receive_sync")
        rid = req.py_request_id
        self.call_log.append(f"request_and_receive_sync:{rid}")
        # The blocking receive settles before returning: consume a scripted
        # outcome if one was queued ahead of time, defaulting to success.
        outcome = self._recv_outcomes.pop(rid, "complete")
        if outcome == "complete":
            req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        else:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR

    def request_and_receive_async(self, req: LlmRequest) -> None:
        self._assert_alive("request_and_receive_async")
        rid = req.py_request_id
        self.call_log.append(f"request_and_receive_async:{rid}")
        assert rid not in self._pending_recvs, f"double receive for request {rid}"
        req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        self._pending_recvs[rid] = req

    def check_context_transfer_status(
        self, at_least_request_num: Optional[int], mark_complete: bool = False
    ) -> CtxTransferStatus:
        self._assert_alive("check_context_transfer_status")
        self._assert_finite_poll("check_context_transfer_status", at_least_request_num)
        self.call_log.append(f"check_context_transfer_status:{at_least_request_num}")
        completed, errors = [], []
        for rid, outcome in list(self._send_outcomes.items()):
            del self._send_outcomes[rid]
            req = self._pending_sends.pop(rid)
            if outcome == "complete":
                if mark_complete:
                    req.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE
                completed.append(rid)
            else:
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                errors.append(rid)
        return CtxTransferStatus(completed, errors)

    def check_gen_transfer_status(self, at_least_request_num: Optional[int]) -> GenTransferStatus:
        self._assert_alive("check_gen_transfer_status")
        self._assert_finite_poll("check_gen_transfer_status", at_least_request_num)
        self.call_log.append(f"check_gen_transfer_status:{at_least_request_num}")
        completed, errors, cancelled = [], [], []
        for rid, outcome in list(self._recv_outcomes.items()):
            del self._recv_outcomes[rid]
            req = self._pending_recvs.pop(rid)
            if outcome == "complete":
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                completed.append(rid)
            elif outcome == "error":
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                errors.append(rid)
            else:
                # Cancelled sessions are closed; per the contract the caller
                # decides the final request state.
                cancelled.append(req)
        return GenTransferStatus(completed, errors, cancelled)

    def check_gen_transfer_complete(self) -> bool:
        return not self._pending_recvs

    def cancel_request(self, req: LlmRequest) -> bool:
        self._assert_alive("cancel_request")
        rid = req.py_request_id
        self.call_log.append(f"cancel_request:{rid}")
        self._pending_sends.pop(rid, None)
        self._pending_recvs.pop(rid, None)
        self._send_outcomes.pop(rid, None)
        self._recv_outcomes.pop(rid, None)
        return True

    def prepare_context_requests(self, requests: List[LlmRequest]) -> None:
        # Mirror BindKvCacheTransceiver: a no-op placeholder so the executor
        # can invoke it unconditionally.
        ...

    def get_disaggregated_params(self) -> Dict[str, object]:
        return {}

    def shutdown(self) -> None:
        self._is_shut_down = True

    # -- internal invariants --------------------------------------------------

    def _assert_alive(self, method: str) -> None:
        assert not self._is_shut_down, f"{method} called after shutdown"

    @staticmethod
    def _assert_finite_poll(method: str, at_least_request_num: Optional[int]) -> None:
        assert at_least_request_num is not None, (
            f"{method}(None) is not portable: the C++ runtime rejects it under "
            "in-flight cancellation and may block unboundedly otherwise. Use a "
            "finite poll (0 or N) in an explicit re-poll loop."
        )
