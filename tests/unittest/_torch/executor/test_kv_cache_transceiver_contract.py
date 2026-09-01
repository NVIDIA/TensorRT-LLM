# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance tests for the KvCacheTransceiver contract.

Exercises ``FakeKvCacheTransceiver`` against the typed results, request-state
postconditions, and signature requirements documented on the ABC, and checks
that every implementation (C++ binding wrapper, Python V2, fake) exposes the
same call signatures. The fake is the executable specification later
executor-level tests build on, so its own strictness (negative tests) is
verified here too.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fake_kv_cache_transceiver import FakeKvCacheTransceiver

from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    BindKvCacheTransceiver,
    CtxTransferStatus,
    GenTransferStatus,
    KvCacheTransceiver,
)
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


def _req(rid: int) -> SimpleNamespace:
    return SimpleNamespace(py_request_id=rid, state=None)


def test_fake_implements_contract() -> None:
    fake = FakeKvCacheTransceiver(kv_transfer_timeout_ms=1000)
    assert isinstance(fake, KvCacheTransceiver)
    assert fake.kv_transfer_timeout_ms == 1000
    assert FakeKvCacheTransceiver(kv_transfer_timeout_ms=None).kv_transfer_timeout_ms is None


def test_all_implementations_expose_contract_signatures() -> None:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    for cls in (BindKvCacheTransceiver, KvCacheTransceiverV2, FakeKvCacheTransceiver):
        ctx_params = inspect.signature(cls.check_context_transfer_status).parameters
        assert "at_least_request_num" in ctx_params, cls
        assert "mark_complete" in ctx_params, cls
        assert ctx_params["mark_complete"].default is False, cls
        gen_params = inspect.signature(cls.check_gen_transfer_status).parameters
        assert "at_least_request_num" in gen_params, cls
        assert "cancel_request" in vars(cls) or hasattr(cls, "cancel_request"), cls


def _bind_with_impl(impl) -> BindKvCacheTransceiver:
    """BindKvCacheTransceiver around a stub impl, skipping the C++ constructor."""
    bind = object.__new__(BindKvCacheTransceiver)
    bind.impl = impl
    return bind


def test_bind_wraps_cpp_context_status_and_forwards_arguments() -> None:
    impl = Mock()
    impl.check_context_transfer_status.return_value = ([1, 2], [3])
    bind = _bind_with_impl(impl)

    status = bind.check_context_transfer_status(0)
    assert isinstance(status, CtxTransferStatus)
    assert status == CtxTransferStatus([1, 2], [3])
    impl.check_context_transfer_status.assert_called_once_with(0, False)

    bind.check_context_transfer_status(None, mark_complete=True)
    impl.check_context_transfer_status.assert_called_with(None, True)


def test_bind_normalizes_void_cpp_gen_status_to_empty_typed_result() -> None:
    impl = Mock()
    impl.check_gen_transfer_status.return_value = None
    bind = _bind_with_impl(impl)

    status = bind.check_gen_transfer_status(0)
    assert isinstance(status, GenTransferStatus)
    assert status == GenTransferStatus([], [], [])
    impl.check_gen_transfer_status.assert_called_once_with(0)


def test_v2_early_return_paths_are_typed() -> None:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    v2 = object.__new__(KvCacheTransceiverV2)
    v2._ever_had_send_session = False
    v2._ctx_need_tp_sync = False
    v2._ctx_need_pp_sync = False
    ctx_status = v2.check_context_transfer_status(0)
    assert isinstance(ctx_status, CtxTransferStatus)
    assert ctx_status == CtxTransferStatus([], [])

    v2._ever_had_recv_session = False
    v2._gen_need_sync = False
    gen_status = v2.check_gen_transfer_status(0)
    assert isinstance(gen_status, GenTransferStatus)
    assert gen_status == GenTransferStatus([], [], [])


def test_send_lifecycle_reports_typed_completion() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(1)

    fake.respond_and_send_async(req)
    assert req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    status = fake.check_context_transfer_status(0)
    assert isinstance(status, CtxTransferStatus)
    assert status == CtxTransferStatus([], [])

    fake.finish_send(req)
    status = fake.check_context_transfer_status(0)
    assert status.completed_request_ids == [1]
    assert status.error_request_ids == []
    # Without mark_complete the state transition is the caller's job.
    assert req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS


def test_send_mark_complete_transitions_state() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(2)
    fake.respond_and_send_async(req)
    fake.finish_send(req)

    status = fake.check_context_transfer_status(0, mark_complete=True)
    assert status.completed_request_ids == [2]
    assert req.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE


def test_send_error_reported_with_error_state() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(3)
    fake.respond_and_send_async(req)
    fake.finish_send(req, outcome="error")

    status = fake.check_context_transfer_status(0)
    assert status == CtxTransferStatus([], [3])
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_recv_async_lifecycle_postconditions() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(4)

    fake.request_and_receive_async(req)
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert not fake.check_gen_transfer_complete()

    fake.finish_recv(req)
    status = fake.check_gen_transfer_status(0)
    assert isinstance(status, GenTransferStatus)
    assert status.completed_request_ids == [4]
    assert status.error_request_ids == []
    assert status.cancelled_requests == []
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert fake.check_gen_transfer_complete()


def test_recv_error_sets_error_state() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(5)
    fake.request_and_receive_async(req)
    fake.finish_recv(req, outcome="error")

    status = fake.check_gen_transfer_status(0)
    assert status.error_request_ids == [5]
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_remote_cancel_returns_request_and_leaves_state_to_caller() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(6)
    fake.request_and_receive_async(req)
    fake.cancel_recv_remotely(req)

    status = fake.check_gen_transfer_status(0)
    assert status.cancelled_requests == [req]
    assert status.completed_request_ids == []
    # The transceiver must not decide user-cancel vs remote-cancel.
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert fake.check_gen_transfer_complete()


def test_recv_sync_settles_before_returning() -> None:
    fake = FakeKvCacheTransceiver()
    ok = _req(7)
    fake.request_and_receive_sync(ok)
    assert ok.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE


def test_recv_sync_failure_sets_error_state() -> None:
    fake = FakeKvCacheTransceiver()
    bad = _req(12)
    fake.script_sync_recv(bad, outcome="error")
    fake.request_and_receive_sync(bad)
    assert bad.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_recv_sync_rejects_double_receive() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(13)
    fake.request_and_receive_sync(req)
    with pytest.raises(AssertionError, match="double receive"):
        fake.request_and_receive_sync(req)


def test_unconsumed_sync_script_does_not_leak_into_status_poll() -> None:
    fake = FakeKvCacheTransceiver()
    stale = _req(14)
    fake.script_sync_recv(stale, outcome="error")
    # The sync receive never happens; the async poll must not see the script.
    assert fake.check_gen_transfer_status(0) == GenTransferStatus([], [], [])


def test_cancel_request_clears_pending_transfers() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(8)
    fake.request_and_receive_async(req)

    assert fake.cancel_request(req) is True
    assert fake.check_gen_transfer_complete()
    assert fake.check_gen_transfer_status(0) == GenTransferStatus([], [], [])


def test_cancelled_request_may_receive_again() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(15)
    fake.request_and_receive_async(req)
    fake.cancel_request(req)
    fake.request_and_receive_async(req)
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS


def test_inflight_cancellation_support_is_configurable() -> None:
    assert FakeKvCacheTransceiver().supports_inflight_request_cancellation() is False
    fake = FakeKvCacheTransceiver(supports_inflight_cancellation=True)
    assert fake.supports_inflight_request_cancellation() is True


def test_results_stay_positionally_compatible() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(9)
    fake.respond_and_send_async(req)
    fake.finish_send(req)

    completed, errors = fake.check_context_transfer_status(0)
    assert (completed, errors) == ([9], [])

    gen = fake.check_gen_transfer_status(0)
    completed, errors, cancelled = gen
    assert gen == ([], [], [])


def test_drain_is_finite_polls_in_a_loop() -> None:
    """Portable draining re-polls with finite arguments until nothing remains.

    A single poll is never a guaranteed drain (V2's bounded blocking wait can
    time out and keep sessions in progress; see
    ``test_context_transfer_status_timeout_retains_session_and_request``), so
    the re-poll loop below is the pattern coordinator code must use.
    """
    fake = FakeKvCacheTransceiver()
    req = _req(10)
    fake.request_and_receive_async(req)

    status = fake.check_gen_transfer_status(0)
    assert status == GenTransferStatus([], [], [])
    assert not fake.check_gen_transfer_complete()
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    # The drain loop's next iteration reaps the transfer once it settles.
    fake.finish_recv(req)
    status = fake.check_gen_transfer_status(0)
    assert status.completed_request_ids == [10]
    assert fake.check_gen_transfer_complete()


def test_block_all_rejected_as_nonportable() -> None:
    """The fake rejects None outright.

    None is runtime-specific (the C++ runtime rejects it under in-flight
    cancellation and may block unboundedly otherwise), so coordinator code
    that reaches for it must fail at the offending call, not pass here and
    break on C++.
    """
    fake = FakeKvCacheTransceiver()
    with pytest.raises(AssertionError, match="not portable"):
        fake.check_context_transfer_status(None)
    with pytest.raises(AssertionError, match="not portable"):
        fake.check_gen_transfer_status(None)


def test_double_send_raises() -> None:
    fake = FakeKvCacheTransceiver()
    req = _req(11)
    fake.respond_and_send_async(req)
    with pytest.raises(AssertionError, match="double send"):
        fake.respond_and_send_async(req)


def test_use_after_shutdown_raises() -> None:
    fake = FakeKvCacheTransceiver()
    fake.shutdown()
    with pytest.raises(AssertionError, match="after shutdown"):
        fake.check_context_transfer_status(0)
