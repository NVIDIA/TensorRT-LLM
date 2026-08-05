# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.lifecycle import (
    CancelResult,
    LifecycleCapability,
    LifecycleCapabilityError,
    LogicalDisposition,
    PhysicalDisposition,
    ShutdownResult,
    TransceiverCapabilities,
    TransceiverLifecycle,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    BindKvCacheTransceiver,
    KvCacheTransceiver,
)


def _request(request_id: int = 17):
    return SimpleNamespace(
        request_id=request_id,
        py_disaggregated_params=None,
    )


def _python_transceiver() -> KvCacheTransceiverV2:
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._wait_reqs = {}
    transceiver._ever_had_send_session = False
    transceiver._ever_had_recv_session = False
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._gen_need_sync = False
    transceiver._bounce_transfer_enabled = False
    transceiver._mapping = SimpleNamespace(enable_attention_dp=False)
    transceiver._transfer_worker = Mock()
    return transceiver


@pytest.mark.parametrize(
    ("physical", "safe_to_reuse"),
    [
        (PhysicalDisposition.NOT_EXPOSED, True),
        (PhysicalDisposition.ACTIVE, False),
        (PhysicalDisposition.QUIESCING, False),
        (PhysicalDisposition.QUIESCED_SUCCESS, True),
        (PhysicalDisposition.QUIESCED_FAILURE, True),
        (PhysicalDisposition.IN_DOUBT, False),
    ],
)
def test_physical_reuse_requires_positive_quiescence_evidence(
    physical: PhysicalDisposition, safe_to_reuse: bool
) -> None:
    result = CancelResult(LogicalDisposition.ACCEPTED, physical)
    assert result.safe_to_reuse is safe_to_reuse


def test_logical_acceptance_does_not_imply_physical_reuse() -> None:
    result = CancelResult(
        logical=LogicalDisposition.ACCEPTED,
        physical=PhysicalDisposition.QUIESCING,
        retryable=True,
    )
    assert not result.safe_to_reuse


def test_capability_require_reports_the_complete_missing_set() -> None:
    capabilities = TransceiverCapabilities(
        protocol_version=1,
        supported=frozenset({LifecycleCapability.DIRECT_TRANSFER}),
        qualified_legacy_mode=False,
    )

    with pytest.raises(
        LifecycleCapabilityError,
        match="ALLOCATION_GENERATION_LEASES, SUBMISSION_FENCE",
    ):
        capabilities.require(
            [
                LifecycleCapability.DIRECT_TRANSFER,
                LifecycleCapability.SUBMISSION_FENCE,
                LifecycleCapability.ALLOCATION_GENERATION_LEASES,
            ]
        )


def test_unqualified_default_does_not_silently_enable_legacy_mode() -> None:
    capabilities = TransceiverCapabilities()

    assert not capabilities.qualified_legacy_mode


def test_shutdown_result_rejects_unknown_count_without_in_doubt_state() -> None:
    with pytest.raises(ValueError, match="only IN_DOUBT shutdown"):
        ShutdownResult(
            physical=PhysicalDisposition.QUIESCED_SUCCESS,
            in_doubt_context_count=None,
        )


def test_unknown_shutdown_accounting_never_releases_managers() -> None:
    result = ShutdownResult(physical=PhysicalDisposition.IN_DOUBT)

    assert result.in_doubt_context_count is None
    assert not result.safe_to_release_managers


def test_base_transceiver_arbitrary_kv_state_falls_back_to_empty() -> None:
    assert KvCacheTransceiver.get_data_transceiver_state(None) == b""


def test_python_cancel_before_session_reports_not_exposed() -> None:
    transceiver = _python_transceiver()
    req = _request()
    transceiver._wait_reqs[req.request_id] = req

    result = transceiver.cancel_session(req, "client cancelled")

    assert result == CancelResult(
        logical=LogicalDisposition.ACCEPTED,
        physical=PhysicalDisposition.NOT_EXPOSED,
        retryable=False,
        reason="client cancelled",
    )
    assert transceiver._wait_reqs == {}


def test_python_active_cancel_latches_logical_failure_and_remains_retryable() -> None:
    transceiver = _python_transceiver()
    req = _request()
    session = Mock()
    session.has_transferring_tasks.return_value = True
    session.resources_drained.return_value = False
    session.status = SessionStatus.TRANSFERRING
    transceiver._send_sessions[req.request_id] = session
    transceiver._send_reqs[req.request_id] = req

    result = transceiver.cancel_session(req, "deadline")

    assert result.logical is LogicalDisposition.ACCEPTED
    assert result.physical is PhysicalDisposition.ACTIVE
    assert result.retryable
    assert not result.safe_to_reuse
    session.cancel.assert_called_once_with()
    session.close.assert_not_called()
    assert transceiver._send_sessions == {req.request_id: session}
    assert transceiver._send_reqs == {req.request_id: req}


def test_python_drained_published_session_cancel_reports_quiesced_failure() -> None:
    transceiver = _python_transceiver()
    req = _request()
    session = Mock()
    session.has_transferring_tasks.return_value = False
    transceiver._recv_sessions[req.request_id] = session
    transceiver._recv_reqs[req.request_id] = req

    result = transceiver.cancel_session(req, "client disconnected")

    assert result.logical is LogicalDisposition.ACCEPTED
    assert result.physical is PhysicalDisposition.QUIESCED_FAILURE
    assert result.safe_to_reuse
    session.cancel.assert_called_once_with()
    session.close.assert_called_once_with()
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}


def test_python_missing_session_does_not_fabricate_not_exposed() -> None:
    transceiver = _python_transceiver()

    result = transceiver.cancel_session(_request(), "late abort")

    assert result.logical is LogicalDisposition.NOT_FOUND
    assert result.physical is PhysicalDisposition.IN_DOUBT
    assert result.retryable


def test_python_legacy_cancel_result_is_preserved() -> None:
    req = _request()

    transferring = _python_transceiver()
    active_session = Mock()
    active_session.has_transferring_tasks.return_value = True
    transferring._send_sessions[req.request_id] = active_session
    transferring._send_reqs[req.request_id] = req
    assert not transferring.cancel_request(req)

    waiting = _python_transceiver()
    waiting._wait_reqs[req.request_id] = req
    assert waiting.cancel_request(req)

    missing = _python_transceiver()
    assert missing.cancel_request(req)


def test_python_capabilities_do_not_claim_unimplemented_safety() -> None:
    transceiver = _python_transceiver()
    transceiver._bounce_transfer_enabled = True
    transceiver._mapping.enable_attention_dp = True

    capabilities = transceiver.capabilities()

    assert capabilities.supports(
        LifecycleCapability.DIRECT_TRANSFER,
        LifecycleCapability.GENERATION_FIRST,
    )
    assert not capabilities.supports(
        LifecycleCapability.BOUNCE_TRANSFER,
        LifecycleCapability.ATTENTION_DATA_PARALLEL,
    )
    assert not capabilities.supports(
        LifecycleCapability.ALLOCATION_GENERATION_LEASES,
        LifecycleCapability.SUBMISSION_FENCE,
        LifecycleCapability.PER_OPERATION_QUIESCENCE,
    )
    assert capabilities.qualified_legacy_mode
    assert isinstance(transceiver, TransceiverLifecycle)


def test_python_structured_shutdown_reports_bounded_in_doubt_ownership() -> None:
    transceiver = _python_transceiver()
    req = _request()
    session = Mock()
    transceiver._send_sessions[req.request_id] = session
    transceiver._send_reqs[req.request_id] = req
    transceiver.shutdown = Mock(return_value=False)

    result = transceiver.shutdown_lifecycle(0.0)

    assert result == ShutdownResult(
        physical=PhysicalDisposition.IN_DOUBT,
        in_doubt_context_count=None,
        reason=(
            "Python transceiver did not prove physical drain before the lifecycle shutdown deadline"
        ),
    )
    transceiver.shutdown.assert_called_once_with()
    session.close.assert_not_called()
    assert transceiver._send_sessions == {req.request_id: session}
    assert transceiver._send_reqs == {req.request_id: req}


def test_python_structured_shutdown_reports_quiesced_after_drain() -> None:
    transceiver = _python_transceiver()
    transceiver.shutdown = Mock(return_value=True)

    result = transceiver.shutdown_lifecycle(0.25)

    assert result == ShutdownResult(
        physical=PhysicalDisposition.QUIESCED_SUCCESS,
        in_doubt_context_count=0,
    )
    transceiver.shutdown.assert_called_once_with()


def test_cpp_adapter_preserves_structured_dispositions() -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver._supports_inflight_request_cancellation = True
    transceiver.impl = SimpleNamespace(
        cancel_session=Mock(
            return_value=SimpleNamespace(
                logical=SimpleNamespace(name="ACCEPTED"),
                physical=SimpleNamespace(name="QUIESCING"),
                retryable=True,
                reason="operation cancellation requested",
            )
        )
    )
    req = _request()

    result = transceiver.cancel_session(req, "deadline")

    assert result == CancelResult(
        logical=LogicalDisposition.ACCEPTED,
        physical=PhysicalDisposition.QUIESCING,
        retryable=True,
        reason="operation cancellation requested",
    )
    transceiver.impl.cancel_session.assert_called_once_with(req, "deadline")


def test_cpp_adapter_maps_explicit_capabilities() -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver._supports_inflight_request_cancellation = True
    capability_fields = {capability.value.lower(): False for capability in LifecycleCapability}
    capability_fields["direct_transfer"] = True
    transceiver.impl = SimpleNamespace(
        capabilities=Mock(
            return_value=SimpleNamespace(
                protocol_version=0,
                qualified_legacy_mode=True,
                **capability_fields,
            )
        )
    )

    capabilities = transceiver.capabilities()

    assert capabilities.supports(
        LifecycleCapability.DIRECT_TRANSFER,
        LifecycleCapability.IN_FLIGHT_CANCELLATION,
    )
    assert not capabilities.supports(LifecycleCapability.ATTEMPT_IDENTITY)
    assert capabilities.qualified_legacy_mode


def test_cpp_adapter_without_capability_api_is_explicit_qualified_v0() -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver._supports_inflight_request_cancellation = False
    transceiver.impl = SimpleNamespace()

    capabilities = transceiver.capabilities()

    assert capabilities.protocol_version == 0
    assert capabilities.supported == frozenset()
    assert capabilities.qualified_legacy_mode


@pytest.mark.parametrize(
    ("protocol_version", "qualified_legacy_mode"),
    [
        (1, False),
        (0, False),
        (1, True),
    ],
)
def test_cpp_adapter_rejects_unqualified_or_generation_safe_advertisement(
    protocol_version: int,
    qualified_legacy_mode: bool,
) -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver._supports_inflight_request_cancellation = False
    transceiver.impl = SimpleNamespace(
        capabilities=Mock(
            return_value=SimpleNamespace(
                protocol_version=protocol_version,
                qualified_legacy_mode=qualified_legacy_mode,
            )
        )
    )

    with pytest.raises(RuntimeError, match="qualified only"):
        transceiver.capabilities()


def test_cpp_adapter_does_not_invoke_destructive_legacy_cancel() -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver.impl = SimpleNamespace(cancel_request=Mock(return_value=True))

    result = transceiver.cancel_session(_request(), "legacy runtime")

    assert result.logical is LogicalDisposition.REJECTED
    assert result.physical is PhysicalDisposition.IN_DOUBT
    assert not result.safe_to_reuse
    transceiver.impl.cancel_request.assert_not_called()
    assert isinstance(transceiver, TransceiverLifecycle)


def test_cpp_shutdown_adapter_preserves_in_doubt_count() -> None:
    transceiver = object.__new__(BindKvCacheTransceiver)
    transceiver.impl = SimpleNamespace(
        shutdown_lifecycle=Mock(
            return_value=SimpleNamespace(
                physical=SimpleNamespace(name="IN_DOUBT"),
                in_doubt_context_count=3,
                fatal=True,
                reason="poisoned transfer buffer",
            )
        )
    )

    result = transceiver.shutdown_lifecycle(0.25)

    assert result == ShutdownResult(
        physical=PhysicalDisposition.IN_DOUBT,
        in_doubt_context_count=3,
        fatal=True,
        reason="poisoned transfer buffer",
    )
    transceiver.impl.shutdown_lifecycle.assert_called_once_with(250)
