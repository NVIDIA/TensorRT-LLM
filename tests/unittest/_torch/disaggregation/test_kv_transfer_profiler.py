# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Observational KV profiling contracts; transport is controlled by CPU doubles."""

import gc
import weakref
from collections.abc import Iterator, ValuesView
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fake_kv_cache_transceiver import FakeKvCacheTransceiver
from profiler_test_utils import decode_profiler_metadata

from tensorrt_llm import _profiler
from tensorrt_llm._torch.disaggregation import kv_cache_transceiver as factory
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import (
    BindKvCacheTransceiver,
    CtxTransferStatus,
    GenTransferStatus,
    KvCacheTransceiver,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only


class _Request(SimpleNamespace):
    """Weak-referenceable request stand-in, like the Python LlmRequest subclass."""


def _request(
    rid: int = 5, transfer_id: int | None = 17, ctx_id: int | None = None
) -> SimpleNamespace:
    """Build a request with independent public, peer, and native status
    aliases.
    """
    return _Request(
        request_id=rid,
        py_request_id=transfer_id if transfer_id is not None else rid,
        py_disaggregated_params=SimpleNamespace(
            disagg_request_id=transfer_id, ctx_request_id=ctx_id
        ),
        prompt_len=128,
        state=None,
        py_kv_send_session_retired=False,
    )


def _wrap(backend: KvCacheTransceiver, monkeypatch: pytest.MonkeyPatch) -> KvCacheTransceiver:
    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", frozenset({"kv"}))
    wrap = getattr(factory, "_maybe_profile_kv_cache_transceiver", None)
    assert callable(wrap), "KV factory lacks opt-in observational wrapping"
    return wrap(backend, "NIXL", SimpleNamespace(rank=3))


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    captured: list[tuple[str, str]] = []

    @contextmanager
    def record(name: str, args: str) -> Iterator[None]:
        captured.append((name, args))
        yield

    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: True)
    monkeypatch.setattr(torch.profiler, "record_function", record)
    return captured


class _ScriptedTransceiver(FakeKvCacheTransceiver):
    """Adds contract branches that the shared strict fake deliberately excludes."""

    def __init__(self) -> None:
        super().__init__(123, supports_inflight_cancellation=True, consumes_transfer_buffer=False)
        self.cancels = iter([False, False, True])
        self.owned: bool | None = False
        self.submit_error: Exception | None = None
        self.shutdown_error: Exception | None = None
        self.send_terminal: str | None = None
        self.observed: set[int] = set()
        self.observed_status_ids: dict[int, int] = {}
        self.polls: list[tuple[str, int | None, bool]] = []

    @property
    def pipeline_transfer_enabled(self) -> bool:
        return True

    @property
    def receive_completion_uses_request_state(self) -> bool:
        return True

    def has_inflight_transfer(self, req: SimpleNamespace) -> bool:
        return bool(self.owned)

    def transfer_resource_ownership(self, req: SimpleNamespace) -> bool | None:
        return self.owned

    def has_retired_send_session(self, req: SimpleNamespace) -> bool:
        return True

    def has_poisoned_transfer_buffer(self) -> bool:
        return True

    def observe_send_terminal_outcome(self, req: SimpleNamespace, enabled: bool = True) -> None:
        """Track request identities whose scripted send outcomes may be
        observed.
        """
        if enabled:
            self.observed.add(id(req))
            self.observed_status_ids[id(req)] = req.py_request_id
        else:
            self.observed.discard(id(req))
            self.observed_status_ids.pop(id(req), None)

    def get_send_terminal_outcome(self, req: SimpleNamespace) -> str | None:
        return self.send_terminal

    def get_send_terminal_outcomes(self) -> dict[int, str]:
        """Return scripted outcomes only for sends with active observation
        claims.
        """
        if self.send_terminal is None:
            return {}
        return {rid: self.send_terminal for rid in self.observed_status_ids.values()}

    def respond_and_send_async(self, req: SimpleNamespace) -> None:
        """Record send submission, preserving a scripted admission failure."""
        self.call_log.append("send")
        if self.submit_error is not None:
            raise self.submit_error
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    def check_context_transfer_status(
        self, at_least_request_num: int | None, mark_complete: bool = False
    ) -> CtxTransferStatus:
        self.polls.append(("context", at_least_request_num, mark_complete))
        return CtxTransferStatus([], [])

    def cancel_request(self, req: SimpleNamespace) -> bool:
        return next(self.cancels)

    def get_data_transceiver_state(self) -> bytes:
        return b"native-state"

    def get_status_dump(self) -> str:
        return "native-status"

    def shutdown(self) -> None:
        """Record shutdown and propagate its scripted failure for retry
        tests.
        """
        self.call_log.append("shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.mark.parametrize("regions", [frozenset(), frozenset({"workflow", "mpi"})])
def test_disabled_wrapper_preserves_identity(
    monkeypatch: pytest.MonkeyPatch, regions: frozenset[str]
) -> None:
    backend = FakeKvCacheTransceiver()
    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", regions)
    wrap = getattr(factory, "_maybe_profile_kv_cache_transceiver", None)
    assert callable(wrap), "KV factory lacks disabled identity contract"
    assert wrap(backend, "NIXL", None) is backend


def test_context_completion_is_correlated_once_and_metadata_cached(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify one send completion reuses its submission metadata and aliases."""
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    req.prompt_len = 256
    backend.finish_send(req)
    status = wrapper.check_context_transfer_status(0)
    wrapper.check_context_transfer_status(0)
    assert status == CtxTransferStatus([17], [])
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_complete",
    ]
    assert events[0][1] is events[1][1]
    region, fields = decode_profiler_metadata(events[0][1])
    assert region == "kv"
    assert fields["request_id"] == 5 and fields["transfer_id"] == 17
    assert fields["prompt_tokens"] == 128
    assert (fields["role"], fields["direction"], fields["backend"], fields["rank"]) == (
        "context",
        "send",
        "NIXL",
        3,
    )
    assert not wrapper._records


@pytest.mark.parametrize(
    "outcome,event", [("complete", "complete"), ("error", "failed"), ("cancel", "cancelled")]
)
def test_python_receive_named_outcomes(
    monkeypatch: pytest.MonkeyPatch, events: list, outcome: str, event: str
) -> None:
    """Verify native completion IDs produce the matching receive terminal
    marker.
    """
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.request_and_receive_async(req)
    backend.finish_recv(req, outcome)
    result = wrapper.check_gen_transfer_status(0)
    wrapper.check_gen_transfer_status(0)
    if outcome == "cancel":
        assert result.cancelled_requests == [req]
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert events[-1][0] == f"trtllm::kv_transfer_{event}"
    assert len(events) == 2
    assert not wrapper._records


@pytest.mark.parametrize(
    "state,event",
    [
        (LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE, "complete"),
        (LlmRequestState.DISAGG_TRANS_ERROR, "failed"),
    ],
)
def test_cpp_void_receive_is_observed_from_request_mutation(
    monkeypatch: pytest.MonkeyPatch, events: list, state: LlmRequestState, event: str
) -> None:
    req = _request()
    impl = Mock()
    impl.check_gen_transfer_status.side_effect = lambda bound: setattr(req, "state", state)
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = impl
    backend.kv_transfer_timeout_ms = None
    wrapper = _wrap(backend, monkeypatch)
    wrapper.request_and_receive_async(req)
    result = wrapper.check_gen_transfer_status(0)
    assert result == GenTransferStatus([], [], [])
    assert events[-1][0] == f"trtllm::kv_transfer_{event}"
    assert not wrapper._records


@pytest.mark.parametrize("failed", [False, True])
def test_cancellation_false_keeps_observation_until_safe_true(
    monkeypatch: pytest.MonkeyPatch, events: list, failed: bool
) -> None:
    """Verify refused cancellation keeps the observation until safe
    cancellation.
    """
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    assert wrapper.cancel_request(req) is False
    assert wrapper.cancel_request(req) is False
    assert len(events) == 1 and id(req) in backend.observed
    if failed:
        req.state = LlmRequestState.DISAGG_TRANS_ERROR
    assert wrapper.cancel_request(req) is True
    assert events[-1][0] == "trtllm::kv_transfer_cancelled"
    assert not wrapper._records and not backend.observed


def test_v2_failed_receive_safe_cancellation_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify real V2 cancellation preserves the failed receive outcome once."""
    owner = object.__new__(KvCacheTransceiverV2)
    owner._fp4_mla_bridge_enabled = True
    owner._wait_reqs = {}
    owner._send_sessions = {}
    owner._recv_sessions = {}
    owner._recv_reqs = {}
    monkeypatch.setattr(
        owner,
        "request_and_receive_async",
        lambda request: setattr(
            request, "state", LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        ),
    )
    wrapper = _wrap(owner, monkeypatch)
    req = _request()
    wrapper.request_and_receive_async(req)
    session = Mock(status=SessionStatus.ERROR)
    session.has_transferring_tasks.return_value = False
    session.close.return_value = True
    owner._recv_sessions[17] = session
    owner._recv_reqs[17] = req
    assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert wrapper.has_inflight_transfer(req)

    assert wrapper.cancel_request(req) is True
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not wrapper.has_inflight_transfer(req)
    assert wrapper.cancel_request(req) is True
    session.cancel.assert_called_once_with()
    session.close.assert_called_once_with()
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_failed",
    ]
    assert events[0][1] is events[1][1]
    assert not wrapper._records


def test_cancelled_receive_status_preserves_existing_failure_once(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify a safely cancelled receive with an error still emits failure
    once.
    """
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.request_and_receive_async(req)
    req.state = LlmRequestState.DISAGG_TRANS_ERROR
    backend.finish_recv(req, "cancel")

    assert wrapper.check_gen_transfer_status(0) == GenTransferStatus([], [], [req])
    assert req.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert wrapper.check_gen_transfer_status(0) == GenTransferStatus([], [], [])
    assert wrapper.cancel_request(req) is True
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_failed",
    ]
    assert events[0][1] is events[1][1]
    assert not wrapper._records


def test_precise_owner_outcome_overrides_ambiguous_bridge_error(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    backend = _ScriptedTransceiver()
    backend.check_context_transfer_status = lambda bound, mark_complete=False: CtxTransferStatus(
        [], [17]
    )
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    backend.send_terminal = "cancelled"
    result = wrapper.check_context_transfer_status(0)
    assert result == CtxTransferStatus([], [17])
    assert events[-1][0] == "trtllm::kv_transfer_cancelled"
    assert not backend.observed


def test_retired_boolean_is_not_a_cancellation_oracle(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    wrapper.respond_and_send_async(_request())
    wrapper.check_context_transfer_status(0)
    assert len(events) == 1
    assert wrapper._records


@pytest.mark.parametrize("ownership,pending", [(False, False), (True, True), (None, True)])
def test_submission_failure_preserves_partial_or_unknown_admission(
    monkeypatch: pytest.MonkeyPatch, events: list, ownership: bool | None, pending: bool
) -> None:
    """Verify failed admission retains observations while ownership is
    unresolved.
    """
    backend = _ScriptedTransceiver()
    backend.owned = ownership
    backend.submit_error = RuntimeError("admission failed")
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(req)
    assert caught.value is backend.submit_error
    assert bool(wrapper._records) is pending
    assert bool(backend.observed) is pending
    if pending:
        backend.send_terminal = "failed"
        wrapper.check_context_transfer_status(0)
        assert events[-1][0] == "trtllm::kv_transfer_failed"
        assert not wrapper._records and not backend.observed


def test_shutdown_failure_preserves_observations_and_original_error_for_retry(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify failed shutdown retains observations and the original retryable
    error.
    """
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    wrapper.respond_and_send_async(_request())
    backend.shutdown_error = RuntimeError("physical owner remains")
    with pytest.raises(RuntimeError) as caught:
        wrapper.shutdown()
    assert caught.value is backend.shutdown_error
    assert wrapper._records and backend.observed and len(events) == 1
    backend.shutdown_error = None
    wrapper.shutdown()
    wrapper.shutdown()
    assert not wrapper._records and not backend.observed
    assert len(events) == 2 and events[-1][0] == "trtllm::kv_transfer_cancelled"
    assert backend.call_log.count("shutdown") == 3


def test_duplicates_still_delegate_without_duplicate_submit(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    wrapper.respond_and_send_async(req)
    assert backend.call_log == ["send", "send"]
    assert len(events) == 1


def test_backend_duplicate_rejection_preserves_original_pending_record(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify duplicate admission failure preserves the first pending
    observation.
    """
    wrapper = _wrap(FakeKvCacheTransceiver(), monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    with pytest.raises(AssertionError, match="double send"):
        wrapper.respond_and_send_async(req)
    assert len(events) == 1 and wrapper._records


def test_numeric_alias_domains_and_directions_do_not_overwrite(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify overlapping numeric aliases keep separate send and receive
    records.
    """
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    first, second = _request(5, 17), _request(17, 23)
    wrapper.respond_and_send_async(first)
    wrapper.respond_and_send_async(second)
    wrapper.request_and_receive_async(first)
    backend.finish_send(first)
    wrapper.check_context_transfer_status(0)
    assert len(wrapper._records) == 2
    backend.finish_send(second)
    wrapper.check_context_transfer_status(0)
    backend.finish_recv(first)
    wrapper.check_gen_transfer_status(0)
    assert not wrapper._records
    assert [
        decode_profiler_metadata(payload)[1]["request_id"]
        for name, payload in events
        if name.endswith("complete")
    ] == [5, 17, 5]


@pytest.mark.parametrize("transfer_id,ctx_id,want", [(None, 23, 23), (None, None, 5), (0, 23, 0)])
def test_transfer_fallback_does_not_change_status_identity(
    monkeypatch: pytest.MonkeyPatch,
    events: list,
    transfer_id: int | None,
    ctx_id: int | None,
    want: int,
) -> None:
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request(5, transfer_id, ctx_id)
    wrapper.respond_and_send_async(req)
    backend.finish_send(req)
    wrapper.check_context_transfer_status(0)
    assert decode_profiler_metadata(events[0][1])[1]["transfer_id"] == want
    assert not wrapper._records


@pytest.mark.parametrize("bound,mark", [(None, False), (0, True), (2, False)])
def test_poll_arguments_and_empty_record_calls_preserve_rank_symmetry(
    monkeypatch: pytest.MonkeyPatch, bound: int | None, mark: bool
) -> None:
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    status = wrapper.check_context_transfer_status(bound, mark)
    assert status == CtxTransferStatus([], [])
    assert backend.polls == [("context", bound, mark)]


def test_all_resource_capabilities_are_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the wrapper preserves every resource-owner capability result."""
    backend = _ScriptedTransceiver()
    backend.owned = True
    wrapper = _wrap(backend, monkeypatch)
    assert isinstance(wrapper, KvCacheTransceiver)
    assert wrapper.kv_transfer_timeout_ms == 123
    backend.kv_transfer_timeout_ms = None
    assert wrapper.kv_transfer_timeout_ms is None
    assert wrapper.pipeline_transfer_enabled and not wrapper.consumes_transfer_buffer
    req = _request()
    assert wrapper.has_inflight_transfer(req) and wrapper.has_retired_send_session(req)
    assert (
        wrapper.has_poisoned_transfer_buffer() and wrapper.supports_inflight_request_cancellation()
    )
    assert wrapper.get_data_transceiver_state() == b"native-state"
    assert wrapper.get_status_dump() == "native-status"
    assert wrapper.get_disaggregated_params() == {}
    assert wrapper.check_gen_transfer_complete()
    wrapper.prepare_context_requests([req])
    req.context_current_position = req.prompt_len
    wrapper.commit_blocks_for_reuse(req)
    assert backend.call_log[-2:] == ["prepare_context_requests:[17]", "commit_blocks_for_reuse:17"]


def test_wrapper_overrides_every_public_transceiver_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the wrapper explicitly implements the public transceiver
    contract.
    """
    wrapper = _wrap(FakeKvCacheTransceiver(), monkeypatch)
    public_members = {
        name
        for name, member in vars(KvCacheTransceiver).items()
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    missing = public_members - vars(type(wrapper)).keys()
    assert not missing, f"Wrapper inherits transceiver defaults: {sorted(missing)}"


def test_deferred_receive_counts_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Mock(spec=KvCacheTransceiver)
    backend.set_gen_deferred_receives.return_value = None
    wrapper = _wrap(backend, monkeypatch)
    assert wrapper.set_gen_deferred_receives(3) is None
    assert wrapper.set_gen_deferred_receives(0) is None
    assert backend.mock_calls == [
        ("set_gen_deferred_receives", (3,), {}),
        ("set_gen_deferred_receives", (0,), {}),
    ]


@pytest.mark.parametrize("supported", [False, True])
def test_idle_cancellation_capability_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, supported: bool
) -> None:
    backend = Mock(spec=KvCacheTransceiver)
    backend.supports_idle_request_cancellation.return_value = supported
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    assert wrapper.supports_idle_request_cancellation() is supported
    assert wrapper.supports_idle_request_cancellation(req=req) is supported
    assert backend.mock_calls == [
        ("supports_idle_request_cancellation", (None,), {}),
        ("supports_idle_request_cancellation", (req,), {}),
    ]


def test_deferred_receive_and_idle_cancellation_errors_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Mock(spec=KvCacheTransceiver)
    wrapper = _wrap(backend, monkeypatch)
    error = RuntimeError("backend failure")
    backend.set_gen_deferred_receives.side_effect = error
    backend.supports_idle_request_cancellation.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        wrapper.set_gen_deferred_receives(1)
    assert caught.value is error
    with pytest.raises(RuntimeError) as caught:
        wrapper.supports_idle_request_cancellation(_request())
    assert caught.value is error


@pytest.mark.parametrize("outcome,event", [("complete", "complete"), ("error", "failed")])
def test_sync_receive_settles_without_phantom_pending_record(
    monkeypatch: pytest.MonkeyPatch, events: list, outcome: str, event: str
) -> None:
    """Verify synchronous receive settles immediately without a pending
    record.
    """
    backend = FakeKvCacheTransceiver()
    req = _request()
    backend.script_sync_recv(req, outcome)
    wrapper = _wrap(backend, monkeypatch)
    wrapper.request_and_receive_sync(req)
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_sync",
        f"trtllm::kv_transfer_{event}",
    ]
    assert not wrapper._records


def test_inactive_profiler_never_extracts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    del req.prompt_len
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: False)
    encode = Mock(side_effect=AssertionError("inactive metadata"))
    monkeypatch.setattr(_profiler, "_encode_metadata", encode)
    wrapper.respond_and_send_async(req)
    assert not wrapper._records
    backend.finish_send(req)
    wrapper.check_context_transfer_status(0)
    encode.assert_not_called()
    assert not wrapper._records


@pytest.mark.parametrize("direction", ["send", "receive"])
@pytest.mark.parametrize("outcome", ["complete", "failed"])
def test_native_accepted_inflight_cancel_waits_for_terminal_status(
    monkeypatch: pytest.MonkeyPatch, events: list, direction: str, outcome: str
) -> None:
    """Verify accepted native cancellation waits for later terminal evidence."""
    monkeypatch.setattr(factory, "_disagg_inflight_cancel_enabled_cache", True)
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = Mock()
    backend.impl.cancel_request.return_value = True
    backend._supports_inflight_request_cancellation = True
    backend.kv_transfer_timeout_ms = 1000
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    req.state = (
        LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
        if direction == "send"
        else LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    )
    submit = (
        wrapper.respond_and_send_async if direction == "send" else wrapper.request_and_receive_async
    )
    submit(req)
    assert wrapper.cancel_request(req) is True
    assert len(events) == 1 and wrapper._records
    if direction == "send":
        backend.impl.check_context_transfer_status.return_value = (
            ([req.request_id], []) if outcome == "complete" else ([], [req.request_id])
        )
        wrapper.check_context_transfer_status(0)
    else:
        state = (
            LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            if outcome == "complete"
            else LlmRequestState.DISAGG_TRANS_ERROR
        )
        backend.impl.check_gen_transfer_status.side_effect = lambda bound: setattr(
            req, "state", state
        )
        wrapper.check_gen_transfer_status(0)
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        f"trtllm::kv_transfer_{outcome}",
    ]
    assert not wrapper._records


def test_factory_profiled_bridge_rejection_releases_coordinator_claim(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    from coordinator_harness import CoordinatorHarness, TransferRequest

    harness = CoordinatorHarness(kv_transfer_timeout_ms=1000)
    owner = object.__new__(KvCacheTransceiverV2)
    owner._fp4_mla_bridge_enabled = True
    owner._send_sessions = {}
    owner._send_reqs = {}
    owner._recv_sessions = {}
    owner._enable_pipelined_transfer = False
    owner.kv_transfer_timeout_ms = 1000
    constructor = Mock(return_value=owner)
    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", frozenset({"kv"}))
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2", constructor
    )
    wrapped = factory.create_kv_cache_transceiver(
        SimpleNamespace(rank=0),
        None,
        object.__new__(factory.KVCacheManagerV2),
        None,
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON"),
    )
    harness.coordinator._transceiver = wrapped
    request = TransferRequest(1, prompt_len=128, py_kv_send_session_retired=False)
    harness.active.append(request)
    harness.send(request)
    assert not harness.in_transfer(request)
    assert request.py_kv_transfer_start_time is None
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert not wrapped._records


def test_profile_window_end_keeps_inactive_calls_constant_time(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify inactive calls avoid scanning the emptied observation table."""

    class NoEmptyScan(dict):
        def values(self) -> ValuesView[object]:
            assert self, "inactive empty-record traversal"
            return super().values()

    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    wrapper._records = NoEmptyScan()
    send = _request(5, 17)
    receive = _request(6, 18)
    references = [weakref.ref(send), weakref.ref(receive)]
    wrapper.respond_and_send_async(send)
    wrapper.request_and_receive_async(receive)
    assert len(wrapper._records_by_status) == 2
    assert len(wrapper._receive_state_records) == 1
    assert backend.observed == {id(send)}
    backend.finish_recv(receive)
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: False)
    assert wrapper.check_gen_transfer_status(0).completed_request_ids == [18]
    assert not wrapper._records
    assert not wrapper._records_by_status
    assert not wrapper._receive_state_records
    assert not backend.observed and not backend.observed_status_ids
    del send, receive
    gc.collect()
    assert all(reference() is None for reference in references)

    for _ in range(3):
        wrapper.check_context_transfer_status(0)
        wrapper.check_gen_transfer_status(0)
    inactive = _request(7, 19)
    wrapper.respond_and_send_async(inactive)
    assert not wrapper._records and not backend.observed
    assert [name for name, _ in events] == ["trtllm::kv_transfer_submit"] * 2

    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: True)
    send = _request(8, 20)
    receive = _request(9, 21)
    wrapper.respond_and_send_async(send)
    wrapper.request_and_receive_async(receive)
    assert len(wrapper._records_by_status) == 2
    assert len(wrapper._receive_state_records) == 1
    backend.send_terminal = "complete"
    backend.finish_recv(receive)
    wrapper.check_context_transfer_status(0)
    wrapper.check_gen_transfer_status(0)
    assert not wrapper._records
    assert not wrapper._records_by_status
    assert not wrapper._receive_state_records
    assert not backend.observed and not backend.observed_status_ids
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_complete",
        "trtllm::kv_transfer_complete",
    ]


@pytest.mark.parametrize("direction", ["send", "receive"])
def test_pending_python_transfers_are_indexed_without_full_observation_scans(
    monkeypatch: pytest.MonkeyPatch, events: list, direction: str
) -> None:
    """Verify Python status polling uses indexes instead of scanning all
    records.
    """

    class NoFullScan(dict):
        def values(self):
            raise AssertionError("full pending-record scan")

    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    submit = (
        wrapper.respond_and_send_async if direction == "send" else wrapper.request_and_receive_async
    )
    pending = [_request(rid, rid + 10000) for rid in range(1024)]
    for req in pending:
        submit(req)
    wrapper._records = NoFullScan(wrapper._records)
    extra = _request(1024, 11024)
    submit(extra)
    wrapper.check_context_transfer_status(0)
    wrapper.check_gen_transfer_status(0)
    if direction == "send":
        backend.finish_send(extra)
        wrapper.check_context_transfer_status(0)
    else:
        backend.finish_recv(extra)
        wrapper.check_gen_transfer_status(0)
    assert len(wrapper._records) == 1024
    assert len(wrapper._records_by_status) == 1024
    assert (direction, 11024) not in wrapper._records_by_status


def test_observation_registration_failure_warns_once_and_preserves_admission(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    from tensorrt_llm.logger import logger

    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    warnings = Mock()
    monkeypatch.setattr(logger, "log", warnings)
    monkeypatch.setattr(logger, "_appeared_keys", set())
    req = _request()
    del req.py_disaggregated_params
    wrapper.respond_and_send_async(req)
    wrapper.respond_and_send_async(req)
    assert backend.call_log == ["send", "send"]
    assert not wrapper._records
    warnings.assert_called_once()
    assert "registration" in warnings.call_args.args[1]


def test_send_outcome_observation_failure_warns_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    backend.get_send_terminal_outcomes = Mock(side_effect=RuntimeError("outcome observation"))
    result = CtxTransferStatus([17], [])
    backend.check_context_transfer_status = lambda bound, mark_complete=False: result
    warn = Mock()
    monkeypatch.setattr(_profiler, "warn_profiler_failure", warn)
    assert wrapper.check_context_transfer_status(0) is result
    assert events[-1][0] == "trtllm::kv_transfer_complete"
    assert warn.call_args.args[:3] == ("trtllm::kv_transfer", "kv", "send outcomes")


def test_real_cpu_profiler_captures_paired_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the real CPU profiler records paired submit and completion
    markers.
    """
    backend = FakeKvCacheTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as capture:
        wrapper.respond_and_send_async(req)
        backend.finish_send(req)
        wrapper.check_context_transfer_status(0)
    names = [event.name for event in capture.events() if event.name.startswith("trtllm::kv")]
    assert names == ["trtllm::kv_transfer_submit", "trtllm::kv_transfer_complete"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("runtime", ["CPP", "PYTHON"])
def test_factory_wraps_after_unchanged_runtime_selection(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, runtime: str
) -> None:
    """Verify profiling wraps the resource owner after normal runtime
    selection.
    """
    backend = FakeKvCacheTransceiver()
    constructor = Mock(return_value=backend)
    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", frozenset({"kv"}) if enabled else frozenset())
    monkeypatch.setattr(factory, "BindKvCacheTransceiver", constructor)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2", constructor
    )
    mapping, dist, attention = SimpleNamespace(rank=3), Mock(), Mock()
    manager = Mock(spec=factory.KVCacheManagerV2) if runtime == "PYTHON" else Mock()
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime=runtime)
    result = factory.create_kv_cache_transceiver(mapping, dist, manager, attention, config)
    assert (result is backend) is not enabled
    expected = (
        (mapping, dist, manager, config)
        if runtime == "PYTHON"
        else (mapping, dist, manager, attention, config, None)
    )
    assert constructor.call_args.args == expected


@pytest.mark.parametrize("body_error", [None, RuntimeError("body"), KeyboardInterrupt("body")])
@pytest.mark.parametrize("active", [False, True])
def test_factory_wrapper_context_manager_preserves_shutdown_and_body_error(
    monkeypatch: pytest.MonkeyPatch, events: list, active: bool, body_error: BaseException | None
) -> None:
    """Return the factory wrapper and shut down once without suppressing body errors."""
    backend = _ScriptedTransceiver()
    monkeypatch.setattr(_profiler, "_CUSTOM_REGIONS", frozenset({"kv"}))
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: active)
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2",
        Mock(return_value=backend),
    )
    wrapper = factory.create_kv_cache_transceiver(
        SimpleNamespace(rank=3),
        Mock(),
        Mock(spec=factory.KVCacheManagerV2),
        Mock(),
        CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON"),
    )
    wrapper_shutdown = Mock(wraps=wrapper.shutdown)
    backend_shutdown = Mock(wraps=backend.shutdown)
    monkeypatch.setattr(wrapper, "shutdown", wrapper_shutdown)
    monkeypatch.setattr(backend, "shutdown", backend_shutdown)
    expected = pytest.raises(type(body_error)) if body_error is not None else nullcontext()
    with expected as caught:
        with wrapper as entered:
            assert entered is wrapper and entered is not backend
            entered.respond_and_send_async(_request())
            if body_error is not None:
                raise body_error
    if body_error is not None:
        assert caught.value is body_error
    wrapper_shutdown.assert_called_once_with()
    backend_shutdown.assert_called_once_with()
    assert backend.call_log == ["send", "shutdown"]
    assert not wrapper._records and not backend.observed
    assert [name for name, _ in events] == (
        ["trtllm::kv_transfer_submit", "trtllm::kv_transfer_cancelled"] if active else []
    )


def test_v2_outcome_observation_requires_opt_in_and_safe_close() -> None:
    """Verify V2 records terminal outcomes only after opt-in and safe
    closure.
    """
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    session = Mock()
    session._enforce_physical_ownership = True
    session.resources_drained.return_value = False
    owner._send_sessions = {17: session}
    owner._send_reqs = {17: req}
    getter = getattr(owner, "get_send_terminal_outcome", None)
    assert callable(getter), "safe retirement has no precise observational outcome"
    assert getter(req) is None
    owner.observe_send_terminal_outcome(req)
    assert owner.get_send_terminal_outcomes() == {}
    with pytest.raises(RuntimeError, match="physical resources"):
        owner._retire_send_session(17, outcome="cancelled")
    assert getter(req) is None and not req.py_kv_send_session_retired
    session.resources_drained.return_value = True
    owner._retire_send_session(17, outcome="cancelled")
    assert getter(req) == "cancelled"
    assert owner.get_send_terminal_outcomes() == {17: "cancelled"}
    owner.observe_send_terminal_outcome(req, enabled=False)
    assert getter(req) is None
    assert owner.get_send_terminal_outcomes() == {}


def test_v2_failed_close_reports_failure_not_cancellation() -> None:
    """Verify V2 failed-session closure produces the failure terminal
    outcome.
    """
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    getter = getattr(owner, "get_send_terminal_outcome", None)
    assert callable(getter), "failed close has no precise observational outcome"
    session = Mock()
    session._enforce_physical_ownership = False
    owner.observe_send_terminal_outcome(req)
    owner._close_failed_sessions({17: session}, {17: req}, [17], mark_retired=True)
    assert req.py_kv_send_session_retired and getter(req) == "failed"


def test_v2_pipeline_retirement_is_visible_before_final_request_registration() -> None:
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    session = Mock()
    session._enforce_physical_ownership = False
    owner._send_sessions = {17: session}
    owner._send_reqs = {}
    owner.observe_send_terminal_outcome(req)
    owner._retire_send_session(17, outcome="cancelled")
    assert owner.get_send_terminal_outcome(req) == "cancelled"


def test_v2_actual_partial_send_admission_keeps_record_until_safe_retirement(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify partial real send admission stays observed until safe
    retirement.
    """
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    req.set_kv_cache_transfer_start = lambda value: None
    session = Mock()
    session._enforce_physical_ownership = False
    owner._send_sessions = {}
    owner._send_reqs = {}
    owner._recv_sessions = {}
    owner._enable_pipelined_transfer = False
    owner._fp4_mla_bridge_enabled = False
    owner._transfer_worker = SimpleNamespace(create_tx_session=lambda request: session)
    owner._create_cache_extent = lambda request: SimpleNamespace(
        local=SimpleNamespace(is_last=False)
    )
    error = RuntimeError("publication partially admitted")

    def publish(extent: object) -> None:
        raise error

    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.PeerPublish",
        lambda native_session, request: SimpleNamespace(publish=publish),
    )
    wrapper = _wrap(owner, monkeypatch)
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(req)
    assert caught.value is error and wrapper._records
    assert owner.transfer_resource_ownership(req) is True
    owner._retire_send_session(17, outcome="cancelled")
    monkeypatch.setattr(
        owner,
        "check_context_transfer_status",
        lambda bound, mark_complete=False: CtxTransferStatus([], []),
    )
    wrapper.check_context_transfer_status(0)
    assert events[-1][0] == "trtllm::kv_transfer_cancelled" and not wrapper._records


def test_v2_rejected_receive_admission_cleans_observation(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify rejected receive admission removes its observation after safe
    cleanup.
    """
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    req.set_kv_cache_transfer_start = lambda value: None
    owner._send_sessions = {}
    owner._recv_sessions = {}
    owner._fp4_mla_bridge_enabled = False
    error = RuntimeError("destination admission rejected")

    def reject(request: object) -> None:
        raise error

    owner._create_cache_extent = reject
    wrapper = _wrap(owner, monkeypatch)
    with pytest.raises(RuntimeError) as caught:
        wrapper.request_and_receive_async(req)
    assert caught.value is error and not wrapper._records
    assert owner.transfer_resource_ownership(req) is False
    assert events[-1][0] == "trtllm::kv_transfer_failed"


def test_v2_no_opt_in_does_not_retain_terminal_outcomes() -> None:
    owner = object.__new__(KvCacheTransceiverV2)
    req = _request()
    session = Mock()
    session._enforce_physical_ownership = False
    owner._send_sessions = {17: session}
    owner._send_reqs = {17: req}
    getter = getattr(owner, "get_send_terminal_outcome", None)
    assert callable(getter), "disabled outcome observation contract missing"
    owner._retire_send_session(17, outcome="completed")
    assert getter(req) is None


def test_unknown_native_admission_does_not_strongly_retain_failed_requests(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify uncertain native admission does not keep failed requests alive."""
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = Mock()
    error = RuntimeError("native admission is unknown")
    backend.impl.respond_and_send_async.side_effect = error
    backend.kv_transfer_timeout_ms = None
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    reference = weakref.ref(req)
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(req)
    assert caught.value is error and wrapper._records
    del caught, error
    backend.impl.reset_mock(side_effect=True)
    gc.collect()
    del req
    gc.collect()
    assert reference() is None
    assert not wrapper._records
    assert len(events) == 1


@pytest.mark.parametrize("outcome", ["complete", "failed"])
@pytest.mark.parametrize("new_request_id", [5, 6])
def test_uncertain_native_generation_cannot_consume_reused_status_id(
    monkeypatch: pytest.MonkeyPatch, events: list, outcome: str, new_request_id: int
) -> None:
    """Verify a reused native status ID cannot settle an uncertain older
    generation.
    """
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = Mock()
    backend.kv_transfer_timeout_ms = None
    error = RuntimeError("unknown partial native admission")
    backend.impl.respond_and_send_async.side_effect = error
    wrapper = _wrap(backend, monkeypatch)
    old = _request(5, 17)
    new = _request(new_request_id, 23)
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(old)
    assert caught.value is error
    backend.impl.respond_and_send_async.side_effect = None
    wrapper.respond_and_send_async(new)
    result = ([new_request_id], []) if outcome == "complete" else ([], [new_request_id])
    backend.impl.check_context_transfer_status.return_value = result
    wrapper.check_context_transfer_status(0)
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_submit",
        f"trtllm::kv_transfer_{outcome}",
    ]
    fields = [decode_profiler_metadata(payload)[1] for _, payload in events]
    assert [(field["request_id"], field["transfer_id"]) for field in fields] == [
        (5, 17),
        (new_request_id, 23),
        (new_request_id, 23),
    ]
    assert backend.impl.method_calls == [
        ("respond_and_send_async", (old,), {}),
        ("respond_and_send_async", (new,), {}),
        ("check_context_transfer_status", (0, False), {}),
    ]
    # Keep the old object alive: weak-reference expiry cannot hide the collision.
    assert old.request_id == 5
    assert len(wrapper._records) == (0 if new_request_id == 5 else 1)


def test_uncertain_native_send_does_not_shadow_reused_receive_domain(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify an uncertain send cannot hide a receive with the same numeric
    alias.
    """
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = Mock()
    backend.kv_transfer_timeout_ms = None
    error = RuntimeError("unknown send admission")
    backend.impl.respond_and_send_async.side_effect = error
    wrapper = _wrap(backend, monkeypatch)
    old = _request(5, 17)
    new = _request(5, 23)
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(old)
    assert caught.value is error
    wrapper.request_and_receive_async(new)
    new.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    backend.impl.check_gen_transfer_status.return_value = None
    wrapper.check_gen_transfer_status(0)
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_complete",
    ]
    assert decode_profiler_metadata(events[-1][1])[1]["direction"] == "receive"
    assert len(wrapper._records) == 1 and old.request_id == 5


def test_profile_metadata_failure_never_skips_transfer_or_replaces_error(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify metadata failure preserves admission and the original backend
    error.
    """
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    del req.prompt_len
    error = RuntimeError("backend admission")
    backend.submit_error = error
    with pytest.raises(RuntimeError) as caught:
        wrapper.respond_and_send_async(req)
    assert caught.value is error and backend.call_log == ["send"]
    assert not wrapper._records


def test_reused_request_after_terminal_gets_a_fresh_observation(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify a reused request receives a new observation after terminal
    cleanup.
    """
    backend = _ScriptedTransceiver()
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.respond_and_send_async(req)
    backend.send_terminal = "complete"
    wrapper.check_context_transfer_status(0)
    backend.send_terminal = None
    req.prompt_len = 256
    wrapper.respond_and_send_async(req)
    backend.send_terminal = "failed"
    wrapper.check_context_transfer_status(0)
    assert [name for name, _ in events] == [
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_complete",
        "trtllm::kv_transfer_submit",
        "trtllm::kv_transfer_failed",
    ]
    assert decode_profiler_metadata(events[2][1])[1]["prompt_tokens"] == 256


def test_receive_terminal_state_does_not_override_live_resource_owner(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify terminal request state cannot settle a still-owned receive
    session.
    """
    backend = _ScriptedTransceiver()
    backend.owned = True
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    wrapper.request_and_receive_async(req)
    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    backend.check_gen_transfer_status = lambda bound: GenTransferStatus([], [], [])
    wrapper.check_gen_transfer_status(0)
    assert len(events) == 1 and wrapper._records
    backend.owned = False
    wrapper.check_gen_transfer_status(0)
    assert events[-1][0] == "trtllm::kv_transfer_complete" and not wrapper._records


@pytest.mark.parametrize("bound", [None, 0, 2])
def test_receive_status_arguments_and_result_identity_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, bound: int | None
) -> None:
    backend = _ScriptedTransceiver()
    result = GenTransferStatus([], [], [])
    calls: list[int | None] = []

    def status(value: int | None) -> GenTransferStatus:
        calls.append(value)
        return result

    backend.check_gen_transfer_status = status
    wrapper = _wrap(backend, monkeypatch)
    assert wrapper.check_gen_transfer_status(bound) is result
    assert calls == [bound]


def test_unknown_admission_cannot_reuse_a_stale_terminal_request_state(
    monkeypatch: pytest.MonkeyPatch, events: list
) -> None:
    """Verify stale terminal state cannot settle uncertain new admission."""
    backend = object.__new__(BindKvCacheTransceiver)
    backend.impl = Mock()
    backend.impl.respond_and_send_async.side_effect = RuntimeError("unknown admission")
    backend.impl.check_gen_transfer_status.return_value = None
    backend.kv_transfer_timeout_ms = None
    wrapper = _wrap(backend, monkeypatch)
    req = _request()
    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    backend.impl.request_and_receive_async.side_effect = RuntimeError("unknown admission")
    with pytest.raises(RuntimeError):
        wrapper.request_and_receive_async(req)
    wrapper.check_gen_transfer_status(0)
    assert wrapper._records and len(events) == 1
