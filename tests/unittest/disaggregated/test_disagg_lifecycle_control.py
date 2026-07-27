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

import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.disaggregation.handoff import HandoffEventState, HandoffLifecycleEvent
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle, TransceiverLifecycleAdvertisement
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_lifecycle_control import (
    CONTEXT_ARTIFACT_ABORT_PATH,
    ArtifactControlAction,
    ArtifactObligationRequest,
    ContextArtifactAbortRequest,
    DisaggLifecycleControl,
    GenerationGrantAbortRequest,
    GenerationGrantRenewRequest,
    GenerationGrantRequest,
    RequestLifecycleMetadata,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1.0

    def __call__(self) -> float:
        return self.now


class _Promise:
    def __init__(self) -> None:
        self.abort_count = 0
        self._handoff_ready = asyncio.Event()
        self._disagg_handoff_event = None

    def abort(self) -> None:
        self.abort_count += 1

    def publish_handoff(self, event: HandoffLifecycleEvent) -> None:
        self._disagg_handoff_event = event
        self._handoff_ready.set()

    async def _wait_disagg_handoff_event(self) -> HandoffLifecycleEvent:
        await self._handoff_ready.wait()
        return self._disagg_handoff_event


class _Response:
    def __init__(self, payload: dict) -> None:
        self.status = 200
        self._body = json.dumps(payload)

    async def text(self) -> str:
        return self._body


class _ResponseContext:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self) -> _Response:
        return self._response

    async def __aexit__(self, *_args) -> None:
        return None


class _Session:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads

    def post(self, *_args, **_kwargs) -> _ResponseContext:
        return _ResponseContext(_Response(self._payloads.pop(0)))


class _ContextControlSession:
    def __init__(self, control: DisaggLifecycleControl) -> None:
        self._control = control
        self.requests: list[ContextArtifactAbortRequest] = []

    def post(self, url: str, *, json: dict) -> _ResponseContext:
        assert url.endswith(CONTEXT_ARTIFACT_ABORT_PATH)
        request = ContextArtifactAbortRequest.model_validate(json)
        self.requests.append(request)
        response = self._control.abort_context_artifact(request)
        return _ResponseContext(_Response(response.model_dump(mode="json")))


class _ArtifactPeerSession:
    def __init__(
        self,
        clock: _Clock,
        renewal_attempts: list[tuple[float, bool]],
    ) -> None:
        self._clock = clock
        self._renewal_attempts = renewal_attempts
        self.renewal_calls = 0

    def post(self, *_args, **kwargs) -> _ResponseContext:
        action = kwargs["json"]["action"]
        if action == ArtifactControlAction.RENEW.value:
            if self.renewal_calls < len(self._renewal_attempts):
                attempt_time_s, succeeds = self._renewal_attempts[self.renewal_calls]
            else:
                attempt_time_s, succeeds = self._clock.now, True
            self.renewal_calls += 1
            self._clock.now = attempt_time_s
            if not succeeds:
                raise OSError("CTX endpoint is unreachable")
            state = "ACTIVE"
        elif action == ArtifactControlAction.RELEASE.value:
            state = "RELEASED"
        else:
            state = "ABANDONED"
        return _ResponseContext(
            _Response(
                _artifact_response(
                    _uuid(31),
                    state=state,
                )
            )
        )


def _uuid(value: int) -> UUID:
    return UUID(int=value)


def _advertisement() -> TransceiverLifecycleAdvertisement:
    return TransceiverLifecycleAdvertisement(
        protocol_version=1,
        capabilities=(),
        qualified_legacy_mode=False,
        backend="python",
        instance_id=str(_uuid(91)),
        world_size=1,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        attention_dp=False,
    )


_GRANT_ADMISSION_FIELDS = {
    "context_control_endpoint",
    "context_transceiver_lifecycle",
    "ctx_dp_rank",
    "schedule_style",
}
_GRANT_REQUEST_ONLY_FIELDS = {
    *_GRANT_ADMISSION_FIELDS,
    "ttl_s",
}


def _grant_request(
    grant: int = 5,
    *,
    context_control_endpoint: str | None = None,
) -> GenerationGrantRequest:
    return GenerationGrantRequest(
        lifecycle_protocol_version=1,
        logical_request_id=17,
        prefill_artifact_id=_uuid(2),
        artifact_version=0,
        handoff_attempt_uuid=_uuid(3),
        consumer_grant_id=_uuid(grant),
        transfer_session_id=_uuid(6),
        ttl_s=60.0,
        context_control_endpoint=context_control_endpoint,
        context_transceiver_lifecycle=_advertisement(),
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )


def _params(request: GenerationGrantRequest, decision) -> SimpleNamespace:
    return SimpleNamespace(
        logical_request_id=request.logical_request_id,
        prefill_artifact_id=request.prefill_artifact_id,
        artifact_version=request.artifact_version,
        handoff_attempt_uuid=request.handoff_attempt_uuid,
        consumer_grant_id=request.consumer_grant_id,
        transfer_session_id=request.transfer_session_id,
        generation_endpoint_name=decision.generation_endpoint_name,
        generation_endpoint_rank=decision.generation_endpoint_rank,
        generation_endpoint_incarnation=decision.generation_endpoint_incarnation,
        context_control_endpoint=request.context_control_endpoint,
        context_transceiver_lifecycle=request.context_transceiver_lifecycle,
        schedule_style=request.schedule_style,
        ctx_dp_rank=request.ctx_dp_rank,
    )


def _abort_request(
    request: GenerationGrantRequest,
    decision,
) -> GenerationGrantAbortRequest:
    return GenerationGrantAbortRequest(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        generation_endpoint_name=decision.generation_endpoint_name,
        generation_endpoint_rank=decision.generation_endpoint_rank,
        generation_endpoint_incarnation=decision.generation_endpoint_incarnation,
    )


def _artifact_response(
    incarnation: UUID,
    *,
    accepted: bool = True,
    state: str = "ACTIVE",
) -> dict:
    return {
        "lifecycle_protocol_version": 1,
        "accepted": accepted,
        "state": state,
        "reason": "",
        "ttl_s": 60.0 if state == "ACTIVE" else None,
        "context_endpoint_incarnation": str(incarnation),
    }


@pytest.mark.asyncio
async def test_gen_grant_is_idempotent_and_consumed_at_scheduler_insertion() -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        clock=clock,
    )
    request = _grant_request()

    first = control.issue_generation_grant(request)
    replay = control.issue_generation_grant(request)

    assert first.accepted
    assert replay == first
    promise = _Promise()
    params = _params(request, first)
    control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[request.consumer_grant_id]
    promise.publish_handoff(
        HandoffLifecycleEvent.from_params(
            params,
            HandoffEventState.HANDOFF_COMMITTED,
        )
    )
    await ticket.handoff_task
    await control.finish_generation(params, success=True)

    replacement = control.issue_generation_grant(_grant_request(7))
    assert replacement.accepted
    assert promise.abort_count == 0


def test_endpoint_owns_grant_ttl_and_replay_does_not_extend_it() -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=10.0,
        clock=clock,
    )
    request = _grant_request()
    request.ttl_s = 999.0

    first = control.issue_generation_grant(request)
    clock.now = 4.0
    request.ttl_s = 1.0
    replay = control.issue_generation_grant(request)

    assert first.ttl_s == 10.0
    assert replay.ttl_s == 7.0


def test_control_protocol_version_is_required_and_must_be_one() -> None:
    payload = _grant_request().model_dump()
    payload.pop("lifecycle_protocol_version")
    with pytest.raises(ValidationError):
        GenerationGrantRequest.model_validate(payload)

    payload["lifecycle_protocol_version"] = 2
    with pytest.raises(ValidationError):
        GenerationGrantRequest.model_validate(payload)


def test_generation_grant_rejects_when_gen_owns_the_only_queue_credit() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )

    assert control.issue_generation_grant(_grant_request()).accepted
    rejected = control.issue_generation_grant(_grant_request(7))

    assert not rejected.accepted
    assert "credit" in rejected.reason


@pytest.mark.asyncio
async def test_explicit_grant_abort_releases_credit_and_aborts_bound_request() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    promise = _Promise()
    control.mark_generation_scheduler_inserted(
        _params(request, decision),
        promise,
    )

    response = await control.abort_generation_grant(_abort_request(request, decision))

    assert response.accepted
    assert response.state == "REVOKED"
    assert promise.abort_count == 1
    assert control.issue_generation_grant(_grant_request(7)).accepted


@pytest.mark.asyncio
async def test_conflicting_grant_abort_preserves_the_exact_bound_request() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    promise = _Promise()
    control.mark_generation_scheduler_inserted(
        _params(request, decision),
        promise,
    )
    conflicting = _abort_request(request, decision).model_copy(
        update={"generation_endpoint_incarnation": _uuid(99)}
    )

    with pytest.raises(RuntimeError, match="conflicting generation identity"):
        await control.abort_generation_grant(conflicting)

    assert request.consumer_grant_id in control._generation_tickets
    assert promise.abort_count == 0
    response = await control.abort_generation_grant(_abort_request(request, decision))
    assert response.accepted
    assert request.consumer_grant_id not in control._generation_tickets
    assert promise.abort_count == 1


@pytest.mark.asyncio
async def test_conflicting_generation_finish_preserves_the_exact_bound_request() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    promise = _Promise()
    params = _params(request, decision)
    control.mark_generation_scheduler_inserted(params, promise)
    conflicting_params = _params(request, decision)
    conflicting_params.generation_endpoint_incarnation = _uuid(99)

    with pytest.raises(ValueError, match="different bound request"):
        await control.finish_generation(conflicting_params, success=False)

    assert request.consumer_grant_id in control._generation_tickets
    assert promise.abort_count == 0
    response = await control.abort_generation_grant(_abort_request(request, decision))
    assert response.accepted
    assert request.consumer_grant_id not in control._generation_tickets
    assert promise.abort_count == 1


@pytest.mark.asyncio
async def test_abort_cleanup_only_allows_the_exact_concurrent_finish() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    promise = _Promise()
    params = _params(request, decision)
    control.mark_generation_scheduler_inserted(params, promise)
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    stop_ticket_tasks = control._stop_ticket_tasks

    async def block_ticket_cleanup(ticket) -> None:
        cleanup_started.set()
        await allow_cleanup.wait()
        await stop_ticket_tasks(ticket)

    control._stop_ticket_tasks = block_ticket_cleanup
    abort_task = asyncio.create_task(
        control.abort_generation_grant(_abort_request(request, decision))
    )
    await cleanup_started.wait()
    conflicting_params = _params(request, decision)
    conflicting_params.generation_endpoint_incarnation = _uuid(99)

    with pytest.raises(RuntimeError, match="conflicting generation identity"):
        await control.finish_generation(conflicting_params, success=False)

    assert request.consumer_grant_id in control._grant_sessions
    await control.finish_generation(params, success=True)
    assert request.consumer_grant_id not in control._grant_sessions
    allow_cleanup.set()
    response = await abort_task
    assert response.accepted
    assert request.consumer_grant_id not in control._grant_sessions
    assert promise.abort_count == 1


@pytest.mark.asyncio
async def test_conflicting_preinsertion_abort_does_not_cancel_context() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    scheduled_aborts = []
    control._schedule_context_artifact_abort = lambda *args, **kwargs: scheduled_aborts.append(
        (args, kwargs)
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    conflicting = _abort_request(request, decision).model_copy(
        update={"generation_endpoint_incarnation": _uuid(99)}
    )

    with pytest.raises(RuntimeError, match="conflicting generation identity"):
        await control.abort_generation_grant(conflicting)

    assert scheduled_aborts == []
    assert request.consumer_grant_id in control._grant_sessions
    response = await control.abort_generation_grant(_abort_request(request, decision))
    assert response.accepted
    assert len(scheduled_aborts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_trigger", ["admission", "sweep"])
async def test_grant_expiry_retires_every_bound_generation_resource(
    expiry_trigger: str,
) -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=2.0,
        clock=clock,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    promise = _Promise()
    params = _params(request, decision)
    metadata = control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[request.consumer_grant_id]
    clock.now = 3.0

    replacement = None
    if expiry_trigger == "admission":
        replacement = control.issue_generation_grant(_grant_request(7))
    else:
        control._sweep_once()
    await asyncio.gather(
        *(task for task in (ticket.renewal_task, ticket.handoff_task) if task is not None),
        return_exceptions=True,
    )

    assert promise.abort_count == 1
    assert ticket.commit_gate.state.value == "ABORTED"
    assert ticket.renewal_task is not None
    assert ticket.renewal_task.cancelled()
    assert ticket.handoff_task is not None
    assert ticket.handoff_task.cancelled()
    assert request.consumer_grant_id not in control._generation_tickets
    assert request.consumer_grant_id not in control._grant_sessions
    assert request.consumer_grant_id not in control._grant_requests
    assert request.consumer_grant_id not in control._context_endpoint_incarnations
    assert not control._generation_grants.scheduler_inserted(metadata.grant)
    if replacement is None:
        replacement = control.issue_generation_grant(_grant_request(7))
    assert replacement.accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action", ["expiry", "revocation"])
async def test_terminal_grant_before_gen_insertion_aborts_active_context(
    terminal_action: str,
) -> None:
    clock = _Clock()
    context_control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        endpoint_lifecycle=_advertisement,
        max_live_generation_grants=1,
        clock=clock,
    )
    session = _ContextControlSession(context_control)
    generation_control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=2.0,
        clock=clock,
        session=session,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    request.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
    context_params = SimpleNamespace(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint=request.context_control_endpoint,
    )
    context_promise = _Promise()
    context_control.mark_context_scheduler_inserted(
        context_params,
        context_promise,
    )
    decision = generation_control.issue_generation_grant(request)

    if terminal_action == "expiry":
        clock.now = 3.0
        generation_control._sweep_once()
        await asyncio.gather(
            *tuple(generation_control._cleanup_tasks),
            return_exceptions=True,
        )
    else:
        await generation_control.abort_generation_grant(_abort_request(request, decision))
        await asyncio.gather(
            *tuple(generation_control._cleanup_tasks),
            return_exceptions=True,
        )

    assert context_promise.abort_count == 1
    assert len(session.requests) == 1
    assert session.requests[0].context_endpoint_incarnation == context_control.endpoint_incarnation
    with pytest.raises(RuntimeError, match="ABANDONED"):
        context_control.register_context_artifact(
            context_params,
            context_promise,
        )
    assert request.consumer_grant_id not in context_control._artifact_tickets
    assert request.consumer_grant_id not in context_control._context_attempt_tickets


@pytest.mark.asyncio
async def test_stale_generation_endpoint_cannot_consume_an_admitted_grant() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    params.generation_endpoint_incarnation = _uuid(99)

    with pytest.raises(RuntimeError, match="conflicting generation identity"):
        control.mark_generation_scheduler_inserted(params, _Promise())

    assert request.consumer_grant_id in control._grant_sessions
    assert request.consumer_grant_id in control._grant_requests
    promise = _Promise()
    correct_params = _params(request, decision)
    control.mark_generation_scheduler_inserted(correct_params, promise)
    response = await control.abort_generation_grant(_abort_request(request, decision))
    assert response.accepted
    assert promise.abort_count == 1


def test_serialized_generation_endpoint_incarnation_is_accepted() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    params.generation_endpoint_incarnation = str(params.generation_endpoint_incarnation)

    metadata = control.validate_generation_grant_active(params)

    assert (
        str(metadata.grant.generation_endpoint.incarnation)
        == params.generation_endpoint_incarnation
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_control_endpoint", "ctx-other.example:8000"),
        ("schedule_style", DisaggScheduleStyle.GENERATION_FIRST),
        ("ctx_dp_rank", 1),
    ],
)
def test_scheduler_submission_must_match_the_admitted_request_contract(
    field: str,
    value: object,
) -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    setattr(params, field, value)

    with pytest.raises(ValueError, match="conflicts with its admitted"):
        control.validate_generation_grant_active(params)
    with pytest.raises(ValueError, match="conflicts with its admitted"):
        control.mark_generation_scheduler_inserted(params, _Promise())


def test_artifact_renewal_before_context_ready_is_applied_on_registration() -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
        artifact_ttl_s=5.0,
        artifact_renew_interval_s=1.0,
        clock=clock,
    )
    request = _grant_request()
    renew = ArtifactObligationRequest(
        **request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name="gen.example:8000",
        generation_endpoint_rank=0,
        generation_endpoint_incarnation=_uuid(8),
        action=ArtifactControlAction.RENEW,
        sequence=4,
    )

    response = control.handle_artifact_obligation(renew)
    params = SimpleNamespace(
        **renew.model_dump(exclude={"ttl_s", "action", "sequence"}),
        context_control_endpoint="ctx.example:8000",
    )
    control.register_context_artifact(params, _Promise())

    assert response.accepted
    assert response.state == "ACTIVE"
    assert response.ttl_s == 5.0
    assert response.context_endpoint_incarnation == control.endpoint_incarnation


def test_context_first_unbound_artifacts_share_the_hard_artifact_capacity() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=4,
        max_live_artifact_obligations=1,
    )
    first_request = _grant_request(5)
    first_params = SimpleNamespace(
        **first_request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint="ctx.example:8000",
    )
    first_promise = _Promise()
    control.register_context_artifact(first_params, first_promise)

    second_request = _grant_request(7)
    second_params = SimpleNamespace(
        **second_request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint="ctx.example:8000",
    )
    second_promise = _Promise()
    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        control.register_context_artifact(second_params, second_promise)

    assert second_promise.abort_count == 1
    assert control._artifact_obligations.reserved_obligation_count == 1

    renewal = ArtifactObligationRequest(
        **first_request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name="gen.example:8000",
        generation_endpoint_rank=0,
        generation_endpoint_incarnation=_uuid(8),
        action=ArtifactControlAction.RENEW,
        sequence=0,
        context_endpoint_incarnation=control.endpoint_incarnation,
    )
    response = control.handle_artifact_obligation(renewal)

    assert response.accepted
    assert response.artifact_ready
    assert control._artifact_obligations.live_obligation_count == 1
    assert control._artifact_obligations.reserved_obligation_count == 1


def test_unbound_context_artifact_expiry_aborts_its_request() -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
        artifact_ttl_s=2.0,
        artifact_renew_interval_s=1.0,
        clock=clock,
    )
    request = _grant_request()
    params = SimpleNamespace(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint="ctx.example:8000",
    )
    promise = _Promise()
    control.register_context_artifact(params, promise)

    clock.now = 3.0
    control._sweep_once()

    assert promise.abort_count == 1
    renew = ArtifactObligationRequest(
        **request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name="gen.example:8000",
        generation_endpoint_rank=0,
        generation_endpoint_incarnation=_uuid(8),
        action=ArtifactControlAction.RENEW,
        sequence=0,
    )
    response = control.handle_artifact_obligation(renew)
    assert not response.accepted
    assert response.state == "ABANDONED"


def test_endpointless_abort_tombstones_context_artifact_before_gen_admission() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    abort = ContextArtifactAbortRequest(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_endpoint_incarnation=control.endpoint_incarnation,
        reason="all generation endpoints rejected admission",
    )

    response = control.abort_context_artifact(abort)
    replay = control.abort_context_artifact(abort)

    assert response.accepted
    assert replay.accepted
    params = SimpleNamespace(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint="ctx.example:8000",
    )
    promise = _Promise()
    with pytest.raises(RuntimeError, match="ABANDONED"):
        control.register_context_artifact(params, promise)
    assert promise.abort_count == 1


def test_terminal_artifacts_do_not_consume_active_session_history_capacity() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
    )
    control._max_artifact_session_history = 1

    for grant_id in (5, 7):
        request = _grant_request(grant_id)
        response = control.abort_context_artifact(
            ContextArtifactAbortRequest(
                **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
                context_endpoint_incarnation=control.endpoint_incarnation,
                reason="generation admission failed",
            )
        )
        assert response.accepted
        assert request.consumer_grant_id not in control._artifact_sessions


def test_replay_filter_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="replay filter capacity"):
        DisaggLifecycleControl(
            role=ServerRole.CONTEXT,
            endpoint_name=lambda: "ctx.example:8000",
            max_live_generation_grants=1,
            replay_filter_capacity=0,
        )


@pytest.mark.asyncio
async def test_artifact_acknowledgement_binds_context_endpoint_incarnation() -> None:
    first_incarnation = _uuid(31)
    session = _Session(
        [
            _artifact_response(first_incarnation),
            _artifact_response(_uuid(32)),
        ]
    )
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        session=session,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    metadata = RequestLifecycleMetadata.from_params(params)

    await control._send_artifact_control(
        metadata,
        ArtifactControlAction.RENEW,
        sequence=0,
    )
    with pytest.raises(RuntimeError, match="incarnation changed"):
        await control._send_artifact_control(
            metadata,
            ArtifactControlAction.RENEW,
            sequence=1,
        )


@pytest.mark.asyncio
async def test_terminal_artifact_renewal_response_aborts_generation() -> None:
    session = _Session(
        [
            _artifact_response(
                _uuid(31),
                accepted=False,
                state="ABANDONED",
            )
        ]
    )
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        session=session,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    promise = _Promise()

    metadata = control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[metadata.grant.consumer_grant_id]
    assert ticket.renewal_task is not None
    await ticket.renewal_task

    assert promise.abort_count == 1
    assert request.consumer_grant_id not in control._generation_tickets
    assert request.consumer_grant_id not in control._grant_sessions
    assert control._generation_grants.live_grant_count == 0


@pytest.mark.asyncio
async def test_artifact_renewal_recovers_from_transient_ctx_unreachability() -> None:
    clock = _Clock()
    session = _ArtifactPeerSession(
        clock,
        [
            (4.0, False),
            (4.5, True),
        ],
    )
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=20.0,
        artifact_ttl_s=5.0,
        artifact_renew_interval_s=0.001,
        clock=clock,
        session=session,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    promise = _Promise()
    metadata = control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[request.consumer_grant_id]

    for _ in range(100):
        if request.consumer_grant_id in control._context_endpoint_incarnations:
            break
        await asyncio.sleep(0.001)

    assert session.renewal_calls >= 2
    assert promise.abort_count == 0
    assert request.consumer_grant_id in control._generation_tickets
    assert control._generation_grants.validate_active(metadata.grant, now_s=clock.now)

    promise.publish_handoff(
        HandoffLifecycleEvent.from_params(
            params,
            HandoffEventState.HANDOFF_COMMITTED,
        )
    )
    assert ticket.handoff_task is not None
    await ticket.handoff_task
    await control.finish_generation(params, success=True)


@pytest.mark.asyncio
async def test_artifact_renewal_fails_closed_after_peer_ack_ttl() -> None:
    clock = _Clock()
    session = _ArtifactPeerSession(
        clock,
        [
            (7.0, False),
        ],
    )
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=100.0,
        artifact_ttl_s=5.0,
        artifact_renew_interval_s=1.0,
        clock=clock,
        session=session,
    )
    request = _grant_request(context_control_endpoint="ctx.example:8000")
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    promise = _Promise()
    control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[request.consumer_grant_id]
    renewal_task = ticket.renewal_task
    handoff_task = ticket.handoff_task
    assert renewal_task is not None
    assert handoff_task is not None

    await renewal_task

    assert session.renewal_calls == 1
    assert promise.abort_count == 1
    assert request.consumer_grant_id not in control._generation_tickets
    assert request.consumer_grant_id not in control._grant_sessions
    assert request.consumer_grant_id not in control._grant_requests
    assert ticket.renewal_task is None
    assert ticket.handoff_task is None
    assert handoff_task.cancelled()
    assert control._generation_grants.live_grant_count == 0
    assert control.issue_generation_grant(_grant_request(7)).accepted


def test_control_messages_bind_the_grant_to_one_transfer_session() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    base = dict(
        **request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name="gen.example:8000",
        generation_endpoint_rank=0,
        generation_endpoint_incarnation=_uuid(8),
        action=ArtifactControlAction.RENEW,
        sequence=0,
    )
    control.handle_artifact_obligation(ArtifactObligationRequest(**base))

    base["transfer_session_id"] = _uuid(77)
    base["sequence"] = 1
    with pytest.raises(ValueError, match="different transfer session"):
        control.handle_artifact_obligation(ArtifactObligationRequest(**base))


def test_grant_renewal_requires_the_exact_admitted_session() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    renew = GenerationGrantRenewRequest(
        **request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name=decision.generation_endpoint_name,
        generation_endpoint_rank=decision.generation_endpoint_rank,
        generation_endpoint_incarnation=decision.generation_endpoint_incarnation,
    )
    renew.transfer_session_id = _uuid(88)

    with pytest.raises(ValueError, match="different transfer session"):
        control.renew_generation_grant(renew)


def test_expired_grant_is_rejected_before_scheduler_submission() -> None:
    clock = _Clock()
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
        grant_ttl_s=2.0,
        clock=clock,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    clock.now = 3.0

    with pytest.raises(RuntimeError, match="expired before scheduler"):
        control.validate_generation_grant_active(_params(request, decision))


def test_revoked_grant_race_aborts_request_after_scheduler_enqueue() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    promise = _Promise()
    control.validate_generation_grant_active(params)
    control._generation_grants.revoke(
        RequestLifecycleMetadata.from_params(params).grant,
        "revoked between validation and enqueue",
    )

    with pytest.raises(RuntimeError, match="terminal before scheduler"):
        control.mark_generation_scheduler_inserted(params, promise)

    assert promise.abort_count == 1


@pytest.mark.asyncio
async def test_handoff_commit_wins_a_later_generation_abort() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.GENERATION,
        endpoint_name=lambda: "gen.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    decision = control.issue_generation_grant(request)
    params = _params(request, decision)
    promise = _Promise()
    control.validate_generation_grant_active(params)
    metadata = control.mark_generation_scheduler_inserted(params, promise)
    ticket = control._generation_tickets[request.consumer_grant_id]
    promise.publish_handoff(
        HandoffLifecycleEvent.from_params(
            params,
            HandoffEventState.HANDOFF_COMMITTED,
        )
    )

    await ticket.handoff_task
    response = await control.abort_generation_grant(_abort_request(request, decision))

    assert ticket.commit_gate.state.value == "COMMITTED"
    assert response.state == "REVOKED"
    assert promise.abort_count == 1
    assert metadata.session == ticket.metadata.session


def test_artifact_release_before_context_registration_is_terminal_success() -> None:
    control = DisaggLifecycleControl(
        role=ServerRole.CONTEXT,
        endpoint_name=lambda: "ctx.example:8000",
        max_live_generation_grants=1,
    )
    request = _grant_request()
    release = ArtifactObligationRequest(
        **request.model_dump(exclude=_GRANT_ADMISSION_FIELDS),
        generation_endpoint_name="gen.example:8000",
        generation_endpoint_rank=0,
        generation_endpoint_incarnation=_uuid(8),
        action=ArtifactControlAction.RELEASE,
        sequence=0,
    )

    response = control.handle_artifact_obligation(release)
    params = SimpleNamespace(
        **request.model_dump(exclude=_GRANT_REQUEST_ONLY_FIELDS),
        context_control_endpoint="ctx.example:8000",
    )
    promise = _Promise()
    metadata = control.register_context_artifact(params, promise)

    assert response.accepted
    assert response.state == "RELEASED"
    assert metadata.consumer_grant_id == request.consumer_grant_id
    assert promise.abort_count == 1
