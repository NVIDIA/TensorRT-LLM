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
"""Internal control plane for disaggregated request obligations.

This module owns logical queue and artifact obligations. It never settles an
allocator lease: deadline expiry may abort a request and start fencing, but
physical reuse remains the transceiver/allocator contract's responsibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional
from urllib.parse import urljoin
from uuid import UUID, uuid4

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tensorrt_llm._torch.disaggregation.control_plane import (
    ArtifactObligationRegistry,
    GenerationAdmissionRegistry,
    TerminalIdentityFilter,
)
from tensorrt_llm._torch.disaggregation.handoff import HandoffEventState, HandoffLifecycleEvent
from tensorrt_llm._torch.disaggregation.obligations import (
    ArtifactObligationIdentity,
    ArtifactObligationState,
    GenerationGrantIdentity,
    GenerationGrantState,
    ObligationConflictError,
    ReceiveCommitGate,
    ReceiveCommitState,
)
from tensorrt_llm._torch.disaggregation.protocol import (
    AttemptIdentity,
    EndpointIdentity,
    TransferProtocolIdentity,
)
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle, TransceiverLifecycleAdvertisement
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.disagg_auth import build_internal_disagg_lifecycle_auth_headers

GENERATION_GRANT_PATH = "/_internal/disagg_lifecycle/generation_grant"
GENERATION_GRANT_RENEW_PATH = "/_internal/disagg_lifecycle/generation_grant/renew"
GENERATION_GRANT_ABORT_PATH = "/_internal/disagg_lifecycle/generation_grant/abort"
ARTIFACT_OBLIGATION_PATH = "/_internal/disagg_lifecycle/artifact_obligation"
CONTEXT_ARTIFACT_ABORT_PATH = "/_internal/disagg_lifecycle/context_artifact/abort"


class _StrictControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lifecycle_protocol_version: Literal[1]


class AttemptControlFields(_StrictControlModel):
    logical_request_id: int = Field(ge=0)
    prefill_artifact_id: UUID
    artifact_version: int = Field(ge=0)
    handoff_attempt_uuid: UUID
    consumer_grant_id: UUID
    transfer_session_id: UUID

    def to_attempt(self) -> AttemptIdentity:
        return AttemptIdentity(
            logical_request_id=self.logical_request_id,
            prefill_artifact_id=self.prefill_artifact_id,
            artifact_version=self.artifact_version,
            handoff_attempt_uuid=self.handoff_attempt_uuid,
        )

    def to_session(self) -> TransferProtocolIdentity:
        return TransferProtocolIdentity(
            attempt=self.to_attempt(),
            transfer_session_id=self.transfer_session_id,
        )


class GenerationGrantRequest(AttemptControlFields):
    ttl_s: float = Field(gt=0)
    context_control_endpoint: Optional[str] = Field(default=None, min_length=1)
    context_transceiver_lifecycle: TransceiverLifecycleAdvertisement
    schedule_style: DisaggScheduleStyle
    ctx_dp_rank: Optional[int] = Field(default=None, ge=0)


class GenerationGrantDecisionResponse(_StrictControlModel):
    accepted: bool
    reason: str = ""
    generation_endpoint_name: Optional[str] = None
    generation_endpoint_rank: Optional[int] = Field(default=None, ge=0)
    generation_endpoint_incarnation: Optional[UUID] = None
    generation_transceiver_lifecycle: Optional[TransceiverLifecycleAdvertisement] = None
    ttl_s: Optional[float] = Field(default=None, gt=0)


class GenerationGrantIdentityFields(AttemptControlFields):
    generation_endpoint_name: str = Field(min_length=1)
    generation_endpoint_rank: int = Field(ge=0)
    generation_endpoint_incarnation: UUID

    def to_grant(self) -> GenerationGrantIdentity:
        return GenerationGrantIdentity(
            consumer_grant_id=self.consumer_grant_id,
            attempt=self.to_attempt(),
            generation_endpoint=EndpointIdentity(
                self.generation_endpoint_name,
                self.generation_endpoint_rank,
                self.generation_endpoint_incarnation,
            ),
        )


class GenerationGrantRenewRequest(GenerationGrantIdentityFields):
    ttl_s: float = Field(gt=0)
    sequence: int = Field(ge=0)


class GenerationGrantAbortRequest(GenerationGrantIdentityFields):
    reason: str = Field(default="generation request abandoned", min_length=1)


class ContextArtifactAbortRequest(AttemptControlFields):
    context_endpoint_incarnation: UUID
    reason: str = Field(default="context artifact abandoned", min_length=1)


class ArtifactControlAction(str, Enum):
    RENEW = "renew"
    RELEASE = "release"
    ABORT = "abort"


class ArtifactObligationRequest(GenerationGrantRenewRequest):
    action: ArtifactControlAction
    sequence: int = Field(default=0, ge=0)
    context_endpoint_incarnation: UUID

    def to_artifact(self) -> ArtifactObligationIdentity:
        return ArtifactObligationIdentity(self.to_grant())


class ObligationResponse(_StrictControlModel):
    accepted: bool
    state: str
    reason: str = ""
    ttl_s: Optional[float] = Field(default=None, gt=0)
    artifact_ready: Optional[bool] = None
    context_endpoint_incarnation: Optional[UUID] = None


@dataclass(frozen=True, slots=True)
class RequestAttemptMetadata:
    """Attempt fields available before a GEN endpoint accepts the grant."""

    session: TransferProtocolIdentity
    consumer_grant_id: UUID
    context_control_endpoint: Optional[str]

    @classmethod
    def from_params(
        cls,
        params: object,
    ) -> "RequestAttemptMetadata":
        if params is None:
            raise ValueError("disaggregated lifecycle metadata is missing")
        required = {
            "lifecycle_protocol_version": 1,
            "logical_request_id": getattr(params, "logical_request_id", None),
            "prefill_artifact_id": getattr(params, "prefill_artifact_id", None),
            "artifact_version": getattr(params, "artifact_version", None),
            "handoff_attempt_uuid": getattr(params, "handoff_attempt_uuid", None),
            "consumer_grant_id": getattr(params, "consumer_grant_id", None),
            "transfer_session_id": getattr(params, "transfer_session_id", None),
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "disaggregated lifecycle metadata is incomplete: " + ", ".join(missing)
            )
        fields = AttemptControlFields.model_validate(required)
        return cls(
            session=fields.to_session(),
            consumer_grant_id=fields.consumer_grant_id,
            context_control_endpoint=getattr(params, "context_control_endpoint", None),
        )


@dataclass(frozen=True, slots=True)
class RequestLifecycleMetadata:
    """Validated lifecycle metadata after GEN has accepted responsibility."""

    session: TransferProtocolIdentity
    grant: GenerationGrantIdentity
    context_control_endpoint: Optional[str]
    context_endpoint_incarnation: UUID

    @classmethod
    def from_params(
        cls,
        params: object,
    ) -> "RequestLifecycleMetadata":
        attempt = RequestAttemptMetadata.from_params(params)
        endpoint_name = getattr(params, "generation_endpoint_name", None)
        endpoint_rank = getattr(params, "generation_endpoint_rank", None)
        endpoint_incarnation = getattr(params, "generation_endpoint_incarnation", None)
        if endpoint_name is None or endpoint_rank is None or endpoint_incarnation is None:
            raise ValueError("generation endpoint identity is missing from lifecycle metadata")
        try:
            endpoint_incarnation = (
                endpoint_incarnation
                if isinstance(endpoint_incarnation, UUID)
                else UUID(str(endpoint_incarnation))
            )
        except (TypeError, ValueError) as error:
            raise ValueError("generation endpoint incarnation must be a non-nil UUID") from error
        grant = GenerationGrantIdentity(
            consumer_grant_id=attempt.consumer_grant_id,
            attempt=attempt.session.attempt,
            generation_endpoint=EndpointIdentity(
                endpoint_name,
                endpoint_rank,
                endpoint_incarnation,
            ),
        )
        context_lifecycle = TransceiverLifecycleAdvertisement.from_value(
            getattr(params, "context_transceiver_lifecycle", None)
        )
        return cls(
            session=attempt.session,
            grant=grant,
            context_control_endpoint=attempt.context_control_endpoint,
            context_endpoint_incarnation=UUID(context_lifecycle.instance_id),
        )


@dataclass(slots=True)
class _GenerationTicket:
    metadata: RequestLifecycleMetadata
    promise: object
    commit_gate: ReceiveCommitGate
    renewal_task: Optional[asyncio.Task] = None
    handoff_task: Optional[asyncio.Task] = None
    artifact_action: Optional[ArtifactControlAction] = None
    control_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _ArtifactTicket:
    metadata: RequestAttemptMetadata
    promise: object
    expires_at_s: float


@dataclass(slots=True)
class _ContextAttemptTicket:
    metadata: RequestAttemptMetadata
    promise: object


@dataclass(frozen=True, slots=True)
class _ArtifactSessionRecord:
    session: TransferProtocolIdentity
    retain_until_s: float


@dataclass(frozen=True, slots=True)
class _ArtifactAttemptTombstone:
    session: TransferProtocolIdentity
    state: ArtifactObligationState
    reason: str
    retain_until_s: float


@dataclass(frozen=True, slots=True)
class _GenerationAttemptTombstone:
    session: TransferProtocolIdentity
    request: GenerationGrantRequest
    identity: GenerationGrantIdentity
    state: GenerationGrantState
    reason: str
    retain_until_s: float


@dataclass(frozen=True, slots=True)
class _ArtifactRenewalAck:
    ttl_s: float
    artifact_ready: bool


class _ArtifactControlProtocolError(RuntimeError):
    """A syntactically valid transport produced an unsafe CTX response."""


class DisaggLifecycleControl:
    """Server-local authority for Phase-5 grants and artifact obligations."""

    def __init__(
        self,
        *,
        role: ServerRole,
        endpoint_name: Callable[[], str],
        max_live_generation_grants: int,
        max_live_artifact_obligations: Optional[int] = None,
        grant_ttl_s: float = 600.0,
        artifact_ttl_s: float = 60.0,
        artifact_renew_interval_s: float = 20.0,
        sweep_interval_s: float = 1.0,
        replay_horizon_s: Optional[float] = None,
        replay_filter_capacity: int = 262144,
        clock: Callable[[], float] = time.monotonic,
        session: Optional[aiohttp.ClientSession] = None,
        endpoint_lifecycle: Optional[Callable[[], TransceiverLifecycleAdvertisement]] = None,
        internal_disagg_auth_key: Optional[str] = None,
    ) -> None:
        if grant_ttl_s <= 0 or artifact_ttl_s <= 0:
            raise ValueError("obligation TTLs must be positive")
        if not 0 < artifact_renew_interval_s < artifact_ttl_s:
            raise ValueError("artifact renewal interval must be below its TTL")
        if sweep_interval_s <= 0:
            raise ValueError("sweep interval must be positive")
        if replay_horizon_s is None:
            replay_horizon_s = 2.0 * max(grant_ttl_s, artifact_ttl_s)
        if replay_horizon_s < max(grant_ttl_s, artifact_ttl_s):
            raise ValueError("replay horizon must cover every endpoint-owned obligation TTL")
        if replay_filter_capacity <= 0:
            raise ValueError("replay filter capacity must be positive")
        if max_live_artifact_obligations is None:
            max_live_artifact_obligations = max_live_generation_grants
        if max_live_artifact_obligations <= 0:
            raise ValueError("max_live_artifact_obligations must be positive")
        self._role = role
        self._endpoint_name = endpoint_name
        self._endpoint_lifecycle = endpoint_lifecycle
        initial_lifecycle = (
            TransceiverLifecycleAdvertisement.from_value(endpoint_lifecycle())
            if endpoint_lifecycle is not None
            else None
        )
        self._endpoint_incarnation = (
            UUID(initial_lifecycle.instance_id) if initial_lifecycle is not None else uuid4()
        )
        self._replay_horizon_s = replay_horizon_s
        self._clock = clock
        self._generation_grants = GenerationAdmissionRegistry(
            max_live_grants=max_live_generation_grants,
            replay_filter_capacity=replay_filter_capacity,
            replay_horizon_s=replay_horizon_s,
            clock=clock,
        )
        self._artifact_obligations = ArtifactObligationRegistry(
            max_pending_renewals=max_live_artifact_obligations,
            max_live_obligations=max_live_artifact_obligations,
            replay_filter_capacity=replay_filter_capacity,
            replay_horizon_s=replay_horizon_s,
            clock=clock,
        )
        self._grant_ttl_s = grant_ttl_s
        self._artifact_ttl_s = artifact_ttl_s
        self._artifact_renew_interval_s = artifact_renew_interval_s
        self._sweep_interval_s = sweep_interval_s
        self._session = session
        self._owns_session = session is None
        self._internal_disagg_auth_key = internal_disagg_auth_key
        self._sweeper: Optional[asyncio.Task] = None
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._generation_tickets: dict[UUID, _GenerationTicket] = {}
        self._context_attempt_tickets: dict[UUID, _ContextAttemptTicket] = {}
        self._artifact_tickets: dict[UUID, _ArtifactTicket] = {}
        self._unbound_artifact_tickets: dict[UUID, _ArtifactTicket] = {}
        self._artifact_grants: dict[UUID, GenerationGrantIdentity] = {}
        self._grant_sessions: dict[UUID, TransferProtocolIdentity] = {}
        self._grant_requests: dict[UUID, GenerationGrantRequest] = {}
        self._grant_generation_lifecycles: dict[UUID, TransceiverLifecycleAdvertisement] = {}
        self._generation_attempt_tombstones: dict[UUID, _GenerationAttemptTombstone] = {}
        self._artifact_sessions: dict[UUID, _ArtifactSessionRecord] = {}
        self._artifact_attempt_tombstones: dict[UUID, _ArtifactAttemptTombstone] = {}
        self._artifact_terminal_ids = TerminalIdentityFilter(
            replay_filter_capacity,
            replay_horizon_s=replay_horizon_s,
            clock=clock,
        )
        self._context_endpoint_incarnations: dict[UUID, UUID] = {}
        self._max_generation_attempt_history = replay_filter_capacity
        self._max_artifact_session_history = 4096
        self._max_artifact_terminal_history = replay_filter_capacity

    @property
    def endpoint_incarnation(self) -> UUID:
        return self._endpoint_incarnation

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def shutdown(self) -> None:
        sweeper, self._sweeper = self._sweeper, None
        if sweeper is not None:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
        tickets = tuple(self._generation_tickets.values())
        self._generation_tickets.clear()
        for ticket in tickets:
            if ticket.renewal_task is not None:
                ticket.renewal_task.cancel()
            if ticket.handoff_task is not None:
                ticket.handoff_task.cancel()
        if tickets:
            await asyncio.gather(
                *(
                    task
                    for ticket in tickets
                    for task in (ticket.renewal_task, ticket.handoff_task)
                    if task is not None
                ),
                return_exceptions=True,
            )
        context_tickets = tuple(self._context_attempt_tickets.values())
        self._context_attempt_tickets.clear()
        for ticket in context_tickets:
            self._abort_promise(ticket.promise)
        cleanup_tasks = tuple(self._cleanup_tasks)
        for task in cleanup_tasks:
            task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._cleanup_tasks.clear()
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None

    def issue_generation_grant(
        self,
        request: GenerationGrantRequest,
    ) -> GenerationGrantDecisionResponse:
        self._require_role(ServerRole.GENERATION)
        now_s = self._clock()
        self._purge_generation_replay_state(now_s)
        for expired in self._generation_grants.sweep_expired(now_s):
            self._retire_expired_generation_grant(
                expired,
                reason="generation intent grant expired",
            )
        endpoint_name = self._endpoint_name()
        generation_lifecycle = self._current_endpoint_lifecycle()
        endpoint_incarnation = (
            UUID(generation_lifecycle.instance_id)
            if generation_lifecycle is not None
            else self._endpoint_incarnation
        )
        endpoint = EndpointIdentity(endpoint_name, 0, endpoint_incarnation)
        identity = GenerationGrantIdentity(
            request.consumer_grant_id,
            request.to_attempt(),
            endpoint,
        )
        existing_session = self._grant_sessions.get(request.consumer_grant_id)
        session = request.to_session()
        if existing_session is not None:
            if existing_session != session:
                raise ValueError("consumer_grant_id was replayed for a different transfer session")
            existing_request = self._grant_requests[request.consumer_grant_id]
            if not self._same_grant_request(existing_request, request):
                raise ValueError("consumer_grant_id was replayed with conflicting admission facts")
            existing_lifecycle = self._grant_generation_lifecycles.get(request.consumer_grant_id)
            if existing_lifecycle != generation_lifecycle:
                raise ValueError(
                    "consumer_grant_id was replayed against a different generation "
                    "transceiver incarnation"
                )
        else:
            terminal = self._generation_attempt_tombstones.get(request.consumer_grant_id)
            if terminal is not None:
                if terminal.session != session:
                    raise ValueError(
                        "consumer_grant_id was replayed for a different terminal transfer session"
                    )
                if not self._same_grant_request(terminal.request, request):
                    raise ValueError(
                        "consumer_grant_id was replayed with conflicting terminal admission facts"
                    )
                return GenerationGrantDecisionResponse(
                    lifecycle_protocol_version=1,
                    accepted=False,
                    reason=terminal.reason or "generation grant is already terminal",
                )
        decision = self._generation_grants.issue(
            identity,
            issued_at_s=now_s,
            expires_at_s=now_s + self._grant_ttl_s,
        )
        if not decision.accepted:
            return GenerationGrantDecisionResponse(
                lifecycle_protocol_version=1,
                accepted=False,
                reason=decision.reason,
            )
        if existing_session is None:
            self._grant_sessions[request.consumer_grant_id] = session
            self._grant_requests[request.consumer_grant_id] = request.model_copy(deep=True)
            if generation_lifecycle is not None:
                self._grant_generation_lifecycles[request.consumer_grant_id] = generation_lifecycle
        assert decision.expires_at_s is not None
        remaining_ttl_s = decision.expires_at_s - now_s
        if remaining_ttl_s <= 0:
            raise RuntimeError("generation grant expired during admission")
        return GenerationGrantDecisionResponse(
            lifecycle_protocol_version=1,
            accepted=True,
            generation_endpoint_name=endpoint.instance_name,
            generation_endpoint_rank=endpoint.instance_rank,
            generation_endpoint_incarnation=endpoint.incarnation,
            generation_transceiver_lifecycle=generation_lifecycle,
            ttl_s=remaining_ttl_s,
        )

    def renew_generation_grant(
        self,
        request: GenerationGrantRenewRequest,
    ) -> ObligationResponse:
        self._require_role(ServerRole.GENERATION)
        now_s = self._clock()
        self._purge_generation_replay_state(now_s)
        terminal = self._require_session(
            request.consumer_grant_id,
            request.to_session(),
        )
        if terminal is not None:
            if terminal.identity != request.to_grant():
                raise ValueError("generation grant renewal belongs to a different terminal request")
            return ObligationResponse(
                lifecycle_protocol_version=1,
                accepted=False,
                state=terminal.state.value,
                reason=terminal.reason,
            )
        decision = self._generation_grants.renew(
            request.to_grant(),
            sequence=request.sequence,
            now_s=now_s,
            ttl_s=self._grant_ttl_s,
        )
        state = decision.state
        if state is not GenerationGrantState.ACTIVE:
            self._retire_expired_generation_grant(
                request.to_grant(),
                reason="generation intent grant expired before renewal",
            )
        remaining_ttl_s = None if decision.expires_at_s is None else decision.expires_at_s - now_s
        return ObligationResponse(
            lifecycle_protocol_version=1,
            accepted=state is GenerationGrantState.ACTIVE,
            state=state.value,
            ttl_s=remaining_ttl_s if remaining_ttl_s and remaining_ttl_s > 0 else None,
        )

    async def abort_generation_grant(
        self,
        request: GenerationGrantAbortRequest,
    ) -> ObligationResponse:
        """Revoke an admitted grant and fan out abort to any bound request."""
        self._require_role(ServerRole.GENERATION)
        identity = request.to_grant()
        now_s = self._clock()
        self._purge_generation_replay_state(now_s)
        terminal = self._require_session(
            request.consumer_grant_id,
            request.to_session(),
        )
        if terminal is not None:
            if terminal.identity != identity:
                raise ValueError("generation grant abort belongs to a different terminal request")
            if terminal.state is not GenerationGrantState.REVOKED:
                raise ObligationConflictError("cannot revoke a grant that was already released")
            return ObligationResponse(
                lifecycle_protocol_version=1,
                accepted=True,
                state=terminal.state.value,
            )
        grant_state = self._generation_grants.validate_identity(identity)
        if grant_state is GenerationGrantState.RELEASED:
            raise ObligationConflictError("cannot revoke a grant that was already released")
        ticket = self._generation_tickets.get(request.consumer_grant_id)
        if ticket is not None:
            if ticket.metadata.grant != identity:
                raise ValueError("generation grant abort belongs to a different bound request")
            self._generation_tickets.pop(request.consumer_grant_id, None)
            gate_state = ticket.commit_gate.state
            if gate_state is ReceiveCommitState.OPEN:
                gate_state = ticket.commit_gate.abort(request.reason)
            self._abort_promise(ticket.promise)
            await self._stop_ticket_tasks(ticket)
            if ticket.metadata.context_control_endpoint:
                try:
                    await self._settle_ticket_artifact(
                        ticket,
                        ArtifactControlAction.RELEASE
                        if gate_state is ReceiveCommitState.COMMITTED
                        else ArtifactControlAction.ABORT,
                    )
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    OSError,
                    RuntimeError,
                ) as error:
                    logger.warning(
                        "Failed to fan out artifact abort for grant %s; "
                        "CTX obligation expiry remains authoritative: %s",
                        request.consumer_grant_id,
                        error,
                    )
        else:
            admitted_request = self._grant_requests.get(request.consumer_grant_id)
            if admitted_request is not None and admitted_request.context_control_endpoint:
                self._schedule_context_artifact_abort(
                    admitted_request,
                    reason=request.reason,
                )
        state = self._generation_grants.revoke(
            identity,
            request.reason,
            now_s=now_s,
        )
        self._remember_generation_terminal(
            identity,
            state=state,
            reason=request.reason,
            now_s=now_s,
        )
        self._grant_sessions.pop(request.consumer_grant_id, None)
        self._grant_requests.pop(request.consumer_grant_id, None)
        self._grant_generation_lifecycles.pop(request.consumer_grant_id, None)
        self._context_endpoint_incarnations.pop(request.consumer_grant_id, None)
        return ObligationResponse(
            lifecycle_protocol_version=1,
            accepted=True,
            state=state.value,
        )

    def handle_artifact_obligation(
        self,
        request: ArtifactObligationRequest,
    ) -> ObligationResponse:
        self._require_role(ServerRole.CONTEXT)
        self._require_endpoint_incarnation(
            request.context_endpoint_incarnation,
            operation="artifact control",
        )
        now_s = self._clock()
        self._purge_artifact_replay_state(now_s)
        terminal = self._artifact_attempt_tombstones.get(request.consumer_grant_id)
        if terminal is not None:
            if terminal.session != request.to_session():
                raise ValueError(
                    "artifact control replay belongs to a different terminal transfer session"
                )
            expected_action = (
                request.action is ArtifactControlAction.ABORT
                and terminal.state is ArtifactObligationState.ABANDONED
            ) or (
                request.action is ArtifactControlAction.RELEASE
                and terminal.state is ArtifactObligationState.RELEASED
            )
            return self._artifact_response(
                accepted=expected_action,
                state=terminal.state,
                reason=terminal.reason,
                artifact_ready=False,
            )
        if self._artifact_terminal_ids.contains(request.consumer_grant_id):
            return self._artifact_response(
                accepted=False,
                state=ArtifactObligationState.ABANDONED,
                reason=("artifact identity is terminal or replay protection is saturated"),
                artifact_ready=False,
            )
        self._remember_artifact_session(
            request.consumer_grant_id,
            request.to_session(),
            now_s=now_s,
        )
        identity = request.to_artifact()
        existing_grant = self._artifact_grants.get(request.consumer_grant_id)
        if existing_grant is not None and existing_grant != identity.grant:
            raise ValueError("artifact control replay names a different generation endpoint")
        unbound = self._unbound_artifact_tickets.get(
            request.consumer_grant_id,
            None,
        )
        if unbound is not None:
            if unbound.metadata.session != request.to_session():
                raise ValueError("artifact control belongs to a different CTX transfer session")
        if unbound is not None:
            self._artifact_obligations.register(
                identity,
                now_s=now_s,
                expires_at_s=now_s + self._artifact_ttl_s,
            )
            self._unbound_artifact_tickets.pop(
                request.consumer_grant_id,
                None,
            )
            self._artifact_tickets[request.consumer_grant_id] = unbound
        self._artifact_grants[request.consumer_grant_id] = identity.grant
        renewal_expires_at_s = None
        artifact_ready = False
        if request.action is ArtifactControlAction.RENEW:
            decision = self._artifact_obligations.renew_or_defer(
                identity,
                sequence=request.sequence,
                now_s=now_s,
                ttl_s=self._artifact_ttl_s,
            )
            state = decision.state
            renewal_expires_at_s = decision.expires_at_s
            artifact_ready = decision.artifact_ready
        elif request.action is ArtifactControlAction.RELEASE:
            state = self._artifact_obligations.release(identity, now_s=now_s)
            self._artifact_tickets.pop(request.consumer_grant_id, None)
            self._unbound_artifact_tickets.pop(
                request.consumer_grant_id,
                None,
            )
            context_ticket = self._context_attempt_tickets.pop(
                request.consumer_grant_id,
                None,
            )
            if context_ticket is not None:
                self._abort_promise(context_ticket.promise)
        else:
            state = self._artifact_obligations.abandon(identity, now_s=now_s)
            ticket = self._artifact_tickets.pop(request.consumer_grant_id, None)
            if ticket is None:
                ticket = self._unbound_artifact_tickets.pop(
                    request.consumer_grant_id,
                    None,
                )
            if ticket is not None:
                self._abort_promise(ticket.promise)
            context_ticket = self._context_attempt_tickets.pop(
                request.consumer_grant_id,
                None,
            )
            if context_ticket is not None:
                self._abort_promise(context_ticket.promise)
        accepted = (
            state is ArtifactObligationState.ACTIVE
            if request.action is ArtifactControlAction.RENEW
            else (
                state is ArtifactObligationState.RELEASED
                if request.action is ArtifactControlAction.RELEASE
                else state is ArtifactObligationState.ABANDONED
            )
        )
        if state is not ArtifactObligationState.ACTIVE:
            self._remember_artifact_terminal(
                request.consumer_grant_id,
                request.to_session(),
                state=state,
                reason=f"artifact obligation became {state.value}",
                now_s=now_s,
            )
        return self._artifact_response(
            accepted=accepted,
            state=state,
            reason="" if accepted else f"artifact obligation is {state.value}",
            expires_at_s=renewal_expires_at_s,
            artifact_ready=artifact_ready,
            now_s=now_s,
        )

    def abort_context_artifact(
        self,
        request: ContextArtifactAbortRequest,
    ) -> ObligationResponse:
        """Retire a CTX artifact before a generation endpoint is known."""
        self._require_role(ServerRole.CONTEXT)
        self._require_endpoint_incarnation(
            request.context_endpoint_incarnation,
            operation="context artifact abort",
        )
        now_s = self._clock()
        self._purge_artifact_replay_state(now_s)
        session = request.to_session()
        existing = self._artifact_attempt_tombstones.get(request.consumer_grant_id)
        if existing is not None:
            if existing.session != session:
                raise ValueError("context artifact abort belongs to a different terminal session")
            return self._artifact_response(
                accepted=existing.state is ArtifactObligationState.ABANDONED,
                state=existing.state,
                reason=existing.reason,
            )
        if self._artifact_terminal_ids.contains(request.consumer_grant_id):
            return self._artifact_response(
                accepted=False,
                state=ArtifactObligationState.ABANDONED,
                reason=("artifact identity is terminal or replay protection is saturated"),
            )
        self._remember_artifact_session(
            request.consumer_grant_id,
            session,
            now_s=now_s,
        )
        self._retire_context_artifact(
            request.consumer_grant_id,
            session,
            reason=request.reason,
            now_s=now_s,
        )
        return self._artifact_response(
            accepted=True,
            state=ArtifactObligationState.ABANDONED,
            reason=request.reason,
        )

    def mark_generation_scheduler_inserted(
        self,
        params: object,
        promise: object,
    ) -> RequestLifecycleMetadata:
        """Consume an exact grant immediately after ``generate_async`` enqueues."""
        self._require_role(ServerRole.GENERATION)
        metadata = RequestLifecycleMetadata.from_params(params)
        self._require_session(metadata.grant.consumer_grant_id, metadata.session)
        self._require_admitted_request_contract(params)
        try:
            inserted = self._generation_grants.mark_scheduler_inserted(
                metadata.grant,
                now_s=self._clock(),
            )
        except ObligationConflictError:
            raise
        except (KeyError, RuntimeError) as error:
            retired = self._retire_expired_generation_grant(
                metadata.grant,
                reason="generation intent grant became terminal before scheduler insertion",
            )
            if retired is None or retired.promise is not promise:
                self._abort_promise(promise)
            raise RuntimeError(
                "generation intent grant became terminal before scheduler insertion"
            ) from error
        if not inserted:
            retired = self._retire_expired_generation_grant(
                metadata.grant,
                reason="generation intent grant expired before scheduler insertion",
            )
            if retired is None or retired.promise is not promise:
                self._abort_promise(promise)
            raise RuntimeError("generation intent grant expired before scheduler insertion")
        grant_id = metadata.grant.consumer_grant_id
        existing = self._generation_tickets.get(grant_id)
        if existing is not None:
            if existing.metadata != metadata or existing.promise is not promise:
                raise RuntimeError("generation grant is already bound to another request")
            return metadata
        ticket = _GenerationTicket(
            metadata=metadata,
            promise=promise,
            commit_gate=ReceiveCommitGate(metadata.session),
        )
        self._generation_tickets[grant_id] = ticket
        if metadata.context_control_endpoint:
            ticket.renewal_task = asyncio.create_task(self._renew_artifact(ticket))
        ticket.handoff_task = asyncio.create_task(self._watch_handoff_event(ticket))
        return metadata

    def validate_generation_grant_active(
        self,
        params: object,
    ) -> RequestLifecycleMetadata:
        """Fence a stale/revoked grant before submitting it to the scheduler."""
        self._require_role(ServerRole.GENERATION)
        metadata = RequestLifecycleMetadata.from_params(params)
        self._require_session(metadata.grant.consumer_grant_id, metadata.session)
        self._require_admitted_request_contract(params)
        if not self._generation_grants.validate_active(
            metadata.grant,
            now_s=self._clock(),
        ):
            self._retire_expired_generation_grant(
                metadata.grant,
                reason="generation intent grant expired before scheduler submission",
            )
            raise RuntimeError("generation intent grant expired before scheduler submission")
        return metadata

    def mark_context_scheduler_inserted(
        self,
        params: object,
        promise: object,
    ) -> RequestAttemptMetadata:
        """Bind an active CTX request so peer abort can stop prefill compute."""
        self._require_role(ServerRole.CONTEXT)
        metadata = RequestAttemptMetadata.from_params(params)
        grant_id = metadata.consumer_grant_id
        now_s = self._clock()
        self._purge_artifact_replay_state(now_s)
        terminal = self._artifact_attempt_tombstones.get(grant_id)
        if terminal is not None:
            if terminal.session != metadata.session:
                raise ValueError(
                    "context scheduler insertion belongs to a different terminal transfer session"
                )
            self._abort_promise(promise)
            raise RuntimeError(f"context request is already {terminal.state.value}")
        if self._artifact_terminal_ids.contains(grant_id):
            self._abort_promise(promise)
            raise RuntimeError(
                "context request identity is terminal or replay protection is saturated"
            )
        self._remember_artifact_session(
            grant_id,
            metadata.session,
            now_s=now_s,
        )
        existing = self._context_attempt_tickets.get(grant_id)
        if existing is not None:
            if existing.metadata != metadata or existing.promise is not promise:
                raise RuntimeError("context request is already bound to another scheduler request")
            return metadata
        self._context_attempt_tickets[grant_id] = _ContextAttemptTicket(
            metadata,
            promise,
        )
        return metadata

    def register_context_artifact(
        self,
        params: object,
        promise: object,
    ) -> RequestAttemptMetadata:
        """Register the artifact after CTX reports it immutable and ready."""
        self._require_role(ServerRole.CONTEXT)
        metadata = RequestAttemptMetadata.from_params(params)
        now_s = self._clock()
        grant_id = metadata.consumer_grant_id
        context_ticket = self._context_attempt_tickets.get(grant_id)
        if context_ticket is not None and (
            context_ticket.metadata != metadata or context_ticket.promise is not promise
        ):
            raise RuntimeError(
                "artifact registration belongs to a different active context request"
            )
        self._context_attempt_tickets.pop(grant_id, None)
        self._purge_artifact_replay_state(now_s)
        terminal = self._artifact_attempt_tombstones.get(grant_id)
        if terminal is not None:
            if terminal.session != metadata.session:
                raise ValueError(
                    "artifact registration belongs to a different terminal transfer session"
                )
            self._abort_promise(promise)
            if terminal.state is ArtifactObligationState.RELEASED:
                # HANDOFF_COMMITTED can race the CTX response callback. The
                # exact release tombstone is sufficient to retire the
                # just-materialized artifact without recreating an obligation.
                return metadata
            raise RuntimeError(f"artifact obligation is already {terminal.state.value}")
        if self._artifact_terminal_ids.contains(grant_id):
            self._abort_promise(promise)
            raise RuntimeError("artifact identity is terminal or replay protection is saturated")
        self._remember_artifact_session(
            grant_id,
            metadata.session,
            now_s=now_s,
        )
        existing = self._artifact_tickets.get(grant_id)
        if existing is None:
            existing = self._unbound_artifact_tickets.get(grant_id)
        if existing is not None and (
            existing.metadata != metadata or existing.promise is not promise
        ):
            raise RuntimeError("artifact obligation is already bound to another request")
        grant = self._artifact_grants.get(grant_id)
        if grant is None:
            try:
                self._artifact_obligations.reserve_unbound(
                    grant_id,
                    now_s=now_s,
                )
            except RuntimeError:
                self._abort_promise(promise)
                self._retire_context_artifact(
                    grant_id,
                    metadata.session,
                    reason="context artifact retention capacity is exhausted",
                    now_s=now_s,
                )
                raise
            self._unbound_artifact_tickets[grant_id] = _ArtifactTicket(
                metadata,
                promise,
                now_s + self._artifact_ttl_s,
            )
            return metadata
        try:
            self._artifact_obligations.register(
                ArtifactObligationIdentity(grant),
                now_s=now_s,
                expires_at_s=now_s + self._artifact_ttl_s,
            )
        except RuntimeError:
            self._abort_promise(promise)
            raise
        self._artifact_tickets[grant_id] = _ArtifactTicket(
            metadata,
            promise,
            now_s + self._artifact_ttl_s,
        )
        return metadata

    async def finish_generation(
        self,
        params: object,
        *,
        success: bool,
        reason: str = "",
    ) -> None:
        self._require_role(ServerRole.GENERATION)
        metadata = RequestLifecycleMetadata.from_params(params)
        grant_id = metadata.grant.consumer_grant_id
        now_s = self._clock()
        self._purge_generation_replay_state(now_s)
        terminal = self._require_session(grant_id, metadata.session)
        if terminal is not None:
            if terminal.identity != metadata.grant:
                raise ValueError(
                    "generation terminal cleanup belongs to a different terminal request"
                )
            return
        ticket = self._generation_tickets.get(grant_id)
        handoff_committed = False
        if ticket is not None:
            if ticket.metadata != metadata:
                raise ValueError("generation terminal cleanup belongs to a different bound request")
            self._generation_tickets.pop(grant_id, None)
            gate_state = ticket.commit_gate.state
            handoff_event = getattr(ticket.promise, "_disagg_handoff_event", None)
            if gate_state is ReceiveCommitState.OPEN and handoff_event is not None:
                gate_state = self._apply_handoff_event(ticket, handoff_event)
            if gate_state is ReceiveCommitState.OPEN:
                gate_state = ticket.commit_gate.abort(
                    reason or "generation ended before handoff commit"
                )
            handoff_committed = gate_state is ReceiveCommitState.COMMITTED
            await self._stop_ticket_tasks(ticket)
            if metadata.context_control_endpoint:
                try:
                    await self._settle_ticket_artifact(
                        ticket,
                        ArtifactControlAction.RELEASE
                        if handoff_committed
                        else ArtifactControlAction.ABORT,
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as error:
                    logger.warning(
                        "Failed to send terminal artifact control for grant %s; "
                        "CTX obligation expiry remains authoritative: %s",
                        grant_id,
                        error,
                    )
        if success and not handoff_committed:
            success = False
            reason = reason or "generation completed without HANDOFF_COMMITTED"
            logger.error(
                "Generation request %s ended without a committed disaggregated handoff",
                grant_id,
            )
        terminal_reason = "" if success else reason or "generation request failed"
        if success:
            terminal_state = self._generation_grants.release(
                metadata.grant,
                now_s=now_s,
            )
        else:
            terminal_state = self._generation_grants.revoke(
                metadata.grant,
                terminal_reason,
                now_s=now_s,
            )
        self._remember_generation_terminal(
            metadata.grant,
            state=terminal_state,
            reason=terminal_reason,
            now_s=now_s,
        )
        self._grant_sessions.pop(grant_id, None)
        self._grant_requests.pop(grant_id, None)
        self._grant_generation_lifecycles.pop(grant_id, None)
        self._context_endpoint_incarnations.pop(grant_id, None)

    def abandon_context_artifact(self, params: object) -> None:
        self._require_role(ServerRole.CONTEXT)
        metadata = RequestAttemptMetadata.from_params(params)
        grant_id = metadata.consumer_grant_id
        now_s = self._clock()
        self._purge_artifact_replay_state(now_s)
        terminal = self._artifact_attempt_tombstones.get(grant_id)
        if terminal is not None:
            if terminal.session != metadata.session:
                raise ValueError(
                    "context artifact abandonment belongs to a different terminal session"
                )
            return
        if self._artifact_terminal_ids.contains(grant_id):
            return
        self._remember_artifact_session(
            grant_id,
            metadata.session,
            now_s=now_s,
        )
        self._retire_context_artifact(
            grant_id,
            metadata.session,
            reason="context artifact explicitly abandoned",
            now_s=now_s,
        )

    async def _watch_handoff_event(self, ticket: _GenerationTicket) -> None:
        """Retire the CTX obligation at the receive-commit boundary."""
        try:
            waiter = getattr(ticket.promise, "_wait_disagg_handoff_event", None)
            if not callable(waiter):
                raise RuntimeError(
                    "generation result does not expose disaggregated handoff evidence"
                )
            event = await waiter()
            gate_state = self._apply_handoff_event(ticket, event)
        except asyncio.CancelledError:
            raise
        except (
            RuntimeError,
            ValueError,
        ) as error:
            await self._fail_handoff_closed(ticket, error)
            return

        await self._stop_ticket_tasks(ticket)
        if not ticket.metadata.context_control_endpoint:
            return
        try:
            await self._settle_ticket_artifact(
                ticket,
                ArtifactControlAction.RELEASE
                if gate_state is ReceiveCommitState.COMMITTED
                else ArtifactControlAction.ABORT,
            )
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as error:
            if gate_state is ReceiveCommitState.COMMITTED:
                logger.warning(
                    "HANDOFF_COMMITTED for grant %s, but CTX release "
                    "acknowledgement was unavailable; CTX obligation expiry "
                    "remains authoritative: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    error,
                )
                return
            await self._fail_handoff_closed(ticket, error)

    def _apply_handoff_event(
        self,
        ticket: _GenerationTicket,
        event: object,
    ) -> ReceiveCommitState:
        """Apply exact receive evidence before any request-finalization race."""
        if not isinstance(event, HandoffLifecycleEvent):
            raise RuntimeError("generation result returned invalid handoff evidence")
        if (
            event.session != ticket.metadata.session
            or event.consumer_grant_id != ticket.metadata.grant.consumer_grant_id
        ):
            raise RuntimeError("handoff evidence belongs to a different request attempt")
        if event.state is HandoffEventState.HANDOFF_COMMITTED:
            return ticket.commit_gate.commit()
        gate_state = ticket.commit_gate.state
        if gate_state is ReceiveCommitState.OPEN:
            gate_state = ticket.commit_gate.abort(event.reason or event.state.value)
        if gate_state is ReceiveCommitState.ABORTED:
            self._abort_promise(ticket.promise)
        return gate_state

    async def _fail_handoff_closed(
        self,
        ticket: _GenerationTicket,
        error: Exception,
    ) -> None:
        """Fail an uncommitted handoff while preserving a prior commit fact."""
        gate_state = ticket.commit_gate.state
        if gate_state is ReceiveCommitState.COMMITTED:
            logger.warning(
                "Ignoring post-commit lifecycle cleanup failure for grant %s: %s",
                ticket.metadata.grant.consumer_grant_id,
                error,
            )
            return
        if gate_state is ReceiveCommitState.OPEN:
            gate_state = ticket.commit_gate.abort(str(error))
        self._abort_promise(ticket.promise)
        await self._stop_ticket_tasks(ticket)
        if gate_state is ReceiveCommitState.ABORTED and ticket.metadata.context_control_endpoint:
            try:
                await self._settle_ticket_artifact(
                    ticket,
                    ArtifactControlAction.ABORT,
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
            ) as abort_error:
                logger.warning(
                    "Failed to abort CTX artifact after handoff protocol failure for grant %s: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    abort_error,
                )
        reason = f"handoff protocol failed: {error}"
        now_s = self._clock()
        try:
            state = self._generation_grants.revoke(
                ticket.metadata.grant,
                reason,
                now_s=now_s,
            )
        except (KeyError, RuntimeError, ObligationConflictError):
            state = GenerationGrantState.REVOKED
        self._remember_generation_terminal(
            ticket.metadata.grant,
            state=state,
            reason=reason,
            now_s=now_s,
        )
        self._retire_expired_generation_grant(
            ticket.metadata.grant,
            reason=reason,
            current_task=asyncio.current_task(),
        )
        logger.error(
            "Disaggregated handoff failed closed for grant %s: %s",
            ticket.metadata.grant.consumer_grant_id,
            error,
        )

    async def _stop_ticket_tasks(self, ticket: _GenerationTicket) -> None:
        """Cancel ticket background work without awaiting the current task."""
        current = asyncio.current_task()
        tasks = [
            task
            for task in (ticket.renewal_task, ticket.handoff_task)
            if task is not None and task is not current
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if ticket.renewal_task is not current:
            ticket.renewal_task = None
        if ticket.handoff_task is not current:
            ticket.handoff_task = None

    async def _settle_ticket_artifact(
        self,
        ticket: _GenerationTicket,
        action: ArtifactControlAction,
    ) -> None:
        """Serialize the receive commit/abort race and exact CTX acknowledgement."""
        async with ticket.control_lock:
            existing = ticket.artifact_action
            if existing is action:
                return
            if existing is ArtifactControlAction.RELEASE:
                # A later abort can terminate decode, but cannot rewrite a
                # handoff that committed first.
                return
            if existing is ArtifactControlAction.ABORT:
                raise _ArtifactControlProtocolError(
                    "cannot release an artifact after its handoff was aborted"
                )
            await self._send_artifact_control(
                ticket.metadata,
                action,
                sequence=0,
            )
            ticket.artifact_action = action

    async def _renew_artifact(self, ticket: _GenerationTicket) -> None:
        sequence = 0
        peer_ack_deadline_s = self._clock() + self._artifact_ttl_s
        while True:
            now_s = self._clock()
            try:
                grant_is_active = self._generation_grants.validate_active(
                    ticket.metadata.grant,
                    now_s=now_s,
                )
            except (KeyError, RuntimeError):
                self._retire_expired_generation_grant(
                    ticket.metadata.grant,
                    reason="generation intent grant became terminal during artifact renewal",
                    current_task=asyncio.current_task(),
                )
                return
            if not grant_is_active:
                self._retire_expired_generation_grant(
                    ticket.metadata.grant,
                    reason="generation intent grant expired during artifact renewal",
                    current_task=asyncio.current_task(),
                )
                return
            remaining_peer_ttl_s = peer_ack_deadline_s - now_s
            if remaining_peer_ttl_s <= 0:
                await self._fail_generation_ticket_closed(
                    ticket,
                    "context artifact acknowledgement expired",
                )
                return
            sent_at_s = self._clock()
            renewal_acknowledged = False
            try:
                acknowledgement = await asyncio.wait_for(
                    self._send_artifact_control(
                        ticket.metadata,
                        ArtifactControlAction.RENEW,
                        sequence=sequence,
                    ),
                    timeout=remaining_peer_ttl_s,
                )
            except _ArtifactControlProtocolError as error:
                logger.error(
                    "Artifact obligation failed closed for grant %s: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    error,
                )
                await self._fail_generation_ticket_closed(
                    ticket,
                    "context artifact obligation was rejected",
                )
                return
            except RuntimeError as error:
                logger.error(
                    "Artifact renewal control failed closed for grant %s: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    error,
                )
                await self._fail_generation_ticket_closed(
                    ticket,
                    "context artifact renewal control is unavailable",
                )
                return
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                now_s = self._clock()
                if now_s >= peer_ack_deadline_s:
                    logger.error(
                        "Artifact acknowledgement expired for grant %s after "
                        "the CTX endpoint was unreachable: %s",
                        ticket.metadata.grant.consumer_grant_id,
                        error,
                    )
                    await self._fail_generation_ticket_closed(
                        ticket,
                        "context artifact acknowledgement expired",
                    )
                    return
                logger.warning(
                    "Artifact renewal failed for grant %s: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    error,
                )
            else:
                acknowledged_at_s = self._clock()
                if acknowledgement is None:
                    await self._fail_generation_ticket_closed(
                        ticket,
                        "context artifact acknowledgement omitted renewal facts",
                    )
                    return
                peer_ack_deadline_s = sent_at_s + acknowledgement.ttl_s
                if acknowledged_at_s >= peer_ack_deadline_s:
                    await self._fail_generation_ticket_closed(
                        ticket,
                        "context artifact acknowledgement arrived after its deadline",
                    )
                    return
                # Artifact retention and GEN queue ownership are independent
                # obligations. Only the coordinator/supervisor renews the GEN
                # intent grant; a responsive CTX must not keep orphan GEN state
                # alive after that owner disappears.
                renewal_acknowledged = True
            if renewal_acknowledged:
                sequence += 1
            remaining_peer_ttl_s = peer_ack_deadline_s - self._clock()
            await asyncio.sleep(
                min(
                    self._artifact_renew_interval_s,
                    max(remaining_peer_ttl_s / 3.0, 0.0),
                )
            )

    async def _fail_generation_ticket_closed(
        self,
        ticket: _GenerationTicket,
        reason: str,
    ) -> None:
        """Revoke a GEN ticket after its peer-backed obligation is lost."""
        if ticket.commit_gate.state is ReceiveCommitState.COMMITTED:
            return
        try:
            self._generation_grants.revoke(
                ticket.metadata.grant,
                reason,
                now_s=self._clock(),
            )
        except (KeyError, RuntimeError, ObligationConflictError):
            pass
        retired = self._retire_expired_generation_grant(
            ticket.metadata.grant,
            reason=reason,
            current_task=asyncio.current_task(),
        )
        if retired is not None:
            await self._stop_ticket_tasks(retired)
            if retired.renewal_task is asyncio.current_task():
                retired.renewal_task = None

    async def _send_artifact_control(
        self,
        metadata: RequestLifecycleMetadata,
        action: ArtifactControlAction,
        *,
        sequence: int,
    ) -> Optional[_ArtifactRenewalAck]:
        if self._session is None or metadata.context_control_endpoint is None:
            raise RuntimeError("lifecycle control is not started or CTX endpoint is missing")
        attempt = metadata.session.attempt
        request = ArtifactObligationRequest(
            lifecycle_protocol_version=1,
            logical_request_id=attempt.logical_request_id,
            prefill_artifact_id=attempt.prefill_artifact_id,
            artifact_version=attempt.artifact_version,
            handoff_attempt_uuid=attempt.handoff_attempt_uuid,
            consumer_grant_id=metadata.grant.consumer_grant_id,
            transfer_session_id=metadata.session.transfer_session_id,
            generation_endpoint_name=metadata.grant.generation_endpoint.instance_name,
            generation_endpoint_rank=metadata.grant.generation_endpoint.instance_rank,
            generation_endpoint_incarnation=metadata.grant.generation_endpoint.incarnation,
            ttl_s=self._artifact_ttl_s,
            action=action,
            sequence=sequence,
            context_endpoint_incarnation=metadata.context_endpoint_incarnation,
        )
        url = self._control_url(
            metadata.context_control_endpoint,
            ARTIFACT_OBLIGATION_PATH,
        )
        body = request.model_dump(mode="json")
        headers = build_internal_disagg_lifecycle_auth_headers(
            self._internal_disagg_auth_key,
            ARTIFACT_OBLIGATION_PATH,
            body,
        )
        async with self._session.post(
            url,
            json=body,
            **({"headers": headers} if headers else {}),
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise _ArtifactControlProtocolError(
                    f"artifact control endpoint returned HTTP {response.status}: {body[:1024]}"
                )
            try:
                result = ObligationResponse.model_validate_json(body)
            except ValidationError as error:
                raise _ArtifactControlProtocolError(
                    "artifact control endpoint returned an invalid response"
                ) from error
            if result.context_endpoint_incarnation is None:
                raise _ArtifactControlProtocolError(
                    "artifact control response omitted CTX endpoint incarnation"
                )
            if result.context_endpoint_incarnation != metadata.context_endpoint_incarnation:
                raise _ArtifactControlProtocolError(
                    "artifact control response came from a stale context endpoint incarnation"
                )
            grant_id = metadata.grant.consumer_grant_id
            acknowledged = self._context_endpoint_incarnations.get(grant_id)
            if acknowledged is None:
                self._context_endpoint_incarnations[grant_id] = result.context_endpoint_incarnation
            elif acknowledged != result.context_endpoint_incarnation:
                raise _ArtifactControlProtocolError(
                    "CTX endpoint incarnation changed for an acknowledged artifact"
                )
            expected_state = {
                ArtifactControlAction.RENEW: ArtifactObligationState.ACTIVE,
                ArtifactControlAction.RELEASE: ArtifactObligationState.RELEASED,
                ArtifactControlAction.ABORT: ArtifactObligationState.ABANDONED,
            }[action]
            if not result.accepted or result.state != expected_state.value:
                raise _ArtifactControlProtocolError(
                    result.reason
                    or (
                        "artifact control response did not acknowledge "
                        f"{action.value} as {expected_state.value}"
                    )
                )
            if action is ArtifactControlAction.RENEW:
                if result.ttl_s is None or result.artifact_ready is None:
                    raise _ArtifactControlProtocolError(
                        "active artifact renewal omitted its endpoint-owned TTL "
                        "or artifact-ready fact"
                    )
                return _ArtifactRenewalAck(
                    ttl_s=result.ttl_s,
                    artifact_ready=result.artifact_ready,
                )
            return None

    async def _sweep_loop(self) -> None:
        while True:
            self._sweep_once()
            await asyncio.sleep(self._sweep_interval_s)

    def _sweep_once(self) -> None:
        """Apply one deterministic expiry pass without inferring reuse."""
        now_s = self._clock()
        for identity in self._generation_grants.sweep_expired(now_s):
            self._retire_expired_generation_grant(
                identity,
                reason="generation intent grant expired",
            )
        for identity in self._artifact_obligations.sweep_expired(now_s):
            grant_id = identity.grant.consumer_grant_id
            ticket = self._artifact_tickets.pop(grant_id, None)
            if ticket is not None:
                self._abort_promise(ticket.promise)
            session_record = self._artifact_sessions.get(grant_id)
            if session_record is not None:
                self._remember_artifact_terminal(
                    grant_id,
                    session_record.session,
                    state=ArtifactObligationState.ABANDONED,
                    reason="artifact obligation expired",
                    now_s=now_s,
                )
        for grant_id, ticket in tuple(self._unbound_artifact_tickets.items()):
            if now_s >= ticket.expires_at_s:
                self._retire_context_artifact(
                    grant_id,
                    ticket.metadata.session,
                    reason="unbound context artifact expired",
                    now_s=now_s,
                )
        self._purge_artifact_replay_state(now_s)

    @staticmethod
    def _abort_promise(promise: object) -> None:
        abort = getattr(promise, "abort", None)
        if callable(abort):
            try:
                abort()
            except (AssertionError, RuntimeError) as error:
                logger.error("Failed to fan out lifecycle abort: %s", error)

    def _retire_expired_generation_grant(
        self,
        identity: GenerationGrantIdentity,
        *,
        reason: str,
        current_task: Optional[asyncio.Task] = None,
    ) -> Optional[_GenerationTicket]:
        """Apply every local side effect after a GEN grant becomes terminal."""
        grant_id = identity.consumer_grant_id
        now_s = self._clock()
        self._remember_generation_terminal(
            identity,
            state=GenerationGrantState.REVOKED,
            reason=reason,
            now_s=now_s,
        )
        admitted_request = self._grant_requests.get(grant_id)
        ticket = self._generation_tickets.pop(grant_id, None)
        if ticket is not None:
            if ticket.metadata.grant != identity:
                raise ObligationConflictError(
                    "terminal generation grant belongs to a different bound request"
                )
            if ticket.commit_gate.state is ReceiveCommitState.OPEN:
                try:
                    ticket.commit_gate.abort(reason)
                except ObligationConflictError:
                    pass
            self._abort_promise(ticket.promise)
            if ticket.metadata.context_control_endpoint:
                self._schedule_artifact_abort(ticket)
            for task in (ticket.renewal_task, ticket.handoff_task):
                if task is not None and task is not current_task:
                    task.cancel()
        elif admitted_request is not None and admitted_request.context_control_endpoint:
            self._schedule_context_artifact_abort(
                admitted_request,
                reason=reason,
            )
        self._grant_sessions.pop(grant_id, None)
        self._grant_requests.pop(grant_id, None)
        self._grant_generation_lifecycles.pop(grant_id, None)
        self._context_endpoint_incarnations.pop(grant_id, None)
        return ticket

    def _schedule_artifact_abort(self, ticket: _GenerationTicket) -> None:
        """Best-effort peer accelerator; CTX lease expiry remains the backstop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Cannot deliver CTX artifact abort for grant %s without a running event loop",
                ticket.metadata.grant.consumer_grant_id,
            )
            return

        async def _deliver() -> None:
            try:
                await self._settle_ticket_artifact(
                    ticket,
                    ArtifactControlAction.ABORT,
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
            ) as error:
                logger.warning(
                    "Failed to deliver CTX artifact abort for grant %s; "
                    "obligation expiry remains authoritative: %s",
                    ticket.metadata.grant.consumer_grant_id,
                    error,
                )

        task = loop.create_task(_deliver())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    def _schedule_context_artifact_abort(
        self,
        request: GenerationGrantRequest,
        *,
        reason: str,
    ) -> None:
        """Deliver an abort for a grant that expired before GEN insertion."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Cannot deliver pre-insertion CTX abort for grant %s without a running event loop",
                request.consumer_grant_id,
            )
            return

        async def _deliver() -> None:
            try:
                await self._send_context_artifact_abort(
                    request,
                    reason=reason,
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
            ) as error:
                logger.warning(
                    "Failed to deliver pre-insertion CTX abort for grant %s; "
                    "the coordinator connection and local request cancellation "
                    "remain backstops: %s",
                    request.consumer_grant_id,
                    error,
                )

        task = loop.create_task(_deliver())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def _send_context_artifact_abort(
        self,
        request: GenerationGrantRequest,
        *,
        reason: str,
    ) -> None:
        """Abort CTX compute when a GEN grant dies before scheduler insertion."""
        if self._session is None or request.context_control_endpoint is None:
            raise RuntimeError("lifecycle control is not started or CTX endpoint is missing")
        context_lifecycle = TransceiverLifecycleAdvertisement.from_value(
            request.context_transceiver_lifecycle
        )
        abort = ContextArtifactAbortRequest(
            lifecycle_protocol_version=1,
            logical_request_id=request.logical_request_id,
            prefill_artifact_id=request.prefill_artifact_id,
            artifact_version=request.artifact_version,
            handoff_attempt_uuid=request.handoff_attempt_uuid,
            consumer_grant_id=request.consumer_grant_id,
            transfer_session_id=request.transfer_session_id,
            context_endpoint_incarnation=UUID(context_lifecycle.instance_id),
            reason=reason,
        )
        url = self._control_url(
            request.context_control_endpoint,
            CONTEXT_ARTIFACT_ABORT_PATH,
        )
        body = abort.model_dump(mode="json")
        headers = build_internal_disagg_lifecycle_auth_headers(
            self._internal_disagg_auth_key,
            CONTEXT_ARTIFACT_ABORT_PATH,
            body,
        )
        async with self._session.post(
            url,
            json=body,
            **({"headers": headers} if headers else {}),
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise _ArtifactControlProtocolError(
                    "context artifact abort endpoint returned HTTP "
                    f"{response.status}: {body[:1024]}"
                )
            try:
                result = ObligationResponse.model_validate_json(body)
            except ValidationError as error:
                raise _ArtifactControlProtocolError(
                    "context artifact abort endpoint returned an invalid response"
                ) from error
            if result.context_endpoint_incarnation != UUID(context_lifecycle.instance_id):
                raise _ArtifactControlProtocolError(
                    "context artifact abort response came from a stale context endpoint incarnation"
                )
            if not result.accepted or result.state != ArtifactObligationState.ABANDONED.value:
                raise _ArtifactControlProtocolError(
                    result.reason or "context endpoint did not acknowledge the artifact abort"
                )

    def _require_session(
        self,
        grant_id: UUID,
        session: TransferProtocolIdentity,
    ) -> Optional[_GenerationAttemptTombstone]:
        existing = self._grant_sessions.get(grant_id)
        if existing is None:
            terminal = self._generation_attempt_tombstones.get(grant_id)
            if terminal is None:
                raise KeyError(f"generation grant {grant_id} has no admitted transfer session")
            if terminal.session != session:
                raise ValueError(
                    "generation grant belongs to a different terminal transfer session"
                )
            return terminal
        if existing != session:
            raise ValueError("generation grant belongs to a different transfer session")
        return None

    def _require_admitted_request_contract(self, params: object) -> None:
        """Bind scheduler submission to the immutable facts GEN admitted."""
        grant_id = getattr(params, "consumer_grant_id", None)
        try:
            grant_uuid = grant_id if isinstance(grant_id, UUID) else UUID(str(grant_id))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "consumer_grant_id must identify an admitted generation grant"
            ) from error
        admitted = self._grant_requests.get(grant_uuid)
        if admitted is None:
            raise KeyError(f"generation grant {grant_uuid} has no admitted request contract")
        actual_lifecycle = getattr(params, "context_transceiver_lifecycle", None)
        if actual_lifecycle is not None:
            actual_lifecycle = TransceiverLifecycleAdvertisement.from_value(actual_lifecycle)
        actual = (
            getattr(params, "context_control_endpoint", None),
            actual_lifecycle,
            getattr(params, "schedule_style", None),
            getattr(params, "ctx_dp_rank", None),
        )
        expected = (
            admitted.context_control_endpoint,
            admitted.context_transceiver_lifecycle,
            admitted.schedule_style,
            admitted.ctx_dp_rank,
        )
        if actual != expected:
            raise ValueError(
                "generation request conflicts with its admitted context endpoint, "
                "transceiver lifecycle, schedule style, or writer rank"
            )
        admitted_generation_lifecycle = self._grant_generation_lifecycles.get(grant_uuid)
        current_generation_lifecycle = self._current_endpoint_lifecycle()
        if (
            admitted_generation_lifecycle is not None or current_generation_lifecycle is not None
        ) and admitted_generation_lifecycle != current_generation_lifecycle:
            raise ValueError(
                "generation request reached a different transceiver incarnation "
                "than the endpoint that admitted it"
            )

    def _current_endpoint_lifecycle(
        self,
    ) -> Optional[TransceiverLifecycleAdvertisement]:
        if self._endpoint_lifecycle is None:
            return None
        lifecycle = TransceiverLifecycleAdvertisement.from_value(self._endpoint_lifecycle())
        if UUID(lifecycle.instance_id) != self._endpoint_incarnation:
            raise RuntimeError(
                "lifecycle-control endpoint no longer matches its transceiver incarnation"
            )
        return lifecycle

    def _require_endpoint_incarnation(
        self,
        expected: UUID,
        *,
        operation: str,
    ) -> None:
        self._current_endpoint_lifecycle()
        if expected != self._endpoint_incarnation:
            raise ValueError(f"{operation} targets a stale endpoint incarnation")

    @staticmethod
    def _same_grant_request(
        actual: GenerationGrantRequest,
        expected: GenerationGrantRequest,
    ) -> bool:
        return actual.model_dump(exclude={"ttl_s"}) == expected.model_dump(exclude={"ttl_s"})

    def _remember_generation_terminal(
        self,
        identity: GenerationGrantIdentity,
        *,
        state: GenerationGrantState,
        reason: str,
        now_s: float,
    ) -> None:
        grant_id = identity.consumer_grant_id
        existing = self._generation_attempt_tombstones.get(grant_id)
        if existing is not None:
            if (
                existing.identity != identity
                or existing.state is not state
                or existing.session != self._grant_sessions.get(grant_id, existing.session)
            ):
                raise ObligationConflictError(
                    "generation retirement conflicts with its terminal request"
                )
            self._generation_attempt_tombstones[grant_id] = _GenerationAttemptTombstone(
                existing.session,
                existing.request,
                existing.identity,
                existing.state,
                existing.reason or reason,
                max(
                    existing.retain_until_s,
                    now_s + self._replay_horizon_s,
                ),
            )
            return
        session = self._grant_sessions.get(grant_id)
        request = self._grant_requests.get(grant_id)
        if session is None or request is None:
            return
        if len(self._generation_attempt_tombstones) >= (self._max_generation_attempt_history):
            self._generation_attempt_tombstones.pop(next(iter(self._generation_attempt_tombstones)))
        self._generation_attempt_tombstones[grant_id] = _GenerationAttemptTombstone(
            session,
            request.model_copy(deep=True),
            identity,
            state,
            reason,
            now_s + self._replay_horizon_s,
        )

    def _purge_generation_replay_state(self, now_s: float) -> None:
        for grant_id, tombstone in tuple(self._generation_attempt_tombstones.items()):
            if now_s >= tombstone.retain_until_s:
                self._generation_attempt_tombstones.pop(grant_id, None)

    def _require_role(self, expected: ServerRole) -> None:
        if self._role != expected:
            raise RuntimeError(
                f"{expected.name} lifecycle operation reached a {self._role.name} server"
            )

    def _remember_artifact_session(
        self,
        grant_id: UUID,
        session: TransferProtocolIdentity,
        *,
        now_s: float,
    ) -> None:
        self._purge_artifact_replay_state(now_s)
        existing = self._artifact_sessions.get(grant_id)
        if existing is not None:
            if existing.session != session:
                raise ValueError("artifact control replay belongs to a different transfer session")
            self._artifact_sessions[grant_id] = _ArtifactSessionRecord(
                session,
                max(
                    existing.retain_until_s,
                    now_s + self._replay_horizon_s,
                ),
            )
            return
        if self._artifact_terminal_ids.contains(grant_id):
            raise RuntimeError("artifact identity is terminal or replay protection is saturated")
        if len(self._artifact_sessions) >= self._max_artifact_session_history:
            raise RuntimeError("artifact replay-protection capacity is exhausted")
        self._artifact_sessions[grant_id] = _ArtifactSessionRecord(
            session,
            now_s + self._replay_horizon_s,
        )

    def _retire_context_artifact(
        self,
        grant_id: UUID,
        session: TransferProtocolIdentity,
        *,
        reason: str,
        now_s: float,
    ) -> None:
        record = self._artifact_sessions.get(grant_id)
        if record is not None and record.session != session:
            raise ValueError("context artifact retirement belongs to a different transfer session")
        grant = self._artifact_grants.get(grant_id)
        if grant is not None:
            self._artifact_obligations.abandon(
                ArtifactObligationIdentity(grant),
                now_s=now_s,
            )
        else:
            self._artifact_obligations.release_unbound(grant_id)
        ticket = self._artifact_tickets.pop(grant_id, None)
        if ticket is None:
            ticket = self._unbound_artifact_tickets.pop(grant_id, None)
        if ticket is not None:
            if ticket.metadata.session != session:
                raise ValueError("context artifact retirement belongs to a different bound request")
            self._abort_promise(ticket.promise)
        context_ticket = self._context_attempt_tickets.pop(grant_id, None)
        if context_ticket is not None:
            if context_ticket.metadata.session != session:
                raise ValueError(
                    "context artifact retirement belongs to a different active context request"
                )
            self._abort_promise(context_ticket.promise)
        self._remember_artifact_terminal(
            grant_id,
            session,
            state=ArtifactObligationState.ABANDONED,
            reason=reason,
            now_s=now_s,
        )

    def _remember_artifact_terminal(
        self,
        grant_id: UUID,
        session: TransferProtocolIdentity,
        *,
        state: ArtifactObligationState,
        reason: str,
        now_s: float,
    ) -> None:
        self._artifact_terminal_ids.add(grant_id, now_s=now_s)
        self._artifact_sessions.pop(grant_id, None)
        self._artifact_grants.pop(grant_id, None)
        context_ticket = self._context_attempt_tickets.pop(grant_id, None)
        if context_ticket is not None:
            if context_ticket.metadata.session != session:
                raise ObligationConflictError(
                    "artifact retirement conflicts with its active context request"
                )
            self._abort_promise(context_ticket.promise)
        existing = self._artifact_attempt_tombstones.get(grant_id)
        if existing is not None:
            if existing.session != session or existing.state is not state:
                raise ObligationConflictError(
                    "artifact retirement conflicts with its terminal session"
                )
            self._artifact_attempt_tombstones[grant_id] = _ArtifactAttemptTombstone(
                session,
                state,
                existing.reason or reason,
                max(
                    existing.retain_until_s,
                    now_s + self._replay_horizon_s,
                ),
            )
            return
        if len(self._artifact_attempt_tombstones) < (self._max_artifact_terminal_history):
            self._artifact_attempt_tombstones[grant_id] = _ArtifactAttemptTombstone(
                session,
                state,
                reason,
                now_s + self._replay_horizon_s,
            )

    def _purge_artifact_replay_state(self, now_s: float) -> None:
        self._artifact_terminal_ids.advance(now_s)
        for grant_id, tombstone in tuple(self._artifact_attempt_tombstones.items()):
            if now_s >= tombstone.retain_until_s:
                self._artifact_attempt_tombstones.pop(grant_id, None)
        for grant_id, record in tuple(self._artifact_sessions.items()):
            if now_s < record.retain_until_s:
                continue
            if grant_id in self._artifact_tickets or grant_id in self._unbound_artifact_tickets:
                continue
            self._artifact_sessions.pop(grant_id, None)
            self._artifact_grants.pop(grant_id, None)

    def _artifact_response(
        self,
        *,
        accepted: bool,
        state: ArtifactObligationState,
        reason: str = "",
        expires_at_s: Optional[float] = None,
        artifact_ready: Optional[bool] = None,
        now_s: Optional[float] = None,
    ) -> ObligationResponse:
        ttl_s = None
        if state is ArtifactObligationState.ACTIVE:
            if expires_at_s is None:
                expires_at_s = self._clock() + self._artifact_ttl_s
            if now_s is None:
                now_s = self._clock()
            ttl_s = expires_at_s - now_s
            if ttl_s <= 0:
                accepted = False
                state = ArtifactObligationState.ABANDONED
                reason = reason or "artifact obligation expired before acknowledgement"
                ttl_s = None
        return ObligationResponse(
            lifecycle_protocol_version=1,
            accepted=accepted,
            state=state.value,
            reason=reason,
            ttl_s=ttl_s,
            artifact_ready=artifact_ready,
            context_endpoint_incarnation=self._endpoint_incarnation,
        )

    @staticmethod
    def _control_url(endpoint: str, path: str) -> str:
        base = endpoint if "://" in endpoint else f"http://{endpoint}"
        return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


__all__ = [
    "ARTIFACT_OBLIGATION_PATH",
    "CONTEXT_ARTIFACT_ABORT_PATH",
    "GENERATION_GRANT_ABORT_PATH",
    "GENERATION_GRANT_PATH",
    "GENERATION_GRANT_RENEW_PATH",
    "ArtifactControlAction",
    "ArtifactObligationRequest",
    "ContextArtifactAbortRequest",
    "DisaggLifecycleControl",
    "GenerationGrantAbortRequest",
    "GenerationGrantDecisionResponse",
    "GenerationGrantRenewRequest",
    "GenerationGrantRequest",
    "ObligationResponse",
    "RequestLifecycleMetadata",
]
