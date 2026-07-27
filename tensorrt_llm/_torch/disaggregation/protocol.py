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

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, TypeAlias
from uuid import UUID

import msgpack

from tensorrt_llm._torch.disaggregation.lifecycle import LifecycleCapability


class ProtocolVersion(IntEnum):
    """Disaggregated lifecycle wire-protocol versions."""

    QUALIFIED_LEGACY = 0
    GENERATION_SAFE = 1


class ProtocolIdentityError(ValueError):
    """Raised when a lifecycle message does not carry a valid identity."""


class StaleProtocolMessageError(ProtocolIdentityError):
    """Raised before a stale message can mutate a newer session."""


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolIdentityError(f"{name} must be an integer >= {minimum}")
    return value


def _require_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolIdentityError(f"{name} must be a non-empty string")
    return value


def _require_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, UUID) or value.int == 0:
        raise ProtocolIdentityError(f"{name} must be a non-nil UUID")
    return value


def _decode_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, bytes) or len(value) != 16:
        raise ProtocolIdentityError(f"{name} must be a 16-byte UUID")
    return _require_uuid(name, UUID(bytes=value))


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolIdentityError(f"{name} must be a map")
    if any(not isinstance(key, str) for key in value):
        raise ProtocolIdentityError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: frozenset[str],
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual))
        unexpected = ", ".join(sorted(actual - expected))
        details = []
        if missing:
            details.append(f"missing [{missing}]")
        if unexpected:
            details.append(f"unexpected [{unexpected}]")
        raise ProtocolIdentityError(f"{name} has invalid fields: {'; '.join(details)}")


@dataclass(frozen=True, slots=True)
class EndpointIdentity:
    """One immutable worker/process incarnation."""

    instance_name: str
    instance_rank: int
    incarnation: UUID

    def __post_init__(self) -> None:
        _require_nonempty_str("instance_name", self.instance_name)
        _require_int("instance_rank", self.instance_rank)
        _require_uuid("incarnation", self.incarnation)


@dataclass(frozen=True, slots=True)
class AllocationWireIdentity:
    """Wire-neutral identity of one allocator-issued allocation generation.

    ``allocator_domain_id`` is opaque. Python allocators use their UUID string,
    while the C++ allocator uses the decimal spelling of its uint64 domain.
    """

    allocator_domain_id: str
    request_id: int | None
    allocation_generation: int

    def __post_init__(self) -> None:
        _require_nonempty_str("allocator_domain_id", self.allocator_domain_id)
        if self.request_id is not None:
            _require_int("request_id", self.request_id)
        _require_int("allocation_generation", self.allocation_generation, minimum=1)

    @classmethod
    def from_local(cls, identity: object) -> "AllocationWireIdentity":
        """Convert either runtime allocator identity without importing it."""
        try:
            domain = getattr(identity, "allocator_domain_id")
            request_id = getattr(identity, "request_id")
            generation = getattr(identity, "allocation_generation")
        except AttributeError as error:
            raise ProtocolIdentityError(
                "local allocation identity is missing required fields"
            ) from error
        if domain is None or isinstance(domain, bool):
            raise ProtocolIdentityError("local allocation identity has an invalid allocator domain")
        return cls(
            allocator_domain_id=str(domain),
            request_id=request_id,
            allocation_generation=generation,
        )


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """One globally unique placement attempt for an immutable artifact."""

    logical_request_id: int
    prefill_artifact_id: UUID
    artifact_version: int
    handoff_attempt_uuid: UUID

    def __post_init__(self) -> None:
        _require_int("logical_request_id", self.logical_request_id)
        _require_uuid("prefill_artifact_id", self.prefill_artifact_id)
        _require_int("artifact_version", self.artifact_version)
        _require_uuid("handoff_attempt_uuid", self.handoff_attempt_uuid)


@dataclass(frozen=True, slots=True)
class TransferProtocolIdentity:
    """One exact transfer session for an immutable request attempt.

    Endpoint identities are operation-scoped rather than session-scoped: one
    receiver session can publish to multiple source writer ranks.
    """

    attempt: AttemptIdentity
    transfer_session_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptIdentity):
            raise ProtocolIdentityError("attempt must be an AttemptIdentity")
        _require_uuid("transfer_session_id", self.transfer_session_id)

    @property
    def logical_request_id(self) -> int:
        return self.attempt.logical_request_id


@dataclass(frozen=True, slots=True)
class PublicationIdentity:
    """Receiver-authorized destination publication for one exact writer."""

    session: TransferProtocolIdentity
    source_endpoint: EndpointIdentity
    destination_endpoint: EndpointIdentity
    destination_allocation: AllocationWireIdentity
    operation_id: UUID
    slice_id: int
    writer_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.session, TransferProtocolIdentity):
            raise ProtocolIdentityError("session must be a TransferProtocolIdentity")
        if not isinstance(self.source_endpoint, EndpointIdentity):
            raise ProtocolIdentityError("source_endpoint must be an EndpointIdentity")
        if not isinstance(self.destination_endpoint, EndpointIdentity):
            raise ProtocolIdentityError("destination_endpoint must be an EndpointIdentity")
        if not isinstance(self.destination_allocation, AllocationWireIdentity):
            raise ProtocolIdentityError("destination_allocation must be an AllocationWireIdentity")
        _require_uuid("operation_id", self.operation_id)
        _require_int("slice_id", self.slice_id)
        _require_int("writer_rank", self.writer_rank)
        if self.source_endpoint.instance_rank != self.writer_rank:
            raise ProtocolIdentityError("source_endpoint instance_rank must match writer_rank")

    @property
    def logical_request_id(self) -> int:
        return self.session.logical_request_id


@dataclass(frozen=True, slots=True)
class OperationIdentity:
    """Publication plus the exact source allocation used by its writer."""

    publication: PublicationIdentity
    source_allocation: AllocationWireIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.publication, PublicationIdentity):
            raise ProtocolIdentityError("publication must be a PublicationIdentity")
        if not isinstance(self.source_allocation, AllocationWireIdentity):
            raise ProtocolIdentityError("source_allocation must be an AllocationWireIdentity")

    @property
    def logical_request_id(self) -> int:
        return self.publication.logical_request_id


# Compatibility name for early callers of the Phase-4 scaffolding. A transfer
# protocol identity intentionally excludes endpoint and allocation generations:
# endpoints are bound by each publication, while allocations are derived from
# the immutable leases protecting each operation.
TransferSessionIdentity = TransferProtocolIdentity


@dataclass(frozen=True, slots=True)
class QualifiedLegacyIdentity:
    """Explicit protocol-v0 identity for a pre-negotiated legacy session."""

    logical_request_id: int

    def __post_init__(self) -> None:
        _require_int("logical_request_id", self.logical_request_id)


PublicationWireIdentity: TypeAlias = QualifiedLegacyIdentity | PublicationIdentity
ResultWireIdentity: TypeAlias = QualifiedLegacyIdentity | PublicationIdentity | OperationIdentity
WireIdentity: TypeAlias = PublicationWireIdentity | ResultWireIdentity
ProtocolSessionKey: TypeAlias = int | TransferProtocolIdentity


_LEGACY_KEYS = frozenset({"protocol_version", "qualified_legacy_mode", "logical_request_id"})
_V1_KEYS = frozenset({"protocol_version", "kind", "identity"})
_PUBLICATION_KEYS = frozenset(
    {
        "session",
        "source_endpoint",
        "destination_endpoint",
        "destination_allocation",
        "operation_id",
        "slice_id",
        "writer_rank",
    }
)
_OPERATION_KEYS = frozenset({"publication", "source_allocation"})
_SESSION_KEYS = frozenset({"attempt", "transfer_session_id"})
_ATTEMPT_KEYS = frozenset(
    {
        "logical_request_id",
        "prefill_artifact_id",
        "artifact_version",
        "handoff_attempt_uuid",
    }
)
_ENDPOINT_KEYS = frozenset({"instance_name", "instance_rank", "incarnation"})
_ALLOCATION_KEYS = frozenset({"allocator_domain_id", "request_id", "allocation_generation"})
_MAX_WIRE_IDENTITY_BYTES = 4096
PROTOCOL_V1_REQUIRED_CAPABILITIES = frozenset(
    capability.value
    for capability in (
        LifecycleCapability.ALLOCATION_GENERATION_LEASES,
        LifecycleCapability.ATTEMPT_IDENTITY,
        LifecycleCapability.CANCEL_BEFORE_CREATE_TOMBSTONES,
        LifecycleCapability.ENDPOINT_INCARNATION,
        LifecycleCapability.EXACT_WRITER_TRACKING,
        LifecycleCapability.PER_OPERATION_QUIESCENCE,
        LifecycleCapability.PUBLICATION_GATE,
        LifecycleCapability.SUBMISSION_FENCE,
        LifecycleCapability.TERMINAL_RESULT_REPLAY,
    )
)


def validate_protocol_advertisement(
    protocol_version: int,
    *,
    endpoint_incarnation: UUID | None,
    capabilities: frozenset[str],
    qualified_legacy_mode: bool,
) -> None:
    """Validate that a peer never advertises partially implemented protocol-v1."""
    version = _require_int("protocol_version", protocol_version)
    if version == ProtocolVersion.QUALIFIED_LEGACY:
        if not qualified_legacy_mode:
            raise ProtocolIdentityError(
                "protocol-v0 advertisement must explicitly qualify legacy mode"
            )
        return
    if version != ProtocolVersion.GENERATION_SAFE:
        raise ProtocolIdentityError(f"unsupported protocol_version {version}")
    _require_uuid("endpoint_incarnation", endpoint_incarnation)
    if qualified_legacy_mode:
        raise ProtocolIdentityError(
            "protocol-v1 advertisement cannot also select qualified legacy mode"
        )
    missing = PROTOCOL_V1_REQUIRED_CAPABILITIES - capabilities
    if missing:
        raise ProtocolIdentityError(
            "protocol-v1 advertisement is missing identity capabilities: "
            + ", ".join(sorted(missing))
        )


def _encode_endpoint(identity: EndpointIdentity) -> dict[str, object]:
    return {
        "instance_name": identity.instance_name,
        "instance_rank": identity.instance_rank,
        "incarnation": identity.incarnation.bytes,
    }


def _decode_endpoint(value: object) -> EndpointIdentity:
    data = _require_mapping("endpoint identity", value)
    _require_exact_keys("endpoint identity", data, _ENDPOINT_KEYS)
    return EndpointIdentity(
        instance_name=_require_nonempty_str("instance_name", data["instance_name"]),
        instance_rank=_require_int("instance_rank", data["instance_rank"]),
        incarnation=_decode_uuid("incarnation", data["incarnation"]),
    )


def _encode_allocation(identity: AllocationWireIdentity) -> dict[str, object]:
    return {
        "allocator_domain_id": identity.allocator_domain_id,
        "request_id": identity.request_id,
        "allocation_generation": identity.allocation_generation,
    }


def _decode_allocation(value: object) -> AllocationWireIdentity:
    data = _require_mapping("allocation identity", value)
    _require_exact_keys("allocation identity", data, _ALLOCATION_KEYS)
    request_id = data["request_id"]
    if request_id is not None:
        request_id = _require_int("request_id", request_id)
    return AllocationWireIdentity(
        allocator_domain_id=_require_nonempty_str(
            "allocator_domain_id", data["allocator_domain_id"]
        ),
        request_id=request_id,
        allocation_generation=_require_int(
            "allocation_generation",
            data["allocation_generation"],
            minimum=1,
        ),
    )


def _encode_attempt(identity: AttemptIdentity) -> dict[str, object]:
    return {
        "logical_request_id": identity.logical_request_id,
        "prefill_artifact_id": identity.prefill_artifact_id.bytes,
        "artifact_version": identity.artifact_version,
        "handoff_attempt_uuid": identity.handoff_attempt_uuid.bytes,
    }


def _decode_attempt(value: object) -> AttemptIdentity:
    data = _require_mapping("attempt identity", value)
    _require_exact_keys("attempt identity", data, _ATTEMPT_KEYS)
    return AttemptIdentity(
        logical_request_id=_require_int("logical_request_id", data["logical_request_id"]),
        prefill_artifact_id=_decode_uuid("prefill_artifact_id", data["prefill_artifact_id"]),
        artifact_version=_require_int("artifact_version", data["artifact_version"]),
        handoff_attempt_uuid=_decode_uuid("handoff_attempt_uuid", data["handoff_attempt_uuid"]),
    )


def _encode_session(identity: TransferProtocolIdentity) -> dict[str, object]:
    return {
        "attempt": _encode_attempt(identity.attempt),
        "transfer_session_id": identity.transfer_session_id.bytes,
    }


def _decode_session(value: object) -> TransferProtocolIdentity:
    data = _require_mapping("transfer session identity", value)
    _require_exact_keys("transfer session identity", data, _SESSION_KEYS)
    return TransferProtocolIdentity(
        attempt=_decode_attempt(data["attempt"]),
        transfer_session_id=_decode_uuid("transfer_session_id", data["transfer_session_id"]),
    )


def _encode_publication(identity: PublicationIdentity) -> dict[str, object]:
    return {
        "session": _encode_session(identity.session),
        "source_endpoint": _encode_endpoint(identity.source_endpoint),
        "destination_endpoint": _encode_endpoint(identity.destination_endpoint),
        "destination_allocation": _encode_allocation(identity.destination_allocation),
        "operation_id": identity.operation_id.bytes,
        "slice_id": identity.slice_id,
        "writer_rank": identity.writer_rank,
    }


def _decode_publication(value: object) -> PublicationIdentity:
    data = _require_mapping("publication identity", value)
    _require_exact_keys("publication identity", data, _PUBLICATION_KEYS)
    return PublicationIdentity(
        session=_decode_session(data["session"]),
        source_endpoint=_decode_endpoint(data["source_endpoint"]),
        destination_endpoint=_decode_endpoint(data["destination_endpoint"]),
        destination_allocation=_decode_allocation(data["destination_allocation"]),
        operation_id=_decode_uuid("operation_id", data["operation_id"]),
        slice_id=_require_int("slice_id", data["slice_id"]),
        writer_rank=_require_int("writer_rank", data["writer_rank"]),
    )


def encode_wire_identity(identity: WireIdentity) -> bytes:
    """Encode an exact v1 publication/result or qualified v0 identity."""
    if isinstance(identity, QualifiedLegacyIdentity):
        data: dict[str, object] = {
            "protocol_version": int(ProtocolVersion.QUALIFIED_LEGACY),
            "qualified_legacy_mode": True,
            "logical_request_id": identity.logical_request_id,
        }
    elif isinstance(identity, PublicationIdentity):
        data = {
            "protocol_version": int(ProtocolVersion.GENERATION_SAFE),
            "kind": "publication",
            "identity": _encode_publication(identity),
        }
    elif isinstance(identity, OperationIdentity):
        data = {
            "protocol_version": int(ProtocolVersion.GENERATION_SAFE),
            "kind": "operation",
            "identity": {
                "publication": _encode_publication(identity.publication),
                "source_allocation": _encode_allocation(identity.source_allocation),
            },
        }
    else:
        raise ProtocolIdentityError(f"unsupported wire identity type: {type(identity)!r}")
    payload = msgpack.packb(data, use_bin_type=True)
    if len(payload) > _MAX_WIRE_IDENTITY_BYTES:
        raise ProtocolIdentityError("wire identity payload is too large")
    return payload


def decode_wire_identity(
    payload: bytes,
    *,
    allow_qualified_legacy: bool = False,
) -> WireIdentity:
    """Decode and fully validate a lifecycle wire identity.

    Legacy identities are rejected unless the caller already negotiated the
    qualified compatibility path. Generation-safe messages have no permissive
    fallback when any required field is absent.
    """
    if not isinstance(payload, bytes):
        raise ProtocolIdentityError("wire identity payload must be bytes")
    if len(payload) > _MAX_WIRE_IDENTITY_BYTES:
        raise ProtocolIdentityError("wire identity payload is too large")
    try:
        unpacked = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    except (ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as error:
        raise ProtocolIdentityError("wire identity is not valid msgpack") from error
    data = _require_mapping("wire identity", unpacked)
    version = _require_int("protocol_version", data.get("protocol_version"))
    if version == ProtocolVersion.QUALIFIED_LEGACY:
        _require_exact_keys("legacy wire identity", data, _LEGACY_KEYS)
        if data["qualified_legacy_mode"] is not True:
            raise ProtocolIdentityError("protocol-v0 identity is not qualified")
        if not allow_qualified_legacy:
            raise ProtocolIdentityError("qualified protocol-v0 identity was not negotiated")
        return QualifiedLegacyIdentity(
            logical_request_id=_require_int("logical_request_id", data["logical_request_id"])
        )
    if version != ProtocolVersion.GENERATION_SAFE:
        raise ProtocolIdentityError(f"unsupported protocol_version {version}")
    _require_exact_keys("generation-safe wire identity", data, _V1_KEYS)
    kind = _require_nonempty_str("kind", data["kind"])
    if kind == "publication":
        return _decode_publication(data["identity"])
    if kind != "operation":
        raise ProtocolIdentityError(f"unsupported protocol-v1 identity kind {kind!r}")
    operation = _require_mapping("operation identity", data["identity"])
    _require_exact_keys("operation identity", operation, _OPERATION_KEYS)
    return OperationIdentity(
        publication=_decode_publication(operation["publication"]),
        source_allocation=_decode_allocation(operation["source_allocation"]),
    )


def require_exact_identity(
    received: WireIdentity,
    expected: WireIdentity,
) -> None:
    """Reject stale or cross-session input before the caller mutates state."""
    if type(received) is not type(expected) or received != expected:
        raise StaleProtocolMessageError(
            "wire identity does not match the active transfer operation"
        )


def require_result_for_publication(
    result: ResultWireIdentity,
    publication: PublicationWireIdentity,
) -> None:
    """Validate a result before it can mutate its destination publication."""
    if isinstance(publication, QualifiedLegacyIdentity):
        require_exact_identity(result, publication)
        return
    if isinstance(result, PublicationIdentity):
        require_exact_identity(result, publication)
        return
    if not isinstance(result, OperationIdentity) or result.publication != publication:
        raise StaleProtocolMessageError(
            "result identity does not match the active destination publication"
        )


def protocol_session_key(
    logical_request_id: int,
    identity: (
        TransferProtocolIdentity
        | PublicationIdentity
        | OperationIdentity
        | QualifiedLegacyIdentity
        | None
    ),
) -> ProtocolSessionKey:
    """Return the replay/tombstone key for the negotiated protocol."""
    request_id = _require_int("logical_request_id", logical_request_id)
    if identity is None or isinstance(identity, QualifiedLegacyIdentity):
        return request_id
    if isinstance(identity, OperationIdentity):
        session = identity.publication.session
    elif isinstance(identity, PublicationIdentity):
        session = identity.session
    elif isinstance(identity, TransferProtocolIdentity):
        session = identity
    else:
        raise ProtocolIdentityError(f"unsupported session identity type: {type(identity)!r}")
    if session.logical_request_id != request_id:
        raise ProtocolIdentityError(
            "logical_request_id does not match the transfer session identity"
        )
    return session


_TRANSFER_PARAM_IDENTITY_FIELDS = (
    "logical_request_id",
    "prefill_artifact_id",
    "artifact_version",
    "handoff_attempt_uuid",
    "consumer_grant_id",
    "transfer_session_id",
)


def _canonical_param_uuid(name: str, value: object) -> UUID:
    if not isinstance(value, str):
        raise ProtocolIdentityError(f"{name} must be a canonical non-nil UUID string")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ProtocolIdentityError(f"{name} must be a canonical non-nil UUID string") from error
    if parsed.int == 0 or str(parsed) != value:
        raise ProtocolIdentityError(f"{name} must be a canonical non-nil UUID string")
    return parsed


def transfer_protocol_identity_from_params(
    params: object | None,
) -> TransferProtocolIdentity | None:
    """Build the immutable transfer identity from one atomic params tuple.

    The consumer grant is control-plane identity, not part of the transfer
    session key, but requiring and validating it here prevents a request from
    entering protocol-v1 with a partially propagated lifecycle tuple.
    """
    if params is None:
        return None
    values = {name: getattr(params, name, None) for name in _TRANSFER_PARAM_IDENTITY_FIELDS}
    present = {name for name, value in values.items() if value is not None}
    if not present:
        return None
    expected = set(_TRANSFER_PARAM_IDENTITY_FIELDS)
    if present != expected:
        missing = ", ".join(sorted(expected - present))
        raise ProtocolIdentityError(
            f"transfer protocol identity fields must be provided together; missing: {missing}"
        )

    logical_request_id = _require_int(
        "logical_request_id",
        values["logical_request_id"],
    )
    artifact_version = _require_int(
        "artifact_version",
        values["artifact_version"],
    )
    parsed_uuids = {
        name: _canonical_param_uuid(name, values[name])
        for name in (
            "prefill_artifact_id",
            "handoff_attempt_uuid",
            "consumer_grant_id",
            "transfer_session_id",
        )
    }
    if len(set(parsed_uuids.values())) != len(parsed_uuids):
        raise ProtocolIdentityError("transfer protocol UUID fields must be distinct")
    return TransferProtocolIdentity(
        attempt=AttemptIdentity(
            logical_request_id=logical_request_id,
            prefill_artifact_id=parsed_uuids["prefill_artifact_id"],
            artifact_version=artifact_version,
            handoff_attempt_uuid=parsed_uuids["handoff_attempt_uuid"],
        ),
        transfer_session_id=parsed_uuids["transfer_session_id"],
    )
