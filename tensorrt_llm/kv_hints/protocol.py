# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV hint wire envelopes and immutable request-admission snapshots.

The envelope validates transport metadata. Action payloads stay opaque; accepting
an envelope does not activate a backend, perform a fetch, or apply source policy.
"""

import json
from dataclasses import dataclass
from typing import TypeAlias

import msgspec
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .metrics import count_ignored_kv_hints

__all__ = [
    "KvHintAction",
    "KvHintsEnvelope",
    "KvHintsInput",
    "KvHintsSnapshot",
    "admit_kv_hints",
    "snapshot_kv_hints",
]

MAX_KV_HINTS_BYTES = 64 * 1024
_SUPPORTED_PROTOCOL_VERSION = "0.1"


class KvHintAction(BaseModel):
    """One producer-versioned action carried without interpreting its payload."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False, revalidate_instances="always")
    __pydantic_extra__: dict[str, JsonValue]

    action_id: str = Field(min_length=1, description="Producer's action identifier.")
    action_type: str = Field(min_length=1, description="Opaque action type, such as kv.fetch.")
    action_version: str = Field(min_length=1, description="Producer's action schema version.")
    payload: dict[str, JsonValue] = Field(
        description="Opaque JSON object owned by the action consumer."
    )


class KvHintsEnvelope(BaseModel):
    """Versioned KV hints transported as prototype per-request metadata.

    Unknown action types, action versions and optional JSON fields are preserved. Consumers must
    negotiate and validate executable actions separately from this wire format.
    """

    model_config = ConfigDict(extra="allow", allow_inf_nan=False, revalidate_instances="always")
    __pydantic_extra__: dict[str, JsonValue]

    protocol_version: str = Field(
        min_length=1,
        description="KV hint envelope protocol version; unsupported versions are ignored at admission.",
    )
    message_id: str = Field(min_length=1, description="Producer's hint message identifier.")
    actions: list[KvHintAction] = Field(description="Actions in their producer-specified order.")


KvHintsInput: TypeAlias = KvHintsEnvelope | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class KvHintsSnapshot:
    """Immutable admitted metadata, including every nested payload value.

    Attributes:
        wire_json: Validated envelope encoded as UTF-8 JSON bytes.
    """

    wire_json: bytes

    def envelope(self) -> KvHintsEnvelope:
        """Decode an independent copy for a consumer without exposing mutable state.

        Returns:
            A validated envelope whose mutation cannot change this snapshot.
        """
        return KvHintsEnvelope.model_validate_json(self.wire_json)


def admit_kv_hints(hints: KvHintsInput | None) -> KvHintsEnvelope | None:
    """Apply version, size, schema and proxy transport checks to optional hints.

    Args:
        hints: A typed envelope or raw JSON object from a trusted caller.

    Returns:
        A validated envelope, or None after counting an unsupported, oversized
        or malformed hint. Unsupported raw versions need not follow the supported
        schema. Direct construction of a typed envelope still validates its fields.
    """
    admitted = _admit_kv_hints(hints)
    return None if admitted is None else admitted[0]


def _admit_kv_hints(hints: KvHintsInput | None) -> tuple[KvHintsEnvelope, bytes] | None:
    """Return the admitted model and its checked, decodable JSON encoding."""
    if hints is None:
        return None
    if isinstance(hints, KvHintsEnvelope):
        version = getattr(hints, "protocol_version", None)
    elif isinstance(hints, dict):
        version = hints.get("protocol_version")
    else:
        count_ignored_kv_hints("malformed")
        return None
    if isinstance(version, str) and version and version != _SUPPORTED_PROTOCOL_VERSION:
        count_ignored_kv_hints("unknown_version")
        return None
    # Raw HTTP/Python input is bounded before nested validation. The validated
    # encoding below enforces the same limit after any model normalization.
    if isinstance(hints, dict):
        try:
            wire_json = json.dumps(
                hints, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            count_ignored_kv_hints("malformed")
            return None
        if len(wire_json) > MAX_KV_HINTS_BYTES:
            count_ignored_kv_hints("oversized")
            return None
    try:
        validated = KvHintsEnvelope.model_validate(hints)
        wire_json = validated.model_dump_json().encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        count_ignored_kv_hints("malformed")
        return None
    if len(wire_json) > MAX_KV_HINTS_BYTES:
        count_ignored_kv_hints("oversized")
        return None
    try:
        validated = KvHintsEnvelope.model_validate_json(wire_json)
        msgspec.msgpack.encode(validated.model_dump(mode="json"))
    except (OverflowError, TypeError, ValueError, RecursionError):
        count_ignored_kv_hints("malformed")
        return None
    return validated, wire_json


def snapshot_kv_hints(hints: KvHintsInput | None) -> KvHintsSnapshot | None:
    """Validate and freeze optional hints before queueing a generation request.

    Args:
        hints: The optional producer envelope or raw JSON object.

    Returns:
        Immutable UTF-8 JSON, or None for absent or ignored hints. Malformed
        metadata is counted and discarded before the request is queued.
    """
    admitted = _admit_kv_hints(hints)
    return None if admitted is None else KvHintsSnapshot(admitted[1])
