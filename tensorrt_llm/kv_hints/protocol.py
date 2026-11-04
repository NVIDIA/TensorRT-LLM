# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV hint wire envelopes and immutable request-admission snapshots.

The envelope validates transport metadata. Action payloads stay opaque; accepting
an envelope does not activate a backend, perform a fetch, or apply source policy.
"""

import json
from typing import NoReturn, TypeAlias

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


def _reject_mutation(*args: object, **kwargs: object) -> NoReturn:
    raise TypeError("KV hint metadata is immutable")


class _FrozenDict(dict[str, object]):
    """Read-only JSON object retaining dict compatibility for consumers."""

    __setitem__ = __delitem__ = __ior__ = _reject_mutation
    clear = pop = popitem = setdefault = update = _reject_mutation

    def __reduce__(self) -> tuple[type, tuple[dict[str, object]]]:
        return type(self), (dict(self),)


class _FrozenList(list[object]):
    """Read-only JSON array retaining list compatibility for consumers."""

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _reject_mutation
    append = clear = extend = insert = pop = remove = reverse = sort = _reject_mutation

    def __reduce__(self) -> tuple[type, tuple[list[object]]]:
        return type(self), (list(self),)


def _freeze(value: object) -> object:
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, dict):
        return _FrozenDict((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return value


class _KvHintActionSnapshot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, object]
    _extra: dict[str, object] | None = None

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "payload", _freeze(self.payload))
        msgspec.structs.force_setattr(self, "_extra", _freeze(self._extra))


class KvHintsSnapshot(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Admitted msgspec envelope with read-only nested request metadata.

    Consumers can pass this envelope directly to a compatible provider after
    selection and authorization. Optional wire metadata remains in ``_extra``
    for lossless request transport and reconstruction of the public envelope.
    """

    protocol_version: str
    message_id: str
    actions: list[_KvHintActionSnapshot]
    _extra: dict[str, object] | None = None

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "actions", _freeze(self.actions))
        msgspec.structs.force_setattr(self, "_extra", _freeze(self._extra))

    @property
    def wire_json(self) -> bytes:
        """Render the public wire format on demand; request transport stores no JSON."""
        return self.envelope().model_dump_json().encode("utf-8")

    def envelope(self) -> KvHintsEnvelope:
        """Build an independent public model for inspection or wire serialization.

        Returns:
            A validated envelope whose mutation cannot change this snapshot.
        """
        fields = msgspec.to_builtins(self)
        for action in fields["actions"]:
            action.update(action.pop("_extra", None) or {})
        fields.update(fields.pop("_extra", None) or {})
        return KvHintsEnvelope.model_validate(fields)


def admit_kv_hints(hints: KvHintsInput | None) -> KvHintsEnvelope | None:
    """Apply version, size, schema and proxy transport checks to optional hints.

    Args:
        hints: A typed envelope or raw JSON object from a trusted caller.

    Returns:
        A validated envelope, or None after counting an unsupported, oversized
        or malformed hint. Unsupported raw versions need not follow the supported
        schema. Direct construction of a typed envelope still validates its fields.
    """
    return _admit_kv_hints(hints)


def _admit_kv_hints(hints: KvHintsInput | None) -> KvHintsEnvelope | None:
    """Return the model after schema, wire-size and transport checks."""
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
        msgspec.msgpack.encode(validated.model_dump(mode="json"))
    except (OverflowError, TypeError, ValueError, RecursionError):
        count_ignored_kv_hints("malformed")
        return None
    return validated


def snapshot_kv_hints(hints: KvHintsInput | None) -> KvHintsSnapshot | None:
    """Validate and convert hints to a read-only msgspec envelope at admission.

    Args:
        hints: The optional producer envelope or raw JSON object.

    Returns:
        A msgspec envelope, or None for absent or ignored hints. Malformed
        metadata is counted and discarded before the request is queued.
    """
    envelope = _admit_kv_hints(hints)
    if envelope is None:
        return None
    return KvHintsSnapshot(
        protocol_version=envelope.protocol_version,
        message_id=envelope.message_id,
        actions=[
            _KvHintActionSnapshot(
                action_id=action.action_id,
                action_type=action.action_type,
                action_version=action.action_version,
                payload=action.payload,
                _extra=action.__pydantic_extra__ or None,
            )
            for action in envelope.actions
        ],
        _extra=envelope.__pydantic_extra__ or None,
    )
