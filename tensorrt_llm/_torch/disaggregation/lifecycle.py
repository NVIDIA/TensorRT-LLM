# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Protocol, Sequence, runtime_checkable


class LogicalDisposition(str, Enum):
    """Logical result of a lifecycle operation.

    Logical acceptance never implies that an allocation can be reused.
    """

    ACCEPTED = "ACCEPTED"
    ALREADY_TERMINAL = "ALREADY_TERMINAL"
    NOT_FOUND = "NOT_FOUND"
    REJECTED = "REJECTED"


class PhysicalDisposition(str, Enum):
    """Physical disposition of resources associated with a transfer."""

    NOT_EXPOSED = "NOT_EXPOSED"
    ACTIVE = "ACTIVE"
    QUIESCING = "QUIESCING"
    QUIESCED_SUCCESS = "QUIESCED_SUCCESS"
    QUIESCED_FAILURE = "QUIESCED_FAILURE"
    IN_DOUBT = "IN_DOUBT"

    @property
    def is_reusable(self) -> bool:
        """Whether this disposition proves that local memory may be reused."""
        return self in (
            PhysicalDisposition.NOT_EXPOSED,
            PhysicalDisposition.QUIESCED_SUCCESS,
            PhysicalDisposition.QUIESCED_FAILURE,
        )


class LifecycleCapability(str, Enum):
    """Independently negotiable transceiver lifecycle capabilities."""

    ATTEMPT_IDENTITY = "ATTEMPT_IDENTITY"
    ENDPOINT_INCARNATION = "ENDPOINT_INCARNATION"
    ALLOCATION_GENERATION_LEASES = "ALLOCATION_GENERATION_LEASES"
    CANCEL_BEFORE_CREATE_TOMBSTONES = "CANCEL_BEFORE_CREATE_TOMBSTONES"
    PUBLICATION_GATE = "PUBLICATION_GATE"
    IN_FLIGHT_CANCELLATION = "IN_FLIGHT_CANCELLATION"
    EXACT_WRITER_TRACKING = "EXACT_WRITER_TRACKING"
    SUBMISSION_FENCE = "SUBMISSION_FENCE"
    PER_OPERATION_QUIESCENCE = "PER_OPERATION_QUIESCENCE"
    ENDPOINT_WIDE_QUIESCENCE = "ENDPOINT_WIDE_QUIESCENCE"
    DIRECT_TRANSFER = "DIRECT_TRANSFER"
    BOUNCE_TRANSFER = "BOUNCE_TRANSFER"
    MULTI_WRITER = "MULTI_WRITER"
    GENERATION_FIRST = "GENERATION_FIRST"
    PIPELINE_PARALLEL = "PIPELINE_PARALLEL"
    TENSOR_PARALLEL = "TENSOR_PARALLEL"
    ATTENTION_DATA_PARALLEL = "ATTENTION_DATA_PARALLEL"
    TERMINAL_RESULT_REPLAY = "TERMINAL_RESULT_REPLAY"


@dataclass(frozen=True)
class TransceiverCapabilities:
    """Versioned capability set advertised before address publication."""

    protocol_version: int = 0
    supported: FrozenSet[LifecycleCapability] = field(default_factory=frozenset)
    qualified_legacy_mode: bool = False

    def __post_init__(self) -> None:
        if self.protocol_version < 0:
            raise ValueError("protocol_version must be non-negative")

    def supports(self, *capabilities: LifecycleCapability) -> bool:
        return all(capability in self.supported for capability in capabilities)

    def missing(self, required: Sequence[LifecycleCapability]) -> FrozenSet[LifecycleCapability]:
        return frozenset(required).difference(self.supported)

    def require(self, required: Sequence[LifecycleCapability]) -> None:
        missing = self.missing(required)
        if missing:
            names = ", ".join(sorted(capability.value for capability in missing))
            raise LifecycleCapabilityError(
                f"transceiver lifecycle capabilities are missing: {names}"
            )


@dataclass(frozen=True)
class CancelResult:
    """Separate logical cancellation from physical resource disposition."""

    logical: LogicalDisposition
    physical: PhysicalDisposition
    retryable: bool = False
    reason: str = ""

    @property
    def safe_to_reuse(self) -> bool:
        return self.physical.is_reusable


@dataclass(frozen=True)
class ShutdownResult:
    """Result of bounded transceiver shutdown or drain."""

    physical: PhysicalDisposition
    in_doubt_context_count: int | None = None
    fatal: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.in_doubt_context_count is not None and self.in_doubt_context_count < 0:
            raise ValueError("in_doubt_context_count must be non-negative")
        if self.physical is not PhysicalDisposition.IN_DOUBT and self.in_doubt_context_count != 0:
            raise ValueError(
                "only IN_DOUBT shutdown may have unknown or non-zero context accounting"
            )

    @property
    def safe_to_release_managers(self) -> bool:
        return self.physical.is_reusable and self.in_doubt_context_count == 0 and not self.fatal


class LifecycleCapabilityError(RuntimeError):
    """Raised when a session requires an unadvertised safety capability."""


@runtime_checkable
class TransceiverLifecycle(Protocol):
    """Additive backend-neutral lifecycle surface.

    Runtime adapters may retain their existing data paths. Unsupported
    operations must fail closed with ``REJECTED`` or ``IN_DOUBT`` rather than
    fabricate quiescence. Session creation and publication authorization join
    this surface once generation-safe identity and allocation leases land.
    """

    def capabilities(self) -> TransceiverCapabilities: ...

    def cancel_session(self, session: Any, reason: str) -> CancelResult: ...

    def poll_session(self, session: Any) -> PhysicalDisposition: ...

    def fence_submission(self, session: Any) -> LogicalDisposition: ...

    def quiesce_session(self, session: Any) -> PhysicalDisposition: ...

    def shutdown_lifecycle(self, deadline_s: float | None) -> ShutdownResult: ...
