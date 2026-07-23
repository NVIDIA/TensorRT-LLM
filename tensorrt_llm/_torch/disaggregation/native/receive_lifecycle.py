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
"""Receiver-side ownership state for native disaggregated KV transfers.

This module deliberately depends only on the Python standard library.  It owns
the concurrency-sensitive lifecycle decisions, while ``transfer.py`` and the
bounce implementation execute the returned actions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, TypeAlias

TransferKey: TypeAlias = tuple[int, int]


class WriterMode(Enum):
    """The path a writer used to access receiver memory."""

    PENDING = "pending"
    DIRECT = "direct"
    BOUNCE = "bounce"
    NO_REMOTE_ACCESS = "no_remote_access"
    UNKNOWN = "unknown"


class ExposureState(Enum):
    """Whether a writer could have observed a receiver target address."""

    UNEXPOSED = "unexposed"
    POSSIBLY_EXPOSED = "possibly_exposed"
    PUBLISHED = "published"
    NEVER_EXPOSED = "never_exposed"


class WriterResult(Enum):
    """A writer's terminal data result."""

    SUCCESS = "success"
    FAILED = "failed"


class LogicalState(Enum):
    """Outcome visible to the request/session consumer."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhysicalState(Enum):
    """Whether memory may still be touched by a transfer accessor."""

    ACTIVE = "active"
    DRAINING = "draining"
    IN_DOUBT = "in_doubt"
    DRAINED = "drained"


class BounceState(Enum):
    """Receiver bounce-slot scatter lifecycle."""

    NONE = "none"
    IDLE = "idle"
    SCATTERING = "scattering"
    DONE = "done"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


class LifecycleAction(Enum):
    """A side effect that the transfer/bounce adapters must execute."""

    NOTIFY_SUCCESS = "notify_success"
    NOTIFY_FAILURE = "notify_failure"
    NOTIFY_CANCELLED = "notify_cancelled"
    START_BOUNCE_SCATTER = "start_bounce_scatter"
    RELEASE_BOUNCE = "release_bounce"
    CONTEXT_DRAINED = "context_drained"


@dataclass(frozen=True)
class LifecycleUpdate:
    """Result of one atomic lifecycle transition."""

    key: TransferKey
    accepted: bool
    logical_state: LogicalState
    physical_state: PhysicalState
    writers_terminal: bool
    bounce_pending: bool
    actions: tuple[LifecycleAction, ...] = ()
    publication_allowed: bool = False
    duplicate: bool = False
    conflict: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class WriterSnapshot:
    """Immutable diagnostic view of one expected writer."""

    rank: int
    exposure: ExposureState
    mode: WriterMode
    result: WriterResult | None
    conflict: bool
    backend_quiesced: bool


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable diagnostic view of a receive transfer context."""

    key: TransferKey
    logical_state: LogicalState
    physical_state: PhysicalState
    bounce_state: BounceState
    writers_terminal: bool
    bounce_pending: bool
    bounce_settled: bool
    consumer_attached: bool
    protocol_conflict: bool
    expected_writers: frozenset[int]
    writers: tuple[WriterSnapshot, ...]


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable diagnostic view of the complete registry."""

    accepting: bool
    backend_quiesced: bool
    contexts: tuple[ContextSnapshot, ...]


@dataclass
class WriterRecord:
    """Mutable state for one member of the exact expected-writer set."""

    rank: int
    exposure: ExposureState = ExposureState.UNEXPOSED
    mode: WriterMode = WriterMode.PENDING
    result: WriterResult | None = None
    conflict: bool = False
    backend_quiesced: bool = False

    @property
    def is_quiescent(self) -> bool:
        """Return whether this writer can no longer access receiver memory."""

        return self.backend_quiesced or (self.result is not None and not self.conflict)

    def snapshot(self) -> WriterSnapshot:
        """Return an immutable diagnostic view."""

        return WriterSnapshot(
            rank=self.rank,
            exposure=self.exposure,
            mode=self.mode,
            result=self.result,
            conflict=self.conflict,
            backend_quiesced=self.backend_quiesced,
        )


@dataclass
class RecvTransferContext:
    """Canonical receiver-side state for one request slice."""

    key: TransferKey
    expected_writers: frozenset[int]
    has_bounce_slot: bool
    consumer_attached: bool = True
    writers: dict[int, WriterRecord] = field(init=False)
    logical_state: LogicalState = field(default=LogicalState.PENDING, init=False)
    physical_state: PhysicalState = field(default=PhysicalState.ACTIVE, init=False)
    bounce_state: BounceState = field(init=False)
    bounce_settled: bool = field(default=False, init=False)
    protocol_conflict: bool = field(default=False, init=False)
    conflict_requires_backend_quiescence: bool = field(default=False, init=False)
    in_doubt: bool = field(default=False, init=False)
    logical_reason: str | None = field(default=None, init=False)
    _notified_logical_state: LogicalState | None = field(default=None, init=False)
    _bounce_release_emitted: bool = field(default=False, init=False)
    _drained_emitted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.writers = {rank: WriterRecord(rank) for rank in self.expected_writers}
        self.bounce_state = BounceState.IDLE if self.has_bounce_slot else BounceState.NONE

    def snapshot(self) -> ContextSnapshot:
        """Return an immutable diagnostic view."""

        return ContextSnapshot(
            key=self.key,
            logical_state=self.logical_state,
            physical_state=self.physical_state,
            bounce_state=self.bounce_state,
            writers_terminal=self.writers_terminal,
            bounce_pending=self.bounce_pending,
            bounce_settled=self.bounce_settled,
            consumer_attached=self.consumer_attached,
            protocol_conflict=self.protocol_conflict,
            expected_writers=self.expected_writers,
            writers=tuple(self.writers[rank].snapshot() for rank in sorted(self.writers)),
        )

    @property
    def writers_terminal(self) -> bool:
        """Return whether every expected writer is physically quiescent."""

        return all(writer.is_quiescent for writer in self.writers.values())

    @property
    def bounce_pending(self) -> bool:
        """Return whether a requested bounce scatter has not settled."""

        return self.bounce_state is BounceState.SCATTERING


class RecvTransferRegistry:
    """Thread-safe root owner for receiver transfer lifecycle state.

    Address publication and cancellation arbitrate under the same lock.
    ``begin_publication`` changes the writer to ``POSSIBLY_EXPOSED`` before
    returning permission to send, so cancellation can never mistake an
    authorized publication for an idle reservation.
    """

    def __init__(self) -> None:
        self._contexts: dict[TransferKey, RecvTransferContext] = {}
        self._request_keys: dict[int, set[TransferKey]] = {}
        self._accepting = True
        self._backend_quiesced = False
        self._lock = threading.RLock()

    def prepare(
        self,
        key: TransferKey,
        expected_writers: Iterable[int],
        *,
        has_bounce_slot: bool,
        consumer_attached: bool = True,
    ) -> LifecycleUpdate:
        """Create the canonical context before any target can be published."""

        writers = frozenset(expected_writers)
        if not writers:
            raise ValueError("expected_writers must not be empty")
        if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in writers):
            raise ValueError("expected_writers must contain non-negative integer ranks")

        with self._lock:
            if not self._accepting:
                return self._rejected_update(key, "registry admission is closed")
            existing = self._contexts.get(key)
            if existing is not None:
                same = (
                    existing.expected_writers == writers
                    and existing.has_bounce_slot == has_bounce_slot
                )
                return self._update(
                    existing,
                    accepted=False,
                    duplicate=same,
                    conflict=not same,
                    reason=("context already prepared" if same else "conflicting context identity"),
                )
            context = RecvTransferContext(
                key=key,
                expected_writers=writers,
                has_bounce_slot=has_bounce_slot,
                consumer_attached=consumer_attached,
            )
            self._contexts[key] = context
            self._request_keys.setdefault(key[0], set()).add(key)
            return self._update(context)

    def begin_publication(self, key: TransferKey, writer_rank: int) -> LifecycleUpdate:
        """Atomically authorize at most one target publication for a writer."""

        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return self._rejected_update(key, "unknown transfer context")
            writer = context.writers.get(writer_rank)
            if writer is None:
                return self._unexpected_local_writer(context, writer_rank)
            if context.logical_state is not LogicalState.PENDING:
                return self._update(context, accepted=False, reason="publication gate is closed")
            if writer.exposure is not ExposureState.UNEXPOSED:
                return self._update(
                    context,
                    accepted=False,
                    duplicate=True,
                    reason="writer publication already decided",
                )
            writer.exposure = ExposureState.POSSIBLY_EXPOSED
            return self._update(context, publication_allowed=True)

    def mark_published(self, key: TransferKey, writer_rank: int) -> LifecycleUpdate:
        """Record that an authorized publication completed successfully."""

        with self._lock:
            context, writer, error = self._lookup_writer(key, writer_rank)
            if error is not None:
                return error
            assert context is not None and writer is not None
            if writer.exposure is ExposureState.PUBLISHED:
                return self._update(context, accepted=False, duplicate=True)
            if writer.exposure is not ExposureState.POSSIBLY_EXPOSED:
                return self._publication_conflict(context, writer, "publication was not authorized")
            writer.exposure = ExposureState.PUBLISHED
            return self._update(context)

    def mark_publication_ambiguous(self, key: TransferKey, writer_rank: int) -> LifecycleUpdate:
        """Keep an authorized writer possibly exposed after an ambiguous send."""

        with self._lock:
            context, writer, error = self._lookup_writer(key, writer_rank)
            if error is not None:
                return error
            assert context is not None and writer is not None
            if writer.exposure is ExposureState.POSSIBLY_EXPOSED:
                return self._update(context, accepted=False, duplicate=True)
            if writer.exposure is ExposureState.PUBLISHED:
                return self._update(context, accepted=False, duplicate=True)
            return self._publication_conflict(context, writer, "publication was not authorized")

    def mark_never_published(self, key: TransferKey, writer_rank: int) -> LifecycleUpdate:
        """Record definitive proof that a writer never received a target."""

        with self._lock:
            context, writer, error = self._lookup_writer(key, writer_rank)
            if error is not None:
                return error
            assert context is not None and writer is not None
            if writer.exposure is ExposureState.NEVER_EXPOSED:
                return self._update(context, accepted=False, duplicate=True)
            if writer.exposure is ExposureState.PUBLISHED:
                return self._publication_conflict(
                    context, writer, "published target cannot be undone"
                )
            if writer.result is not None:
                return self._publication_conflict(
                    context, writer, "writer already reported a result"
                )
            writer.exposure = ExposureState.NEVER_EXPOSED
            writer.mode = WriterMode.NO_REMOTE_ACCESS
            writer.result = WriterResult.FAILED
            return self._evaluate(context)

    def record_result(
        self,
        key: TransferKey,
        writer_rank: int,
        result: WriterResult,
        mode: WriterMode,
    ) -> LifecycleUpdate:
        """Record one exact writer's terminal result and return newly safe actions."""

        if mode is WriterMode.PENDING:
            raise ValueError("a terminal result cannot use WriterMode.PENDING")
        if result is WriterResult.SUCCESS and mode not in (WriterMode.DIRECT, WriterMode.BOUNCE):
            raise ValueError("a successful result must identify DIRECT or BOUNCE mode")
        if mode is WriterMode.NO_REMOTE_ACCESS and result is not WriterResult.FAILED:
            raise ValueError("NO_REMOTE_ACCESS cannot report successful data delivery")

        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return self._rejected_update(key, "unknown transfer context")
            writer = context.writers.get(writer_rank)
            if writer is None:
                return self._unexpected_result_writer(context, writer_rank)
            if writer.result is not None:
                return self._dedupe_or_conflict_result(context, writer, result, mode)
            exposed = writer.exposure in (
                ExposureState.POSSIBLY_EXPOSED,
                ExposureState.PUBLISHED,
            )
            allowed_modes = (
                (WriterMode.DIRECT, WriterMode.BOUNCE)
                if context.has_bounce_slot
                else (WriterMode.DIRECT,)
            )
            if exposed and mode not in allowed_modes:
                writer.mode = mode
                writer.result = result
                return self._writer_result_conflict(
                    context,
                    writer,
                    f"writer {writer_rank} reported {mode.value}, expected one of "
                    f"{tuple(item.value for item in allowed_modes)}",
                )
            if not exposed and mode is not WriterMode.NO_REMOTE_ACCESS:
                writer.mode = mode
                writer.result = result
                return self._writer_result_conflict(
                    context,
                    writer,
                    f"writer {writer_rank} reported access without publication",
                )
            writer.mode = mode
            writer.result = result
            return self._evaluate(context)

    def record_protocol_conflict(
        self, key: TransferKey, writer_rank: int, reason: str
    ) -> LifecycleUpdate:
        """Fail closed on a writer frame that is not terminal evidence.

        Receiving such a frame proves that the claimed identity may belong to
        an exposed or stale operation.  Without generation-safe replay, neither
        an expected-but-unpublished writer nor an unexpected writer permits the
        no-publication fast path.  Backend-wide quiescence is required before
        the target can be retired.
        """

        if not reason:
            raise ValueError("protocol conflict reason must not be empty")
        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return self._rejected_update(key, "unknown transfer context")
            writer = context.writers.get(writer_rank)
            if writer is not None:
                writer.conflict = True
                if writer.exposure is ExposureState.UNEXPOSED:
                    writer.exposure = ExposureState.POSSIBLY_EXPOSED
                if writer.mode is WriterMode.PENDING:
                    writer.mode = WriterMode.UNKNOWN
            context.protocol_conflict = True
            context.conflict_requires_backend_quiescence = True
            context.in_doubt = True
            return self._fail_and_evaluate(context, reason, conflict=True)

    def finish_bounce_scatter(self, key: TransferKey, succeeded: bool) -> LifecycleUpdate:
        """Acknowledge that bounce scatter/settlement and slot release finished."""

        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return self._rejected_update(key, "unknown transfer context")
            if context.bounce_settled:
                same = (context.bounce_state is not BounceState.FAILED) == succeeded
                return self._update(
                    context,
                    accepted=False,
                    duplicate=same,
                    conflict=not same,
                    reason=(None if same else "conflicting scatter result"),
                )
            if not context.has_bounce_slot:
                return self._update(
                    context,
                    accepted=False,
                    conflict=True,
                    reason="context has no bounce resource",
                )
            if not context.writers_terminal:
                return self._update(
                    context,
                    accepted=False,
                    conflict=True,
                    reason="bounce resource settled before its accessors drained",
                )
            context.bounce_settled = True
            context._bounce_release_emitted = True
            context.bounce_state = BounceState.DONE if succeeded else BounceState.FAILED
            if not succeeded:
                return self._fail_and_evaluate(context, "bounce scatter failed")
            return self._evaluate(context)

    def fail_context(self, key: TransferKey, reason: str) -> LifecycleUpdate:
        """Latch logical failure after local preparation or publication fails.

        Unexposed writers are closed immediately.  Any writer that crossed the
        publication boundary remains owned until it reports terminal or the
        backend supplies global quiescence evidence.
        """

        if not reason:
            raise ValueError("failure reason must not be empty")
        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return self._rejected_update(key, "unknown transfer context")
            if context.logical_state is not LogicalState.PENDING:
                return self._update(
                    context,
                    accepted=False,
                    duplicate=True,
                    reason="logical outcome is already terminal",
                )
            context.logical_state = LogicalState.FAILED
            context.logical_reason = reason
            context.in_doubt = any(
                writer.exposure in (ExposureState.POSSIBLY_EXPOSED, ExposureState.PUBLISHED)
                and not writer.is_quiescent
                for writer in context.writers.values()
            )
            return self._evaluate(context, reason=reason)

    def cancel_request(self, unique_rid: int) -> tuple[LifecycleUpdate, ...]:
        """Logically cancel every slice while preserving exposed resources."""

        with self._lock:
            return tuple(
                self._cancel_context(context, "request cancelled")
                for context in self._contexts_for_request(unique_rid)
            )

    def timeout_request(self, unique_rid: int) -> tuple[LifecycleUpdate, ...]:
        """Logically fail every slice while preserving unknown physical access."""

        with self._lock:
            updates = []
            for context in self._contexts_for_request(unique_rid):
                if context.logical_state is LogicalState.PENDING:
                    context.logical_state = LogicalState.FAILED
                    context.logical_reason = "request timed out"
                    context.in_doubt = True
                updates.append(self._evaluate(context))
            return tuple(updates)

    def detach_consumer(self, unique_rid: int) -> tuple[LifecycleUpdate, ...]:
        """Detach request/session callbacks without changing physical ownership."""

        with self._lock:
            updates = []
            for context in self._contexts_for_request(unique_rid):
                context.consumer_attached = False
                updates.append(self._update(context))
            return tuple(updates)

    def begin_shutdown(self) -> tuple[LifecycleUpdate, ...]:
        """Close admission and logically cancel every active context."""

        with self._lock:
            self._accepting = False
            return tuple(
                self._cancel_context(context, "receiver shutdown")
                for context in sorted(self._contexts.values(), key=lambda item: item.key)
                if context.physical_state is not PhysicalState.DRAINED
            )

    def mark_backend_quiesced(self) -> tuple[LifecycleUpdate, ...]:
        """Apply backend-wide proof that no remote accessor can remain active."""

        with self._lock:
            self._backend_quiesced = True
            updates = []
            for context in sorted(self._contexts.values(), key=lambda item: item.key):
                if context.physical_state is PhysicalState.DRAINED:
                    continue
                missing_data = False
                for writer in context.writers.values():
                    writer.backend_quiesced = True
                    if writer.result is None:
                        missing_data = True
                context.conflict_requires_backend_quiescence = False
                if missing_data and context.logical_state is LogicalState.PENDING:
                    context.logical_state = LogicalState.FAILED
                    context.logical_reason = "backend quiesced before all writer results"
                updates.append(self._evaluate(context))
            return tuple(updates)

    def context_snapshot(self, key: TransferKey) -> ContextSnapshot | None:
        """Return an immutable view of a context, if it exists."""

        with self._lock:
            context = self._contexts.get(key)
            return None if context is None else context.snapshot()

    def target_mode(self, key: TransferKey) -> WriterMode | None:
        """Return the target mode offered to this context without allocating a snapshot."""

        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                return None
            return WriterMode.BOUNCE if context.has_bounce_slot else WriterMode.DIRECT

    def snapshot(self) -> RegistrySnapshot:
        """Return an immutable view of registry admission and all contexts."""

        with self._lock:
            return RegistrySnapshot(
                accepting=self._accepting,
                backend_quiesced=self._backend_quiesced,
                contexts=tuple(
                    context.snapshot()
                    for context in sorted(self._contexts.values(), key=lambda item: item.key)
                ),
            )

    def is_request_drained(self, unique_rid: int) -> bool:
        """Return whether all contexts for a request are physically drained."""

        with self._lock:
            return all(
                context.physical_state is PhysicalState.DRAINED
                for context in self._contexts_for_request(unique_rid)
            )

    def is_drained(self) -> bool:
        """Return whether every registered context is physically drained."""

        with self._lock:
            return all(
                context.physical_state is PhysicalState.DRAINED
                for context in self._contexts.values()
            )

    def retire_request(self, unique_rid: int) -> tuple[TransferKey, ...]:
        """Remove drained request contexts while retaining any unsafe context."""

        with self._lock:
            retired = tuple(
                context.key
                for context in self._contexts_for_request(unique_rid)
                if context.physical_state is PhysicalState.DRAINED
            )
            for key in retired:
                del self._contexts[key]
                self._request_keys[unique_rid].discard(key)
            if not self._request_keys.get(unique_rid):
                self._request_keys.pop(unique_rid, None)
            return retired

    def retire_request_if_drained(self, unique_rid: int) -> bool:
        """Atomically retire all request contexts, or none if any is still active."""

        with self._lock:
            contexts = self._contexts_for_request(unique_rid)
            if any(context.physical_state is not PhysicalState.DRAINED for context in contexts):
                return False
            for context in contexts:
                del self._contexts[context.key]
            self._request_keys.pop(unique_rid, None)
            return True

    def _contexts_for_request(self, unique_rid: int) -> list[RecvTransferContext]:
        keys = self._request_keys.get(unique_rid, ())
        return [self._contexts[key] for key in sorted(keys) if key in self._contexts]

    def _lookup_writer(
        self, key: TransferKey, writer_rank: int
    ) -> tuple[RecvTransferContext | None, WriterRecord | None, LifecycleUpdate | None]:
        context = self._contexts.get(key)
        if context is None:
            return None, None, self._rejected_update(key, "unknown transfer context")
        writer = context.writers.get(writer_rank)
        if writer is None:
            return context, None, self._unexpected_local_writer(context, writer_rank)
        return context, writer, None

    def _cancel_context(self, context: RecvTransferContext, reason: str) -> LifecycleUpdate:
        if context.logical_state is LogicalState.PENDING:
            context.logical_state = LogicalState.CANCELLED
            context.logical_reason = reason
            context.in_doubt = True
        return self._evaluate(context)

    def _close_unexposed_writers(self, context: RecvTransferContext) -> None:
        if context.logical_state is LogicalState.PENDING:
            return
        for writer in context.writers.values():
            if writer.exposure is ExposureState.UNEXPOSED and writer.result is None:
                writer.exposure = ExposureState.NEVER_EXPOSED
                writer.mode = WriterMode.NO_REMOTE_ACCESS
                writer.result = WriterResult.FAILED

    def _evaluate(
        self,
        context: RecvTransferContext,
        *,
        accepted: bool = True,
        duplicate: bool = False,
        conflict: bool = False,
        reason: str | None = None,
    ) -> LifecycleUpdate:
        actions: list[LifecycleAction] = []

        if context.logical_state is LogicalState.PENDING:
            results = [writer.result for writer in context.writers.values()]
            if any(result is WriterResult.FAILED for result in results):
                context.logical_state = LogicalState.FAILED
                context.logical_reason = reason or "writer failed"
            elif all(result is WriterResult.SUCCESS for result in results):
                has_bounce_result = any(
                    writer.mode is WriterMode.BOUNCE for writer in context.writers.values()
                )
                if has_bounce_result:
                    if context.bounce_state is BounceState.IDLE:
                        context.bounce_state = BounceState.SCATTERING
                        actions.append(LifecycleAction.START_BOUNCE_SCATTER)
                    elif context.bounce_state is BounceState.DONE:
                        context.logical_state = LogicalState.SUCCEEDED
                        context.logical_reason = "all writers and bounce scatter succeeded"
                    elif context.bounce_state is BounceState.FAILED:
                        context.logical_state = LogicalState.FAILED
                        context.logical_reason = "bounce scatter failed"
                else:
                    context.logical_state = LogicalState.SUCCEEDED
                    context.logical_reason = "all writers succeeded"

        self._close_unexposed_writers(context)

        if context.logical_state is not LogicalState.PENDING:
            actions.extend(self._notification_actions(context))

        all_quiescent = context.writers_terminal
        scatter_active = context.bounce_pending
        if context.logical_state is not LogicalState.PENDING and context.has_bounce_slot:
            if context.bounce_state is BounceState.IDLE:
                context.bounce_state = BounceState.SUPPRESSED

        ready_for_resource_settlement = (
            all_quiescent
            and not scatter_active
            and not context.conflict_requires_backend_quiescence
        )
        if ready_for_resource_settlement and context.has_bounce_slot and not context.bounce_settled:
            context.physical_state = PhysicalState.DRAINING
            if not context._bounce_release_emitted:
                context._bounce_release_emitted = True
                actions.append(LifecycleAction.RELEASE_BOUNCE)
        elif ready_for_resource_settlement:
            context.physical_state = PhysicalState.DRAINED
            if not context._drained_emitted:
                context._drained_emitted = True
                actions.append(LifecycleAction.CONTEXT_DRAINED)
        elif context.in_doubt or context.conflict_requires_backend_quiescence:
            context.physical_state = PhysicalState.IN_DOUBT
        elif context.logical_state is not LogicalState.PENDING or scatter_active:
            context.physical_state = PhysicalState.DRAINING
        else:
            context.physical_state = PhysicalState.ACTIVE

        return self._update(
            context,
            accepted=accepted,
            duplicate=duplicate,
            conflict=conflict,
            actions=tuple(actions),
            reason=reason,
        )

    def _notification_actions(self, context: RecvTransferContext) -> tuple[LifecycleAction, ...]:
        if not context.consumer_attached:
            return ()
        notified = context._notified_logical_state
        # A protocol contradiction discovered while the context is still live
        # overrides an earlier optimistic success.  Emit one corrective failure
        # so aggregate/session state cannot remain successful while ownership is
        # physically in doubt.
        if context.logical_state is LogicalState.FAILED and notified is not LogicalState.FAILED:
            context._notified_logical_state = LogicalState.FAILED
            return (LifecycleAction.NOTIFY_FAILURE,)
        if notified is not None:
            return ()
        context._notified_logical_state = context.logical_state
        if context.logical_state is LogicalState.SUCCEEDED:
            return (LifecycleAction.NOTIFY_SUCCESS,)
        if context.logical_state is LogicalState.CANCELLED:
            return (LifecycleAction.NOTIFY_CANCELLED,)
        return (LifecycleAction.NOTIFY_FAILURE,)

    def _fail_and_evaluate(
        self, context: RecvTransferContext, reason: str, *, conflict: bool = False
    ) -> LifecycleUpdate:
        if context.logical_state is LogicalState.PENDING or (
            conflict and context.logical_state is LogicalState.SUCCEEDED
        ):
            context.logical_state = LogicalState.FAILED
            context.logical_reason = reason
        return self._evaluate(context, accepted=False, conflict=conflict, reason=reason)

    def _dedupe_or_conflict_result(
        self,
        context: RecvTransferContext,
        writer: WriterRecord,
        result: WriterResult,
        mode: WriterMode,
    ) -> LifecycleUpdate:
        if writer.result is result:
            if writer.mode is mode or mode is WriterMode.UNKNOWN:
                return self._update(context, accepted=False, duplicate=True)
            if writer.mode is WriterMode.UNKNOWN:
                writer.mode = mode
                return self._update(context, accepted=False, duplicate=True)
        writer.conflict = True
        context.protocol_conflict = True
        context.conflict_requires_backend_quiescence = True
        context.in_doubt = True
        return self._fail_and_evaluate(
            context,
            f"contradictory result from writer {writer.rank}",
            conflict=True,
        )

    def _unexpected_result_writer(
        self, context: RecvTransferContext, writer_rank: int
    ) -> LifecycleUpdate:
        context.protocol_conflict = True
        context.conflict_requires_backend_quiescence = True
        context.in_doubt = True
        return self._fail_and_evaluate(
            context,
            f"result from unexpected writer {writer_rank}",
            conflict=True,
        )

    def _writer_result_conflict(
        self, context: RecvTransferContext, writer: WriterRecord, reason: str
    ) -> LifecycleUpdate:
        writer.conflict = True
        context.protocol_conflict = True
        context.conflict_requires_backend_quiescence = True
        context.in_doubt = True
        return self._fail_and_evaluate(context, reason, conflict=True)

    def _unexpected_local_writer(
        self, context: RecvTransferContext, writer_rank: int
    ) -> LifecycleUpdate:
        context.protocol_conflict = True
        return self._fail_and_evaluate(
            context,
            f"unexpected writer {writer_rank}",
            conflict=True,
        )

    def _publication_conflict(
        self, context: RecvTransferContext, writer: WriterRecord, reason: str
    ) -> LifecycleUpdate:
        writer.conflict = True
        context.protocol_conflict = True
        context.conflict_requires_backend_quiescence = True
        context.in_doubt = True
        return self._fail_and_evaluate(context, reason, conflict=True)

    @staticmethod
    def _update(
        context: RecvTransferContext,
        *,
        accepted: bool = True,
        actions: tuple[LifecycleAction, ...] = (),
        publication_allowed: bool = False,
        duplicate: bool = False,
        conflict: bool = False,
        reason: str | None = None,
    ) -> LifecycleUpdate:
        return LifecycleUpdate(
            key=context.key,
            accepted=accepted,
            logical_state=context.logical_state,
            physical_state=context.physical_state,
            writers_terminal=context.writers_terminal,
            bounce_pending=context.bounce_pending,
            actions=actions,
            publication_allowed=publication_allowed,
            duplicate=duplicate,
            conflict=conflict,
            reason=reason,
        )

    @staticmethod
    def _rejected_update(key: TransferKey, reason: str) -> LifecycleUpdate:
        return LifecycleUpdate(
            key=key,
            accepted=False,
            logical_state=LogicalState.PENDING,
            physical_state=PhysicalState.ACTIVE,
            writers_terminal=False,
            bounce_pending=False,
            reason=reason,
        )


__all__ = [
    "BounceState",
    "ContextSnapshot",
    "ExposureState",
    "LifecycleAction",
    "LifecycleUpdate",
    "LogicalState",
    "PhysicalState",
    "RecvTransferContext",
    "RecvTransferRegistry",
    "RegistrySnapshot",
    "TransferKey",
    "WriterMode",
    "WriterRecord",
    "WriterResult",
    "WriterSnapshot",
]
