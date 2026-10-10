# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind manager staging loans to backend evidence and shared retirement deadlines.

All adapter methods run serially on the manager owner thread. Only backend
``quiesce`` runs on daemon threads; those threads never touch a lender or lease.
This module supplies resource binding, without activating a scheduler path.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...pyexecutor.kv_cache.sharing import Lease, PartsHold, RegionView, StagingLender
from ..base.shared import (
    Attempt,
    CacheExtent,
    CancelDisposition,
    Cancelled,
    Delivered,
    Failed,
    Fetches,
    Outcome,
    Publishes,
    RegistersPools,
    Registration,
    Route,
    SubmissionRejected,
    request_cancel,
)
from ..lifecycle.retirement import RetirementDeadline
from .shared import STAGING_EXTENT_NAMESPACE, SharedRuntimeProfile, build_extent, served_masks

if TYPE_CHECKING:
    import torch


def cuda_copy_completion(stream: torch.cuda.Stream) -> Callable[[], Callable[[], bool]]:
    """Create an event recorder for the manager's exact execution stream.

    Args:
        stream: Stream used by the manager to enqueue staging copies. Supplying
            another stream does not establish the required completion ordering.

    Returns:
        Owner-thread callback that records an event and returns its nonblocking,
        request-independent completion query. The callback retains the stream.
    """
    import torch

    def record() -> Callable[[], bool]:
        """Record completion after all previously queued manager-stream work.

        Returns:
            A query retaining the event until its owner releases the callback.
        """
        event = torch.cuda.Event()
        event.record(stream)
        return event.query

    return record


@dataclass(eq=False)
class SharedLeaseOperation:
    """One retained loan and the independent evidence needed to release it.

    Callers inspect ``outcome`` and ``released`` independently. A logical result
    never authorizes destination reuse. The adapter owns this object until it
    releases the lease; callers may drop their request and operation references.
    ``cancel_disposition`` and ``cancel_error`` describe the one-shot cancellation
    request, not logical cancellation or physical access completion.
    """

    retirement: RetirementDeadline
    lease: Lease
    view: RegionView
    extent: CacheExtent
    backend: Fetches | Publishes
    writing: bool
    adapter: SharedStagingAdapter = field(repr=False)
    attempt: Attempt | None = None
    outcome: Outcome | None = None
    released: bool = False
    cancel_disposition: CancelDisposition | None = None
    cancel_error: str | None = None
    _cancel_requested: bool = False
    _proof: Future[bool] | None = field(default=None, repr=False)
    _access_ended: bool = False
    _copy_complete: Callable[[], bool] | None = field(default=None, repr=False)
    _marks_started: bool = False
    _copy_unproven: bool = False


class SharedStagingAdapter:
    """Retain one manager's staging allocations through registration and delivery.

    The caller supplies the manager's existing lender and a separate retirement
    owner for registration teardown. Each submitted lease uses its own retirement
    owner; it must not share that owner's timeout callback with another session.
    Registration does not start a request-transfer timer. Shutdown and ambiguous
    registration cleanup enter the supplied registration owner's drain deadline.
    """

    def __init__(
        self,
        manager: object,
        lender: StagingLender,
        profile: SharedRuntimeProfile,
        register_backend: RegistersPools,
        registration_retirement: RetirementDeadline,
        record_copy_completion: Callable[[], Callable[[], bool]],
    ) -> None:
        """Root the manager and take a hold before any registration can occur.

        Args:
            manager: Owner of the staging lender and manager execution stream.
            lender: Already attached manager-owned host staging lender.
            profile: Explicit compatibility facts, validated before taking holds.
            register_backend: Backend registering the lender's physical parts.
            registration_retirement: Dedicated lifecycle owner for teardown.
            record_copy_completion: Records on the manager stream immediately
                after ``mark_arrived`` and returns a nonblocking completion query.
                It must retain its evidence independently of the request.

        Raises:
            ValueError: The profile is unsupported.
            TypeError: The backend lacks pool-registration support.
        """
        profile.validate()
        if not isinstance(register_backend, RegistersPools):
            raise TypeError("backend does not support pool registration")
        self._manager = manager
        self._lender = lender
        self._register_backend = register_backend
        self._registration_retirement = registration_retirement
        self._record_copy_completion = record_copy_completion
        self._parts = tuple(lender.parts)
        self._hold: PartsHold | None = lender.hold_parts()
        self._registrations: list[Registration] = []
        self._operations: set[SharedLeaseOperation] = set()
        self._retirements: weakref.WeakSet[RetirementDeadline] = weakref.WeakSet()
        self._registered = False
        self._stopping = False
        self._closed = False

    def register(self) -> None:
        """Register each physical part once, retaining successful earlier handles.

        Part names describe layouts, not allocation identities. Handles remain
        associated with the exact retained part tuple and adapter instance.

        Raises:
            RuntimeError: Shutdown started or registration previously failed.
            Exception: A provider registration error, with existing roots retained.
        """
        if self._stopping:
            raise RuntimeError("staging adapter admission is closed")
        if self._registered:
            return
        try:
            for part in self._parts[len(self._registrations) :]:
                self._registrations.append(
                    self._register_backend.register_pool(part.address, part.nbytes)
                )
        except Exception:
            # Registration is an external boundary; a raised provider error must
            # not drop earlier successful handles or their manager-owned memory.
            self._stopping = True
            self._registration_retirement.retain_unproven(self, "pool registration failed")
            raise
        self._registered = True

    def submit_fetch(
        self,
        lease: Lease,
        backend: Fetches,
        retirement: RetirementDeadline,
        *,
        route: Route | None = None,
        is_last: bool = True,
    ) -> SharedLeaseOperation:
        """Take a ready write lease and submit into its retained host staging.

        Args:
            lease: Write lease obtained from this adapter's lender.
            backend: The same provider instance that registered the staging parts.
            retirement: Dedicated lifecycle owner, including its timeout callback.
            route: Optional provider route; the caller owns route closure.
            is_last: Explicit end marker for sequence-aware providers.

        Returns:
            Binding retained independently from the caller's request.

        Raises:
            ValueError: Lease is not ready or its view is incompatible; ownership
                remains with the caller when validation fails.
            RuntimeError: Adapter admission is closed or registration incomplete.
            TypeError: Backend lacks the required capability.
        """
        if not isinstance(backend, Fetches):
            raise TypeError("backend does not support fetching")
        operation = self._prepare(lease, backend, retirement, True, is_last)
        self._submit(operation, lambda: backend.fetch(operation.extent, route=route))
        return operation

    def submit_publish(
        self,
        lease: Lease,
        backend: Publishes,
        retirement: RetirementDeadline,
        *,
        is_last: bool = True,
    ) -> SharedLeaseOperation:
        """Take a ready read lease after its device-to-host copy has completed.

        Args:
            lease: Read lease obtained from this adapter's lender.
            backend: The same provider instance that registered the staging parts.
            retirement: Dedicated lifecycle owner, including its timeout callback.
            is_last: Explicit end marker for sequence-aware providers.

        Returns:
            Binding retained through outcome, backend access, and lease release.

        Raises:
            ValueError: Lease is not ready or its view is incompatible; ownership
                remains with the caller when validation fails.
            RuntimeError: Adapter admission is closed or registration incomplete.
            TypeError: Backend lacks the required capability.
        """
        if not isinstance(backend, Publishes):
            raise TypeError("backend does not support publication")
        operation = self._prepare(lease, backend, retirement, False, is_last)
        self._submit(operation, lambda: backend.publish(operation.extent))
        return operation

    def _prepare(
        self,
        lease: Lease,
        backend: Fetches | Publishes,
        retirement: RetirementDeadline,
        writing: bool,
        is_last: bool,
    ) -> SharedLeaseOperation:
        """Validate a ready lease before transferring it into adapter ownership.

        Args:
            lease: Manager loan whose staging view must already be ready.
            backend: Provider retained for submission and access-end evidence.
            retirement: Unused deadline owner for this operation.
            writing: Whether successful delivery copies into manager pages.
            is_last: Extent's sequence-end marker.

        Returns:
            Provisional physical owner, retained before submission.

        Raises:
            RuntimeError: Adapter unavailable or deadline already bound here.
            ValueError: View not ready or incompatible with retained parts.
        """
        if self._stopping or not self._registered:
            raise RuntimeError("staging adapter is not accepting submissions")
        if backend is not self._register_backend:
            raise ValueError("submission backend must own this adapter's registrations")
        if retirement.controller is not self._registration_retirement.controller:
            raise ValueError("operation must use this adapter's retirement watchdog")
        if retirement in self._retirements or retirement is self._registration_retirement:
            raise RuntimeError("each lease requires its own retirement owner")
        if any(active.lease is lease for active in self._operations):
            raise RuntimeError("lease is already owned by this adapter")
        view = lease.poll()
        if view is None:
            raise ValueError(lease.failure or "lease is not ready")
        extent = build_extent(view, self._parts, name=STAGING_EXTENT_NAMESPACE, is_last=is_last)
        operation = SharedLeaseOperation(retirement, lease, view, extent, backend, writing, self)
        self._retirements.add(retirement)
        self._operations.add(operation)

        def timeout() -> None:
            """Latch logical timeout metadata under the lifecycle arbiter lock."""
            if operation.outcome is None:
                operation.outcome = Failed("transfer timeout")

        retirement.bind_timeout_outcome(timeout)
        return operation

    def _submit(self, operation: SharedLeaseOperation, submit: Callable[[], Attempt]) -> None:
        """Expose all roots before entering a potentially escaping provider call.

        Args:
            operation: Prepared owner whose lease is already ready.
            submit: Immediate provider submission returning its retained Attempt.
        """
        if not operation.retirement.expose(operation):
            self._commit(operation, Cancelled(by_peer=False))
            operation._access_ended = True
            self._finish(operation)
            return
        try:
            operation.attempt = submit()
            if not isinstance(operation.attempt, Attempt):
                operation.attempt = None
                raise TypeError("submission escaped without a valid Attempt")
        except SubmissionRejected as error:
            self._commit(operation, Failed(str(error)))
            operation._access_ended = True
            self._finish(operation)
            return
        except Exception as error:
            # An unknown external exception can follow escaped DMA. No handle
            # exists with which to establish no-future-access evidence.
            self._commit(operation, Failed(str(error)))
            operation.retirement.retain_unproven(operation, "submission escaped without an Attempt")
            return
        self.retry_quiescence(operation)

    def _commit(self, operation: SharedLeaseOperation, outcome: Outcome) -> None:
        """Latch an outcome without using it as permission to release memory.

        Args:
            operation: Binding whose logical answer may still be pending.
            outcome: First observed terminal logical answer.
        """
        with operation.retirement.lock:
            operation.retirement.check()
            if operation.outcome is None:
                operation.outcome = outcome
                if isinstance(outcome, Delivered):
                    operation.retirement.complete_pieces()
                else:
                    operation.retirement.request_drain("delivery failed or cancelled")

    def cancel(self, operation: SharedLeaseOperation, *, by_peer: bool = False) -> None:
        """Latch cancellation, then request it without waiting for physical work.

        The first locally committed outcome wins, even if an unobserved backend
        outcome is already available. The provider call runs outside the arbiter
        lock and cannot authorize release or extend retirement deadlines.

        Args:
            operation: Operation returned by this adapter.
            by_peer: Whether cancellation originated at the peer.

        Raises:
            ValueError: Operation belongs to another adapter.
        """
        self._require_operation(operation)
        self._commit(operation, Cancelled(by_peer=by_peer))
        self._request_cancel(operation)

    def _request_cancel(self, operation: SharedLeaseOperation) -> None:
        """Send one best-effort request for a failed or cancelled operation.

        Provider failures remain diagnostic and cannot obstruct evidence
        collection. Timeout callbacks leave this call to owner-thread progress.

        Args:
            operation: Retained binding whose logical outcome is already latched.
        """
        with operation.retirement.lock:
            if (
                operation.released
                or operation._cancel_requested
                or operation.attempt is None
                or not isinstance(operation.outcome, (Failed, Cancelled))
            ):
                return
            operation._cancel_requested = True
            attempt = operation.attempt
        try:
            disposition = request_cancel(attempt)
            if not isinstance(disposition, CancelDisposition):
                raise TypeError("backend returned an invalid cancellation disposition")
        except Exception as error:
            # Provider exception types are unrestricted at this external boundary.
            # A failed request must not block proof consumption or other operations.
            operation.cancel_error = f"{type(error).__name__}: {error}"
        else:
            operation.cancel_disposition = disposition

    def retry_quiescence(self, operation: SharedLeaseOperation) -> None:
        """Start one background proof, including explicit retries after ambiguity.

        False, a provider exception, or worker-start failure is not a pending
        promise. It starts drain and retains roots; only an explicit later retry
        can provide new evidence.
        An in-flight proof is never duplicated and never blocks the owner thread.

        Args:
            operation: Operation returned by this adapter.

        Raises:
            ValueError: Operation belongs to another adapter.
        """
        self._require_operation(operation)
        if operation.released or operation._access_ended or operation._proof is not None:
            return
        if operation.attempt is None:
            return
        proof: Future[bool] = Future()
        operation._proof = proof
        attempt = operation.attempt

        def wait() -> None:
            """Collect only backend evidence without calling manager-owned objects."""
            try:
                proof.set_result(operation.backend.quiesce((attempt,)))
            except Exception as error:
                # Provider exceptions are transported to the owner thread as
                # disputed evidence, never interpreted as completed access.
                proof.set_exception(error)

        try:
            threading.Thread(target=wait, name="shared-kv-quiesce", daemon=True).start()
        except RuntimeError as error:
            # Submission has already escaped. Preserve its returned operation
            # and let owner-thread progress handle failed proof scheduling.
            proof.set_exception(error)

    def progress(self) -> None:
        """Consume nonblocking outcomes and access/copy evidence on the owner thread."""
        for operation in tuple(self._operations):
            operation.retirement.check()
            if operation.outcome is None and operation.attempt is not None:
                try:
                    outcome = operation.attempt.poll()
                    if isinstance(outcome, Delivered):
                        served_masks(operation.view, outcome.served)
                    elif outcome is not None and not isinstance(outcome, (Failed, Cancelled)):
                        raise TypeError("backend returned an invalid logical outcome")
                except Exception as error:
                    # Poll is an external nonblocking boundary. Invalid results
                    # invalidate bytes but do not establish safe physical access.
                    self._commit(operation, Failed(str(error)))
                else:
                    if outcome is not None:
                        self._commit(operation, outcome)
            self._request_cancel(operation)
            proof = operation._proof
            if proof is not None and proof.done():
                operation._proof = None
                try:
                    ended = proof.result()
                except Exception:
                    ended = False
                if ended is True:
                    operation._access_ended = True
                else:
                    operation.retirement.retain_unproven(operation, "backend access end unproven")
            self._finish(operation)

    def _finish(self, operation: SharedLeaseOperation) -> None:
        """Release only after backend access and every local copy have ended.

        Args:
            operation: Retained binding progressed exclusively by its owner.
        """
        if operation.retirement.controller.fatal is not None:
            return
        if operation.released or not operation._access_ended or operation.outcome is None:
            return
        if operation.writing and not operation._marks_started:
            operation._marks_started = True
            served = (
                operation.outcome.served
                if isinstance(operation.outcome, Delivered)
                else frozenset()
            )
            try:
                operation.lease.mark_arrived(served_masks(operation.view, served))
                operation._copy_complete = self._record_copy_completion()
            except Exception:
                # The copy call may have queued work before failing. Repeating
                # mark_arrived or inferring completion from request exit is unsafe.
                operation._copy_unproven = True
                operation.retirement.retain_unproven(operation, "local copy evidence unavailable")
        if operation._copy_unproven:
            return
        if operation._copy_complete is not None:
            try:
                complete = operation._copy_complete()
            except Exception:
                operation._copy_unproven = True
                operation.retirement.retain_unproven(operation, "local copy query failed")
                return
            if complete is not True:
                return
        if not operation.retirement.settle(operation):
            return
        try:
            operation.lease.release()
        except Exception:
            operation.retirement.retain_unproven(operation, "lease release failed")
            return
        operation.released = True
        self._operations.discard(operation)
        operation.retirement.close()

    def close(self) -> bool:
        """Stop admission, drain physical access/copies, then revoke registrations.

        Successful registration handles are removed individually. Failed closure
        retains its handle and the parts hold, and a later call retries it. A
        fatal lifecycle decision refuses teardown even if late evidence arrives.

        Returns:
            True after every handle and the parts hold were successfully released;
            False while access, copy completion, or registration closure is unsafe.
        """
        if self._closed:
            return True
        self._stopping = True
        retirement = self._registration_retirement
        retirement.retain_unproven(self, "staging adapter shutdown")
        for operation in tuple(self._operations):
            self.cancel(operation)
        self.progress()
        if self._operations or retirement.controller.fatal is not None:
            return False
        while self._registrations:
            try:
                self._registrations[-1].close()
            except Exception:
                # The same provider handle must remain available for retry.
                return False
            self._registrations.pop()
        if not retirement.settle(self):
            return False
        if self._hold is not None:
            try:
                self._hold.release()
            except Exception:
                retirement.retain_unproven(self, "parts hold release failed")
                return False
            self._hold = None
        retirement.close()
        self._closed = True
        return True

    def _require_operation(self, operation: SharedLeaseOperation) -> None:
        """Reject a foreign operation before changing its logical or physical state.

        Args:
            operation: Binding supplied to an explicit owner operation.

        Raises:
            ValueError: The binding belongs to another adapter instance.
        """
        if operation.adapter is not self:
            raise ValueError("operation belongs to another staging adapter")
