# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-use admission tokens for native context cancellation.

Only unsubmitted admissions occupy the pending table. Replay metadata remains
for a bounded monotonic window. A fixed signed-expiry floor prevents removed
authorizations from reopening admission, including after a UTC rollback.
Admitted work carries its token in the existing executor request; its normal
owner remains responsible for KV retirement. A cancellation receipt never
reports physical settlement.

Protected control traffic requires a trusted or encrypted network segment.
Message signatures authenticate intent; they do not encrypt tokens or payloads.

A runtime advertising logical context cancellation must close pending handoff
and new submissions for the cancelled request without releasing its memory.
Its existing owners release claims only after evidence that all local and
remote access has ended. A peer that never fetches remains bounded by the
transfer timeout, followed by the physical-quiescence grace and fail-close
policy when access is still unproven; unused admission expires separately.
"""

import hashlib
import hmac
import json
import re
import secrets
import time
from collections import OrderedDict
from threading import Lock
from typing import Callable, Literal, NamedTuple

CancellationReceipt = Literal["not_submitted", "queued"]
CONTEXT_RESERVATION_AUTH_LIFETIME_MS = 60_000
_MAX_AUTHORIZATION_HORIZON_MS = 120_000
_PENDING_RESERVATION_LIMIT = 1024
_RESERVATION_LIFETIME_SECONDS = 60.0
_AUTHORIZATION_REPLAY_LIMIT = 65_536
_AUTHORIZATION_RETENTION_SECONDS = 120.0
_AUTHORIZATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


class _AcceptedAuthorization(NamedTuple):
    """Worker-local replay record; redelivery never extends its lifetime."""

    request_id: int
    expires_at_ms: int
    token: str
    retain_until: float


class ContextCancellationReservations:
    """Serialize one-shot context admission against cancellation on rank zero."""

    def __init__(self) -> None:
        """Create an incarnation-specific signing key and bounded reservations."""
        self._key = secrets.token_bytes(32)
        self._lock = Lock()
        self._pending: OrderedDict[str, float] = OrderedDict()
        # All HTTP lanes share this worker-owned ledger. Consumed authorizations
        # do not occupy pending admission, but remain recognizable after UTC changes.
        self._authorizations: OrderedDict[str, _AcceptedAuthorization] = OrderedDict()
        self._pruned_authorization_expiry_ms = -1

    def reserve(
        self, request_id: int, authorization_id: str, authorization_expires_at_ms: int
    ) -> str:
        """Reserve one context admission without allocating transfer resources.

        Args:
            request_id: The immutable disaggregated pairing identity.
            authorization_id: Signed 128-bit coordinator nonce shared by redeliveries.
            authorization_expires_at_ms: Signed UTC admission-authorization expiry.

        Returns:
            An opaque token valid only at this worker incarnation.

        Raises:
            ValueError: The request identity is invalid.
            RuntimeError: Pending admission or replay-accounting capacity is full.
        """
        if type(request_id) is not int or not 0 <= request_id < 2**64:
            raise ValueError("Context cancellation requires an integer request ID")
        if (
            not isinstance(authorization_id, str)
            or _AUTHORIZATION_ID_PATTERN.fullmatch(authorization_id) is None
            or type(authorization_expires_at_ms) is not int
        ):
            raise ValueError("Invalid context reservation authorization")
        with self._lock:
            now = time.monotonic()
            now_ms = time.time_ns() // 1_000_000
            if not now_ms < authorization_expires_at_ms <= now_ms + _MAX_AUTHORIZATION_HORIZON_MS:
                raise ValueError("Context reservation authorization is outside its validity window")
            # UTC corrections cannot shorten nonce retention. Once a record is
            # removed, its signed expiry stays at or below a fixed rejection floor.
            # Equal retention intervals put records in monotonic deadline order.
            while self._authorizations:
                oldest = next(iter(self._authorizations.values()))
                if oldest.retain_until > now:
                    break
                self._authorizations.popitem(last=False)
                self._pruned_authorization_expiry_ms = max(
                    self._pruned_authorization_expiry_ms, oldest.expires_at_ms
                )
            # Fixed-duration reservations are inserted in deadline order.
            # Stop at the first live entry rather than scanning all pending work.
            while self._pending:
                first = next(iter(self._pending))
                if self._pending[first] > now:
                    break
                self._pending.popitem(last=False)
            previous = self._authorizations.get(authorization_id)
            if previous is not None:
                if (
                    previous.request_id != request_id
                    or previous.expires_at_ms != authorization_expires_at_ms
                ):
                    raise ValueError("Context reservation authorization identity mismatch")
                if previous.token not in self._pending:
                    raise ValueError("Context cancellation admission is closed")
                return previous.token
            # An unrecognized nonce may be a removed authorization. A large or
            # delayed rollback can temporarily reject fresh admissions here;
            # advancing UTC catches this fixed floor without a moving offset.
            if authorization_expires_at_ms <= self._pruned_authorization_expiry_ms:
                raise ValueError("Context reservation authorization is outside its validity window")
            if len(self._pending) >= _PENDING_RESERVATION_LIMIT:
                raise RuntimeError("Context cancellation reservations are full")
            # Keep replay storage bounded independently of the 1,024 pending
            # slots. Never evict a live nonce to admit more work: reject overload.
            if len(self._authorizations) >= _AUTHORIZATION_REPLAY_LIMIT:
                raise RuntimeError("Context reservation replay accounting is full")
            # This bounds only the time to submit. It is not a transfer or
            # quiescence deadline and never revokes cancellation after admission.
            deadline = now + _RESERVATION_LIFETIME_SECONDS
            payload = json.dumps([request_id, authorization_id, deadline], separators=(",", ":"))
            signature = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
            token = f"{payload}.{signature}"
            self._pending[token] = deadline
            self._authorizations[authorization_id] = _AcceptedAuthorization(
                request_id,
                authorization_expires_at_ms,
                token,
                now + _AUTHORIZATION_RETENTION_SECONDS,
            )
            return token

    def _request_id(self, token: str) -> int:
        """Authenticate a token without expiring its cancellation authority.

        Args:
            token: Worker-issued opaque token.

        Returns:
            The authenticated pairing identity.

        Raises:
            ValueError: The token is malformed or belongs to another worker.
        """
        if not isinstance(token, str) or len(token) > 512:
            raise ValueError("Invalid context cancellation token")
        payload, separator, signature = token.rpartition(".")
        expected = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        if not separator or not hmac.compare_digest(signature.encode(), expected.encode()):
            raise ValueError("Invalid context cancellation token")
        return json.loads(payload)[0]

    def submit(self, token: str, request_id: int, enqueue: Callable[[], int]) -> int:
        """Consume admission and enqueue before a concurrent cancel can proceed.

        Args:
            token: Unconsumed reservation token.
            request_id: Pairing identity of the submitted context request.
            enqueue: The existing worker-to-executor admission operation.

        Returns:
            The executor request identity returned by admission.

        Raises:
            ValueError: Admission expired, was cancelled/consumed, or mismatches.
        """
        with self._lock:
            if type(request_id) is not int or self._request_id(token) != request_id:
                raise ValueError("Context cancellation token request ID mismatch")
            deadline = self._pending.pop(token, None)
            if deadline is None or deadline <= time.monotonic():
                raise ValueError("Context cancellation admission is closed")
            # Consume even if enqueue raises: a partial/failed admission must
            # never be retried with this token. Cancellation remains routable.
            return enqueue()

    def cancel(self, token: str, enqueue: Callable[[str], None]) -> CancellationReceipt:
        """Close unused admission or queue cancellation of the exact instance.

        Args:
            token: Token returned before the context POST was dispatched.
            enqueue: Enqueues token cancellation on the existing executor queue.

        Returns:
            ``not_submitted`` only when admission was closed before consumption;
            otherwise ``queued``, not worker handling or resource retirement.

        Raises:
            ValueError: The token is invalid for this worker incarnation.
        """
        with self._lock:
            self._request_id(token)
            if self._pending.pop(token, None) is not None:
                return "not_submitted"
            enqueue(token)
            return "queued"
