# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-use admission tokens for native context cancellation.

Only unsubmitted reservations live here. Admitted work carries its token in
the existing executor request; its normal owner remains responsible for KV
retirement. A cancellation receipt never reports physical settlement.
"""

import hashlib
import hmac
import json
import secrets
import time
from threading import Lock
from typing import Callable, Literal

CancellationReceipt = Literal["not_submitted", "queued"]


class ContextCancellationReservations:
    """Serialize one-shot context admission against cancellation on rank zero."""

    def __init__(self) -> None:
        """Create an incarnation-specific signing key and bounded reservations."""
        self._key = secrets.token_bytes(32)
        self._lock = Lock()
        self._pending: dict[str, float] = {}

    def reserve(self, request_id: int) -> str:
        """Reserve one context admission without allocating transfer resources.

        Args:
            request_id: The immutable disaggregated pairing identity.

        Returns:
            An opaque token valid only at this worker incarnation.

        Raises:
            ValueError: The request identity is invalid.
            RuntimeError: The unsubmitted reservation limit is reached.
        """
        if type(request_id) is not int or request_id < 0:
            raise ValueError("Context cancellation requires an integer request ID")
        with self._lock:
            now = time.monotonic()
            self._pending = {
                token: deadline for token, deadline in self._pending.items() if deadline > now
            }
            if len(self._pending) >= 1024:
                raise RuntimeError("Context cancellation reservations are full")
            # This bounds only the time to submit. It is not a transfer or
            # quiescence deadline and never revokes cancellation after admission.
            deadline = now + 60.0
            payload = json.dumps(
                [request_id, secrets.token_hex(16), deadline], separators=(",", ":")
            )
            signature = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
            token = f"{payload}.{signature}"
            self._pending[token] = deadline
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
