# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor import context_cancellation
from tensorrt_llm.executor.context_cancellation import ContextCancellationReservations

pytestmark = pytest.mark.cpu_only


def test_cancel_before_submit_closes_admission() -> None:
    """A delayed POST cannot resurrect a reservation cancelled before dispatch."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    submit = Mock(return_value=17)
    cancel = Mock()
    assert owner.cancel(token, cancel) == "not_submitted"
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(token, 17, submit)
    submit.assert_not_called()
    cancel.assert_not_called()


def test_consumed_token_routes_after_result_without_retaining_binding() -> None:
    """The token remains cancellable without a completed-result routing entry."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    assert owner.submit(token, 17, lambda: 17) == 17
    assert not owner._pending
    cancel = Mock()
    for _ in range(2):
        assert owner.cancel(token, cancel) == "queued"
    assert cancel.call_count == 2
    cancel.assert_called_with(token)
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(token, 17, Mock())


def test_cancel_cannot_overtake_consumed_submit() -> None:
    """Exercise the real lock; cancellation cannot precede its queued request."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    submitting = Event()
    finish_submit = Event()
    cancelling = Event()
    order = []

    def submit() -> int:
        """Hold the real admission callback at the executor boundary."""
        submitting.set()
        assert finish_submit.wait(5)
        order.append("submit")
        return 17

    def cancel() -> str:
        """Attempt concurrent cancellation while submission owns the lock."""
        cancelling.set()
        return owner.cancel(token, lambda _: order.append("cancel"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(owner.submit, token, 17, submit)
        try:
            assert submitting.wait(5)
            cancellation = pool.submit(cancel)
            assert cancelling.wait(5)
            assert order == []
        finally:
            finish_submit.set()
        assert submission.result(timeout=5) == 17
        assert cancellation.result(timeout=5) == "queued"
    assert order == ["submit", "cancel"]


def test_enqueue_failure_cannot_reopen_admission() -> None:
    """A partially failed admission stays closed and can still be cancelled."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        owner.submit(token, 17, Mock(side_effect=RuntimeError("enqueue failed")))
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(token, 17, Mock())
    assert owner.cancel(token, Mock()) == "queued"


@pytest.mark.parametrize("consumed", [False, True])
def test_expiry_only_closes_future_admission(
    monkeypatch: pytest.MonkeyPatch, consumed: bool
) -> None:
    """Admission expiry never drops an admitted operation's cancellation."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    if consumed:
        owner.submit(token, 17, lambda: 17)
    now[0] = 160.0
    if not consumed:
        with pytest.raises(ValueError, match="admission is closed"):
            owner.submit(token, 17, Mock())
    assert owner.cancel(token, Mock()) == "queued"


def test_full_reservations_reject_new_work_without_evicting_live_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reservation pressure cannot evict an unexpired cancellation capability."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    owner = ContextCancellationReservations()
    tokens = [owner.reserve(index) for index in range(1024)]
    with pytest.raises(RuntimeError, match="reservations are full"):
        owner.reserve(1024)
    assert owner.cancel(tokens[0], Mock()) == "not_submitted"
    owner.reserve(1024)
    now[0] = 161.0
    owner.reserve(1025)
    assert len(owner._pending) == 1
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(tokens[1], 1, Mock())


@pytest.mark.parametrize(
    "kind", ["malformed", "modified", "wrong_worker", "non_ascii", "overlong", "non_string"]
)
def test_invalid_token_cannot_cancel_or_submit(kind: str) -> None:
    """An old worker or modified capability cannot affect the current instance."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    if kind == "malformed":
        token = "not-a-token"
    elif kind == "modified":
        token = token.replace("[17,", "[18,", 1)
    elif kind == "wrong_worker":
        owner = ContextCancellationReservations()
    elif kind == "non_ascii":
        token = "x.\u00e9"
    elif kind == "overlong":
        token = "a" * 513
    else:
        token = 17
    callback = Mock()
    with pytest.raises(ValueError, match="Invalid context cancellation token"):
        owner.cancel(token, callback)
    with pytest.raises(ValueError, match="Invalid context cancellation token"):
        owner.submit(token, 17, callback)
    callback.assert_not_called()


def test_pairing_id_mismatch_and_successor_tokens_are_isolated() -> None:
    """Tokens remain distinct even when a caller repeats a pairing ID."""
    owner = ContextCancellationReservations()
    first = owner.reserve(17)
    second = owner.reserve(17)
    assert first != second
    with pytest.raises(ValueError, match="request ID mismatch"):
        owner.submit(first, 18, Mock())
    assert owner.cancel(first, Mock()) == "not_submitted"
    assert owner.submit(second, 17, lambda: 17) == 17
    queued = Mock()
    assert owner.cancel(first, queued) == "queued"
    queued.assert_called_once_with(first)


def test_failed_cancel_enqueue_does_not_return_a_receipt() -> None:
    """A queue closed after the worker check must not produce a false receipt."""
    owner = ContextCancellationReservations()
    token = owner.reserve(17)
    owner.submit(token, 17, lambda: 17)
    with pytest.raises(RuntimeError, match="closed"):
        owner.cancel(token, Mock(side_effect=RuntimeError("closed")))


@pytest.mark.parametrize("request_id", [True, 1.0, "1"])
def test_submission_requires_exact_integer_identity(request_id: object) -> None:
    """Equal-but-different types cannot bypass the supported request profile."""
    owner = ContextCancellationReservations()
    token = owner.reserve(1)
    with pytest.raises(ValueError, match="request ID mismatch"):
        owner.submit(token, request_id, Mock())


@pytest.mark.parametrize("request_id", [-1, True, "17", None])
def test_invalid_reservation_identity(request_id: object) -> None:
    """Reject identities outside the immutable integer request namespace."""
    with pytest.raises(ValueError, match="integer request ID"):
        ContextCancellationReservations().reserve(request_id)
