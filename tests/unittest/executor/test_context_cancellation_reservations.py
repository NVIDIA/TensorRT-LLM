# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import secrets
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, current_thread
from unittest.mock import Mock

import pytest

from tensorrt_llm.executor import context_cancellation
from tensorrt_llm.executor.context_cancellation import ContextCancellationReservations

pytestmark = pytest.mark.cpu_only


def _reserve(owner: ContextCancellationReservations, request_id: int) -> str:
    """Supply fresh authorization while existing tests control the admission clock."""
    now_ms = owner._authorization_epoch_ms + int(
        (context_cancellation.time.monotonic() - owner._authorization_origin) * 1000
    )
    return owner.reserve(request_id, secrets.token_hex(16), now_ms + 60_000)


def test_cancel_before_submit_closes_admission() -> None:
    """A delayed POST cannot resurrect a reservation cancelled before dispatch."""
    owner = ContextCancellationReservations()
    token = _reserve(owner, 17)
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
    token = _reserve(owner, 17)
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
    token = _reserve(owner, 17)
    submitting = Event()
    finish_submit = Event()
    cancelling = Event()
    order = []

    class ObservedLock:
        """Expose real contention, not merely cancellation-thread readiness."""

        def __init__(self) -> None:
            self.lock = Lock()
            self.holder = None

        def __enter__(self) -> None:
            if submitting.is_set() and self.holder is not current_thread():
                assert self.lock.locked()
                cancelling.set()
            self.lock.acquire()
            self.holder = current_thread()

        def __exit__(self, *args: object) -> None:
            self.holder = None
            self.lock.release()

    owner._lock = ObservedLock()

    def submit() -> int:
        """Hold the real admission callback at the executor boundary."""
        submitting.set()
        assert finish_submit.wait(5)
        order.append("submit")
        return 17

    def cancel() -> str:
        """Attempt concurrent cancellation while submission owns the lock."""
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
    token = _reserve(owner, 17)
    enqueued = []

    def submit() -> int:
        """Record executor admission before the caller observes an enqueue failure."""
        enqueued.append(token)
        raise RuntimeError("enqueue failed")

    with pytest.raises(RuntimeError, match="enqueue failed"):
        owner.submit(token, 17, submit)
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(token, 17, submit)
    assert enqueued == [token]
    cancel = Mock()
    assert owner.cancel(token, cancel) == "queued"
    cancel.assert_called_once_with(token)


@pytest.mark.parametrize("consumed", [False, True])
def test_expiry_only_closes_future_admission(
    monkeypatch: pytest.MonkeyPatch, consumed: bool
) -> None:
    """Admission expiry never drops an admitted operation's cancellation."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    owner = ContextCancellationReservations()
    token = _reserve(owner, 17)
    if consumed:
        owner.submit(token, 17, lambda: 17)
    now[0] = 160.0
    if not consumed:
        with pytest.raises(ValueError, match="admission is closed"):
            owner.submit(token, 17, Mock())
    assert owner.cancel(token, Mock()) == "queued"


def test_expiry_prunes_only_the_expired_prefix_without_resetting_live_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New reservations preserve the admission windows of existing live tokens."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    owner = ContextCancellationReservations()
    expired = _reserve(owner, 17)
    now[0] = 110.0
    live = _reserve(owner, 18)
    now[0] = 160.0
    newest = _reserve(owner, 19)
    assert list(owner._pending) == [live, newest]
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(expired, 17, Mock())
    now[0] = 170.0
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(live, 18, Mock())
    assert owner.submit(newest, 19, lambda: 19) == 19


def test_full_reservations_reject_new_work_without_evicting_live_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reservation pressure cannot evict an unexpired cancellation capability."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    owner = ContextCancellationReservations()
    tokens = [_reserve(owner, index) for index in range(1024)]
    with pytest.raises(RuntimeError, match="reservations are full"):
        _reserve(owner, 1024)
    assert owner.cancel(tokens[0], Mock()) == "not_submitted"
    _reserve(owner, 1024)
    now[0] = 161.0
    _reserve(owner, 1025)
    assert len(owner._pending) == 1
    with pytest.raises(ValueError, match="admission is closed"):
        owner.submit(tokens[1], 1, Mock())


@pytest.mark.parametrize(
    "kind", ["malformed", "modified", "wrong_worker", "non_ascii", "overlong", "non_string"]
)
def test_invalid_token_cannot_cancel_or_submit(kind: str) -> None:
    """An old worker or modified capability cannot affect the current instance."""
    owner = ContextCancellationReservations()
    token = _reserve(owner, 17)
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
    first = _reserve(owner, 17)
    second = _reserve(owner, 17)
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
    token = _reserve(owner, 17)
    owner.submit(token, 17, lambda: 17)
    with pytest.raises(RuntimeError, match="closed"):
        owner.cancel(token, Mock(side_effect=RuntimeError("closed")))
    recovered = Mock()
    assert owner.cancel(token, recovered) == "queued"
    recovered.assert_called_once_with(token)


def test_forward_clock_correction_accepts_fresh_auth_but_rollback_cannot_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrected UTC can advance replay time, but cannot reopen removed nonces."""
    wall_ms, monotonic = [1_000_000], [100.0]
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: wall_ms[0] * 1_000_000)
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: monotonic[0])
    owner = ContextCancellationReservations()
    first = owner.reserve(17, "a" * 32, 1_060_000)
    owner.submit(first, 17, lambda: 17)
    wall_ms[0] += 90_000
    fresh = owner.reserve(18, "b" * 32, wall_ms[0] + 60_000)
    assert list(owner._pending) == [fresh]
    assert "a" * 32 not in owner._authorizations
    wall_ms[0] -= 90_000
    with pytest.raises(ValueError, match="validity window"):
        owner.reserve(17, "a" * 32, 1_060_000)
    assert owner.submit(fresh, 18, lambda: 18) == 18
    # Frequent probes must not discard fractional monotonic progress.
    for index in range(1, 2002):
        monotonic[0] = 100.0 + index * 0.0005
        with pytest.raises(ValueError, match="validity window"):
            owner.reserve(19, "c" * 32, 1_000_000)
    with pytest.raises(ValueError, match="validity window"):
        owner.reserve(19, "c" * 32, 1_091_000)


@pytest.mark.parametrize("request_id", [True, 1.0, "1"])
def test_submission_requires_exact_integer_identity(request_id: object) -> None:
    """Equal-but-different types cannot bypass the supported request profile."""
    owner = ContextCancellationReservations()
    token = _reserve(owner, 1)
    with pytest.raises(ValueError, match="request ID mismatch"):
        owner.submit(token, request_id, Mock())


@pytest.mark.parametrize("request_id", [-1, True, "17", None, 2**64, 10**500])
def test_invalid_reservation_identity(request_id: object) -> None:
    """Reject identities outside the immutable integer request namespace."""
    owner = ContextCancellationReservations()
    with pytest.raises(ValueError, match="integer request ID"):
        _reserve(owner, request_id)
    assert not owner._pending and not owner._authorizations


def test_largest_native_request_identity_has_usable_token() -> None:
    """The largest native uint64 identity remains valid at every token boundary."""
    owner = ContextCancellationReservations()
    request_id = 2**64 - 1
    token = _reserve(owner, request_id)
    assert len(token) <= 512
    assert owner.submit(token, request_id, lambda: request_id) == request_id
    cancel = Mock()
    assert owner.cancel(token, cancel) == "queued"
    cancel.assert_called_once_with(token)


@pytest.mark.parametrize("close", ["submit", "cancel", "admission_expiry"])
def test_shared_reserve_authorization_is_idempotent_only_while_pending(
    monkeypatch: pytest.MonkeyPatch, close: str
) -> None:
    """Two frontend deliveries share one token, but consumed intent cannot be reopened."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: 1_000_000_000_000)
    owner = ContextCancellationReservations()
    nonce, expiry = "a" * 32, 1_120_000
    ready = Barrier(2)

    def deliver() -> str:
        """Align independent frontend deliveries before the worker admission lock."""
        ready.wait(timeout=5)
        return owner.reserve(17, nonce, expiry)

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Two RPC/frontend lanes enter the same real worker lock concurrently.
        deliveries = [pool.submit(deliver) for _ in range(2)]
        first, second = [result.result(timeout=5) for result in deliveries]
    assert first == second
    assert list(owner._pending) == [first]
    assert len(owner._authorizations) == 1
    if close == "submit":
        assert owner.submit(first, 17, lambda: 17) == 17
    elif close == "cancel":
        assert owner.cancel(first, Mock()) == "not_submitted"
    else:
        now[0] = 161.0
    with pytest.raises(ValueError, match="admission is closed"):
        owner.reserve(17, nonce, expiry)
    assert len(owner._authorizations) == 1


@pytest.mark.parametrize("field", ["request_id", "expiry"])
def test_authorization_nonce_cannot_change_its_identity(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """A valid authorization ID is bound to its original request and signed expiry."""
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: 1_000_000_000_000)
    owner = ContextCancellationReservations()
    nonce, expiry = "a" * 32, 1_060_000
    token = owner.reserve(17, nonce, expiry)
    with pytest.raises(ValueError, match="identity mismatch"):
        owner.reserve(
            18 if field == "request_id" else 17, nonce, expiry + 1 if field == "expiry" else expiry
        )
    assert list(owner._pending) == [token]


@pytest.mark.parametrize(
    ("nonce", "expiry"),
    [
        ("a" * 32, 1_000_000),
        ("a" * 32, 1_120_001),
        ("g" * 32, 1_060_000),
        ("a" * 31, 1_060_000),
        ("a" * 32, True),
    ],
    ids=["expired", "far_future", "non_hex", "short", "non_integer"],
)
def test_invalid_authorization_cannot_allocate_pending_or_replay_state(
    monkeypatch: pytest.MonkeyPatch, nonce: str, expiry: object
) -> None:
    """Unbounded or malformed authorizations are rejected before recording them."""
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: 1_000_000_000_000)
    owner = ContextCancellationReservations()
    with pytest.raises(ValueError, match="authorization"):
        owner.reserve(17, nonce, expiry)
    assert not owner._pending
    assert not owner._authorizations
    assert not owner._authorization_expiries


def test_replay_ledger_capacity_is_separate_and_expiry_does_not_reopen_old_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry order, overload and wall rollback cannot revive a consumed authorization."""
    now = [100.0]
    monkeypatch.setattr(context_cancellation.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: 1_000_000_000_000)
    monkeypatch.setattr(context_cancellation, "_PENDING_RESERVATION_LIMIT", 1)
    monkeypatch.setattr(context_cancellation, "_AUTHORIZATION_REPLAY_LIMIT", 2)
    owner = ContextCancellationReservations()
    first = owner.reserve(17, "a" * 32, 1_090_000)
    owner.submit(first, 17, lambda: 17)
    second = owner.reserve(18, "b" * 32, 1_030_000)
    owner.cancel(second, Mock())
    assert not owner._pending and len(owner._authorizations) == 2
    with pytest.raises(RuntimeError, match="replay accounting is full"):
        owner.reserve(19, "c" * 32, 1_060_000)
    # The second nonce expires first. Pruning must not evict the first live one.
    now[0] = 131.0
    monkeypatch.setattr(context_cancellation.time, "time_ns", lambda: 900_000_000_000)
    third = owner.reserve(19, "c" * 32, 1_091_000)
    assert set(owner._authorizations) == {"a" * 32, "c" * 32}
    assert list(owner._pending) == [third]
    with pytest.raises(ValueError, match="validity window"):
        owner.reserve(18, "b" * 32, 1_030_000)
    with pytest.raises(ValueError, match="admission is closed"):
        owner.reserve(17, "a" * 32, 1_090_000)
    # Admission authorization expiry does not revoke cancellation of old work.
    now[0] = 191.0
    fresh = owner.reserve(20, "d" * 32, 1_151_000)
    assert list(owner._pending) == [fresh]
    assert set(owner._authorizations) == {"d" * 32}
    cancel = Mock()
    assert owner.cancel(first, cancel) == "queued"
    cancel.assert_called_once_with(first)
    with pytest.raises(ValueError, match="validity window"):
        owner.reserve(17, "a" * 32, 1_090_000)
