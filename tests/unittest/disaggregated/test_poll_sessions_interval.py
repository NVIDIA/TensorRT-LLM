# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for KvCacheTransceiverV2._poll_sessions_for_interval.

The idle executor loop calls check_context_transfer_status(1) on every
iteration where no batch is scheduled. With no in-flight send session the
poll's exit condition (completed + failed >= wait_num) is unsatisfiable, so
prior to the fix the helper slept out the full
kv_transfer_sender_future_timeout_ms (default 1000 ms) per idle iteration,
delaying scheduling of newly arrived requests (nvbugs 6647405).
"""

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


def _make_transceiver() -> "KvCacheTransceiverV2":
    """Build a bare instance; _poll_sessions_for_interval only needs _collect_done."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    return object.__new__(KvCacheTransceiverV2)


class _FakeSession:
    """Session stub that flips to completed after an optional delay."""

    def __init__(self, complete_after_s: Optional[float] = None, failed: bool = False) -> None:
        self._failed = failed
        self._complete_at = (
            time.monotonic() + complete_after_s if complete_after_s is not None else None
        )

    def is_completed(self) -> bool:
        return self._complete_at is not None and time.monotonic() >= self._complete_at

    def has_failed(self) -> bool:
        return self._failed

    def wait_complete(self, blocking: bool = False) -> None:
        pass


INTERVAL_MS = 1000


def test_empty_sessions_returns_immediately():
    """No in-flight session: the unsatisfiable target must not sleep out the interval."""
    tc = _make_transceiver()
    start = time.monotonic()
    tc._poll_sessions_for_interval({}, {}, 1, INTERVAL_MS)
    assert time.monotonic() - start < 0.1


def test_wait_num_clamped_to_session_count():
    """A target above len(sessions) waits only for what can actually complete."""
    tc = _make_transceiver()
    sessions = {1: _FakeSession(complete_after_s=0.05)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 2, INTERVAL_MS)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5
    assert sessions[1].is_completed()


def test_waits_for_inflight_session_completion():
    """An in-flight session is still awaited (the PR #17535 semantics are kept)."""
    tc = _make_transceiver()
    sessions = {1: _FakeSession(complete_after_s=0.05)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, INTERVAL_MS)
    elapsed = time.monotonic() - start
    assert 0.04 <= elapsed < 0.5
    assert sessions[1].is_completed()


def test_deadline_still_bounds_never_completing_session():
    """A session that never completes releases the caller at the deadline."""
    tc = _make_transceiver()
    sessions = {1: _FakeSession(complete_after_s=None)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, 100)
    elapsed = time.monotonic() - start
    assert 0.09 <= elapsed < 1.0


def test_failed_session_counts_toward_target():
    """A failed session satisfies the exit condition without waiting."""
    tc = _make_transceiver()
    sessions = {1: _FakeSession(failed=True)}
    start = time.monotonic()
    tc._poll_sessions_for_interval(sessions, {1: object()}, 1, INTERVAL_MS)
    assert time.monotonic() - start < 0.1
