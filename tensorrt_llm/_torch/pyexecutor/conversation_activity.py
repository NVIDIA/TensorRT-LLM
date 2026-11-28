# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .kv_cache_stats import ConversationActivityStats

if TYPE_CHECKING:
    from .llm_request import LlmRequest

_ConversationKey = tuple[str, int | None, str | None]
_SHORT_WINDOW_S = 600
_LONG_WINDOW_S = 3600


def _request_conversation_id(request: "LlmRequest") -> str | None:
    if request.is_dummy_request:
        return None
    conversation_params = request.py_conversation_params
    if conversation_params is None:
        return None
    conversation_id = conversation_params.conversation_id.strip()
    return conversation_id or None


def request_conversation_key(request: "LlmRequest") -> tuple[str, int | None, str | None] | None:
    conversation_id = _request_conversation_id(request)
    if conversation_id is None:
        return None
    # Retention plans must use the same namespace as prefix reuse.
    return conversation_id, request.lora_task_id, request.cache_salt


@dataclass(slots=True)
class _Admission:
    key: _ConversationKey
    last_seen: float
    size: int


class ConversationActivityTracker:
    """Track rolling worker-local admission counts within an identity budget."""

    def __init__(
        self,
        *,
        max_entries: int = 100_000,
        max_identity_bytes: int = 16 * 1024 * 1024,
        now: float | None = None,
    ) -> None:
        if max_entries < 0 or max_identity_bytes < 0:
            raise ValueError("Conversation activity limits must be nonnegative")
        now = time.monotonic() if now is None else now
        self._max_entries = max_entries
        self._max_identity_bytes = max_identity_bytes
        self._identity_bytes = 0
        self._admissions: OrderedDict[_ConversationKey, _Admission] = OrderedDict()
        # Both indices reference the same identity. The short index is a subset.
        self._recent: OrderedDict[_ConversationKey, None] = OrderedDict()
        self._complete_at = {
            seconds: now + seconds for seconds in (_SHORT_WINDOW_S, _LONG_WINDOW_S)
        }
        self._overflow_entries = 0
        self._overflow_bytes = 0

    def _expire(self, now: float) -> None:
        while self._recent:
            key = next(iter(self._recent))
            if self._admissions[key].last_seen + _SHORT_WINDOW_S > now:
                break
            self._recent.popitem(last=False)
        while self._admissions:
            admission = next(iter(self._admissions.values()))
            if admission.last_seen + _LONG_WINDOW_S > now:
                break
            self._admissions.popitem(last=False)
            self._identity_bytes -= admission.size

    def observe_admission(self, request: "LlmRequest", now: float) -> bool:
        """Refresh the identity in both windows and report whether one was supplied."""
        key = request_conversation_key(request)
        if key is None:
            return False
        self._expire(now)
        admission = self._admissions.get(key)
        if admission is None:
            size = sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)
            if self._identity_bytes + size > self._max_identity_bytes:
                self._overflow_bytes += 1
            elif len(self._admissions) >= self._max_entries:
                self._overflow_entries += 1
            else:
                admission = _Admission(key, now, size)
                self._admissions[key] = admission
                self._identity_bytes += size
            if admission is None:
                for seconds in self._complete_at:
                    self._complete_at[seconds] = now + seconds
                return True
        admission.last_seen = now
        self._admissions.move_to_end(admission.key)
        self._recent[admission.key] = None
        self._recent.move_to_end(admission.key)
        return True

    def snapshot_activity(self, now: float) -> ConversationActivityStats:
        """Return current counts, completeness and cumulative overflow."""
        self._expire(now)
        return ConversationActivityStats(
            len(self._recent),
            int(now >= self._complete_at[_SHORT_WINDOW_S]),
            self._overflow_entries,
            self._overflow_bytes,
            len(self._admissions),
            int(now >= self._complete_at[_LONG_WINDOW_S]),
            self._overflow_entries,
            self._overflow_bytes,
        )

    def next_refresh(self, now: float) -> float:
        """Return the next time an idle snapshot can change."""
        deadline = min(
            (value for value in self._complete_at.values() if value > now), default=float("inf")
        )
        if self._recent:
            key = next(iter(self._recent))
            deadline = min(deadline, self._admissions[key].last_seen + _SHORT_WINDOW_S)
        if self._admissions:
            oldest = next(iter(self._admissions.values()))
            deadline = min(deadline, oldest.last_seen + _LONG_WINDOW_S)
        return deadline
