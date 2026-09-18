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
"""How a native transfer's state reads as a contract outcome.

Both directions land here: a send session and a receive session expose the same three things a
handle needs -- the session's own verdict, its exception, and per-task status. What differs is what
``reports_pending`` is owed by: on the receive side the writers' reports, on the send side word
about this side's own writes. Neither answers whether anyone is still touching the memory; that is a
separate question nothing here asks.

TODO: Which ending a piece reports depends on when it is first polled. A piece whose session has
gone terminal while the piece itself is still writing latches the session's verdict, yet the piece
may finish afterwards -- so an early poll says failed and a late one says delivered. Nothing polls
these yet; it has to be settled before the surface is frozen.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.disaggregation.base import Cancelled, Delivered, Failed, Outcome

from .transfer import SessionStatus, TaskStatus


class TaskHandle:
    """One piece of one request, seen through the session carrying it.

    Pieces of the same request share that session, so this piece's own state is read first and the
    session's verdict only answers for a piece that has not ended on its own.
    """

    def __init__(self, session, task, token_end: int):
        self._session = session
        self._task = task
        self._token_end = token_end
        self._ended: Optional[Outcome] = None

    def poll(self) -> Optional[Outcome]:
        if self._ended is not None:
            # Which ending this was cannot change; only whether a writer still owes word can.
            return self._rebuild(self._ended)
        # This piece's own ending wins: bytes that landed landed, whatever became of its siblings,
        # and "cancelled" means stopped short of delivering.
        if self._task.status is TaskStatus.TRANSFERRED:
            outcome = Delivered(token_end=self._token_end)
        elif self._task.status is TaskStatus.ERROR:
            if self._ended_by_cancel():
                outcome = Cancelled(by_peer=self._session.cancelled_by_peer, reports_pending=True)
            else:
                outcome = Failed(reason=self._why_failed(), reports_pending=True)
        elif self._session.status is SessionStatus.CANCELLED:
            outcome = Cancelled(by_peer=self._session.cancelled_by_peer, reports_pending=True)
        elif self._session.status is SessionStatus.ERROR:
            outcome = Failed(reason=self._why_failed(), reports_pending=True)
        else:
            return None
        self._ended = outcome
        return self._rebuild(outcome)

    def _rebuild(self, ended: Outcome) -> Outcome:
        """The latched ending, with today's answer to whether a report is still owed.

        A session is shared by several pieces, so a sibling's failure moves the session's own
        verdict after this piece has already ended; the ending latched here does not move with it.
        """
        if isinstance(ended, Delivered):
            return ended
        owed = self._owed_a_report()
        if isinstance(ended, Cancelled):
            return Cancelled(by_peer=ended.by_peer, reports_pending=owed)
        return Failed(reason=ended.reason, reports_pending=owed)

    def _owed_a_report(self) -> bool:
        """Whether anyone still owes word about this piece.

        The question is how many writers have reported, not how the piece already ended: one
        writer's failure ends it while its siblings are still writing into the destination.
        Closing a session deregisters it, making a report impossible rather than late.
        """
        if getattr(self._session, "_closed", False):
            return False
        outstanding = getattr(self._task, "reports_outstanding", None)
        if outstanding is None:
            # A task that counts nothing leaves its own state as the only evidence there is.
            return self._task.status is TaskStatus.TRANSFERRING
        return outstanding

    def _ended_by_cancel(self) -> bool:
        """Whether this piece's error is the cancellation itself.

        A cancel ends the pieces that had not started by failing them, which is the same task state
        a real transfer error leaves; only the recorded cause tells the two apart, and it is the
        very object the cancel installed rather than one that merely reads like it.
        """
        cancelled = getattr(self._session, "_cancel_exception", None)
        return cancelled is not None and self._task._exception is cancelled

    def _why_failed(self) -> str:
        error = self._task._exception or self._session.exception
        return str(error) if error is not None else "transfer failed without a recorded cause"


class NothingPublished:
    """Submission failed before any peer was told where to write, so there is nothing to wait on."""

    def __init__(self, reason: str):
        self._outcome = Failed(reason=reason, reports_pending=False)

    def poll(self) -> Optional[Outcome]:
        return self._outcome
