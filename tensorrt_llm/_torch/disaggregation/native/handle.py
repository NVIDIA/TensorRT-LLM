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
"""Observe a native task's committed logical result and current report progress.

Task/session transitions commit the result, not polling. Reports and physical
quiescence remain separate questions; an outcome never authorizes memory reuse.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.disaggregation.base import Cancelled, Delivered, Failed, Outcome

from .transfer import KVRecvTask, KVSendTask, RxSession, SessionStatus, TaskStatus, TxSession


class TaskHandle:
    """One piece's decision, including when the first observer arrives late."""

    def __init__(
        self, session: RxSession | TxSession, task: KVRecvTask | KVSendTask, token_end: int
    ) -> None:
        self._session = session
        self._task = task
        self._token_end = token_end

    def poll(self) -> Optional[Outcome]:
        result = self._task.logical_outcome
        if result is None:
            return None
        if result.status is SessionStatus.TRANSFERRED:
            return Delivered(token_end=self._token_end)
        owed = self._owed_a_report()
        if result.status is SessionStatus.CANCELLED:
            return Cancelled(by_peer=result.by_peer, reports_pending=owed)
        return Failed(reason=result.reason, reports_pending=owed)

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


class NothingPublished:
    """Submission failed before any peer was told where to write, so there is nothing to wait on."""

    def __init__(self, reason: str):
        self._outcome = Failed(reason=reason, reports_pending=False)

    def poll(self) -> Optional[Outcome]:
        return self._outcome
