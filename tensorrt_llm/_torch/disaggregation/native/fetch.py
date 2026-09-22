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
"""The native backend seen through the common contract: pulling a peer's cache into our pages."""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.disaggregation.base import Attempt, CacheExtent, Cancelled, Outcome
from tensorrt_llm._torch.disaggregation.base.transfer import get_unique_rid
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

from .handle import NothingPublished, TaskHandle
from .transfer import RxSession, SessionStatus, TransferWorker


def refuse_extent_for_another_request(extent: CacheExtent, request: LlmRequest) -> None:
    """Refuse an extent built for a different request than this adapter is bound to.

    Both sides ignore ``name`` otherwise -- they move ``extent.local`` for the request they hold --
    so passing the wrong extent would transfer the bound request's blocks under the other one's
    name, and nothing would say so.

    Compared against the same derivation the builder used, not against the session's id: the
    session is keyed by the name the peer resolves, which is a different value whenever a request
    carries a context id but no disaggregated one.
    """
    mine = get_unique_rid(request)
    if extent.name != mine:
        raise ValueError(f"extent names request {extent.name}, but this adapter is bound to {mine}")


class CancelledBeforePublication:
    """The session was cancelled before this piece was published, so no peer was told to write.

    The same remote cancel is a ``Cancelled`` once a task carries it; a piece that never got one
    would otherwise read as a failure, making the ending depend on when the cancel landed.
    """

    def __init__(self, by_peer: bool):
        self._outcome = Cancelled(by_peer=by_peer, reports_pending=False)

    def poll(self) -> Optional[Outcome]:
        return self._outcome


class PeerFetch:
    """Somewhere to pull one request's cache from: the peer that ran its context phase.

    One instance per request: the session a piece lands in belongs to the request.
    """

    def __init__(self, worker: TransferWorker, request: LlmRequest):
        self._worker = worker
        self._request = request
        self._session: Optional[RxSession] = None

    @property
    def session(self) -> Optional[RxSession]:
        """The session underneath, for the two jobs the contract does not cover: unpacking the
        auxiliary buffer, and the legacy sweep's own closing and timeout polling.

        ``None`` until the first fetch. Any other use means the contract is missing something.
        """
        return self._session

    def fetch(
        self,
        extent: CacheExtent,
        *,
        src: Optional[str] = None,
        expected_write_bytes: Optional[int] = None,
    ) -> Attempt:
        """``expected_write_bytes`` is the local byte total the remote writers must cover for
        this piece; the session tracks attested written bytes against it so reuse admission can
        be restricted to verifiably written ranges (see ``RxSession.kv_write_verified``)."""
        if src is not None:
            # This backend asks whoever the request names; accepting `src` and not using it would
            # read from a different peer than the caller asked for, and nothing would report it.
            raise NotImplementedError(
                "the native backend pulls from the peer named by the request, not from `src`"
            )
        refuse_extent_for_another_request(extent, self._request)
        if self._session is None:
            self._session = self._worker.create_rx_session(self._request)
        session = self._session
        # The session holds every piece of this request, so position is the only way back to the one
        # this call started.
        mine = len(session._kv_tasks)
        try:
            session.receive(extent.local, expected_write_bytes=expected_write_bytes)
        except Exception as error:
            # Publication reaches peers one at a time, so a failure can leave some already holding
            # the destination; the session goes terminal so the sweep can retire it. The error is
            # then passed on rather than turned into a handle: nothing polls handles yet, and
            # swallowing it here would drop a failure the caller above stops the request on.
            session.fail_admission(error)
            raise
        if len(session._kv_tasks) <= mine:
            # A closed or already terminal session takes no task for the piece, so the destination
            # was published to no one. A cancelled session says so itself, since nothing else here
            # would carry who asked for the stop.
            if session.status is SessionStatus.CANCELLED:
                return CancelledBeforePublication(by_peer=session.cancelled_by_peer)
            return NothingPublished("the session was closed or terminal and took no task")
        return TaskHandle(session, session._kv_tasks[mine], extent.local.token_range.end)
