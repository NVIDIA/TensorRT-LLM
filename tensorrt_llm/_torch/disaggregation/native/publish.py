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
"""The native backend seen through the common contract: offering our pages to the peer that asked."""

from __future__ import annotations

from tensorrt_llm._torch.disaggregation.base import Attempt, CacheExtent
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

from .fetch import refuse_extent_for_another_request
from .handle import TaskHandle
from .transfer import TxSession


class PeerPublish:
    """The peers waiting on one request's context, seen as somewhere to offer its cache.

    Built around a session rather than creating one, unlike the receive side: a send session may
    already exist from an earlier chunk, and whether a new one may be opened is a retirement
    question that stays outside the contract.
    """

    def __init__(self, session: TxSession, request: LlmRequest):
        self._session = session
        self._request = request

    @property
    def session(self) -> TxSession:
        """The session underneath, for the jobs the contract does not cover: the auxiliary buffer,
        and the legacy sweep's own closing and timeout polling."""
        return self._session

    def publish(self, extent: CacheExtent) -> Attempt:
        refuse_extent_for_another_request(extent, self._request)
        # The session holds every piece of this request, so position is the only way back to the one
        # this call started.
        mine = len(self._session.kv_tasks)
        self._session.send(extent.local)
        return TaskHandle(self._session, self._session.kv_tasks[mine], extent.local.token_range.end)
