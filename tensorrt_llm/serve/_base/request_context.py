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
"""What the chat core reads from an HTTP request, as one explicit object."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from starlette.datastructures import Address, Headers, State
from starlette.requests import Request

_UNSET: Any = object()


class RequestContext:
    """What the chat core reads from an HTTP request, made explicit.

    ``from_http()`` wraps the request FastAPI passed to a route and delegates to
    it lazily, so the middleware and the core share one state
    (``server_arrival_time``, ``server_first_token_time``,
    ``perf_metrics_records``) and msgpack bodies keep decoding through the
    request's own ``json()``. ``internal()`` is a request with no client behind
    it, for Anthropic message batches.

    The properties ``headers``, ``state``, ``client``, ``json()`` and
    ``is_disconnected()`` carry the names ``Request`` uses, so helpers that
    still take a ``Request`` accept a context unchanged. The object is always
    truthy, as every ``Request`` is.
    """

    def __init__(
        self,
        *,
        request: Optional[Request] = None,
        state: Optional[State] = None,
        raw_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Use ``from_http()`` or ``internal()``."""
        if request is None and state is None:
            raise ValueError(
                "RequestContext needs a request to delegate to or an internal state; "
                "build it with RequestContext.from_http(request) or "
                "RequestContext.internal(raw_json)"
            )
        self._request = request
        self._state = state
        self._raw_json = raw_json
        self._json_cache: Any = _UNSET

    @classmethod
    def from_http(cls, request: Request) -> RequestContext:
        """Wrap ``request``; nothing is read from it until the core asks."""
        return cls(request=request)

    @classmethod
    def internal(cls, raw_json: Dict[str, Any]) -> RequestContext:
        """A request without headers or a client, whose raw JSON is ``raw_json``.

        The state carries what the middleware would have seeded on a real
        connection: an empty ``perf_metrics_records`` list, and
        ``no_client_connection`` so that the disconnect watcher does not poll a
        connection that can never drop.
        """
        return cls(
            state=State({"perf_metrics_records": [], "no_client_connection": True}),
            raw_json=raw_json,
        )

    @property
    def http_request(self) -> Optional[Request]:
        """The wrapped HTTP request; ``None`` for an internal context."""
        return self._request

    @property
    def headers(self) -> Mapping[str, str]:
        if self._request is None:
            return Headers()
        return self._request.headers

    @property
    def state(self) -> State:
        if self._request is not None:
            return self._request.state
        assert self._state is not None  # the constructor requires one of the two
        return self._state

    @property
    def client(self) -> Optional[Address]:
        if self._request is None:
            return None
        return self._request.client

    async def json(self) -> Any:
        """The request body as JSON.

        On the HTTP path this is the request's own ``json()`` (msgpack-aware on
        workers, cached by the request). An internal context answers its raw
        JSON after one encode-decode round trip, computed once, which is what a
        body read from the wire looks like.
        """
        if self._request is not None:
            return await self._request.json()
        if self._json_cache is _UNSET:
            self._json_cache = json.loads(json.dumps(self._raw_json))
        return self._json_cache

    async def is_disconnected(self) -> bool:
        """Whether the client went away; always ``False`` without a client."""
        if self._request is None:
            return False
        return await self._request.is_disconnected()

    # Typed views of the state, for code written against the context.

    @property
    def no_client_connection(self) -> bool:
        return bool(getattr(self.state, "no_client_connection", False))

    @property
    def server_arrival_time(self) -> Optional[float]:
        value: Optional[float] = getattr(self.state, "server_arrival_time", None)
        return value

    @property
    def server_first_token_time(self) -> Optional[float]:
        value: Optional[float] = getattr(self.state, "server_first_token_time", None)
        return value

    @server_first_token_time.setter
    def server_first_token_time(self, value: Optional[float]) -> None:
        self.state.server_first_token_time = value

    @property
    def perf_metrics_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = self.state.perf_metrics_records
        return records
