# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for trtllm-serve multi-frontend serving (``num_serve_frontends > 1``).

Several frontend processes share one executor and one ``SO_REUSEPORT`` port,
and a client cannot choose which of them answers. Anything that needs exactly
one owner per server therefore lives in the launcher (frontend 0): draining
the engine's iteration statistics and KV-cache events, runtime profiling, and
the in-memory Anthropic Message Batches store. The launcher's uvicorn also
listens on a Unix domain socket in the multi-frontend ipc directory; attached
frontends forward those routes to it (:class:`LauncherForwarder`) and watch it
(:class:`AttachedFrontendWatchdog`) so that a dead launcher or engine does not
leave them answering ``/health`` 200 with nothing behind them.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Optional

import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.logger import logger

LAUNCHER_UDS_NAME = "launcher.sock"

# Routes an attached frontend hands to the launcher, as FastAPI route
# templates. They are registered before the regular routes, and FastAPI
# matches in registration order, so they shadow the local handlers. The
# catch-all covers /v1/messages/batches/{batch_id}[/cancel|/results].
FORWARDED_ROUTES = (
    ("GET", "/metrics"),
    ("POST", "/kv_cache_events"),
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
    ("GET", "/v1/messages/batches"),
    ("POST", "/v1/messages/batches"),
    ("GET", "/v1/messages/batches/{rest:path}"),
    ("POST", "/v1/messages/batches/{rest:path}"),
    ("DELETE", "/v1/messages/batches/{rest:path}"),
)

_FORWARDED_REQUEST_HEADERS = ("content-type", "accept", "authorization")


@dataclass(frozen=True)
class MultiFrontendServing:
    """This process's role in a multi-frontend server.

    ``None`` in place of an instance means single-frontend serving.
    """

    launcher_uds: str
    is_launcher: bool
    # The launcher's pid as seen by an attached frontend (its parent).
    launcher_pid: Optional[int] = None

    @property
    def is_attached(self) -> bool:
        return not self.is_launcher


class LauncherForwarder:
    """Forward HTTP requests from an attached frontend to the launcher."""

    def __init__(self, uds_path: str, timeout_s: float = 120.0):
        self._uds_path = uds_path
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        # Created lazily so it binds to the serving event loop.
        if self._session is None or self._session.closed:
            # force_close: no idle keep-alive connections to race against a
            # launcher that is shutting down; these routes are low-rate.
            self._session = aiohttp.ClientSession(
                connector=aiohttp.UnixConnector(path=self._uds_path, force_close=True),
                timeout=self._timeout,
            )
        return self._session

    async def forward(self, request: Request) -> Response:
        """FastAPI endpoint: replay ``request`` against the launcher verbatim."""
        target = request.url.path
        if request.url.query:
            target = f"{target}?{request.url.query}"
        headers = {
            k: v for k, v in request.headers.items() if k.lower() in _FORWARDED_REQUEST_HEADERS
        }
        body = await request.body()
        try:
            session = await self._get_session()
            # The host is required by the URL syntax and ignored by the
            # Unix connector.
            async with session.request(
                request.method, f"http://launcher{target}", data=body, headers=headers
            ) as resp:
                content = await resp.read()
                return Response(
                    content=content,
                    status_code=resp.status,
                    media_type=resp.headers.get("Content-Type"),
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.warning(
                f"Forwarding {request.method} {target} to the launcher frontend failed: {e!r}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"The launcher frontend that serves {request.url.path} "
                    f"is unavailable: {e!r}"
                },
            )

    async def get_status(self, path: str, timeout_s: float = 5.0) -> Optional[int]:
        """HTTP status of ``GET path`` on the launcher, ``None`` if unreachable."""
        try:
            session = await self._get_session()
            async with session.get(
                f"http://launcher{path}", timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                await resp.read()
                return resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


class AttachedFrontendWatchdog:
    """Detect a lost launcher or engine from an attached frontend.

    An attached frontend owns no worker processes, so the only signals that
    the engine is gone are its parent (the launcher) disappearing, or the
    launcher's own ``/health`` failing, which happens once the launcher's
    executor recorded a fatal engine error. Either way ``mark_dead`` is
    called exactly once; the caller records the fatal error on its executor
    (failing in-flight requests fast) and starts server shutdown.
    """

    def __init__(
        self,
        forwarder: LauncherForwarder,
        launcher_pid: int,
        mark_dead: Callable[[BaseException], None],
        *,
        ppid_interval_s: float = 1.0,
        health_interval_s: float = 5.0,
        health_failures: int = 3,
    ):
        self._forwarder = forwarder
        self._launcher_pid = launcher_pid
        self._mark_dead = mark_dead
        self._ppid_interval = ppid_interval_s
        self._health_interval = health_interval_s
        self._health_failures = health_failures

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        failures = 0
        next_health = loop.time() + self._health_interval
        while True:
            await asyncio.sleep(self._ppid_interval)
            if os.getppid() != self._launcher_pid:
                self._die(f"launcher frontend (pid {self._launcher_pid}) exited")
                return
            if loop.time() < next_health:
                continue
            next_health = loop.time() + self._health_interval
            status = await self._forwarder.get_status("/health")
            if status == 200:
                failures = 0
                continue
            failures += 1
            if failures >= self._health_failures:
                self._die(
                    f"launcher frontend /health failed {failures} "
                    f"consecutive times (last status: {status})"
                )
                return

    def _die(self, why: str) -> None:
        logger.error(f"Attached frontend lost its engine: {why}")
        self._mark_dead(RuntimeError(f"attached frontend lost its engine: {why}"))
