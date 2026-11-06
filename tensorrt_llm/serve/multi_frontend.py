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

The frontends share one executor and one ``SO_REUSEPORT`` port, so a client
cannot pick the process that answers. State that exists once per server lives
in the launcher (frontend 0), which also serves a Unix socket in the
multi-frontend ipc dir: attached frontends forward the routes backed by that
state to it, follow the clock offset it publishes, and watch it for death.
"""

import asyncio
import contextlib
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.logger import logger

LAUNCHER_UDS_NAME = "launcher.sock"
LAUNCHER_UDS_ENV = "TLLM_SERVE_LAUNCHER_UDS"
CLOCK_OFFSET_NAME = "clock_offset"

# Launcher-owned routes. Attached frontends register them ahead of the local
# handlers, which they shadow since FastAPI matches in registration order.
FORWARDED_ROUTES = (
    ("GET", "/metrics"),
    ("POST", "/kv_cache_events"),
    ("POST", "/start_profile"),
    ("POST", "/stop_profile"),
    ("POST", "/steady_clock_offset"),
    ("GET", "/v1/messages/batches"),
    ("POST", "/v1/messages/batches"),
    ("GET", "/v1/messages/batches/{rest:path}"),
    ("POST", "/v1/messages/batches/{rest:path}"),
    ("DELETE", "/v1/messages/batches/{rest:path}"),
)

# Hop-by-hop and framing headers, which aiohttp sets for the replayed request.
# The rest is forwarded: x-trtllm-msgpack selects the body codec, and auth
# headers feed the --middleware the launcher runs again.
_NOT_FORWARDED_REQUEST_HEADERS = frozenset(
    (
        "accept-encoding",
        "connection",
        "content-length",
        "expect",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    )
)


@dataclass(frozen=True)
class MultiFrontendServing:
    """This process's role in a multi-frontend server.

    ``None`` in place of an instance means single-frontend serving.
    """

    launcher_uds: str
    is_launcher: bool
    launcher_pid: Optional[int] = None  # attached frontends: their parent

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
        # Lazy, to bind to the serving loop. These routes are low-rate, so no
        # keep-alive connections to race against a launcher shutting down.
        if self._session is None or self._session.closed:
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
        headers = [
            (k, v)
            for k, v in request.headers.items()
            if k.lower() not in _NOT_FORWARDED_REQUEST_HEADERS
        ]
        body = await request.body()
        try:
            session = await self._get_session()
            # The Unix connector ignores the host.
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


class SharedClockOffset:
    """The disaggregated server's clock offset, published by the launcher.

    The disaggregated server POSTs it once, but every frontend stamps its own
    requests' perf metrics with it.
    """

    def __init__(self, ipc_dir: str):
        self._path = os.path.join(ipc_dir, CLOCK_OFFSET_NAME)
        self._seen = None

    def publish(self, offset: float) -> None:
        """Launcher side: atomically replace the published offset."""
        fd, tmp = tempfile.mkstemp(prefix=f".{CLOCK_OFFSET_NAME}.", dir=os.path.dirname(self._path))
        try:
            with os.fdopen(fd, "w") as f:
                f.write(repr(float(offset)))
            os.replace(tmp, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def poll(self) -> Optional[float]:
        """The published offset if it changed since the last poll, else None."""
        try:
            with open(self._path) as f:
                offset = float(f.read())
        except (OSError, ValueError):
            return None
        if not math.isfinite(offset) or offset == self._seen:
            return None
        self._seen = offset
        return offset


async def follow_clock_offset(
    shared: SharedClockOffset, apply: Callable[[float], None], interval_s: float = 0.25
) -> None:
    """Attached-frontend loop: apply every offset the launcher publishes."""
    while True:
        offset = shared.poll()
        if offset is not None:
            apply(offset)
        await asyncio.sleep(interval_s)


class AttachedFrontendWatchdog:
    """Detect a lost launcher or engine from an attached frontend.

    An attached frontend owns no workers: its only signals are the launcher
    process exiting and the launcher's ``/health`` failing. ``mark_dead`` is
    called once.
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
        # The launcher starts listening only after its attached frontends are
        # ready, so failures count once it has answered.
        launcher_answered = False
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
                launcher_answered = True
                failures = 0
                continue
            if not launcher_answered:
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
