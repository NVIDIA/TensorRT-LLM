# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ASGI middleware that stamps each HTTP request with its arrival time."""

from typing import Optional

from tensorrt_llm._utils import AdjustedSteadyClock


class ServerArrivalTimeMiddleware:
    """
    Custom ASGI middleware to track server arrival time.

    We implement this as a pure ASGI middleware instead of using FastAPI's
    @app.middleware("http") decorator because the decorator internally uses
    BaseHTTPMiddleware, which wraps the ASGI `receive` callable. This wrapping
    breaks Request.is_disconnected() functionality - the wrapped receive doesn't
    properly forward http.disconnect events while the middleware is waiting in
    call_next(), preventing detection of client disconnections during long-running
    non-streaming requests.

    By implementing pure ASGI middleware, we pass through the original receive/send
    callables unchanged, preserving the ability to detect client disconnections.

    See: https://github.com/encode/starlette/discussions/2094
    """

    def __init__(self,
                 app,
                 adjusted_clock: Optional[AdjustedSteadyClock] = None):
        self.app = app
        self._adjusted_clock = adjusted_clock or AdjustedSteadyClock()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add arrival time to scope
            scope.setdefault("state", {})
            scope["state"]["server_arrival_time"] = self._adjusted_clock.now()

        # Pass through the original receive/send - no wrapping!
        await self.app(scope, receive, send)
