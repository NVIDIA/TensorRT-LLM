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
"""ASGI middleware that returns per-request perf metrics and hands completed records to the JSONL writer."""

from typing import Any, Dict, Optional

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.serve.perf_metrics import (
    _CLOCK_SYNC_HEADER_BYTES,
    _RETURN_METRICS_HEADER_BYTES,
    PerfMetricsJsonlWriter,
    _limit_metrics_headers,
    build_clock_sync_header,
    build_metrics_headers,
    build_metrics_sse_event,
)


class PerfMetricsMiddleware:
    """Expose request metrics and optionally persist completed records."""

    def __init__(
        self,
        app: Any,
        expose_headers: bool,
        writer: Optional[PerfMetricsJsonlWriter] = None,
        adjusted_clock: Optional[AdjustedSteadyClock] = None,
    ):
        self._app = app
        self._expose_headers = expose_headers
        self._writer = writer
        self._adjusted_clock = adjusted_clock

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        is_stream = False
        metrics_headers = None
        records = scope.setdefault("state", {}).setdefault("perf_metrics_records", [])
        return_metrics = self._expose_headers and any(
            name.lower() == _RETURN_METRICS_HEADER_BYTES and value.strip() == b"1"
            for name, value in scope.get("headers", [])
        )

        async def send_metrics(message: Dict[str, Any]) -> None:
            nonlocal is_stream, metrics_headers
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                is_stream = any(
                    key.lower() == b"content-type" and b"text/event-stream" in value.lower()
                    for key, value in headers
                )
                if records and return_metrics and not is_stream:
                    metrics_headers = build_metrics_headers(records)
                    public_headers = _limit_metrics_headers(metrics_headers)
                    headers.extend(
                        (name.encode(), value.encode()) for name, value in public_headers.items()
                    )
                if return_metrics and self._adjusted_clock is not None:
                    receive_time = scope.get("state", {}).get("server_arrival_time")
                    if receive_time is not None:
                        clock_sync = build_clock_sync_header(
                            receive_time,
                            self._adjusted_clock.now(),
                        )
                        headers.append((_CLOCK_SYNC_HEADER_BYTES, clock_sync.encode()))
                message["headers"] = headers

            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                if records:
                    if metrics_headers is None:
                        metrics_headers = build_metrics_headers(records)
                    if self._writer is not None:
                        for record in records:
                            self._writer.submit(record)
                    if return_metrics and is_stream:
                        message["body"] = message.get("body", b"") + build_metrics_sse_event(
                            metrics_headers
                        )
                        try:
                            await send(message)
                        except OSError:
                            pass
                        return
            await send(message)

        await self._app(scope, receive, send_metrics)
