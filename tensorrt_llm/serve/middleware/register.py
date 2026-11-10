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
"""Built-in middleware of the aggregated and disaggregated trtllm-serve apps.

Starlette's ``add_middleware`` inserts at the front of the stack, so the middleware added
last runs first. Both apps call ``register_middlewares``; do not add middleware elsewhere.
"""

from typing import Optional

from fastapi import FastAPI

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.serve.middleware.arrival_time import ServerArrivalTimeMiddleware
from tensorrt_llm.serve.middleware.perf_metrics import PerfMetricsMiddleware
from tensorrt_llm.serve.middleware.request_trace import RequestTrace, RequestTraceMiddleware
from tensorrt_llm.serve.perf_metrics import PerfMetricsJsonlWriter


def register_middlewares(
    app: FastAPI,
    *,
    collect_perf_metrics: bool,
    expose_perf_metrics: bool,
    perf_metrics_writer: Optional[PerfMetricsJsonlWriter],
    adjusted_clock: Optional[AdjustedSteadyClock] = None,
    request_trace: Optional[RequestTrace] = None,
) -> None:
    """Install the built-in middleware of a trtllm-serve app.

    The stack, outermost first, is ServerArrivalTimeMiddleware, then
    RequestTraceMiddleware when a request trace is configured, then PerfMetricsMiddleware
    when perf metrics are collected. The trace sits inside the arrival stamp, which it
    records, and outside the perf metrics, so the reply it records is the one the client
    received, metrics headers and SSE metrics event included. ``--middleware`` entries are
    added after the server is built (serve/middleware/user_middleware.py), so they end up
    outside all of them.

    Args:
        app: The app of OpenAIServer or OpenAIDisaggServer.
        collect_perf_metrics: Install PerfMetricsMiddleware (metrics returned to clients or
            written to JSONL).
        expose_perf_metrics: Let PerfMetricsMiddleware return metrics in response headers and
            in the SSE metrics event.
        perf_metrics_writer: Writer for completed records.
        adjusted_clock: The aggregated server's clock, mapped to the disaggregated server's
            clock domain. The disaggregated server passes None, and then neither middleware
            receives the argument.
        request_trace: The request trace built from ``TRTLLM_REQUEST_TRACE_DIR``; None (the
            variable unset) installs no trace middleware.
    """
    if adjusted_clock is None:
        if collect_perf_metrics:
            app.add_middleware(
                PerfMetricsMiddleware,
                expose_headers=expose_perf_metrics,
                writer=perf_metrics_writer,
            )
        if request_trace is not None:
            app.add_middleware(RequestTraceMiddleware, trace=request_trace)
        app.add_middleware(ServerArrivalTimeMiddleware)
        return
    if collect_perf_metrics:
        app.add_middleware(
            PerfMetricsMiddleware,
            expose_headers=expose_perf_metrics,
            writer=perf_metrics_writer,
            adjusted_clock=adjusted_clock,
        )
    if request_trace is not None:
        app.add_middleware(RequestTraceMiddleware, trace=request_trace)
    app.add_middleware(ServerArrivalTimeMiddleware, adjusted_clock=adjusted_clock)
