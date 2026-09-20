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
"""Route-ownership gate for trtllm-serve multi-frontend serving.

With ``num_serve_frontends > 1`` a client cannot choose which frontend process
answers, so every HTTP route of the LLM ``OpenAIServer`` must be classified:

* **launcher-owned** -- backed by state that exists once per server (engine
  stats queue, profiler, batch store). Attached frontends forward these to the
  launcher; the list is ``tensorrt_llm.serve.multi_frontend.FORWARDED_ROUTES``.
* **per-frontend safe** -- correct (or deliberately per-process) when served by
  whichever frontend accepted the connection; listed below with the reason.

A new route that is in neither set fails this test, forcing the author to
decide instead of silently shipping a per-process copy of server-wide state.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.routing import Match

from tensorrt_llm.serve.multi_frontend import FORWARDED_ROUTES
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

# (method, path template) -> why it is fine to answer from any frontend.
PER_FRONTEND_SAFE = {
    (
        "GET",
        "/health",
    ): "liveness of the answering process; engine death propagates via the watchdog",
    ("GET", "/health_generate"): "runs a real request through the shared executor",
    ("GET", "/version"): "constant",
    ("GET", "/v1/models"): "constant",
    ("GET", "/server_info"): "static server/model configuration, identical in every frontend",
    ("POST", "/v1/completions"): "request path: the whole point of multiple frontends",
    ("POST", "/v1/chat/completions"): "request path",
    ("POST", "/v1/messages"): "request path (Anthropic adapter)",
    ("POST", "/v1/messages/count_tokens"): "tokenizer-only, identical in every frontend",
    (
        "POST",
        "/v1/responses",
    ): "request path; the stateful store is disabled group-wide in multi-frontend mode",
    (
        "GET",
        "/v1/responses/{response_id}",
    ): "always 404 in multi-frontend mode (store disabled group-wide)",
    (
        "DELETE",
        "/v1/responses/{response_id}",
    ): "always 404 in multi-frontend mode (store disabled group-wide)",
    ("POST", "/_internal/tokenize"): "tokenizer-only",
    ("GET", "/energy_metrics"): "reads GPU energy counters; same devices from every process",
    ("GET", "/v1/data_transceiver_state"): "fetched from the shared executor over RPC",
    ("GET", "/steady_clock_offset"): "clock probe of the answering process",
    (
        "POST",
        "/steady_clock_offset",
    ): "known limitation: calibrates the answering frontend only (documented)",
    # RL control requires AsyncLLM, which is not the classic IPC executor path
    # multi-frontend mode runs on; the two never coexist.
    ("POST", "/release_memory"): "AsyncLLM-only; never registered together with attached frontends",
    ("POST", "/resume_memory"): "AsyncLLM-only; never registered together with attached frontends",
    ("POST", "/update_weights"): "AsyncLLM-only; never registered together with attached frontends",
}


def _llm_server_routes() -> list[tuple[str, str]]:
    """(method, path) pairs the LLM server registers, without building an LLM."""
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    server.use_harmony = False
    server._enable_rl_control_endpoints = True
    server.resource_governor = None
    server.register_routes()
    return sorted(
        (method, route.path)
        for route in server.app.routes
        if hasattr(route, "methods") and route.methods
        for method in route.methods
        if method != "HEAD"
    )


def _forwarding_app() -> FastAPI:
    app = FastAPI()

    async def _stub():
        return {}

    for method, path in FORWARDED_ROUTES:
        app.add_api_route(path, _stub, methods=[method])
    return app


def _is_forwarded(app: FastAPI, method: str, path_template: str) -> bool:
    # Concretise the registered template so the forwarding templates
    # (including the {rest:path} catch-all) can be matched against it.
    concrete = path_template
    for name in ("{batch_id}", "{response_id}"):
        concrete = concrete.replace(name, "x")
    scope = {"type": "http", "method": method, "path": concrete, "path_params": {}}
    return any(route.matches(scope)[0] == Match.FULL for route in app.routes)


def test_every_llm_route_is_classified() -> None:
    fwd = _forwarding_app()
    unclassified = []
    double = []
    for method, path in _llm_server_routes():
        forwarded = _is_forwarded(fwd, method, path)
        safe = (method, path) in PER_FRONTEND_SAFE
        if forwarded and safe:
            double.append((method, path))
        elif not forwarded and not safe:
            unclassified.append((method, path))
    assert not double, f"routes both forwarded and per-frontend-safe: {double}"
    assert not unclassified, (
        "New OpenAIServer route(s) without a multi-frontend ownership decision: "
        f"{unclassified}. Either add them to FORWARDED_ROUTES in "
        "tensorrt_llm/serve/multi_frontend.py (state lives once per server, e.g. "
        "an engine queue, the profiler or an in-memory store) or to "
        "PER_FRONTEND_SAFE in this test with the reason they are correct when "
        "served by whichever frontend accepted the connection."
    )


def test_every_forwarded_route_exists() -> None:
    """A renamed route must not leave a dangling forwarding template."""
    registered = _llm_server_routes()
    reg_app = FastAPI()

    async def _stub():
        return {}

    for method, path in registered:
        reg_app.add_api_route(path, _stub, methods=[method])
    for method, template in FORWARDED_ROUTES:
        probe = template.replace("{rest:path}", "x/cancel")
        scope = {"type": "http", "method": method, "path": probe, "path_params": {}}
        assert any(route.matches(scope)[0] == Match.FULL for route in reg_app.routes), (
            f"FORWARDED_ROUTES entry {method} {template} matches no registered route"
        )


def test_allowlist_has_no_stale_entries() -> None:
    registered = set(_llm_server_routes())
    stale = [r for r in PER_FRONTEND_SAFE if r not in registered]
    assert not stale, f"PER_FRONTEND_SAFE lists routes that no longer exist: {stale}"
