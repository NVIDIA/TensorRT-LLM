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
"""Every OpenAIServer route must have a multi-frontend owner.

A route either is forwarded to the launcher (state that exists once per server,
``FORWARDED_ROUTES``) or is safe to answer from whichever frontend accepted the
connection (``PER_FRONTEND_SAFE`` below, with the reason). A new route in
neither set fails here, so its author has to decide.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.routing import Match

from tensorrt_llm.serve.multi_frontend import FORWARDED_ROUTES
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

PER_FRONTEND_SAFE = {
    ("GET", "/health"): "liveness of this process; engine death reaches it via the watchdog",
    ("GET", "/health_generate"): "a real request through the shared executor",
    ("GET", "/version"): "constant",
    ("GET", "/v1/models"): "constant",
    ("GET", "/server_info"): "static configuration",
    ("POST", "/v1/completions"): "request path",
    ("POST", "/v1/chat/completions"): "request path",
    ("POST", "/v1/messages"): "request path",
    ("POST", "/v1/messages/count_tokens"): "tokenizer only",
    ("POST", "/v1/responses"): "request path; the stateful store is off with several frontends",
    ("GET", "/v1/responses/{response_id}"): "store is off with several frontends",
    ("DELETE", "/v1/responses/{response_id}"): "store is off with several frontends",
    ("POST", "/_internal/tokenize"): "tokenizer only",
    ("GET", "/energy_metrics"): "the same GPUs from every process",
    ("GET", "/v1/data_transceiver_state"): "read from the shared executor",
    ("GET", "/steady_clock_offset"): "every frontend reads the same host clock",
    # RL control needs AsyncLLM and runtime control needs one frontend.
    ("POST", "/release_memory"): "never served by attached frontends",
    ("POST", "/resume_memory"): "never served by attached frontends",
    ("POST", "/update_weights"): "never served by attached frontends",
    ("GET", "/memory_status"): "never served by attached frontends",
    ("GET", "/openapi.json"): "generated, identical everywhere",
    ("GET", "/docs"): "static page",
    ("GET", "/docs/oauth2-redirect"): "static page",
    ("GET", "/redoc"): "static page",
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
    # Mutually exclusive in practice; both on to enumerate every route.
    server._enable_rl_control_endpoints = True
    server._enable_runtime_control_endpoints = True
    server.resource_governor = None
    server.register_routes()
    return sorted(
        (method, route.path)
        for route in server.app.routes
        if getattr(route, "methods", None)
        for method in route.methods
        if method != "HEAD"
    )


async def _stub():
    return {}


def _forwarding_app(routes=FORWARDED_ROUTES) -> FastAPI:
    # No generated docs routes: they would "match" the server's own.
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    for method, path in routes:
        app.add_api_route(path, _stub, methods=[method])
    return app


def _is_forwarded(app: FastAPI, method: str, path_template: str) -> bool:
    concrete = path_template.replace("{batch_id}", "x").replace("{response_id}", "x")
    scope = {"type": "http", "method": method, "path": concrete, "path_params": {}}
    return any(route.matches(scope)[0] == Match.FULL for route in app.routes)


def test_every_llm_route_is_classified() -> None:
    forwarding = _forwarding_app()
    routes = _llm_server_routes()
    forwarded = {r for r in routes if _is_forwarded(forwarding, *r)}
    assert not forwarded & PER_FRONTEND_SAFE.keys(), "both forwarded and per-frontend safe"
    unclassified = [r for r in routes if r not in forwarded and r not in PER_FRONTEND_SAFE]
    assert not unclassified, (
        f"Routes without a multi-frontend owner: {unclassified}. Add them to "
        "FORWARDED_ROUTES (server-wide state) or to PER_FRONTEND_SAFE with the reason."
    )


def test_no_stale_entries() -> None:
    routes = _llm_server_routes()
    for method, template in FORWARDED_ROUTES:
        app = _forwarding_app([(method, template)])
        assert any(_is_forwarded(app, *r) for r in routes), f"{method} {template} matches nothing"
    assert not PER_FRONTEND_SAFE.keys() - set(routes), "PER_FRONTEND_SAFE lists removed routes"
