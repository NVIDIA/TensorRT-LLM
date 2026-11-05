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

import dataclasses
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.routing import Match
from test_common.serve.routes import effective_routes
from test_common.serve.serving_stubs import make_chat_serving

from tensorrt_llm.serve._base.routing import ServerCapabilities, attach_routers
from tensorrt_llm.serve.api.anthropic.serving import (
    AnthropicBatchesServing,
    AnthropicCountTokensServing,
    AnthropicMessagesServing,
)
from tensorrt_llm.serve.api.openai.chat import api_router as chat_router
from tensorrt_llm.serve.api.openai.completions import api_router as completions_router
from tensorrt_llm.serve.api.openai.completions.serving import CompletionServing
from tensorrt_llm.serve.api.openai.tokenize import api_router as tokenize_router
from tensorrt_llm.serve.api.openai.tokenize.serving import TokenizeServing
from tensorrt_llm.serve.api.ops.metrics import api_router as metrics_router
from tensorrt_llm.serve.api.ops.metrics.serving import MetricsServing
from tensorrt_llm.serve.api.ops.profile import api_router as profile_router
from tensorrt_llm.serve.api.ops.profile.serving import ProfileServing
from tensorrt_llm.serve.api.ops.rl import api_router as rl_router
from tensorrt_llm.serve.api.ops.rl.serving import RLServing
from tensorrt_llm.serve.api.ops.runtime_control import api_router as runtime_control_router
from tensorrt_llm.serve.api.ops.runtime_control.serving import RuntimeControlServing
from tensorrt_llm.serve.api.ops.system.serving import SystemServing
from tensorrt_llm.serve.api.ops.worker import api_router as worker_router
from tensorrt_llm.serve.api.ops.worker.serving import WorkerServing
from tensorrt_llm.serve.multi_frontend import FORWARDED_ROUTES
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import select_postproc_strategy

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


def _completion_serving() -> CompletionServing:
    """A completions serving object with placeholder dependencies: only routed to, never called."""
    return CompletionServing(
        model="m",
        generator=None,
        tokenizer=None,
        model_config=None,
        collect_perf_metrics=False,
        metrics_collector=None,
        adjusted_steady_clock=None,
        iteration_stats_wakeup_event=None,
        iteration_stats_collector_task_getter=lambda: None,
        internal_disagg_auth_key=None,
        server_role=None,
        per_request_spec_decode_stats=False,
        spec_decode_num_spec_tokens=None,
        input_proc_executor=None,
        postproc=select_postproc_strategy(postproc_worker_enabled=False),
    )


def _llm_server_routes() -> list[tuple[str, str]]:
    """(method, path) pairs the LLM server registers, without building an LLM."""
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = SimpleNamespace(
        resource_governor_queue=None,
        args=SimpleNamespace(return_perf_metrics=False),
    )
    server.use_harmony = False
    server.resource_governor = None
    # As the composition root does: the serving objects the routers read go on
    # app.state, the role's own table is registered, then the routers of the role
    # (the system router with /health, /version and /v1/models, the worker,
    # metrics, tokenize, completions, chat, Anthropic, profile, runtime-control
    # and RL control routers) are attached.
    # The routes are enumerated, never called, so the serving objects get
    # collaborators that are never used.
    server.app.state.ops_serving_system = SystemServing(model="m", health_probe=lambda: True)
    setattr(
        server.app.state,
        worker_router.APP_STATE_KEY,
        WorkerServing(
            model="m",
            generator=None,
            adjusted_steady_clock=None,
            shared_clock_offset=None,
            chat_serving=None,
            render_resources=None,
        ),
    )
    setattr(
        server.app.state,
        metrics_router.APP_STATE_KEY,
        MetricsServing(
            model="m",
            generator=None,
            is_visual_gen=False,
            metrics_collector=None,
            iteration_stats_wakeup_event=None,
            iteration_stats_poll_interval=None,
        ),
    )
    setattr(
        server.app.state, profile_router.APP_STATE_KEY, ProfileServing(model="m", generator=None)
    )
    setattr(
        server.app.state,
        runtime_control_router.APP_STATE_KEY,
        RuntimeControlServing(
            model="m",
            generator=None,
            runtime_control_api_key=None,
            runtime_control_replay_cache=None,
        ),
    )
    setattr(
        server.app.state,
        rl_router.APP_STATE_KEY,
        RLServing(model="m", generator=None, rl_control_api_key=None),
    )
    setattr(
        server.app.state,
        tokenize_router.APP_STATE_KEY,
        TokenizeServing(model="m", tokenizer=None),
    )
    setattr(server.app.state, completions_router.APP_STATE_KEY, _completion_serving())
    setattr(server.app.state, chat_router.APP_STATE_KEY, make_chat_serving(model="m"))
    server.app.state.anthropic_serving_messages = AnthropicMessagesServing(backend=None)
    server.app.state.anthropic_serving_count_tokens = AnthropicCountTokensServing(
        render_resources=lambda: None
    )
    server.app.state.anthropic_serving_batches = AnthropicBatchesServing(runner=None)
    server.register_routes()
    # Mutually exclusive in practice; both on to enumerate every route.
    attach_routers(
        server.app,
        dataclasses.replace(
            ServerCapabilities.for_role(None), rl_control=True, runtime_control=True
        ),
    )
    # effective_routes walks included routers, which FastAPI 0.139 keeps as
    # objects without .path or .methods in app.routes.
    return sorted(
        (method, path)
        for path, route in effective_routes(server.app)
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
