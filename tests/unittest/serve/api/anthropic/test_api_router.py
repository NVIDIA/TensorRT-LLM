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
"""The Anthropic router: attachment contract, route provenance per role, msgpack bodies."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve.api.anthropic.api_router import attach_router
from tensorrt_llm.serve.api.anthropic.serving import (
    AnthropicBatchesServing,
    AnthropicCountTokensServing,
    AnthropicMessagesServing,
)
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder
from tensorrt_llm.serve.render import RenderResources

pytestmark = pytest.mark.cpu_only

ANTHROPIC_ROUTER_MODULE = "tensorrt_llm.serve.api.anthropic.api_router"
ANTHROPIC_ROUTES = (
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"),
    ("POST", "/v1/messages/batches"),
    ("GET", "/v1/messages/batches"),
    ("GET", "/v1/messages/batches/{batch_id}"),
    ("DELETE", "/v1/messages/batches/{batch_id}"),
    ("POST", "/v1/messages/batches/{batch_id}/cancel"),
    ("GET", "/v1/messages/batches/{batch_id}/results"),
)


def _anthropic_routes(app: FastAPI) -> dict:
    """(method, path) -> [routes] for every Anthropic path the app registers."""
    found = {key: [] for key in ANTHROPIC_ROUTES}
    for path, route in effective_routes(app):
        for method in getattr(route, "methods", None) or ():
            if (method, path) in found:
                found[(method, path)].append(route)
    return found


def test_attach_requires_every_serving_object():
    # The composition root sets the three serving objects before attaching, so
    # a wiring bug fails at startup, not with an AttributeError per request.
    with pytest.raises(RuntimeError, match="anthropic_serving_messages"):
        attach_router(FastAPI())


def test_serving_classes_require_their_collaborators():
    # No defaults: a serving class built without its collaborators fails at
    # construction instead of falling back silently.
    with pytest.raises(TypeError):
        AnthropicMessagesServing()
    with pytest.raises(TypeError):
        AnthropicCountTokensServing()
    with pytest.raises(TypeError):
        AnthropicBatchesServing()


@pytest.mark.parametrize(
    "role", [None, ServerRole.CONTEXT, ServerRole.GENERATION], ids=["llm", "context", "generation"]
)
def test_the_eight_routes_come_from_the_anthropic_router(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))

    for key, routes in _anthropic_routes(server.app).items():
        assert len(routes) == 1, f"{key} must be registered exactly once, found {len(routes)}"
        route = routes[0]
        assert isinstance(route, _MsgspecRoute), (
            f"{key} must be a _MsgspecRoute; a plain APIRoute breaks msgpack request bodies "
            "from the disaggregated frontend"
        )
        assert route.endpoint.__module__ == ANTHROPIC_ROUTER_MODULE


def test_a_harmony_model_has_no_anthropic_routes(serve_env):
    server = serve_env.build_openai_server(AggConfig(model_type="gpt_oss"))
    assert all(not routes for routes in _anthropic_routes(server.app).values())


@pytest.mark.parametrize("role", [ServerRole.MM_ENCODER, ServerRole.EMBEDDING])
def test_other_roles_have_no_anthropic_routes(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    assert all(not routes for routes in _anthropic_routes(server.app).values())


class _WordTokenizer:
    """Renders the conversation as the words of its messages and counts them."""

    def get_chat_template(self, chat_template=None, tools=None):
        return chat_template or "WORDS"

    def apply_chat_template(self, messages=None, **kwargs):
        if messages is None:
            messages = kwargs.pop("conversation")
        return " ".join(str(m.get("content", "")) for m in messages)

    def encode(self, text, add_special_tokens=True, **kwargs):
        return text.split()


# The TestClient portal thread and the fake server's lazy worker-pool threads
# are alive when the leak check runs; the serve_env fixture closes them in its
# teardown (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
@pytest.mark.threadleak(enabled=False)
def test_count_tokens_decodes_the_disaggregated_frontends_msgpack_body(serve_env):
    """The disaggregated frontend forwards count_tokens as msgpack to a context worker."""
    server = serve_env.build_openai_server(AggConfig())
    resources = SimpleNamespace(tokenizer=_WordTokenizer(), chat_template=None)
    server.app.state.anthropic_serving_count_tokens = AnthropicCountTokensServing(
        render_resources=lambda: RenderResources.from_server(resources)
    )
    body = _msgpack_encoder.encode(
        {"model": "m", "messages": [{"role": "user", "content": "one two three"}]}
    )

    response = serve_env.client(server.app).post(
        "/v1/messages/count_tokens", content=body, headers=MSGPACK_HEADERS
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"input_tokens": 3}
