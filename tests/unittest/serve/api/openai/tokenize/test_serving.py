# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /_internal/tokenize`` through ``TokenizeServing`` and its router.

Pins what the route does today on the LLM roles (default, CONTEXT and
GENERATION) built by the real composition root, and that the other roles do
not serve it.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.api.openai.tokenize import api_router
from tensorrt_llm.serve.api.openai.tokenize.serving import TokenizeServing

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = ("model", "tokenizer")


def _dependencies():
    return {"model": "fake-model", "tokenizer": SimpleNamespace(encode=lambda text: [1])}


def _llm_server(serve_env, role=None):
    server = serve_env.build_openai_server(AggConfig(role=role))
    return server, serve_env.client(server.app)


def test_constructor_requires_each_dependency():
    dependencies = _dependencies()
    # The control: with every dependency present, construction succeeds.
    TokenizeServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            TokenizeServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"TokenizeServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


@pytest.mark.parametrize(
    "role",
    [None, ServerRole.CONTEXT, ServerRole.GENERATION],
    ids=["llm", "context", "generation"],
)
def test_tokenize_route(role, serve_env):
    server, client = _llm_server(serve_env, role)
    response = client.post("/_internal/tokenize", json={"prompt": "hello"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tokens"] == list(server.tokenizer.encode("hello"))
    assert body["count"] == len(body["tokens"]) == 1


def test_tokenize_validation_error_has_the_servers_error_shape(serve_env):
    _, client = _llm_server(serve_env)
    response = client.post("/_internal/tokenize", json={})
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}  # the validation handler of the composition root
    assert "prompt" in body["error"]


def test_tokenizer_failure_is_400_invalid_request(serve_env, monkeypatch):
    server, client = _llm_server(serve_env)
    serving = getattr(server.app.state, api_router.APP_STATE_KEY)

    def boom(text):
        raise ValueError("cannot tokenize")

    monkeypatch.setattr(serving, "tokenizer", SimpleNamespace(encode=boom))
    response = client.post("/_internal/tokenize", json={"prompt": "hello"})
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "InvalidRequestError"
    assert body["message"] == "cannot tokenize"


@pytest.mark.parametrize(
    "role", [ServerRole.MM_ENCODER, ServerRole.EMBEDDING], ids=["mm_encoder", "embedding"]
)
def test_tokenize_is_not_served_on_other_roles(role, serve_env):
    _, client = _llm_server(serve_env, role)
    assert client.post("/_internal/tokenize", json={"prompt": "hello"}).status_code == 404


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())
