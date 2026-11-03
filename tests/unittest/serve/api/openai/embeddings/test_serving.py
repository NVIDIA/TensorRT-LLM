# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /v1/embeddings`` through ``EmbeddingServing`` and its router.

Pins what the embeddings route does today on an EMBEDDING server built by the
real composition root: the request path through the encode batcher, the error
statuses, the msgpack route class, and the batcher reference the health probe
reads.
"""

import array
import base64

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import FakeEncodeOnlyLLM, FakeLlmArgs

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.api.openai.embeddings import api_router
from tensorrt_llm.serve.api.openai.embeddings.batcher import QueueFullError
from tensorrt_llm.serve.api.openai.embeddings.serving import EmbeddingServing
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUIRED = ("model", "generator", "embedding_max_queue_delay", "embedding_max_queue_size")
# "hi" and "hello" are single tokens (5 and 6) of the fake tokenizer.
BODY = {"model": "fake-model", "input": ["hi", "hello"]}


def _dependencies(tmp_path):
    generator = FakeEncodeOnlyLLM(args=FakeLlmArgs(model=str(tmp_path), encode_only=True))
    return {
        "model": "fake-model",
        "generator": generator,
        "embedding_max_queue_delay": 0.001,
        "embedding_max_queue_size": 4,
    }


def _serving(server) -> EmbeddingServing:
    return getattr(server.app.state, api_router.APP_STATE_KEY)


def _embedding_server(serve_env):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.EMBEDDING))
    return server, serve_env.client(server.app)  # the lifespan starts the batcher


def test_constructor_requires_each_dependency(tmp_path):
    dependencies = _dependencies(tmp_path)
    # The control: with every dependency present, construction succeeds.
    EmbeddingServing(**dependencies)
    for name in REQUIRED:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            EmbeddingServing(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"EmbeddingServing must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )
    with pytest.raises(TypeError):
        EmbeddingServing(*dependencies.values())  # keyword-only: no positional construction


def test_batcher_built_from_encode_limits(tmp_path):
    dependencies = _dependencies(tmp_path)
    # The real LLM.encode_limits, read over the fake encoder executor.
    limits = dependencies["generator"].encode_limits
    assert limits is not None
    batcher = EmbeddingServing(**dependencies).embedding_batcher
    assert batcher._max_batch_size == limits.max_batch_size
    assert batcher._max_num_tokens == limits.max_num_tokens
    assert batcher._max_seq_len == limits.max_seq_len
    assert batcher._max_queue_delay == 0.001
    assert batcher._queue.maxsize == 4


def test_embeddings_route_end_to_end(serve_env):
    server, client = _embedding_server(serve_env)
    response = client.post("/v1/embeddings", json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "fake-model"
    assert [item["index"] for item in body["data"]] == [0, 1]
    # The fake encodes each input as [len(token_ids), first token id].
    assert body["data"][0]["embedding"] == [1.0, 5.0]
    assert body["data"][1]["embedding"] == [1.0, 6.0]
    assert body["usage"]["prompt_tokens"] == 2
    assert body["usage"]["total_tokens"] == 2
    # Strings are tokenized through the generator's input processor before encode().
    assert [ids for batch in server.generator.encode_calls for ids in batch] == [[5], [6]]


def test_base64_encoding_format(serve_env):
    _, client = _embedding_server(serve_env)
    response = client.post(
        "/v1/embeddings", json={"model": "fake-model", "input": "hi", "encoding_format": "base64"}
    )
    assert response.status_code == 200, response.text
    packed = base64.b64encode(array.array("f", [1.0, 5.0]).tobytes()).decode("utf-8")
    assert response.json()["data"][0]["embedding"] == packed


def test_embeddings_route_decodes_msgpack(serve_env):
    _, client = _embedding_server(serve_env)
    response = client.post(
        "/v1/embeddings", content=_msgpack_encoder.encode(BODY), headers=MSGPACK_HEADERS
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 2


def test_dimensions_and_empty_input_are_400(serve_env):
    server, client = _embedding_server(serve_env)
    response = client.post(
        "/v1/embeddings", json={"model": "fake-model", "input": "hi", "dimensions": 8}
    )
    assert response.status_code == 400
    assert "Matryoshka" in response.json()["message"]
    response = client.post("/v1/embeddings", json={"model": "fake-model", "input": []})
    assert response.status_code == 400
    assert response.json()["message"] == "`input` must not be empty."
    assert server.generator.encode_calls == []


def test_input_too_long_is_400(serve_env):
    server, client = _embedding_server(serve_env)
    too_long = [5] * (server.generator.encode_limits.max_seq_len + 1)
    response = client.post("/v1/embeddings", json={"model": "fake-model", "input": too_long})
    assert response.status_code == 400
    assert "exceeds max_seq_len" in response.json()["message"]
    assert server.generator.encode_calls == [], "an oversize input is rejected before any encode"


def test_full_queue_is_429(serve_env, monkeypatch):
    server, client = _embedding_server(serve_env)

    async def full(token_ids):
        raise QueueFullError("Embedding request queue is full; retry later.")

    monkeypatch.setattr(_serving(server).embedding_batcher, "submit", full)
    response = client.post("/v1/embeddings", json=BODY)
    assert response.status_code == 429
    assert response.json()["message"] == "Embedding request queue is full; retry later."


def test_health_probe_sees_the_serving_batcher(serve_env, monkeypatch):
    server, client = _embedding_server(serve_env)
    serving = _serving(server)
    message = (
        "the composition root must keep self.embedding_batcher bound to "
        "EmbeddingServing.embedding_batcher, which the /health probe reads"
    )
    assert server.embedding_batcher is serving.embedding_batcher, message
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(serving.embedding_batcher, "is_alive", lambda: False)
    assert client.get("/health").status_code == 503, message


def test_embedding_role_has_no_metrics_route(serve_env):
    _, client = _embedding_server(serve_env)
    assert client.get("/metrics").status_code == 404


def test_attach_router_requires_the_serving_object():
    with pytest.raises(RuntimeError, match=api_router.APP_STATE_KEY):
        api_router.attach_router(FastAPI())
