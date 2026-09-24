# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import pytest
import requests

from ..test_llm import get_model_path
from .openai_server import RemoteRerankServer

pytestmark = pytest.mark.threadleak(enabled=False)

MODEL = "Qwen/Qwen3-Reranker-0.6B"
QUERY = "What is the capital of China?"
DOCUMENTS = [
    "Bananas are typically grown in tropical climates.",
    "The capital of China is Beijing.",
    "Gravity attracts objects with mass toward one another.",
]


@pytest.fixture(scope="module")
def server():
    model_path = get_model_path(MODEL)
    args = ["--max_batch_size", "8", "--max_queue_delay", "0.05"]
    with RemoteRerankServer(model_path, args) as remote_server:
        yield remote_server


def _post(server, path, **overrides):
    request = {
        "model": MODEL,
        "query": QUERY,
        "documents": DOCUMENTS,
    }
    request.update(overrides)
    return requests.post(
        server.url_for(*path.split("/")),
        json=request,
        timeout=300,
    )


def test_v1_rerank_aliases_and_top_n(server):
    for path in ("rerank", "v1/rerank"):
        response = _post(server, path, top_n=1, return_documents=True)
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == MODEL
        assert len(body["results"]) == 1
        assert body["results"][0]["index"] == 1
        assert body["results"][0]["document"]["text"] == DOCUMENTS[1]
        assert 0.0 <= body["results"][0]["relevance_score"] <= 1.0
        assert body["usage"]["prompt_tokens"] > 0


def test_v2_wire_shape(server):
    response = _post(server, "v2/rerank", top_n=2, priority=0)
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"id", "results", "meta"}
    assert len(body["results"]) == 2
    assert body["results"][0]["index"] == 1
    assert all("document" not in result for result in body["results"])
    assert body["meta"]["api_version"] == {"version": "2"}


def test_custom_instruction(server):
    response = _post(
        server,
        "v1/rerank",
        instruction="Judge whether the document is about a country's capital",
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == len(DOCUMENTS)


def test_max_tokens_per_doc_reduces_prompt_tokens(server):
    document = "Beijing is the capital of China. " * 64
    uncapped_response = _post(server, "v1/rerank", documents=[document])
    capped_response = _post(
        server,
        "v1/rerank",
        documents=[document],
        max_tokens_per_doc=8,
    )

    assert uncapped_response.status_code == 200
    assert capped_response.status_code == 200
    uncapped_body = uncapped_response.json()
    capped_body = capped_response.json()
    assert len(uncapped_body["results"]) == 1
    assert len(capped_body["results"]) == 1
    assert capped_body["usage"]["prompt_tokens"] < uncapped_body["usage"]["prompt_tokens"]


def test_v2_rejects_unsupported_priority(server):
    response = _post(server, "v2/rerank", priority=1)
    assert response.status_code == 400


def test_empty_documents_is_validation_error(server):
    response = _post(server, "v1/rerank", documents=[])
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_concurrent_requests(server):
    async def request(index):
        return await asyncio.to_thread(
            _post,
            server,
            "v1/rerank",
            query=f"{QUERY} request {index}",
            top_n=1,
        )

    responses = await asyncio.gather(*(request(index) for index in range(8)))
    assert all(response.status_code == 200 for response in responses)
    assert all(len(response.json()["results"]) == 1 for response in responses)
