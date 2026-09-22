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
"""Serve-level tests for the OpenAI-compatible /rerank, /v1/rerank, and the
Cohere-compatible /v2/rerank endpoints.

Launches `trtllm-serve rerank` on a real Qwen3-Reranker checkpoint (the only
family currently supported, see _RERANK_ARCH_MAP in commands/serve.py) and
exercises the endpoint + dynamic batcher end to end. There is no small
non-Qwen3 stand-in model for this feature (unlike /v1/embeddings, which can
reuse an arbitrary BERT classifier for pure plumbing tests) since the yes/no
scoring mechanism is specific to the Qwen3-Reranker prompt/tokenizer, so every
test here loads real weights.

Prompt truncation math itself is covered separately, without a GPU, by
tests/unittest/llmapi/test_rerank_utils.py.
"""

import asyncio

import pytest
import requests
from utils.util import skip_gpu_memory_less_than

from ..test_llm import get_model_path
from .openai_server import RemoteRerankServer

pytestmark = pytest.mark.threadleak(enabled=False)

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "Berlin is the capital of Germany.",
    "Bananas are a good source of potassium.",
]
# Index 0 is the only document that directly answers the query; it should
# score highest under any correctly-behaving reranker.
RELEVANT_INDEX = 0

# The rerank launch path auto-remaps Qwen3ForCausalLM -> Qwen3ForTextReranking
# (see _RERANK_ARCH_MAP in commands/serve.py), so no special flags are needed.
# Parametrized over the family: 0.6B (small/fast) and 8B (the large variant
# downstream users actually serve); the 8B param is memory-gated. To add 4B,
# append "Qwen3/Qwen3-Reranker-4B".
RERANK_VARIANTS = [
    pytest.param("Qwen3/Qwen3-Reranker-0.6B", id="0.6b"),
    pytest.param(
        "Qwen3/Qwen3-Reranker-8B",
        id="8b",
        marks=skip_gpu_memory_less_than(32 * 1000 * 1000 * 1000),
    ),
]


@pytest.fixture(scope="module", params=RERANK_VARIANTS)
def rerank_model_id(request):
    return request.param


@pytest.fixture(scope="module")
def server(rerank_model_id):
    model_path = get_model_path(rerank_model_id)
    args = [
        "--max_batch_size",
        "8",
        "--max_queue_delay",
        "0.05",
        "--max_queue_size",
        "64",
    ]
    with RemoteRerankServer(model_path, args) as remote_server:
        yield remote_server


def _rerank(server: RemoteRerankServer, path_parts, **json_kwargs):
    return requests.post(server.url_for(*path_parts), json=json_kwargs)


def test_single_document(server: RemoteRerankServer, rerank_model_id):
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=[DOCUMENTS[0]])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["index"] == 0
    assert isinstance(body["results"][0]["relevance_score"], float)
    assert body["results"][0]["document"] is None
    assert body["usage"]["total_tokens"] > 0


def test_documents_sorted_by_relevance(server: RemoteRerankServer,
                                       rerank_model_id):
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=DOCUMENTS)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == len(DOCUMENTS)
    # Results are sorted by descending relevance_score.
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # The one document that actually answers the query should rank first.
    assert results[0]["index"] == RELEVANT_INDEX


def test_top_n_truncates_results(server: RemoteRerankServer, rerank_model_id):
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=DOCUMENTS,
                   top_n=2)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_return_documents(server: RemoteRerankServer, rerank_model_id):
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=DOCUMENTS,
                   return_documents=True)
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert r["document"]["text"] == DOCUMENTS[r["index"]]


def test_document_object_form_accepted(server: RemoteRerankServer,
                                       rerank_model_id):
    # Cohere accepts documents as either plain strings or {"text": ...}
    # objects; both must be accepted (see RerankRequest.document_text).
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=[{
                       "text": DOCUMENTS[0]
                   }])
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_empty_documents_returns_400(server: RemoteRerankServer,
                                     rerank_model_id):
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=[])
    assert resp.status_code == 400


def test_oversized_document_is_truncated_not_rejected(
        server: RemoteRerankServer, rerank_model_id):
    # Unlike /v1/embeddings (which 400s on an oversized input), rerank
    # truncates an overlong document to max_seq_len and still scores it.
    # Only the instruction+query overflowing the budget is a 400 (see
    # test_rerank_utils.py for the exact truncation math, tested without a
    # GPU).
    huge_document = "France " * 20000
    resp = _rerank(server, ("v1", "rerank"),
                   model=rerank_model_id,
                   query=QUERY,
                   documents=[huge_document])
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_instruction_override(server: RemoteRerankServer, rerank_model_id):
    resp = _rerank(
        server,
        ("v1", "rerank"),
        model=rerank_model_id,
        query="capital of France",
        documents=[DOCUMENTS[0]],
        instruction=
        "Given a question, find the passage that directly answers it",
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("path_parts",
                         [("rerank", ), ("v1", "rerank"), ("v2", "rerank")])
def test_all_three_paths_serve_the_same_shape(server: RemoteRerankServer,
                                              rerank_model_id, path_parts):
    resp = _rerank(server,
                   path_parts,
                   model=rerank_model_id,
                   query=QUERY,
                   documents=[DOCUMENTS[0]])
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


@pytest.mark.asyncio
async def test_concurrent_requests_are_batched(server: RemoteRerankServer,
                                                rerank_model_id):
    # Fire many independent requests concurrently. The server coalesces them
    # via the dynamic batcher; here we assert every request still gets a
    # correct, independent ranking.
    num_requests = 8

    def _post():
        return requests.post(
            server.url_for("v1", "rerank"),
            json={
                "model": rerank_model_id,
                "query": QUERY,
                "documents": [DOCUMENTS[0], DOCUMENTS[2]],
            },
        )

    responses = await asyncio.gather(
        *[asyncio.to_thread(_post) for _ in range(num_requests)])
    assert len(responses) == num_requests
    for resp in responses:
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        assert results[0]["index"] == 0  # the relevant doc still ranks first
