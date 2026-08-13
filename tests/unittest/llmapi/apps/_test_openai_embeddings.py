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
"""Serve-level tests for the OpenAI-compatible /v1/embeddings endpoint.

Launches `trtllm-serve embeddings` on a small encoder-only model
(bert-base-uncased-yelp-polarity, a 2-class sequence classifier) and exercises
the endpoint + dynamic batcher end to end. The model is a classifier, so the
returned "embedding" is its [num_labels]=2 logit vector (the endpoint is
model-output-agnostic).
"""

import array
import asyncio
import base64

import openai
import pytest
import requests
from utils.util import skip_gpu_memory_less_than

from ..test_llm import get_model_path
from .openai_server import RemoteEmbeddingServer

pytestmark = pytest.mark.threadleak(enabled=False)

BERT_MODEL = "bert/bert-base-uncased-yelp-polarity"
NUM_CLASSES = 2  # bert-base-uncased-yelp-polarity has 2 output classes
PROMPTS = ["Hello, world!", "TensorRT-LLM serves embeddings.", "foo bar baz"]

# A per-token reward model on a decoder (Qwen2) backbone. Unlike the BERT
# classifier (which pools to a fixed [num_labels] vector), this emits a
# per-token [seq_len, NUM_REWARD_LABELS] tensor, so the flattened embedding
# length is seq_len * NUM_REWARD_LABELS and varies per input. It exercises the
# model-output-agnostic serialization path on a different architecture + a
# variable-length output shape that BERT alone does not cover.
PRM_MODEL = "Qwen2.5-Math-PRM-7B"
NUM_REWARD_LABELS = 2  # Qwen2.5-Math-PRM-7B emits 2 reward logits per token


@pytest.fixture(scope="module")
def server():
    model_path = get_model_path(BERT_MODEL)
    args = ["--max_batch_size", "8", "--max_queue_delay", "0.05", "--max_queue_size", "64"]
    with RemoteEmbeddingServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteEmbeddingServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteEmbeddingServer):
    return server.get_async_client()


def test_single_string(client: openai.OpenAI):
    response = client.embeddings.create(model=BERT_MODEL, input="Hello, world!")
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    assert len(response.data[0].embedding) == NUM_CLASSES
    assert all(isinstance(v, float) for v in response.data[0].embedding)
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_batch_of_strings(client: openai.OpenAI):
    response = client.embeddings.create(model=BERT_MODEL, input=PROMPTS)
    assert len(response.data) == len(PROMPTS)
    # Results are returned in input order, indexed 0..N-1.
    assert [d.index for d in response.data] == list(range(len(PROMPTS)))
    for d in response.data:
        assert len(d.embedding) == NUM_CLASSES
    assert response.usage.prompt_tokens > 0


def test_pretokenized_input(client: openai.OpenAI):
    # A pre-tokenized single input (list[int]) must be accepted as-is.
    token_ids = [101, 7592, 2088, 102]  # [CLS] hello world [SEP]
    response = client.embeddings.create(model=BERT_MODEL, input=token_ids)
    assert len(response.data) == 1
    assert len(response.data[0].embedding) == NUM_CLASSES


def test_base64_encoding_format(server: RemoteEmbeddingServer):
    # Use raw requests to assert the wire format precisely.
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": "Hello, world!", "encoding_format": "base64"},
    )
    assert resp.status_code == 200
    body = resp.json()
    encoded = body["data"][0]["embedding"]
    assert isinstance(encoded, str)
    decoded = array.array("f", base64.b64decode(encoded)).tolist()
    assert len(decoded) == NUM_CLASSES


def test_float_matches_base64(server: RemoteEmbeddingServer):
    url = server.url_for("v1", "embeddings")
    float_resp = requests.post(
        url, json={"model": BERT_MODEL, "input": "consistency check", "encoding_format": "float"}
    ).json()
    b64_resp = requests.post(
        url, json={"model": BERT_MODEL, "input": "consistency check", "encoding_format": "base64"}
    ).json()
    float_vec = float_resp["data"][0]["embedding"]
    b64_vec = array.array("f", base64.b64decode(b64_resp["data"][0]["embedding"])).tolist()
    assert len(float_vec) == len(b64_vec) == NUM_CLASSES
    for a, b in zip(float_vec, b64_vec):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-5)


def test_dimensions_rejected_for_non_matryoshka_model(server: RemoteEmbeddingServer):
    # `dimensions` is a Matryoshka-embedding knob; for a classifier/reward model
    # (raw [num_labels] logits, not a pooled embedding) it is meaningless, so the
    # server rejects it with 400 — matching vLLM's behavior for non-Matryoshka
    # models, rather than silently slicing the logit vector.
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": "truncate me", "dimensions": 1},
    )
    assert resp.status_code == 400


def test_dimensions_must_be_positive(server: RemoteEmbeddingServer):
    # `dimensions` is a Pydantic PositiveInt, so a non-positive value fails
    # request-model validation. The server's RequestValidationError handler
    # renders that as a 400 (consistent `{"error": ...}` envelope) rather than
    # FastAPI's default 422 — see openai_server.validation_exception_handler.
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": "x", "dimensions": 0},
    )
    assert resp.status_code == 400


def test_empty_input_returns_400(server: RemoteEmbeddingServer):
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": []},
    )
    assert resp.status_code == 400


def test_oversized_input_returns_400(server: RemoteEmbeddingServer):
    # A pre-tokenized input far longer than any encoder's max_seq_len must be
    # rejected with 400 (InputTooLongError) rather than crashing the forward.
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": list(range(100000))},
    )
    assert resp.status_code == 400


def test_usage_has_no_completion_tokens(server: RemoteEmbeddingServer):
    # OpenAI's embeddings usage object carries only prompt_tokens/total_tokens.
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": "count my tokens"},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"]
    assert "completion_tokens" not in usage


@pytest.mark.asyncio
async def test_concurrent_requests_are_batched(async_client: openai.AsyncOpenAI):
    # Fire many independent requests concurrently. The server coalesces them via
    # the dynamic batcher; here we assert every request still gets a correct,
    # independently-indexed response.
    num_requests = 16
    responses = await asyncio.gather(
        *[
            async_client.embeddings.create(model=BERT_MODEL, input=f"concurrent request {i}")
            for i in range(num_requests)
        ]
    )
    assert len(responses) == num_requests
    for response in responses:
        assert len(response.data) == 1
        assert response.data[0].index == 0
        assert len(response.data[0].embedding) == NUM_CLASSES


# --------------------------------------------------------------------------- #
# Per-token reward model (decoder backbone) — different arch + output shape.
# A separate, opt-in server because it loads a 7B checkpoint; selected only
# when these tests run. Assertions are model-output-agnostic: the flattened
# embedding is a non-empty float vector whose length is a multiple of
# NUM_REWARD_LABELS (seq_len * NUM_REWARD_LABELS), not a fixed size.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def prm_server():
    model_path = get_model_path(PRM_MODEL)
    args = [
        "--trust_remote_code",
        "--max_batch_size",
        "4",
        "--max_queue_delay",
        "0.05",
        "--max_queue_size",
        "64",
        "--max_num_tokens",
        "8192",
    ]
    with RemoteEmbeddingServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def prm_client(prm_server: RemoteEmbeddingServer):
    return prm_server.get_client()


@pytest.fixture(scope="module")
def prm_async_client(prm_server: RemoteEmbeddingServer):
    return prm_server.get_async_client()


def _assert_reward_embedding(embedding):
    """A per-token reward vector: non-empty floats, length a multiple of 2."""
    assert len(embedding) > 0
    assert len(embedding) % NUM_REWARD_LABELS == 0
    assert all(isinstance(v, float) for v in embedding)


def test_prm_single_string(prm_client: openai.OpenAI):
    response = prm_client.embeddings.create(model=PRM_MODEL, input="Hello, world!")
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    _assert_reward_embedding(response.data[0].embedding)
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_prm_batch_indexing(prm_client: openai.OpenAI):
    response = prm_client.embeddings.create(model=PRM_MODEL, input=PROMPTS)
    assert len(response.data) == len(PROMPTS)
    assert [d.index for d in response.data] == list(range(len(PROMPTS)))
    for d in response.data:
        _assert_reward_embedding(d.embedding)
    assert response.usage.prompt_tokens > 0


@pytest.mark.asyncio
async def test_prm_concurrent_requests_are_batched(prm_async_client: openai.AsyncOpenAI):
    # Concurrent independent requests must coalesce in the dynamic batcher yet
    # still return correct, independently-indexed responses.
    num_requests = 8
    responses = await asyncio.gather(
        *[
            prm_async_client.embeddings.create(model=PRM_MODEL, input=f"concurrent request {i}")
            for i in range(num_requests)
        ]
    )
    assert len(responses) == num_requests
    for response in responses:
        assert len(response.data) == 1
        assert response.data[0].index == 0
        _assert_reward_embedding(response.data[0].embedding)


# --------------------------------------------------------------------------- #
# Qwen3-Embedding (decoder backbone, last-token pool + L2 normalize) — a real
# sentence-embedding model. A separate, opt-in server (loads a multi-GB
# checkpoint). The embeddings launch path auto-remaps Qwen3ForCausalLM ->
# Qwen3ForTextEmbedding, so no special flags are needed. Parametrized over the
# whole family: 0.6B (small/fast) and 8B (the large variant downstream users
# actually serve). Each variant is (model id, hidden_size); they must stay
# paired since the L2-norm assertion checks the embedding width. The 8B param
# is memory-gated so it skips on small GPUs. To add 4B, append
# ("Qwen3/Qwen3-Embedding-4B", 2560).
# --------------------------------------------------------------------------- #

QWEN3_EMB_VARIANTS = [
    pytest.param(("Qwen3/Qwen3-Embedding-0.6B", 1024), id="0.6b"),
    pytest.param(
        ("Qwen3/Qwen3-Embedding-8B", 4096),
        id="8b",
        marks=skip_gpu_memory_less_than(32 * 1000 * 1000 * 1000),
    ),
]


def _assert_unit_norm_embedding(embedding, dim):
    """A sentence embedding: non-empty floats, fixed dim, L2 norm ~= 1.0."""
    assert len(embedding) == dim
    assert all(isinstance(v, float) for v in embedding)
    norm = sum(v * v for v in embedding) ** 0.5
    assert abs(norm - 1.0) < 1e-2


@pytest.fixture(scope="module", params=QWEN3_EMB_VARIANTS)
def qwen3_emb_variant(request):
    """(model id, hidden_size) for the Qwen3-Embedding variant under test."""
    return request.param


@pytest.fixture(scope="module")
def qwen3_emb_server(qwen3_emb_variant):
    model_id, _dim = qwen3_emb_variant
    model_path = get_model_path(model_id)
    args = [
        "--max_batch_size",
        "8",
        "--max_queue_delay",
        "0.05",
    ]
    with RemoteEmbeddingServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def qwen3_emb_client(qwen3_emb_server: RemoteEmbeddingServer):
    return qwen3_emb_server.get_client()


@pytest.fixture(scope="module")
def qwen3_emb_async_client(qwen3_emb_server: RemoteEmbeddingServer):
    return qwen3_emb_server.get_async_client()


def test_qwen3_embedding_single(qwen3_emb_client: openai.OpenAI, qwen3_emb_variant):
    model_id, dim = qwen3_emb_variant
    response = qwen3_emb_client.embeddings.create(
        model=model_id, input="What is the capital of France?"
    )
    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    _assert_unit_norm_embedding(response.data[0].embedding, dim)
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens


def test_qwen3_embedding_batch_indexing(qwen3_emb_client: openai.OpenAI, qwen3_emb_variant):
    model_id, dim = qwen3_emb_variant
    response = qwen3_emb_client.embeddings.create(model=model_id, input=PROMPTS)
    assert len(response.data) == len(PROMPTS)
    assert [d.index for d in response.data] == list(range(len(PROMPTS)))
    for d in response.data:
        _assert_unit_norm_embedding(d.embedding, dim)


@pytest.mark.asyncio
async def test_qwen3_embedding_concurrent_are_batched(
    qwen3_emb_async_client: openai.AsyncOpenAI, qwen3_emb_variant
):
    # Concurrent independent requests must coalesce in the dynamic batcher yet
    # still return correct, independently-indexed unit-norm embeddings.
    model_id, dim = qwen3_emb_variant
    num_requests = 8
    responses = await asyncio.gather(
        *[
            qwen3_emb_async_client.embeddings.create(
                model=model_id, input=f"concurrent request {i}"
            )
            for i in range(num_requests)
        ]
    )
    assert len(responses) == num_requests
    for response in responses:
        assert len(response.data) == 1
        assert response.data[0].index == 0
        _assert_unit_norm_embedding(response.data[0].embedding, dim)
