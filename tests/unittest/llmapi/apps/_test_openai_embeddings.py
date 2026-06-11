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

from ..test_llm import get_model_path
from .openai_server import RemoteEmbeddingServer

pytestmark = pytest.mark.threadleak(enabled=False)

BERT_MODEL = "bert/bert-base-uncased-yelp-polarity"
NUM_CLASSES = 2  # bert-base-uncased-yelp-polarity has 2 output classes
PROMPTS = ["Hello, world!", "TensorRT-LLM serves embeddings.", "foo bar baz"]


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


def test_dimensions_truncation(server: RemoteEmbeddingServer):
    resp = requests.post(
        server.url_for("v1", "embeddings"),
        json={"model": BERT_MODEL, "input": "truncate me", "dimensions": 1},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"][0]["embedding"]) == 1


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
