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

import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.serve.openai_protocol import RerankDocument, RerankRequest, RerankV2Request
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.reranker import build_qwen3_rerank_input, format_qwen3_rerank_prompt

_DEFAULT_QWEN3_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class BoundaryMergingTokenizer(CharacterTokenizer):
    _MERGED_SPACE_D = 0x110000

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        token_ids = []
        index = 0
        while index < len(text):
            if text.startswith(" d", index):
                token_ids.append(self._MERGED_SPACE_D)
                index += 2
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids

    def decode(self, token_ids):
        return "".join(
            " d" if token_id == self._MERGED_SPACE_D else chr(token_id) for token_id in token_ids
        )


class FakeBatcher:
    def __init__(self, logits):
        self.logits = iter(logits)
        self.inputs = []

    def validate_input(self, token_ids):
        assert token_ids

    async def submit(self, token_ids):
        self.inputs.append(token_ids)
        return SimpleNamespace(logits=torch.tensor([next(self.logits)]))


def _server(logits, *, max_seq_len=4096, max_num_tokens=4096):
    server = OpenAIServer.__new__(OpenAIServer)
    server.tokenizer = CharacterTokenizer()
    server.encode_batcher = FakeBatcher(logits)
    server._input_proc_executor = None
    engine = SimpleNamespace(max_seq_len=max_seq_len, max_num_tokens=max_num_tokens)
    server.generator = SimpleNamespace(_encoder_executor=SimpleNamespace(model_engine=engine))
    return server


def test_official_prompt_with_default_and_custom_instruction():
    prompt = format_qwen3_rerank_prompt("query", "document")
    assert f"<Instruct>: {_DEFAULT_QWEN3_RERANK_INSTRUCTION}\n" in prompt
    assert "<Query>: query\n<Document>: document" in prompt
    assert prompt.endswith(_QWEN3_RERANK_SUFFIX)

    custom = format_qwen3_rerank_prompt("query", "document", "custom task")
    assert "<Instruct>: custom task\n" in custom
    assert "<Instruct>: \n" in format_qwen3_rerank_prompt("query", "document", "")


def test_truncation_preserves_query_and_assistant_suffix():
    tokenizer = CharacterTokenizer()
    empty_length = len(
        tokenizer.encode(format_qwen3_rerank_prompt("query", ""), add_special_tokens=False)
    )
    token_ids = build_qwen3_rerank_input(
        tokenizer, "query", "x" * 200, max_seq_len=empty_length + 8
    )
    text = tokenizer.decode(token_ids)

    assert len(token_ids) == empty_length + 8
    assert "<Query>: query\n" in text
    assert text.endswith(_QWEN3_RERANK_SUFFIX)


def test_max_tokens_per_doc_preserves_suffix():
    tokenizer = CharacterTokenizer()
    empty_length = len(
        tokenizer.encode(format_qwen3_rerank_prompt("query", ""), add_special_tokens=False)
    )
    token_ids = build_qwen3_rerank_input(
        tokenizer, "query", "abcdefghij", max_seq_len=4096, max_tokens_per_doc=3
    )

    assert len(token_ids) == empty_length + 3
    assert tokenizer.decode(token_ids).endswith(_QWEN3_RERANK_SUFFIX)


def test_document_limit_uses_tokens_from_the_complete_prompt():
    tokenizer = BoundaryMergingTokenizer()
    token_ids = build_qwen3_rerank_input(
        tokenizer,
        "query",
        "document",
        max_seq_len=4096,
        max_tokens_per_doc=1,
    )

    text = tokenizer.decode(token_ids)
    assert "<Document>: d" in text
    assert "<Document>: do" not in text
    assert text.endswith(_QWEN3_RERANK_SUFFIX)


def test_invalid_limits_and_oversize_query_are_rejected():
    tokenizer = CharacterTokenizer()

    with pytest.raises(ValueError, match="max_seq_len must be greater"):
        build_qwen3_rerank_input(tokenizer, "query", "document", max_seq_len=0)
    with pytest.raises(ValueError, match="max_tokens_per_doc must be greater"):
        build_qwen3_rerank_input(
            tokenizer,
            "query",
            "document",
            max_seq_len=4096,
            max_tokens_per_doc=0,
        )
    with pytest.raises(ValueError, match="instruction and query exceed"):
        build_qwen3_rerank_input(tokenizer, "x" * 1000, "document", max_seq_len=100)


def test_request_schema_limit_validation():
    assert RerankRequest(query="query", documents=["document"], top_n=0).top_n == 0
    with pytest.raises(ValueError):
        RerankRequest(query="query", documents=["document"], top_n=-1)
    with pytest.raises(ValueError):
        RerankV2Request(
            model="reranker",
            query="query",
            documents=["document"],
            top_n=0,
        )
    with pytest.raises(ValueError):
        RerankV2Request(
            model="reranker",
            query="query",
            documents=["document"],
            priority=1000,
        )


@pytest.mark.asyncio
async def test_v1_rerank_sorts_top_n_and_returns_documents():
    server = _server([-2.0, 3.0, 0.0])
    request = RerankRequest(
        model="reranker",
        query="query",
        documents=["low", RerankDocument(text="high"), "middle"],
        top_n=2,
        return_documents=True,
    )
    response = await server._rerank(request)
    body = json.loads(response.body)

    assert [result["index"] for result in body["results"]] == [1, 2]
    assert [result["document"]["text"] for result in body["results"]] == ["high", "middle"]
    assert all(0.0 <= result["relevance_score"] <= 1.0 for result in body["results"])
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"]


@pytest.mark.asyncio
async def test_v1_rerank_forwards_instruction():
    server = _server([0.0])
    instruction = "custom task"
    request = RerankRequest(
        query="query",
        documents=["document"],
        instruction=instruction,
    )

    response = await server._rerank(request)
    prompt = server.tokenizer.decode(server.encode_batcher.inputs[0])

    assert response.status_code == 200
    assert f"<Instruct>: {instruction}\n" in prompt


@pytest.mark.asyncio
async def test_v1_rerank_truncates_to_encoder_token_budget():
    server = _server([0.0], max_seq_len=4096, max_num_tokens=512)
    request = RerankRequest(
        query="query",
        documents=["x" * 500],
    )

    response = await server._rerank(request)

    assert response.status_code == 200
    assert len(server.encode_batcher.inputs) == 1
    assert len(server.encode_batcher.inputs[0]) <= 512


@pytest.mark.asyncio
async def test_v1_top_n_zero_returns_all_results():
    server = _server([-1.0, 1.0])
    request = RerankRequest(
        model="reranker",
        query="query",
        documents=["low", "high"],
        top_n=0,
    )
    response = await server._rerank(request)
    body = json.loads(response.body)

    assert response.status_code == 200
    assert [result["index"] for result in body["results"]] == [1, 0]


@pytest.mark.asyncio
async def test_v2_wire_shape_omits_documents():
    server = _server([1.0, -1.0])
    request = RerankV2Request(model="reranker", query="query", documents=["first", "second"])
    response = await server._rerank(request)
    body = json.loads(response.body)

    assert set(body) == {"id", "results", "meta"}
    assert body["meta"]["api_version"] == {"version": "2"}
    assert [result["index"] for result in body["results"]] == [0, 1]
    assert all("document" not in result for result in body["results"])


@pytest.mark.asyncio
async def test_v2_rejects_nonzero_priority():
    server = _server([0.0])
    request = RerankV2Request(model="reranker", query="query", documents=["document"], priority=1)
    response = await server._rerank(request)

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_model_output_is_internal_error():
    server = _server([[1.0, 2.0]])
    request = RerankRequest(query="query", documents=["document"])
    response = await server._rerank(request)

    assert response.status_code == 500
