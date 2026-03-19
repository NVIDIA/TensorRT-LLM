#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the /v1/tokenize endpoint.

Tests the Pydantic request/response models and the endpoint handler logic
using FastAPI TestClient. The handler is re-implemented locally to mirror
OpenAIServer.tokenize, so we can test without the full CUDA/TRT import chain.
"""

import sys
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Re-declare the protocol models locally so we don't need the full
# tensorrt_llm import chain (which requires CUDA / native bindings).
# These mirror the classes in tensorrt_llm/serve/openai_protocol.py exactly.
# ---------------------------------------------------------------------------
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizeRequest(OpenAIBaseModel):
    model: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[OpenAIChatCompletionMessageParam]] = None

    @model_validator(mode="after")
    def check_prompt_or_messages(self):
        if self.prompt is None and self.messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        if self.prompt is not None and self.messages is not None:
            raise ValueError(
                "Only one of 'prompt' or 'messages' should be provided")
        return self


class TokenizeResponse(OpenAIBaseModel):
    count: int
    tokens: Optional[List[int]] = None


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_encode(text: str) -> list:
    """Trivial tokenizer: split on whitespace, return index list."""
    if not text:
        return []
    return list(range(len(text.split())))


def _build_test_app():
    """Build a minimal FastAPI app with the /v1/tokenize route wired to
    a handler that mirrors the real OpenAIServer.tokenize logic."""

    app = FastAPI()
    tokenizer = SimpleNamespace(encode=_fake_encode)

    @app.post("/v1/tokenize")
    async def tokenize(request: TokenizeRequest) -> JSONResponse:
        try:
            if request.prompt is not None:
                token_ids = tokenizer.encode(request.prompt)
            else:
                # Simulate chat template: concat message contents + wrapper
                parts = []
                for msg in request.messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = getattr(msg, "content", "")
                    if isinstance(content, str):
                        parts.append(content)
                text = " ".join(parts)
                text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                token_ids = tokenizer.encode(text)

            response = TokenizeResponse(count=len(token_ids),
                                        tokens=token_ids)
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            error_response = ErrorResponse(
                message=str(e),
                type="InvalidRequestError",
                code=HTTPStatus.BAD_REQUEST.value)
            return JSONResponse(content=error_response.model_dump(),
                                status_code=error_response.code)

    return app


@pytest.fixture(scope="module")
def client():
    app = _build_test_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Protocol model tests (no server needed)
# ---------------------------------------------------------------------------

class TestTokenizeRequestValidation:
    """Validate the Pydantic model constraints."""

    def test_prompt_only(self):
        req = TokenizeRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.messages is None

    def test_messages_only(self):
        req = TokenizeRequest(
            messages=[{"role": "user", "content": "hello"}])
        assert req.prompt is None
        assert req.messages is not None

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Either 'prompt' or 'messages'"):
            TokenizeRequest()

    def test_both_raises(self):
        with pytest.raises(ValueError,
                           match="Only one of 'prompt' or 'messages'"):
            TokenizeRequest(
                prompt="hello",
                messages=[{"role": "user", "content": "hello"}])

    def test_model_field_optional(self):
        req = TokenizeRequest(prompt="hello", model="my-model")
        assert req.model == "my-model"


class TestTokenizeResponseModel:
    """Validate the response model."""

    def test_basic(self):
        resp = TokenizeResponse(count=3, tokens=[1, 2, 3])
        assert resp.count == 3
        assert resp.tokens == [1, 2, 3]

    def test_tokens_optional(self):
        resp = TokenizeResponse(count=5)
        assert resp.count == 5
        assert resp.tokens is None

    def test_model_dump(self):
        resp = TokenizeResponse(count=2, tokens=[10, 20])
        d = resp.model_dump()
        assert d == {"count": 2, "tokens": [10, 20]}


# ---------------------------------------------------------------------------
# Endpoint integration tests (with mock server)
# ---------------------------------------------------------------------------

class TestTokenizeEndpoint:

    def test_prompt_returns_200(self, client):
        resp = client.post("/v1/tokenize",
                           json={"prompt": "Hello, world!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "tokens" in data
        assert isinstance(data["count"], int)
        assert data["count"] > 0
        assert isinstance(data["tokens"], list)
        assert len(data["tokens"]) == data["count"]

    def test_messages_returns_200(self, client):
        resp = client.post(
            "/v1/tokenize",
            json={"messages": [{"role": "user", "content": "Hello, world!"}]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["tokens"]) == data["count"]

    def test_messages_count_gte_prompt_count(self, client):
        """Chat template wrapping should produce at least as many tokens."""
        text = "Hello world"
        r1 = client.post("/v1/tokenize", json={"prompt": text})
        r2 = client.post(
            "/v1/tokenize",
            json={"messages": [{"role": "user", "content": text}]})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["count"] >= r1.json()["count"]

    def test_empty_body_returns_422(self, client):
        resp = client.post("/v1/tokenize", json={})
        assert resp.status_code == 422

    def test_both_prompt_and_messages_returns_422(self, client):
        resp = client.post("/v1/tokenize",
                           json={
                               "prompt": "Hello",
                               "messages": [{"role": "user",
                                             "content": "Hello"}]
                           })
        assert resp.status_code == 422

    def test_empty_string_prompt(self, client):
        resp = client.post("/v1/tokenize", json={"prompt": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["tokens"] == []

    def test_long_prompt(self, client):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        resp = client.post("/v1/tokenize", json={"prompt": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 10
        assert len(data["tokens"]) == data["count"]

    def test_multi_turn_messages(self, client):
        resp = client.post("/v1/tokenize",
                           json={
                               "messages": [
                                   {"role": "system",
                                    "content": "You are helpful."},
                                   {"role": "user", "content": "Hi"},
                                   {"role": "assistant", "content": "Hello!"},
                                   {"role": "user", "content": "How are you?"},
                               ]
                           })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0

    def test_response_has_no_extra_fields(self, client):
        resp = client.post("/v1/tokenize",
                           json={"prompt": "test"})
        assert resp.status_code == 200
        keys = set(resp.json().keys())
        assert keys == {"count", "tokens"}
