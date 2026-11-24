# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prompt_embeds`` on /v1/completions and as a chat content part (CPU, fake generator)."""

import base64
import io
import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
import torch
from test_common.llm_data import llm_models_root
from test_common.serve.app_factory import AggConfig, ServeTestEnv, isolated_serve_env

from tensorrt_llm.inputs.prompt_embeds import PromptEmbeds
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.prompt_embeds import (
    DISABLED_MESSAGE,
    PROMPT_EMBEDS_MAX_BYTES_ENV,
    ChatPromptEmbedsPart,
    model_hidden_size,
)
from tensorrt_llm.serve.render.chat import render_chat
from tensorrt_llm.serve.render.prompt_embeds import resource_continuation, splice_chat_prompt_embeds
from tensorrt_llm.serve.render.prompt_types import UnsupportedRenderError
from tensorrt_llm.serve.render.resources import RenderResources

# The server's input-processing pool thread outlives a test by design.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Divisible by the default attention head count, so the fake config loads.
HIDDEN = 64


def _b64(tensor: torch.Tensor) -> str:
    """Encode a tensor the way clients send prompt_embeds."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _part(tensor: torch.Tensor) -> Dict[str, Any]:
    """A chat content part carrying ``tensor``."""
    return {"type": "prompt_embeds", "prompt_embeds": {"data": _b64(tensor)}}


@pytest.fixture
def serve_env(tmp_path) -> Iterator[ServeTestEnv]:
    """An isolated serve environment."""
    with isolated_serve_env(tmp_path) as env:
        yield env


def _client(env: ServeTestEnv, enabled: bool = True):
    """Return (server, client) for a fake LLM server."""
    server = env.build_openai_server(AggConfig(enable_prompt_embeds=enabled, hidden_size=HIDDEN))
    return server, env.client(server.app)


def _completion(client, **body: Any):
    """POST /v1/completions."""
    return client.post("/v1/completions", json={"model": "fake-model", "max_tokens": 1, **body})


def _chat(client, messages: List[Dict[str, Any]], **body: Any):
    """POST /v1/chat/completions."""
    return client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "max_tokens": 1, "messages": messages, **body},
    )


def test_completion_rejected_when_disabled(serve_env: ServeTestEnv) -> None:
    """Without --enable_prompt_embeds the server refuses tensor payloads."""
    server, client = _client(serve_env, enabled=False)
    with client:
        response = _completion(client, prompt_embeds=_b64(torch.zeros(3, HIDDEN)))
    assert response.status_code == 400
    assert DISABLED_MESSAGE in response.json()["message"]
    assert server.generator.generate_calls == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_completion_single_payload(serve_env: ServeTestEnv, dtype: torch.dtype) -> None:
    """One payload becomes one whole-prompt embeddings request."""
    rows = torch.randn(3, HIDDEN).to(dtype)
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds=_b64(rows))
    assert response.status_code == 200, response.text
    (call,) = server.generator.generate_calls
    # The server hands generate_async a validated snapshot it built off the loop.
    assert call.inputs["prompt_token_ids"] == [0, 0, 0]
    snapshot = call.inputs["prompt_embeds"]
    assert isinstance(snapshot, PromptEmbeds)
    assert snapshot.embeds.dtype == dtype
    assert torch.equal(snapshot.embeds, rows)
    assert snapshot.positions.tolist() == [0, 1, 2]
    assert response.json()["usage"]["prompt_tokens"] == 3


def test_completion_payloads_then_text_prompts(serve_env: ServeTestEnv) -> None:
    """Embedding prompts are served before text prompts, in order."""
    first, second = torch.randn(2, HIDDEN), torch.randn(4, HIDDEN)
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds=[_b64(first), _b64(second)], prompt="hi")
    assert response.status_code == 200, response.text
    calls = server.generator.generate_calls
    assert len(calls) == 3
    assert torch.equal(calls[0].inputs["prompt_embeds"].embeds, first)
    assert torch.equal(calls[1].inputs["prompt_embeds"].embeds, second)
    assert "prompt_embeds" not in calls[2].inputs
    assert [c["index"] for c in response.json()["choices"]] == [0, 1, 2]


@pytest.mark.parametrize(
    ("tensor", "match"),
    [
        (torch.zeros(3, HIDDEN + 1), "hidden size"),
        (torch.zeros(3, HIDDEN, 1), "shape"),
        (torch.zeros(3, HIDDEN, dtype=torch.int32), "dtype"),
        (torch.zeros(0, HIDDEN), "no rows"),
    ],
)
def test_completion_bad_tensor(serve_env: ServeTestEnv, tensor: torch.Tensor, match: str) -> None:
    """Shape, dtype and width are checked before the request is submitted."""
    server, client = _client(serve_env)
    with client:
        assert model_hidden_size(server.model_config) == HIDDEN
        response = _completion(client, prompt_embeds=_b64(tensor))
    assert response.status_code == 400
    assert match in response.json()["message"]
    assert server.generator.generate_calls == []


def test_completion_malformed_payload(serve_env: ServeTestEnv) -> None:
    """A payload that is not base64 torch.save is a client error."""
    server, client = _client(serve_env)
    with client:
        response = _completion(client, prompt_embeds="not base64!")
    assert response.status_code == 400
    assert server.generator.generate_calls == []


def test_completion_request_byte_limit(serve_env: ServeTestEnv, monkeypatch) -> None:
    """Payloads of one request share the byte limit."""
    monkeypatch.setenv(PROMPT_EMBEDS_MAX_BYTES_ENV, "10000")
    payload = _b64(torch.zeros(16, HIDDEN))  # ~5 KB each
    server, client = _client(serve_env)
    with client:
        assert _completion(client, prompt_embeds=payload).status_code == 200
        response = _completion(client, prompt_embeds=[payload, payload, payload])
    assert response.status_code == 400
    assert "limit" in response.json()["message"]
    assert len(server.generator.generate_calls) == 1


def test_completion_byte_limit_counts_whole_storage(serve_env: ServeTestEnv, monkeypatch) -> None:
    """A small view saved with a large storage is charged for all of it."""
    limit = 1_000_000
    monkeypatch.setenv(PROMPT_EMBEDS_MAX_BYTES_ENV, str(limit))
    # 505,000 stored bytes each: two fit the encoded-length check (which allows
    # for line breaks) but not the storage budget.
    view = torch.zeros(126_250)[: 2 * HIDDEN].view(2, HIDDEN)
    buf = io.BytesIO()
    torch.save(view, buf)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    encoded_limit = 4 * ((limit + 2) // 3)
    assert 2 * len(payload) <= encoded_limit + encoded_limit // 38 + 4
    assert 2 * view.untyped_storage().nbytes() > limit
    server, client = _client(serve_env)
    with client:
        assert _completion(client, prompt_embeds=payload).status_code == 200
        response = _completion(client, prompt_embeds=[payload, payload])
    assert response.status_code == 400
    assert len(server.generator.generate_calls) == 1


def test_completion_line_wrapped_payload_near_limit(serve_env: ServeTestEnv, monkeypatch) -> None:
    """Line breaks in a payload do not count toward the request byte limit."""
    payload = _b64(torch.zeros(16, HIDDEN))
    limit = len(base64.b64decode(payload))
    wrapped = "\r\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    assert len(wrapped) > 4 * ((limit + 2) // 3)
    monkeypatch.setenv(PROMPT_EMBEDS_MAX_BYTES_ENV, str(limit))
    server, client = _client(serve_env)
    with client:
        assert _completion(client, prompt_embeds=wrapped).status_code == 200
    assert len(server.generator.generate_calls) == 1


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({}, "prompt"),
        ({"prompt_embeds": []}, "empty"),
        ({"echo": True}, "echo"),
        ({"truncate_prompt_tokens": 2}, "truncate_prompt_tokens"),
    ],
)
def test_completion_request_validation(
    serve_env: ServeTestEnv, body: Dict[str, Any], match: str
) -> None:
    """Missing prompts and options that need prompt text are rejected."""
    if match in ("echo", "truncate_prompt_tokens"):
        body = {"prompt_embeds": _b64(torch.zeros(2, HIDDEN)), **body}
    server, client = _client(serve_env)
    with client:
        response = _completion(client, **body)
    assert response.status_code == 400
    assert match in response.text
    assert server.generator.generate_calls == []


def test_chat_rejected_when_disabled(serve_env: ServeTestEnv) -> None:
    """Chat parts need --enable_prompt_embeds too."""
    server, client = _client(serve_env, enabled=False)
    with client:
        response = _chat(client, [{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}])
    assert response.status_code == 400
    assert DISABLED_MESSAGE in response.json()["message"]
    assert server.generator.generate_calls == []


def test_chat_part_placement(serve_env: ServeTestEnv) -> None:
    """Each part becomes a run of masked placeholder positions where it renders."""
    first, second = torch.randn(2, HIDDEN), torch.randn(3, HIDDEN)
    server, client = _client(serve_env)
    messages = [
        {"role": "system", "content": [_part(first)]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                _part(second),
                {"type": "text", "text": "ok"},
            ],
        },
    ]
    with client:
        response = _chat(client, messages)
    assert response.status_code == 200, response.text
    (call,) = server.generator.generate_calls
    inputs = call.inputs
    assert set(inputs) == {"prompt_token_ids", "prompt_embeds"}

    def enc(text: str) -> List[int]:
        return server.tokenizer.encode(text, add_special_tokens=False)

    # The fake template renders <|role|>content per message; text parts join with "\n".
    expected_ids = (
        enc("<|system|>") + [0] * 2 + enc("<|user|>hi\n") + [0] * 3 + enc("\nok<|assistant|>")
    )
    expected_mask = [False] * len(enc("<|system|>")) + [True] * 2
    expected_mask += [False] * len(enc("<|user|>hi\n")) + [True] * 3
    expected_mask += [False] * len(enc("\nok<|assistant|>"))
    assert inputs["prompt_token_ids"] == expected_ids
    expected_positions = [i for i, masked in enumerate(expected_mask) if masked]
    assert inputs["prompt_embeds"].positions.tolist() == expected_positions
    assert torch.equal(inputs["prompt_embeds"].embeds, torch.cat([first, second]))
    assert server.generator.preprocess_calls == []


@pytest.mark.parametrize(
    ("extra", "content_extra", "match"),
    [
        ({"prompt_token_ids": [1, 2]}, [], "prompt_token_ids"),
        ({"truncate_prompt_tokens": 4}, [], "truncate_prompt_tokens"),
        ({"echo": True}, [], "echo"),
    ],
)
def test_chat_unsupported_combinations(
    serve_env: ServeTestEnv, extra: Dict[str, Any], content_extra: List[Any], match: str
) -> None:
    """Combinations whose positions the server cannot map are rejected."""
    server, client = _client(serve_env)
    content = [_part(torch.zeros(2, HIDDEN))] + content_extra
    with client:
        response = _chat(client, [{"role": "user", "content": content}], **extra)
    assert response.status_code == 400
    assert match in response.json()["message"]
    assert server.generator.generate_calls == []


def test_chat_part_with_image_rejected(serve_env: ServeTestEnv) -> None:
    """A part beside an image is refused before the image loads (this URL is never fetched)."""
    server = serve_env.build_openai_server(
        AggConfig(model_type="qwen2_vl", enable_prompt_embeds=True, hidden_size=HIDDEN)
    )
    image = {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9/a.png"}}
    content = [_part(torch.zeros(2, HIDDEN)), image]
    with serve_env.client(server.app) as client:
        response = _chat(client, [{"role": "user", "content": content}])
    assert response.status_code == 400
    assert "multimodal" in response.json()["message"]
    assert server.generator.generate_calls == []


def test_chat_part_without_data(serve_env: ServeTestEnv) -> None:
    """A part must carry prompt_embeds.data."""
    server, client = _client(serve_env)
    with client:
        response = _chat(
            client, [{"role": "user", "content": [{"type": "prompt_embeds", "prompt_embeds": {}}]}]
        )
    assert response.status_code == 400
    assert server.generator.generate_calls == []


def test_render_chat_refuses_parts(serve_env: ServeTestEnv) -> None:
    """Callers outside the chat route cannot render prompt_embeds parts."""
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    server, client = _client(serve_env)
    with client:
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}],
        )
        with pytest.raises(UnsupportedRenderError, match="prompt_embeds"):
            render_chat(request, RenderResources.from_server(server))
        # The prompt_token_ids fast path refuses them too, as inference does.
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}],
            prompt_token_ids=[1, 2],
        )
        with pytest.raises(UnsupportedRenderError, match="prompt_embeds"):
            render_chat(request, RenderResources.from_server(server))


def test_parts_need_opt_in() -> None:
    """Parsers other than the chat route's (Responses, governor) refuse the part."""
    messages = [{"role": "user", "content": [_part(torch.zeros(2, HIDDEN))]}]
    with pytest.raises(ValueError, match="only accepted"):
        parse_chat_messages_coroutines(messages, None, model_type_override="llama")


def test_harmony_refuses_parts(serve_env: ServeTestEnv) -> None:
    """Harmony models render messages themselves, so the chat route and renderer refuse parts."""
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    server = serve_env.build_openai_server(
        AggConfig(model_type="gpt_oss", enable_prompt_embeds=True)
    )
    messages = [{"role": "user", "content": [_part(torch.zeros(2, 4))]}]
    with serve_env.client(server.app) as client:
        response = _chat(client, messages)
        assert response.status_code == 400
        assert "harmony" in response.json()["message"]
        with pytest.raises(UnsupportedRenderError, match="prompt_embeds"):
            render_chat(
                ChatCompletionRequest(model="fake-model", messages=messages),
                RenderResources.from_server(server),
            )
    assert server.generator.generate_calls == []


@pytest.mark.parametrize("encoding", ["utf-8", "escaped", "utf-16"])
@pytest.mark.parametrize("enabled", [True, False])
def test_harmony_refuses_parts_in_tool_messages(
    serve_env: ServeTestEnv, enabled: bool, encoding: str
) -> None:
    """A tool message with tool_call_id validates its content lazily; it is still refused.

    ``escaped`` spells the part type with a JSON unicode escape; ``utf-16``
    sends the JSON in UTF-16, which JSON parsing also accepts.
    """
    server = serve_env.build_openai_server(
        AggConfig(model_type="gpt_oss", enable_prompt_embeds=enabled)
    )
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [_part(torch.zeros(2, 4))]},
    ]
    body = json.dumps({"model": "fake-model", "max_tokens": 1, "messages": messages})
    if encoding == "escaped":
        body = body.replace('"prompt_embeds"', '"prompt_\\u0065mbeds"')
        assert '"prompt_embeds"' not in body
    body = body.encode("utf-16" if encoding == "utf-16" else "utf-8")
    with serve_env.client(server.app) as client:
        response = client.post(
            "/v1/chat/completions", content=body, headers={"content-type": "application/json"}
        )
    assert response.status_code == 400
    assert "prompt_embeds" in response.json()["message"]
    assert server.generator.generate_calls == []


def _sentencepiece_tokenizer() -> Any:
    """TinyLlama's SentencePiece tokenizer, or skip when the model is absent."""
    from transformers import AutoTokenizer

    root = llm_models_root()
    path = None if root is None else root / "llama-models-v2" / "TinyLlama-1.1B-Chat-v1.0"
    if path is None or not path.exists():
        pytest.skip("TinyLlama tokenizer not available")
    return AutoTokenizer.from_pretrained(str(path))


@pytest.mark.parametrize("offsets", [True, False])
def test_splice_text_after_part_reads_as_continuation(offsets: bool) -> None:
    """SentencePiece would add a leading-space token to text after a part; it must not.

    Without ``offsets`` the tokenizer offers only ``encode``, as a slow one does.
    """
    hf = _sentencepiece_tokenizer()
    res = SimpleNamespace(
        tokenizer=hf if offsets else SimpleNamespace(is_fast=False, encode=hf.encode)
    )

    def tokenize(text: str, add_special_tokens: bool) -> List[int]:
        return hf.encode(text, add_special_tokens=add_special_tokens)

    part = ChatPromptEmbedsPart(marker="<M>", data="")
    out = splice_chat_prompt_embeds(
        "A<M>\nok</s> fine",
        [part],
        [torch.zeros(1, 4)],
        tokenize,
        False,
        continuation=resource_continuation(res),
    )
    # The ids the text would have had with one token in the part's place.
    in_context = hf.encode("AX\nok</s> fine", add_special_tokens=False)
    head = hf.encode("A", add_special_tokens=False)
    assert out["prompt_token_ids"] == head + [0] + in_context[len(head) + 1 :]
    assert in_context[len(head)] == hf.convert_tokens_to_ids("X")
    # Tokenized on its own, the tail starts with an extra space token.
    assert hf.encode("\nok</s> fine", add_special_tokens=False) != in_context[len(head) + 1 :]


def _identity_tokenize(text: str, add_special_tokens: bool) -> List[int]:
    """One id per character; 99 stands for a BOS token."""
    return ([99] if add_special_tokens else []) + [ord(c) for c in text]


def _bos_eos_tokenize(text: str, add_special_tokens: bool) -> List[int]:
    """One id per character; 99 and 98 stand for BOS and EOS."""
    ids = [ord(c) for c in text]
    return [99] + ids + [98] if add_special_tokens else ids


def test_splice_places_parts_in_rendered_order() -> None:
    """Parts the template renders out of request order take their own rows."""
    parts = [ChatPromptEmbedsPart("<A>", ""), ChatPromptEmbedsPart("<B>", "")]
    rows = [torch.full((1, 4), 1.0), torch.full((2, 4), 2.0)]
    out = splice_chat_prompt_embeds("x<B>y<A>", parts, rows, _identity_tokenize, False)
    assert out["prompt_token_ids"] == [ord("x"), 0, 0, ord("y"), 0]
    assert out["prompt_embeds_mask"] == [False, True, True, False, True]
    assert torch.equal(out["prompt_embeds"], torch.cat([rows[1], rows[0]]))


def test_splice_wraps_whole_prompt_in_special_tokens() -> None:
    """EOS goes after the last segment, not after the first."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    out = splice_chat_prompt_embeds(
        "a<M>b", [part], [torch.zeros(1, 4)], _bos_eos_tokenize, add_special_tokens=True
    )
    assert out["prompt_token_ids"] == [99, ord("a"), 0, ord("b"), 98]
    assert out["prompt_embeds_mask"] == [False, False, True, False, False]


def test_splice_special_tokens_only_on_first_segment() -> None:
    """BOS is added once, before the first segment, even when it is empty."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    out = splice_chat_prompt_embeds(
        "<M>ab", [part], [torch.zeros(2, 4)], _identity_tokenize, add_special_tokens=True
    )
    assert out["prompt_token_ids"] == [99, 0, 0, ord("a"), ord("b")]
    assert out["prompt_embeds_mask"] == [False, True, True, False, False]


def test_splice_refuses_text_without_a_continuation() -> None:
    """Text after a part that no probe can precede is refused, not tokenized on its own."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    with pytest.raises(UnsupportedRenderError, match="continuation"):
        splice_chat_prompt_embeds(
            "a<M>b",
            [part],
            [torch.zeros(1, 4)],
            _identity_tokenize,
            False,
            continuation=lambda text: None,
        )


@pytest.mark.parametrize(
    ("text", "match"),
    [("no marker here", "dropped"), ("<M>x<M>", "repeated")],
)
def test_splice_marker_must_render_once(text: str, match: str) -> None:
    """A template that drops or repeats a part's message is an error."""
    part = ChatPromptEmbedsPart(marker="<M>", data="")
    with pytest.raises(ValueError, match=match):
        splice_chat_prompt_embeds(text, [part], [torch.zeros(1, 4)], _identity_tokenize, False)


def test_splice_rejects_mixed_dtypes() -> None:
    """All parts of one request share a dtype."""
    parts = [ChatPromptEmbedsPart("<A>", ""), ChatPromptEmbedsPart("<B>", "")]
    rows = [torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.float16)]
    with pytest.raises(ValueError, match="dtype"):
        splice_chat_prompt_embeds("<A><B>", parts, rows, _identity_tokenize, False)
