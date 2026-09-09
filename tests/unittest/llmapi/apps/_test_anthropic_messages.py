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
"""End-to-end tests for the Anthropic Messages API against a real server.

The offline suites (test_anthropic_{adapter,routes,batches}.py) drive the
translation layer with stubs. They cannot see the seam between that layer and
the rest of the serving stack: a chat template that renders differently than
the reframer assumes, a tool parser that never populates message.tool_calls, a
token count that disagrees with what the executor actually prefilled. Those
only appear once a real model is behind the endpoint, which is what this file
covers. A server started without --tool_parser, for instance, returns tool
calls as ordinary text, and only a test with a real model behind it can tell.

Requests are raw HTTP: the OpenAI SDK cannot speak this protocol, and pinning
the wire format is half the point of an Anthropic compatibility layer.
"""

import json
from typing import Any, Dict, Iterator, Tuple

import pytest
import requests

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

# An Anthropic request or response body, as it appears on the wire.
JsonDict = Dict[str, Any]

pytestmark = pytest.mark.threadleak(enabled=False)

ANTHROPIC_HEADERS = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    # The server does not authenticate, but real clients always send this and
    # it must not be rejected or echoed back.
    "x-api-key": "not-a-real-key",
}


@pytest.fixture(scope="module", params=["Qwen3/Qwen3-0.6B"])
def model(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def server(model: str) -> Iterator[RemoteOpenAIServer]:
    model_path = get_model_path(model)
    # tool_parser is required for the tool-use test: without it the worker
    # never populates message.tool_calls, so the adapter has nothing to
    # convert and the model's tool call is returned as literal text.
    args = [
        "--tool_parser",
        "qwen3",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",  # co-existence with other servers in the same CI stage
    ]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def base_url(server: RemoteOpenAIServer) -> str:
    return server.url_root


def post(base_url: str, path: str, body: JsonDict, stream: bool = False) -> requests.Response:
    return requests.post(
        f"{base_url}{path}",
        json=body,
        headers=ANTHROPIC_HEADERS,
        stream=stream,
        timeout=300,
    )


def simple_body(model: str, **overrides: Any) -> JsonDict:
    body = {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    }
    body.update(overrides)
    return body


def sse_events(response: requests.Response) -> Iterator[Tuple[str, JsonDict]]:
    """Yield (event_name, payload) from an Anthropic SSE response."""
    event = None
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None or raw == "":
            continue
        if raw.startswith("event:"):
            event = raw[len("event:") :].strip()
        elif raw.startswith("data:"):
            yield event, json.loads(raw[len("data:") :].strip())


def test_messages_non_streaming(base_url: str, model: str) -> None:
    """The documented response shape, not merely a 200."""
    response = post(base_url, "/v1/messages", simple_body(model))
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == model
    assert body["id"].startswith("msg_")
    # stop_reason is the Anthropic vocabulary, not OpenAI's finish_reason.
    assert body["stop_reason"] in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}

    text = "".join(block["text"] for block in body["content"] if block["type"] == "text")
    assert text.strip(), f"no text content in {body['content']}"

    usage = body["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


def test_messages_streaming_reaches_message_stop(base_url: str, model: str) -> None:
    """Event order is the contract: clients build the message from it.

    A stream that ends without message_stop, or that emits deltas outside a
    content block, leaves a client holding a partial message it believes is
    complete.
    """
    response = post(base_url, "/v1/messages", simple_body(model, stream=True), stream=True)
    assert response.status_code == 200, response.text

    events = list(sse_events(response))
    names = [name for name, _ in events]

    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert "content_block_start" in names
    assert "content_block_stop" in names
    assert "message_delta" in names
    assert "error" not in names

    # Every delta has to fall inside an open block.
    depth = 0
    for name in names:
        if name == "content_block_start":
            depth += 1
        elif name == "content_block_stop":
            depth -= 1
        elif name == "content_block_delta":
            assert depth > 0, f"content_block_delta outside a block: {names}"
        assert depth >= 0, f"content_block_stop without a start: {names}"
    assert depth == 0, f"unclosed content block: {names}"

    text = "".join(
        payload["delta"]["text"]
        for name, payload in events
        if name == "content_block_delta" and payload["delta"]["type"] == "text_delta"
    )
    assert text.strip(), "stream produced no text"


def test_streaming_and_non_streaming_agree_on_shape(base_url: str, model: str) -> None:
    """The two paths are separate code; they must not disagree on the envelope.

    Content differs run to run, so this compares the parts that are structural:
    the message envelope from message_start and the terminal stop_reason.
    """
    body = simple_body(model, max_tokens=32)
    plain = post(base_url, "/v1/messages", dict(body)).json()

    response = post(base_url, "/v1/messages", dict(body, stream=True), stream=True)
    events = list(sse_events(response))

    start = next(payload for name, payload in events if name == "message_start")["message"]
    assert start["type"] == plain["type"]
    assert start["role"] == plain["role"]
    assert start["model"] == plain["model"]

    delta = next(payload for name, payload in events if name == "message_delta")
    assert delta["delta"]["stop_reason"] in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}


def test_count_tokens_accounts_for_the_whole_prompt(base_url: str, model: str) -> None:
    """count_tokens must equal what the executor actually prefills.

    Anthropic's semantic is total prompt tokens, while usage.input_tokens
    excludes tokens served from cache -- so the invariant is the sum, not
    equality with input_tokens. A client that sizes its context against a
    wrong number either overflows the window or compacts when it need not.
    """
    body = simple_body(
        model,
        system="You are a terse assistant. Answer in one word.",
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And of Italy?"},
        ],
    )

    counted = post(base_url, "/v1/messages/count_tokens", dict(body))
    assert counted.status_code == 200, counted.text
    input_tokens = counted.json()["input_tokens"]
    assert input_tokens > 0

    reported = post(base_url, "/v1/messages", dict(body)).json()["usage"]
    assert input_tokens == reported["input_tokens"] + reported.get("cache_read_input_tokens", 0)


def test_count_tokens_grows_with_the_prompt(base_url: str, model: str) -> None:
    """A count that ignores its input would still satisfy the invariant above."""
    short = post(base_url, "/v1/messages/count_tokens", simple_body(model)).json()
    long_body = simple_body(
        model,
        messages=[{"role": "user", "content": "words " * 500}],
    )
    long = post(base_url, "/v1/messages/count_tokens", long_body).json()
    assert long["input_tokens"] > short["input_tokens"] * 10


def test_tool_definitions_and_tool_result_round_trip(base_url: str, model: str) -> None:
    """Both directions of tool translation, without depending on the model.

    Only tool_choice "auto" is supported here, so whether the model actually
    calls the tool is its decision and cannot be asserted on without making CI
    flaky. What is deterministic is the translation itself: tool definitions
    must be accepted on the way in, and a conversation containing an assistant
    tool_use plus a user tool_result -- the exact shape every agent loop sends
    on its next turn -- must be accepted and answered.
    """
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    accepted = post(
        base_url,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": tools,
            "tool_choice": {"type": "auto"},
        },
    )
    assert accepted.status_code == 200, accepted.text

    # The follow-up turn is synthesised rather than taken from the reply above,
    # so this exercises tool_use/tool_result translation on every run instead of
    # only when the model happens to call the tool.
    follow_up = post(
        base_url,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_test_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_test_1",
                            "content": "18 degrees and sunny",
                        }
                    ],
                },
            ],
            "tools": tools,
        },
    )
    assert follow_up.status_code == 200, follow_up.text
    result = follow_up.json()
    assert result["type"] == "message"
    assert result["role"] == "assistant"


def test_a_real_tool_call_is_parsed_not_leaked_as_text(base_url: str, model: str) -> None:
    """When the model does call a tool, the call must arrive as a tool_use block.

    This is the failure a missing --tool_parser produces: the worker leaves
    message.tool_calls empty, the adapter has nothing to convert, and the call
    is returned as ordinary text with stop_reason end_turn -- the client reads a
    finished answer and never runs the tool.

    Whether a small model decides to call the tool is not something to assert
    on, so this skips rather than fails when it does not. The check is on the
    shape when it does, and on the absence of leaked call syntax either way.
    """
    response = post(
        base_url,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": "Use the get_weather tool to look up Paris. "
                    "Call the tool; do not answer from memory.",
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given city.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()

    text = "".join(b["text"] for b in result["content"] if b["type"] == "text")
    # Raw call syntax in the text is the leak this test exists to catch, and it
    # is worth failing on whether or not a tool_use block was also produced.
    for marker in ("<tool_call>", '"tool_calls"', "<|tool_call"):
        assert marker not in text, f"unparsed tool call leaked as text: {text[:400]}"

    tool_uses = [b for b in result["content"] if b["type"] == "tool_use"]
    if not tool_uses:
        pytest.skip("model chose not to call the tool; nothing to assert on")

    assert result["stop_reason"] == "tool_use"
    call = tool_uses[0]
    assert call["name"] == "get_weather"
    assert call["id"]
    # A decoded object, not the JSON string the OpenAI protocol carries:
    # clients index into this directly.
    assert isinstance(call["input"], dict)


def test_invalid_request_returns_an_anthropic_error(base_url: str, model: str) -> None:
    """Clients parse the error envelope; a FastAPI default would be unreadable."""
    response = post(base_url, "/v1/messages", {"model": model, "max_tokens": 16})

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_batch_round_trip(base_url: str, model: str) -> None:
    """Create, poll to ended, and read results as newline-delimited JSON."""
    create = post(
        base_url,
        "/v1/messages/batches",
        {
            "requests": [
                {"custom_id": f"req-{i}", "params": simple_body(model, max_tokens=16)}
                for i in range(3)
            ]
        },
    )
    assert create.status_code == 200, create.text
    batch = create.json()
    assert batch["processing_status"] in {"in_progress", "ended"}

    batch_id = batch["id"]
    for _ in range(120):
        status = requests.get(
            f"{base_url}/v1/messages/batches/{batch_id}",
            headers=ANTHROPIC_HEADERS,
            timeout=60,
        ).json()
        if status["processing_status"] == "ended":
            break
        import time

        time.sleep(1)
    else:
        pytest.fail(f"batch {batch_id} did not end in time")

    counts = status["request_counts"]
    assert (
        counts["succeeded"]
        + counts["errored"]
        + counts["canceled"]
        + counts["expired"]
        + counts["processing"]
        == 3
    )

    results = requests.get(
        f"{base_url}/v1/messages/batches/{batch_id}/results",
        headers=ANTHROPIC_HEADERS,
        timeout=60,
    )
    assert results.status_code == 200, results.text
    # Each line must parse on its own: clients stream this file line by line.
    lines = [line for line in results.text.splitlines() if line.strip()]
    assert len(lines) == 3
    seen = {json.loads(line)["custom_id"] for line in lines}
    assert seen == {"req-0", "req-1", "req-2"}


def test_tool_choice_any_is_rejected_with_a_clear_reason(base_url: str, model: str) -> None:
    """The limitation is deliberate, so it must fail loudly and explain itself.

    A client that sends tool_choice "any" has to learn that the server will not
    honour it. Silently downgrading to "auto" would let the model answer in
    prose when the caller required a tool call.
    """
    response = post(
        base_url,
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "any"},
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "any" in body["error"]["message"]
