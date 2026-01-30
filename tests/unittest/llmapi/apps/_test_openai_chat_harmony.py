import json
import os

import openai
import pytest
from utils.llm_data import llm_datasets_root

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)
os.environ['TIKTOKEN_RS_CACHE_DIR'] = os.path.join(llm_datasets_root(),
                                                   'tiktoken_vocab')
os.environ['TIKTOKEN_ENCODINGS_BASE'] = os.path.join(llm_datasets_root(),
                                                     'tiktoken_vocab')


@pytest.fixture(scope="module", ids=["GPT-OSS-20B"])
def model():
    return "gpt_oss/gpt-oss-20b/"


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model: str, num_postprocess_workers: int):
    model_path = get_model_path(model)
    args = ["--num_postprocess_workers", f"{num_postprocess_workers}"]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning(client: openai.AsyncOpenAI, model: str):
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": "Which one is larger as numeric, 9.9 or 9.11?"
        }])
    assert response.choices[0].message.content
    assert response.choices[0].message.reasoning


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning_effort(client: openai.AsyncOpenAI, model: str):
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": "Which one is larger as numeric, 9.9 or 9.11?"
        }],
        reasoning_effort="Medium")
    assert response.choices[0].message.content
    assert response.choices[0].message.reasoning


@pytest.mark.asyncio(loop_scope="module")
async def test_chat(client: openai.AsyncOpenAI, model: str):
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "developer",
            "content": "Respond in Chinese."
        }, {
            "role": "user",
            "content": "Hello!"
        }, {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        }, {
            "role": "user",
            "content": "Tell me a joke."
        }])
    assert response.choices[0].message.content
    assert response.choices[0].message.reasoning


def get_current_weather(location: str, format: str = "celsius") -> dict:
    return {"sunny": True, "temperature": 20 if format == "celsius" else 68}


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_calls(client: openai.AsyncOpenAI, model: str):
    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Gets the current weather in the provided location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description":
                        "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "description": "default: celsius",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            }
        }
    }
    messages = [{"role": "user", "content": "What is the weather like in SF?"}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[tool_get_current_weather],
        extra_body={"top_k": 1},
    )
    message = response.choices[0].message
    print(message)
    assert response.choices[0].finish_reason == "tool_calls"
    assert message.content is None
    assert message.reasoning
    assert message.tool_calls
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "get_current_weather"
    args = json.loads(tool_call.function.arguments)
    answer = get_current_weather(**args)
    messages.extend([{
        "role": "assistant",
        "tool_calls": [tool_call],
        "reasoning": message.reasoning
    }, {
        "role": "tool",
        "content": json.dumps(answer),
        "tool_call_id": tool_call.id
    }])
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={"top_k": 1},
    )
    message = response.choices[0].message
    assert message.content


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming(client: openai.AsyncOpenAI, model: str):
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": "Explain the theory of relativity in brief."
        }],
        stream=True,
    )
    collected_messages = []
    first_iteration = True
    async for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if first_iteration:
                assert delta.role == "assistant", ValueError(
                    "Expected role 'assistant' for first iteration")
                collected_messages.append(delta)
                first_iteration = False
            else:
                assert delta.role is None, ValueError(
                    "Expected no role except for first iteration")
                collected_messages.append(delta)
        else:
            assert hasattr(chunk, "usage"), ValueError(
                "Expected usage info in last streaming response")
            assert chunk.usage.prompt_tokens is not None
            assert chunk.usage.completion_tokens is not None
            assert chunk.usage.total_tokens is not None
            assert chunk.usage.prompt_tokens > 0
            assert chunk.usage.completion_tokens > 0
            assert chunk.usage.total_tokens > 0

    full_response = "".join([
        m.content for m in collected_messages
        if hasattr(m, "content") and m.content
    ])
    full_reasoning_response = "".join([
        m.reasoning for m in collected_messages
        if hasattr(m, "reasoning") and m.reasoning
    ])
    assert full_response
    assert full_reasoning_response


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_tool_call(client: openai.AsyncOpenAI, model: str):
    tool_get_current_weather = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Gets the current weather in the provided location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description":
                        "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "description": "default: celsius",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            }
        }
    }
    messages = [{"role": "user", "content": "What is the weather like in SF?"}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[tool_get_current_weather],
        stream=True,
        extra_body={"top_k": 1},
    )
    tool_name: str
    reasoning_chunks: list[str] = []
    tool_arg_chunks: list[str] = []
    async for chunk in response:
        # Last streaming response will only contains usage info
        if len(chunk.choices) <= 0:
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            function = delta.tool_calls[0].function
            if hasattr(function, "name") and function.name:
                tool_name = function.name
            if hasattr(function, "arguments") and function.arguments:
                args_str = function.arguments
                tool_arg_chunks.append(args_str)
        if hasattr(delta, "reasoning") and delta.reasoning:
            reasoning_chunks.append(delta.reasoning)
    reasoning = "".join(reasoning_chunks)
    tool_args = "".join(tool_arg_chunks)
    assert tool_name == "get_current_weather"
    assert tool_args
    assert reasoning
    args = json.loads(tool_args)
    get_current_weather(**args)
