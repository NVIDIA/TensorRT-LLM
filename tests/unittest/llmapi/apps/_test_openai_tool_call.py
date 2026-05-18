import json
import os
import sys

import openai
import pytest

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type":
                        "string",
                        "description":
                        'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type":
                        "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description":
                        'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
]


def get_current_temperature(location: str, unit: str = "celsius") -> dict:
    return {"temperature": 20 if unit == "celsius" else 68}


@pytest.fixture(scope="module", ids=["Qwen3-0.6B"])
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    args = ["--tool_parser", "qwen3"]
    with RemoteOpenAIServer(model_path, cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_parser(client: openai.AsyncOpenAI, model_name: str):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "What's the temperature in San Francisco now?"
        }],
        tools=TOOLS)
    assert response.choices[0].finish_reason == "tool_calls"
    message = response.choices[0].message
    assert message.content is not None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "get_current_temperature"
    args = json.loads(tool_call.function.arguments)
    get_current_temperature(**args)


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_parser_streaming(client: openai.AsyncOpenAI,
                                     model_name: str):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "What's the temperature in San Francisco now?"
        }],
        tools=TOOLS,
        stream=True)
    tool_id = None
    tool_name = None
    parameters = ""
    finish_reason = None

    async for chunk in response:
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            if tool_call.id:
                if tool_id is not None:
                    raise RuntimeError("tool_id already exists")
                tool_id = tool_call.id
            if tool_call.function.name:
                if tool_name is not None:
                    raise RuntimeError("tool_name already exists")
                tool_name = tool_call.function.name
            if tool_call.function.arguments:
                parameters += tool_call.function.arguments
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
    assert tool_id is not None
    assert tool_name == "get_current_temperature"
    assert finish_reason == "tool_calls"
    assert parameters
    args = json.loads(parameters)
    get_current_temperature(**args)


# --------------------------------------------------------------------------
# Named tool_choice (forced function call) — TRTLLM-12758
# --------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_named_tool_choice_forces_function_call(
        client: openai.AsyncOpenAI, model_name: str):
    """OpenAI/Azure spec: tool_choice={"type":"function","function":{"name":"X"}}
    must force the model to emit a tool_call to function X.
    """
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            # Prompt that on its own would not normally produce a tool call.
            "content": "Just say hi.",
        }],
        tools=TOOLS,
        tool_choice={
            "type": "function",
            "function": {
                "name": "get_current_temperature"
            },
        },
    )
    message = response.choices[0].message
    assert message.tool_calls, (f"Expected forced tool_calls, got: {message}")
    assert len(message.tool_calls) == 1
    forced_call = message.tool_calls[0]
    assert forced_call.function.name == "get_current_temperature"
    # Arguments must be valid JSON matching the schema (location is required).
    args = json.loads(forced_call.function.arguments)
    assert "location" in args


@pytest.mark.asyncio(loop_scope="module")
async def test_named_tool_choice_missing_function_returns_400(
        client: openai.AsyncOpenAI, model_name: str):
    """Forcing a function name that is not present in `tools` must fail with
    HTTP 400 and an error message that identifies the missing function."""
    with pytest.raises(openai.BadRequestError) as exc:
        await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": "Hello"
            }],
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "function_that_does_not_exist"
                },
            },
        )
    assert exc.value.status_code == 400
    assert "function_that_does_not_exist" in str(exc.value)


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_choice_auto_unchanged(client: openai.AsyncOpenAI,
                                          model_name: str):
    """Regression: tool_choice='auto' continues to work unchanged."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "What's the temperature in San Francisco now?",
        }],
        tools=TOOLS,
        tool_choice="auto",
    )
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_choice_none_unchanged(client: openai.AsyncOpenAI,
                                          model_name: str):
    """Regression: tool_choice='none' continues to work unchanged (no tool calls)."""
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{
            "role":
            "user",
            "content":
            "Reply with the single word 'hello' and nothing else.",
        }],
        tools=TOOLS,
        tool_choice="none",
    )
    # finish_reason should not be "tool_calls" when explicitly disabled.
    assert response.choices[0].finish_reason != "tool_calls"
