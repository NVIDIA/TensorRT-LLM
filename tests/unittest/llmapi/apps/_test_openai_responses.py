import json

import openai
import pytest
from openai.types.responses import (ResponseCompletedEvent,
                                    ResponseReasoningTextDeltaEvent,
                                    ResponseTextDeltaEvent)

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["GPT-OSS-20B"])
def model():
    return "gpt_oss/gpt-oss-20b/"


@pytest.fixture(scope="module")
def server(model: str):
    model_path = get_model_path(model)
    with RemoteOpenAIServer(model_path) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


def check_reponse(response, prefix=""):
    reasoning_exist, message_exist = False, False
    for output in response.output:
        if output.type == "reasoning":
            reasoning_exist = True
        elif output.type == "message":
            message_exist = True

    assert reasoning_exist, f"{prefix}Reasoning content not exists!"
    assert message_exist, f"{prefix}Message content not exists!"


def check_tool_calling(response, first_resp=True, prefix=""):
    reasoning_exist, tool_call_exist, message_exist = False, False, False
    function_call = None
    for output in response.output:
        if output.type == "reasoning":
            reasoning_exist = True
        elif output.type == "function_call":
            tool_call_exist = True
            function_call = output
        elif output.type == "message":
            message_exist = True

    if first_resp:
        assert reasoning_exist and tool_call_exist, f"{prefix}Invalid tool calling 1st response"
        assert not message_exist, f"{prefix}Invalid tool calling 1st response"

        return function_call
    else:
        assert reasoning_exist and message_exist, f"{prefix}Invalid tool calling 2nd response"
        assert not tool_call_exist, f"{prefix}Invalid tool calling 2nd response"


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(
        model=model, input="Which one is larger as numeric, 9.9 or 9.11?")

    check_reponse(response, "test_reasoning: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning_effort(client: openai.AsyncOpenAI, model: str):
    for effort in ["low", "medium", "high"]:
        response = await client.responses.create(
            model=model,
            instructions="Use less than 1024 tokens for reasoning",
            input="Which one is larger as numeric, 9.9 or 9.11?",
            reasoning={"effort": effort})
        check_reponse(response, f"test_reasoning_effort_{effort}: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_chat(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(model=model,
                                             input=[{
                                                 "role":
                                                 "developer",
                                                 "content":
                                                 "Respond in Chinese."
                                             }, {
                                                 "role": "user",
                                                 "content": "Hello!"
                                             }, {
                                                 "role":
                                                 "assistant",
                                                 "content":
                                                 "Hello! How can I help you?"
                                             }, {
                                                 "role": "user",
                                                 "content": "Tell me a joke."
                                             }])
    check_reponse(response, "test_chat: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_multi_turn_chat(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(model=model,
                                             input="What is the answer of 1+1?")
    check_reponse(response, "test_multi_turn_chat_1: ")

    response_2 = await client.responses.create(
        model=model,
        input="What is the answer of previous question?",
        previous_response_id=response.id)
    check_reponse(response_2, "test_multi_turn_chat_2: ")


def get_current_weather(location: str, format: str = "celsius") -> dict:
    return {"sunny": True, "temperature": 20 if format == "celsius" else 68}


@pytest.mark.asyncio(loop_scope="module")
async def test_tool_calls(client: openai.AsyncOpenAI, model: str):
    tool_get_current_weather = {
        "type": "function",
        "name": "get_current_weather",
        "description": "Gets the current weather in the provided location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
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
    messages = [{"role": "user", "content": "What is the weather like in SF?"}]
    response = await client.responses.create(
        model=model,
        input=messages,
        tools=[tool_get_current_weather],
    )
    messages.extend(response.output)
    function_call = check_tool_calling(response, True, "test_tool_calls: ")

    assert function_call.name == "get_current_weather"

    args = json.loads(function_call.arguments)
    answer = get_current_weather(**args)
    messages.append({
        "type": "function_call_output",
        "call_id": function_call.call_id,
        "output": json.dumps(answer),
    })

    response = await client.responses.create(model=model,
                                             input=messages,
                                             tools=[tool_get_current_weather])

    check_tool_calling(response, False, "test_tool_calls: ")


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming(client: openai.AsyncOpenAI, model: str):
    stream = await client.responses.create(
        model=model,
        input="Explain the theory of relativity in brief.",
        stream=True,
    )

    reasoning_deltas, message_deltas = list(), list()
    async for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            message_deltas.append(event.delta)
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            reasoning_deltas.append(event.delta)

    full_response = "".join(message_deltas)
    full_reasoning_response = "".join(reasoning_deltas)
    assert full_response
    assert full_reasoning_response


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_tool_call(client: openai.AsyncOpenAI, model: str):
    tool_get_current_weather = {
        "type": "function",
        "name": "get_current_weather",
        "description": "Gets the current weather in the provided location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
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
    messages = [{"role": "user", "content": "What is the weather like in SF?"}]
    stream = await client.responses.create(
        model=model,
        input=messages,
        tools=[tool_get_current_weather],
        stream=True,
    )

    function_call = None
    reasoning_deltas = list()
    async for event in stream:
        if isinstance(event, ResponseCompletedEvent):
            for output in event.response.output:
                if output.type == "function_call":
                    function_call = output
        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            reasoning_deltas.append(event.delta)

    reasoning = "".join(reasoning_deltas)
    tool_args = json.loads(function_call.arguments)

    assert function_call.name == "get_current_weather", "wrong function calling name"
    assert tool_args, "tool args not exists!"
    assert reasoning, "reasoning not exists!"

    get_current_weather(**tool_args)
